/*
===========================================================================
Copyright (C) 2026 the OpenMoHAA team

This file is part of OpenMoHAA source code.

OpenMoHAA source code is free software; you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the License,
or (at your option) any later version.

OpenMoHAA source code is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OpenMoHAA source code; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
===========================================================================
*/

// vk.c -- Vulkan implementation of the backend interface in vk.h
//
// Frames: the renderer draws into a color target and an offscreen
// depth/stencil image of glConfig.vidWidth x vidHeight. The color target is
// the swapchain image the frame presents, acquired at the first render pass
// of the frame (VK_SelectTarget), or, when the swapchain doesn't have the
// size of the render target (a window resize before the vid_restart) or
// can't be drawn to, an offscreen image blitted to the swapchain image at
// present. Like a GL back buffer, the color target's contents are undefined
// until the frame draws or clears it.
//
// Each frame records into a main command buffer (draws, clears, readbacks
// and uploads that must be ordered with draws) and an upload command buffer
// (see vk_image.c); both are submitted at vk_end_frame, or earlier when the
// CPU needs the results (vk_read_pixels) or the staging buffer is full. A
// frame starts lazily on the first operation that needs a command buffer.
//
// State: the vk_set_* functions only record state. vk_draw turns it into a
// pipeline (vk_pipeline.c) and dynamic state, copies the vertexes and
// indexes into a per-frame buffer and records the draw.

#include "vk_local.h"

#define VK_VERTEX_BUFFER_SIZE	( 4 * 1024 * 1024 )		// per frame, grows when needed
#define VK_STAGING_BUFFER_SIZE	( 32 * 1024 * 1024 )	// per frame

// dynamic state to set before the next draw
#define VK_DIRTY_VIEWPORT		1
#define VK_DIRTY_SCISSOR		2
#define VK_DIRTY_DEPTHBIAS		4
#define VK_DIRTY_LINEWIDTH		8
#define VK_DIRTY_STENCIL		16
#define VK_DIRTY_MVP			32		// vertex stage push constants
#define VK_DIRTY_FOG			64		// fragment stage push constants
#define VK_DIRTY_STIPPLE		128		// line stipple, applied with stippled pipelines only
#define VK_DIRTY_DLIGHT			256		// fragment stage push constants of the dynamic lights
#define VK_DIRTY_ALL			511

vkInstance_t	vk;
cvar_t			*r_vkValidation;

// render state recorded by the vk_set_* functions
typedef struct {
	unsigned long	stateBits;
	texEnv_t		texEnv[NUM_TEXTURE_BUNDLES];
	unsigned int	texture[NUM_TEXTURE_BUNDLES];	// image_t::texnum of the bound images
	cullFace_t		cull;
	qboolean		colorMask[4];
	qboolean		polygonOffset;
	float			offsetFactor, offsetUnits;
	float			depthMin, depthMax;
	int				viewport[4];		// window coordinates, origin at the bottom-left
	int				scissor[4];
	float			projection[16];
	float			modelview[16];
	float			lineWidth;
	qboolean		stencilTest;
	stencilFunc_t	stencilFunc;
	int				stencilRef;
	unsigned int	stencilMask;
	unsigned int	stencilWriteMask;
	stencilOp_t		stencilOp[3];		// stencil fail, depth fail, depth pass
	qboolean		fog;
	vec4_t			fogColor;
	float			fogStart, fogEnd;
	qboolean		clipPlaneEnabled;
	float			clipPlane[4];		// eye space
	int				lineStippleFactor;	// 0: no stipple
	unsigned short	lineStipplePattern;
	int				dlightTmu;			// -1: no dynamic lights (vk_set_lightmap_dlights)
	int				dlightBase;			// first light of the list in the frame's dlight buffer
	unsigned int	dlightMask;
	qboolean		dlightBacks;
} vkState_t;

static vkState_t	vks;
static int			vk_dirty;

// what is bound in the main command buffer being recorded
static struct {
	VkPipeline		pipeline;
	VkDescriptorSet	sets[NUM_TEXTURE_BUNDLES];
	VkBuffer		vertexBuffers[4];
	VkDeviceSize	vertexOffsets[4];
	VkBuffer		indexBuffer;
	VkDeviceSize	indexOffset;
	qboolean		dlightSet;		// set 2, the frame's dlight buffer
} vkb;

// the static geometry (vk_set_static_geometry): one device local buffer
// with the streams one after the other
static struct {
	vkBuffer_t		buffer;
	VkDeviceSize	xyzOffset, stOffset[2], colorOffset, indexOffset;
} vk_static;

// the skin data of the skinned draws (vk_add_skin_data): one device local
// buffer of VK_SKIN_DATA_WORDS words, filled from the start
static struct {
	vkBuffer_t		buffer;
	int				words;
	qboolean		full;		// warned
} vk_skin;

static vkPipelineKey_t	vk_lastKey;
static VkPipeline		vk_lastPipeline;

// where the positions and indexes of the last draw with a version were
// copied (see drawGeometry_t::version); valid while vk_vertexEpoch is the
// same: the vertex buffer of the frame is reused from the start after a
// new frame starts or a mid-frame flush
static struct {
	uint64_t		epoch;
	unsigned int	version;
	const float		*xyz;
	int				xyzStride, numVertexes;
	VkBuffer		buffer;
	VkDeviceSize	xyzOffset;
	const glIndex_t	*indexes;
	int				numIndexes;
	VkDeviceSize	indexOffset;
} vk_lastGeometry;
static uint64_t			vk_vertexEpoch = 1;

// the same for the indexes of the last static draw with a version, and the
// last constant color copied into the vertex buffer
static struct {
	uint64_t		epoch;
	unsigned int	version;
	const glIndex_t	*indexes;
	int				numIndexes;
	VkBuffer		buffer;
	VkDeviceSize	offset;
} vk_lastStaticIndexes;
static struct {
	uint64_t		epoch;
	float			color[4];
	VkBuffer		buffer;
	VkDeviceSize	offset;
} vk_lastConstant;

/*
====================================================================

HELPERS

====================================================================
*/

uint32_t VK_FindMemoryType( uint32_t typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred ) {
	const VkPhysicalDeviceMemoryProperties *props = &vk.memoryProperties;
	uint32_t i;

	for ( i = 0; i < props->memoryTypeCount; i++ ) {
		if ( ( typeBits & ( 1u << i ) ) && ( props->memoryTypes[i].propertyFlags & ( required | preferred ) ) == ( required | preferred ) ) {
			return i;
		}
	}
	for ( i = 0; i < props->memoryTypeCount; i++ ) {
		if ( ( typeBits & ( 1u << i ) ) && ( props->memoryTypes[i].propertyFlags & required ) == required ) {
			return i;
		}
	}

	ri.Error( ERR_FATAL, "Vulkan: no memory type with properties 0x%x", required );
	return 0;
}

/*
===============
VK_AllocateMemory

Every device memory object of the backend comes from here (counted by
vkinfo)
===============
*/
VkDeviceMemory VK_AllocateMemory( VkDeviceSize size, uint32_t memoryType ) {
	VkMemoryAllocateInfo allocInfo;
	VkDeviceMemory memory;
	uint64_t start;

	Com_Memset( &allocInfo, 0, sizeof( allocInfo ) );
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = size;
	allocInfo.memoryTypeIndex = memoryType;

	start = VK_Microseconds();
	VK_CHECK( qvkAllocateMemory( vk.device, &allocInfo, NULL, &memory ) );
	vk.stats.allocationUs += VK_Microseconds() - start;
	vk.stats.allocations++;
	vk.liveAllocations++;

	return memory;
}

void VK_FreeMemory( VkDeviceMemory memory ) {
	if ( memory ) {
		qvkFreeMemory( vk.device, memory, NULL );
		vk.liveAllocations--;
	}
}

void VK_CreateBuffer( vkBuffer_t *buffer, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred ) {
	VkBufferCreateInfo info;
	VkMemoryRequirements memReqs;
	void *mapped;

	Com_Memset( buffer, 0, sizeof( *buffer ) );

	Com_Memset( &info, 0, sizeof( info ) );
	info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	info.size = size;
	info.usage = usage;
	info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK( qvkCreateBuffer( vk.device, &info, NULL, &buffer->buffer ) );

	qvkGetBufferMemoryRequirements( vk.device, buffer->buffer, &memReqs );
	buffer->memory = VK_AllocateMemory( memReqs.size, VK_FindMemoryType( memReqs.memoryTypeBits, required, preferred ) );
	VK_CHECK( qvkBindBufferMemory( vk.device, buffer->buffer, buffer->memory, 0 ) );

	if ( required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) {
		VK_CHECK( qvkMapMemory( vk.device, buffer->memory, 0, VK_WHOLE_SIZE, 0, &mapped ) );
		buffer->mapped = mapped;
	}
	buffer->size = size;
}

void VK_DestroyBuffer( vkBuffer_t *buffer ) {
	if ( buffer->buffer ) {
		qvkDestroyBuffer( vk.device, buffer->buffer, NULL );
	}
	VK_FreeMemory( buffer->memory );
	Com_Memset( buffer, 0, sizeof( *buffer ) );
}

void VK_ImageBarrier( VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
	VkImageLayout oldLayout, VkImageLayout newLayout,
	VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
	VkPipelineStageFlags dstStage, VkAccessFlags dstAccess )
{
	VkImageMemoryBarrier barrier;

	Com_Memset( &barrier, 0, sizeof( barrier ) );
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.srcAccessMask = srcAccess;
	barrier.dstAccessMask = dstAccess;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = aspect;
	barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

	qvkCmdPipelineBarrier( cmd, srcStage, dstStage, 0, 0, NULL, 0, NULL, 1, &barrier );
}

// copies into buffer are done before the host reads it (after a fence)
static void VK_HostReadBarrier( VkCommandBuffer cmd, VkBuffer buffer ) {
	VkBufferMemoryBarrier barrier;

	Com_Memset( &barrier, 0, sizeof( barrier ) );
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = buffer;
	barrier.size = VK_WHOLE_SIZE;
	qvkCmdPipelineBarrier( cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &barrier, 0, NULL );
}

/*
====================================================================

DEFERRED DESTRUCTION

====================================================================
*/

/*
===============
VK_DeferDestroy

Releases a resource once the GPU is done with every command recorded so far
===============
*/
static void VK_DeferDestroyEntry( vkDestroyType_t type, uint64_t handle, uint64_t pool, uint64_t size ) {
	vkFrame_t *frame;

	// the list of a frame is processed when the frame slot is reused, after
	// waiting for its fence; between frames, the last submitted frame is
	// the one that may still use the resource
	if ( vk.frame ) {
		frame = vk.frame;
	} else {
		frame = &vk.frames[( vk.frameIndex + VK_MAX_FRAMES - 1 ) % VK_MAX_FRAMES];
	}

	if ( frame->numDestroy == frame->maxDestroy ) {
		vkDeferredDestroy_t *list;

		frame->maxDestroy = frame->maxDestroy ? frame->maxDestroy * 2 : 1024;
		list = ri.Malloc( sizeof( *list ) * frame->maxDestroy );
		if ( frame->numDestroy ) {
			Com_Memcpy( list, frame->destroyList, sizeof( *list ) * frame->numDestroy );
		}
		if ( frame->destroyList ) {
			ri.Free( frame->destroyList );
		}
		frame->destroyList = list;
	}

	frame->destroyList[frame->numDestroy].type = type;
	frame->destroyList[frame->numDestroy].handle = handle;
	frame->destroyList[frame->numDestroy].pool = pool;
	frame->destroyList[frame->numDestroy].size = size;
	frame->numDestroy++;
}

void VK_DeferDestroy( vkDestroyType_t type, uint64_t handle, uint64_t pool ) {
	if ( handle ) {
		VK_DeferDestroyEntry( type, handle, pool, 0 );
	}
}

// a range of a texture memory block, see vk_image.c
void VK_DeferDestroyRange( int block, VkDeviceSize offset, VkDeviceSize size ) {
	VK_DeferDestroyEntry( VK_DESTROY_MEMORY_RANGE, block, offset, size );
}

static void VK_ProcessDestroyList( vkFrame_t *frame ) {
	int i;

	for ( i = 0; i < frame->numDestroy; i++ ) {
		const vkDeferredDestroy_t *d = &frame->destroyList[i];

		switch ( d->type ) {
		case VK_DESTROY_BUFFER:
			qvkDestroyBuffer( vk.device, (VkBuffer)d->handle, NULL );
			break;
		case VK_DESTROY_IMAGE:
			qvkDestroyImage( vk.device, (VkImage)d->handle, NULL );
			break;
		case VK_DESTROY_IMAGE_VIEW:
			qvkDestroyImageView( vk.device, (VkImageView)d->handle, NULL );
			break;
		case VK_DESTROY_MEMORY:
			VK_FreeMemory( (VkDeviceMemory)d->handle );
			break;
		case VK_DESTROY_MEMORY_RANGE:
			VK_FreeMemoryRange( (int)d->handle, d->pool, d->size );
			break;
		case VK_DESTROY_DESCRIPTOR_SET:
			{
				VkDescriptorSet set = (VkDescriptorSet)d->handle;

				VK_CHECK( qvkFreeDescriptorSets( vk.device, (VkDescriptorPool)d->pool, 1, &set ) );
			}
			break;
		case VK_DESTROY_FRAMEBUFFER:
			qvkDestroyFramebuffer( vk.device, (VkFramebuffer)d->handle, NULL );
			break;
		}
	}
	frame->numDestroy = 0;
}

/*
====================================================================

FRAMES AND COMMAND BUFFERS

====================================================================
*/

/*
===============
VK_DepthTexelValue

Window-space depth of a texel copied from the depth aspect of the depth
target
===============
*/
static float VK_DepthTexelValue( const byte *texel ) {
	float depth;

	if ( vk.depthFormat == VK_FORMAT_D16_UNORM_S8_UINT ) {
		return *(const unsigned short *)texel / 65535.0f;
	}
	if ( vk.depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ) {
		return ( *(const unsigned int *)texel & 0xFFFFFF ) / 16777215.0f;
	}
	Com_Memcpy( &depth, texel, sizeof( depth ) );
	return depth;
}

/*
===============
VK_CollectDepthReads

Called once the fence of a frame slot has been waited for: keeps the
results of the depth reads the slot's last frame recorded
===============
*/
static void VK_CollectDepthReads( vkFrame_t *frame ) {
	int i;

	for ( i = 0; i < frame->numDepthReads; i++ ) {
		if ( frame->depthReadInside[i] ) {
			frame->depthResults[i] = VK_DepthTexelValue( frame->depthReadBuffer.mapped + i * 4 );
		} else {
			frame->depthResults[i] = 0.0f;
		}
	}
	frame->numDepthResults = frame->numDepthReads;
	frame->resultSerial = frame->serial;
	frame->numDepthReads = 0;
}

/*
===============
VK_CollectTimestamps

Called once the fence of a frame slot has been waited for: adds the GPU
times of the slot's last frame to the vkinfo counters
===============
*/
static void VK_CollectTimestamps( vkFrame_t *frame ) {
	const uint32_t all = ( 1u << VK_NUM_TIMESTAMPS ) - 1;
	uint64_t ts[VK_NUM_TIMESTAMPS], mask;
	double scale;
	int i;

	if ( frame->timestampsWritten != all ) {
		frame->timestampsWritten = 0;
		return;
	}
	frame->timestampsWritten = 0;
	if ( qvkGetQueryPoolResults( vk.device, frame->queryPool, 0, VK_NUM_TIMESTAMPS, sizeof( ts ), ts, sizeof( ts[0] ),
		VK_QUERY_RESULT_64_BIT ) != VK_SUCCESS ) {
		return;
	}

	mask = vk.timestampValidBits >= 64 ? ~0ull : ( 1ull << vk.timestampValidBits ) - 1;
	for ( i = 0; i < VK_NUM_TIMESTAMPS; i++ ) {
		ts[i] &= mask;
	}
	// nanoseconds per tick, to microseconds
	scale = vk.deviceProperties.limits.timestampPeriod / 1000.0;
	vk.stats.gpuMainUs += ( ( ts[VK_TS_MAIN_END] - ts[VK_TS_MAIN_START] ) & mask ) * scale;
	vk.stats.gpuPresentUs += ( ( ts[VK_TS_PRESENT_END] - ts[VK_TS_PRESENT_START] ) & mask ) * scale;
	vk.stats.gpuFrameUs += ( ( ts[VK_TS_PRESENT_END] - ts[VK_TS_MAIN_START] ) & mask ) * scale;
	vk.stats.gpuFrames++;
}

static void VK_WriteTimestamp( VkCommandBuffer cmd, VkPipelineStageFlagBits stage, int query ) {
	if ( vk.frame->queryPool ) {
		qvkCmdWriteTimestamp( cmd, stage, vk.frame->queryPool, query );
		vk.frame->timestampsWritten |= 1u << query;
	}
}

static void VK_BeginMainCommandBuffer( void ) {
	VkCommandBufferBeginInfo info;

	Com_Memset( &info, 0, sizeof( info ) );
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VK_CHECK( qvkBeginCommandBuffer( vk.frame->mainCmd, &info ) );

	// nothing is bound in a new command buffer
	Com_Memset( &vkb, 0, sizeof( vkb ) );
	vk_dirty = VK_DIRTY_ALL;
}

static void VK_UpdateDepthTarget( void );

void VK_BeginFrameIfNeeded( void ) {
	vkFrame_t *frame;
	uint64_t start;

	if ( vk.frame ) {
		return;
	}

	frame = &vk.frames[vk.frameIndex];

	// the last frame that used this slot must be done
	start = VK_Microseconds();
	VK_CHECK( qvkWaitForFences( vk.device, 1, &frame->fence, VK_TRUE, UINT64_MAX ) );
	vk.stats.waitUs += VK_Microseconds() - start;
	VK_ProcessDestroyList( frame );
	VK_CollectDepthReads( frame );
	VK_CollectTimestamps( frame );

	frame->vertexOffset = 0;
	frame->stagingOffset = 0;
	frame->numDlights = 0;
	frame->skinWords = 0;
	frame->serial = ++vk.frameSerial;
	vk_vertexEpoch++;
	vk.frameStartUs = VK_Microseconds();

	vk.frame = frame;
	vk.uploadCmdRecording = qfalse;
	vk.renderPassActive = qfalse;

	VK_BeginMainCommandBuffer();
	if ( frame->queryPool ) {
		qvkCmdResetQueryPool( frame->mainCmd, frame->queryPool, 0, VK_NUM_TIMESTAMPS );
		VK_WriteTimestamp( frame->mainCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_TS_MAIN_START );
	}

	// before anything of the frame uses the depth image
	VK_UpdateDepthTarget();
}

VkCommandBuffer VK_GetUploadCommandBuffer( void ) {
	VK_BeginFrameIfNeeded();

	if ( !vk.uploadCmdRecording ) {
		VkCommandBufferBeginInfo info;

		Com_Memset( &info, 0, sizeof( info ) );
		info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK( qvkBeginCommandBuffer( vk.frame->uploadCmd, &info ) );
		vk.uploadCmdRecording = qtrue;
	}

	return vk.frame->uploadCmd;
}

/*
===============
VK_AcquireNextImage

Acquires the next swapchain image unless one is held already. The
semaphores come from a ring of VK_ACQUIRE_SEMAPHORES: an image can be
acquired for the next frame right after a present (vk_end_frame), before
that frame starts, and a semaphore can only be signaled again once the
submission that waited for it has completed; with one more semaphore than
frames in flight that has always happened (the frame slot was waited for).
===============
*/
static VkResult VK_AcquireNextImage( uint64_t *us ) {
	int s = vk.nextAcquireSemaphore;
	VkResult result;
	uint64_t start;
	int i;

	if ( vk.heldImage ) {
		return VK_SUCCESS;
	}

	// normally done: see above
	for ( i = 0; i < VK_MAX_FRAMES && vk.acquireWaitSerial[s]; i++ ) {
		if ( vk.frames[i].serial == vk.acquireWaitSerial[s] && &vk.frames[i] != vk.frame ) {
			VK_CHECK( qvkWaitForFences( vk.device, 1, &vk.frames[i].fence, VK_TRUE, UINT64_MAX ) );
		}
	}

	start = VK_Microseconds();
	result = qvkAcquireNextImageKHR( vk.device, vk.swapchain, UINT64_MAX, vk.acquireSemaphores[s], VK_NULL_HANDLE, &vk.heldIndex );
	*us += VK_Microseconds() - start;
	if ( result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR ) {
		vk.heldImage = qtrue;
		vk.heldSemaphore = s;
		vk.acquireWaitSerial[s] = 0;
		vk.nextAcquireSemaphore = ( s + 1 ) % VK_ACQUIRE_SEMAPHORES;
	}
	if ( result == VK_SUBOPTIMAL_KHR ) {
		// still presentable, recreate it next frame
		vk.swapchainOutOfDate = qtrue;
		result = VK_SUCCESS;
	}
	return result;
}

/*
===============
VK_ReleaseHeldImage

Before the swapchain goes away with an image acquired but not presented:
a submission without commands waits for the image's semaphore, which
would otherwise stay signaled
===============
*/
void VK_ReleaseHeldImage( void ) {
	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkSubmitInfo submit;

	if ( !vk.heldImage ) {
		return;
	}

	Com_Memset( &submit, 0, sizeof( submit ) );
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &vk.acquireSemaphores[vk.heldSemaphore];
	submit.pWaitDstStageMask = &waitStage;
	VK_CHECK( qvkQueueSubmit( vk.queue, 1, &submit, VK_NULL_HANDLE ) );
	VK_CHECK( qvkQueueWaitIdle( vk.queue ) );

	vk.heldImage = qfalse;
	vk.acquireWaitPending = qfalse;
}

/*
===============
VK_WaitForHeldImage

The submission that uses the held image waits for its semaphore
===============
*/
static void VK_WaitForHeldImage( VkSubmitInfo *submit, VkPipelineStageFlags *waitStage, VkPipelineStageFlags stage ) {
	*waitStage = stage;
	submit->waitSemaphoreCount = 1;
	submit->pWaitSemaphores = &vk.acquireSemaphores[vk.heldSemaphore];
	submit->pWaitDstStageMask = waitStage;
	vk.acquireWaitSerial[vk.heldSemaphore] = vk.frame->serial;
}

/*
===============
VK_SwapchainFitsTarget

The swapchain images can be the color target: they have its size and
format and can be drawn to and read back
===============
*/
static qboolean VK_SwapchainFitsTarget( void ) {
	return vk.swapchain && vk.swapchainDirect && !vk.swapchainOutOfDate
		&& (int)vk.swapchainExtent.width == vk.width && (int)vk.swapchainExtent.height == vk.height;
}

/*
===============
VK_SelectTarget

Chooses the color target of the frame, at its first render pass (or
readback, or present): the next swapchain image when it fits, acquired now
with the frame's semaphore, otherwise the offscreen image. Called outside
of a render pass.
===============
*/
static void VK_SelectTarget( void ) {
	vkFrame_t *frame = vk.frame;
	uint32_t index;
	VkResult result;
	int width, height;

	if ( vk.target != VK_TARGET_NONE ) {
		return;
	}

	vk.target = VK_TARGET_OFFSCREEN;
	vk.targetImage = vk.colorImage;
	vk.targetFramebuffer = vk.framebuffer;

	// recreate the swapchain when the window changed; there is none while
	// the window has no area (minimized)
	VK_GetDrawableSize( &width, &height );
	if ( !vk.swapchain || vk.swapchainOutOfDate || width != vk.swapchainDrawableWidth || height != vk.swapchainDrawableHeight ) {
		VK_CreateSwapchain();
	}
	if ( !VK_SwapchainFitsTarget() ) {
		return;
	}

	// usually acquired right after the previous present
	result = VK_AcquireNextImage( &vk.stats.acquireUs );
	if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
		// drawn offscreen; VK_Present recreates the swapchain
		vk.swapchainOutOfDate = qtrue;
		return;
	}
	VK_CHECK( result );
	index = vk.heldIndex;

	vk.target = VK_TARGET_SWAPCHAIN;
	vk.targetIndex = index;
	vk.targetImage = vk.swapchainImages[index];
	vk.targetFramebuffer = vk.swapchainFramebuffers[index];
	vk.acquireWaitPending = qtrue;

	// the previous contents don't matter, as with a GL back buffer
	VK_ImageBarrier( frame->mainCmd, vk.targetImage, VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT );
}

static void VK_BeginRenderPass( void ) {
	VkRenderPassBeginInfo info;

	VK_BeginFrameIfNeeded();
	if ( vk.renderPassActive ) {
		return;
	}

	VK_SelectTarget();

	// textures uploaded in order with the draws go back to the sampled layout
	VK_FinishUploads( vk.frame->mainCmd, qfalse );

	Com_Memset( &info, 0, sizeof( info ) );
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	info.renderPass = vk.renderPass;
	info.framebuffer = vk.targetFramebuffer;
	info.renderArea.extent.width = vk.width;
	info.renderArea.extent.height = vk.height;
	qvkCmdBeginRenderPass( vk.frame->mainCmd, &info, VK_SUBPASS_CONTENTS_INLINE );

	vk.renderPassActive = qtrue;
	vk.stats.renderPasses++;
}

void VK_EndRenderPass( void ) {
	if ( vk.renderPassActive ) {
		qvkCmdEndRenderPass( vk.frame->mainCmd );
		vk.renderPassActive = qfalse;
	}
}

/*
===============
VK_SubmitFrameCommands

Submits what the current frame has recorded so far: the upload command
buffer, then the main one. The first submission after the swapchain image
was acquired waits for it; at the end of a frame drawn into the swapchain
image (endOfFrame) the image goes to the presentation layout and signal is
signaled for the present.
===============
*/
static void VK_SubmitFrameCommands( VkFence fence, VkSemaphore signal, qboolean endOfFrame ) {
	vkFrame_t *frame = vk.frame;
	VkCommandBuffer cmds[2];
	VkPipelineStageFlags waitStage;
	VkSubmitInfo submit;
	uint32_t numCmds;
	uint64_t start;

	numCmds = 0;
	if ( vk.uploadCmdRecording ) {
		VK_FinishUploads( frame->uploadCmd, qtrue );
		VK_CHECK( qvkEndCommandBuffer( frame->uploadCmd ) );
		cmds[numCmds++] = frame->uploadCmd;
		vk.uploadCmdRecording = qfalse;
	}

	VK_EndRenderPass();
	VK_FinishUploads( frame->mainCmd, qfalse );
	if ( endOfFrame ) {
		VK_WriteTimestamp( frame->mainCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_TS_MAIN_END );
		if ( vk.target == VK_TARGET_SWAPCHAIN ) {
			VK_ImageBarrier( frame->mainCmd, vk.targetImage, VK_IMAGE_ASPECT_COLOR_BIT,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0 );
			// no copy to time
			VK_WriteTimestamp( frame->mainCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_TS_PRESENT_START );
			VK_WriteTimestamp( frame->mainCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_TS_PRESENT_END );
		}
	}
	VK_CHECK( qvkEndCommandBuffer( frame->mainCmd ) );
	cmds[numCmds++] = frame->mainCmd;

	Com_Memset( &submit, 0, sizeof( submit ) );
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = numCmds;
	submit.pCommandBuffers = cmds;
	if ( vk.acquireWaitPending ) {
		// the image layout transition of VK_SelectTarget waits at the same stage
		VK_WaitForHeldImage( &submit, &waitStage, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT );
		vk.acquireWaitPending = qfalse;
	}
	if ( signal ) {
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &signal;
	}

	if ( fence ) {
		VK_CHECK( qvkResetFences( vk.device, 1, &fence ) );
		vk.lastSubmitFence = fence;
	}
	start = VK_Microseconds();
	VK_CHECK( qvkQueueSubmit( vk.queue, 1, &submit, fence ) );
	vk.stats.submitUs += VK_Microseconds() - start;
	vk.stats.submits++;

	// draws recorded from now on are in a new segment (see vk_image.c)
	vk.segment++;
}

/*
===============
VK_FlushAndWait

Submits the frame so far, waits for the GPU and continues the frame in a
new command buffer
===============
*/
void VK_FlushAndWait( void ) {
	uint64_t start;
	int i;

	if ( !vk.frame ) {
		return;
	}

	VK_SubmitFrameCommands( vk.frame->fence, VK_NULL_HANDLE, qfalse );
	start = VK_Microseconds();
	VK_CHECK( qvkWaitForFences( vk.device, 1, &vk.frame->fence, VK_TRUE, UINT64_MAX ) );
	vk.stats.waitUs += VK_Microseconds() - start;
	vk.stats.flushes++;

	// the GPU is idle: everything waiting for destruction can go, and the
	// buffers of the frame can be reused
	for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
		VK_ProcessDestroyList( &vk.frames[i] );
	}
	vk.frame->vertexOffset = 0;
	vk.frame->stagingOffset = 0;
	vk_vertexEpoch++;

	VK_BeginMainCommandBuffer();
}

/*
===============
VK_QueuePresent

Presents the held swapchain image once its present semaphore is signaled
===============
*/
static void VK_QueuePresent( uint32_t imageIndex ) {
	VkPresentInfoKHR presentInfo;
	VkResult result;
	uint64_t start;

	Com_Memset( &presentInfo, 0, sizeof( presentInfo ) );
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &vk.presentSemaphores[imageIndex];
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &vk.swapchain;
	presentInfo.pImageIndices = &imageIndex;

	// Hold presents at least one com_maxfps period apart (same hook as the GL swap).
	if ( ri.PresentGuard ) {
		ri.PresentGuard();
	}

	start = VK_Microseconds();
	vk.stats.frameToPresentUs += start - vk.frameStartUs;
	result = qvkQueuePresentKHR( vk.queue, &presentInfo );
	vk.stats.presentUs += VK_Microseconds() - start;
	vk.heldImage = qfalse;
	if ( result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ) {
		vk.swapchainOutOfDate = qtrue;
	} else {
		VK_CHECK( result );
	}
}

/*
===============
VK_Present

Frames drawn offscreen: copies the frame to a swapchain image and presents
it; returns qfalse when there is nothing to present to (minimized window),
the frame fence is then not submitted
===============
*/
static qboolean VK_Present( vkFrame_t *frame ) {
	VkCommandBufferBeginInfo beginInfo;
	VkPipelineStageFlags waitStage;
	VkSubmitInfo submit;
	VkImageBlit blit;
	VkCommandBuffer cmd;
	VkImage image;
	uint32_t imageIndex;
	VkResult result;
	uint64_t start;
	int width, height;

	// recreate the swapchain when the window changed; there is none while
	// the window has no area (minimized)
	VK_GetDrawableSize( &width, &height );
	if ( !vk.swapchain || vk.swapchainOutOfDate || width != vk.swapchainDrawableWidth || height != vk.swapchainDrawableHeight ) {
		VK_CreateSwapchain();
	}
	if ( !vk.swapchain ) {
		return qfalse;
	}

	result = VK_AcquireNextImage( &vk.stats.acquireUs );
	if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
		VK_CreateSwapchain();
		if ( !vk.swapchain ) {
			return qfalse;
		}
		result = VK_AcquireNextImage( &vk.stats.acquireUs );
	}
	if ( result == VK_ERROR_OUT_OF_DATE_KHR ) {
		vk.swapchainOutOfDate = qtrue;
		return qfalse;
	}
	VK_CHECK( result );
	imageIndex = vk.heldIndex;

	image = vk.swapchainImages[imageIndex];
	cmd = frame->presentCmd;

	Com_Memset( &beginInfo, 0, sizeof( beginInfo ) );
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VK_CHECK( qvkBeginCommandBuffer( cmd, &beginInfo ) );
	VK_WriteTimestamp( cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_TS_PRESENT_START );

	VK_ImageBarrier( cmd, vk.colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
	VK_ImageBarrier( cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT );

	// the window may have been resized since the last vid_restart: scale
	Com_Memset( &blit, 0, sizeof( blit ) );
	blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.srcSubresource.layerCount = 1;
	blit.srcOffsets[1].x = vk.width;
	blit.srcOffsets[1].y = vk.height;
	blit.srcOffsets[1].z = 1;
	blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	blit.dstSubresource.layerCount = 1;
	blit.dstOffsets[1].x = vk.swapchainExtent.width;
	blit.dstOffsets[1].y = vk.swapchainExtent.height;
	blit.dstOffsets[1].z = 1;
	qvkCmdBlitImage( cmd, vk.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &blit, ( vk.width == (int)vk.swapchainExtent.width && vk.height == (int)vk.swapchainExtent.height ) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR );

	VK_ImageBarrier( cmd, image, VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0 );
	VK_ImageBarrier( cmd, vk.colorImage, VK_IMAGE_ASPECT_COLOR_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT );

	VK_WriteTimestamp( cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_TS_PRESENT_END );
	VK_CHECK( qvkEndCommandBuffer( cmd ) );

	Com_Memset( &submit, 0, sizeof( submit ) );
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	VK_WaitForHeldImage( &submit, &waitStage, VK_PIPELINE_STAGE_TRANSFER_BIT );
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &vk.presentSemaphores[imageIndex];

	VK_CHECK( qvkResetFences( vk.device, 1, &frame->fence ) );
	start = VK_Microseconds();
	VK_CHECK( qvkQueueSubmit( vk.queue, 1, &submit, frame->fence ) );
	vk.stats.submitUs += VK_Microseconds() - start;
	vk.stats.submits++;
	vk.lastSubmitFence = frame->fence;

	VK_QueuePresent( imageIndex );
	return qtrue;
}

static void VK_CreateFrames( void ) {
	VkCommandPoolCreateInfo poolInfo;
	VkCommandBufferAllocateInfo allocInfo;
	VkFenceCreateInfo fenceInfo;
	VkSemaphoreCreateInfo semaphoreInfo;
	VkCommandBuffer cmds[3];
	int i;

	Com_Memset( &semaphoreInfo, 0, sizeof( semaphoreInfo ) );
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
		vkFrame_t *frame = &vk.frames[i];

		Com_Memset( &poolInfo, 0, sizeof( poolInfo ) );
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = vk.queueFamily;
		VK_CHECK( qvkCreateCommandPool( vk.device, &poolInfo, NULL, &frame->commandPool ) );

		Com_Memset( &allocInfo, 0, sizeof( allocInfo ) );
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = frame->commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = ARRAY_LEN( cmds );
		VK_CHECK( qvkAllocateCommandBuffers( vk.device, &allocInfo, cmds ) );
		frame->uploadCmd = cmds[0];
		frame->mainCmd = cmds[1];
		frame->presentCmd = cmds[2];

		// signaled: the first wait for it returns at once
		Com_Memset( &fenceInfo, 0, sizeof( fenceInfo ) );
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		VK_CHECK( qvkCreateFence( vk.device, &fenceInfo, NULL, &frame->fence ) );


		if ( vk.timestampValidBits ) {
			VkQueryPoolCreateInfo queryInfo;

			Com_Memset( &queryInfo, 0, sizeof( queryInfo ) );
			queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
			queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
			queryInfo.queryCount = VK_NUM_TIMESTAMPS;
			VK_CHECK( qvkCreateQueryPool( vk.device, &queryInfo, NULL, &frame->queryPool ) );
		}

		VK_CreateBuffer( &frame->vertexBuffer, VK_VERTEX_BUFFER_SIZE,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
		VK_CreateBuffer( &frame->stagingBuffer, VK_STAGING_BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0 );
		VK_CreateBuffer( &frame->depthReadBuffer, VK_MAX_DEPTH_READS * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT );
		VK_CreateBuffer( &frame->dlightBuffer, VK_MAX_FRAME_DLIGHTS * sizeof( vkDlightGPU_t ), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
		VK_CreateBuffer( &frame->skinBuffer, VK_SKIN_FRAME_WORDS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
	}

	VK_CreateBuffer( &vk_skin.buffer, VK_SKIN_DATA_WORDS * 4,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 );
	vk_skin.words = 0;
	vk_skin.full = qfalse;

	for ( i = 0; i < VK_MAX_SWAPCHAIN_IMAGES; i++ ) {
		VK_CHECK( qvkCreateSemaphore( vk.device, &semaphoreInfo, NULL, &vk.presentSemaphores[i] ) );
	}
	for ( i = 0; i < VK_ACQUIRE_SEMAPHORES; i++ ) {
		VK_CHECK( qvkCreateSemaphore( vk.device, &semaphoreInfo, NULL, &vk.acquireSemaphores[i] ) );
		vk.acquireWaitSerial[i] = 0;
	}
	vk.nextAcquireSemaphore = 0;
	vk.heldImage = qfalse;

	vk.frameIndex = 0;
	vk.frame = NULL;
}

/*
===============
VK_CreateDlightSets

The descriptor sets of the frames' dlight buffers, skin data and frame data
of the skinned draws (set 2 of the shaders), from vk.dlightPool
(VK_InitPipelines), which frees them
===============
*/
static void VK_CreateDlightSets( void ) {
	VkDescriptorSetAllocateInfo allocInfo;
	VkDescriptorBufferInfo bufferInfo[3];
	VkWriteDescriptorSet writes[3];
	int i, j;

	for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
		vkFrame_t *frame = &vk.frames[i];

		Com_Memset( &allocInfo, 0, sizeof( allocInfo ) );
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = vk.dlightPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &vk.dlightSetLayout;
		VK_CHECK( qvkAllocateDescriptorSets( vk.device, &allocInfo, &frame->dlightSet ) );

		bufferInfo[0].buffer = frame->dlightBuffer.buffer;
		bufferInfo[1].buffer = vk_skin.buffer.buffer;
		bufferInfo[2].buffer = frame->skinBuffer.buffer;
		// one write per binding: their stages differ
		Com_Memset( writes, 0, sizeof( writes ) );
		for ( j = 0; j < 3; j++ ) {
			bufferInfo[j].offset = 0;
			bufferInfo[j].range = VK_WHOLE_SIZE;
			writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[j].dstSet = frame->dlightSet;
			writes[j].dstBinding = j;
			writes[j].descriptorCount = 1;
			writes[j].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[j].pBufferInfo = &bufferInfo[j];
		}
		qvkUpdateDescriptorSets( vk.device, 3, writes, 0, NULL );
	}
}

static void VK_DestroyFrames( void ) {
	int i;

	for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
		vkFrame_t *frame = &vk.frames[i];

		VK_ProcessDestroyList( frame );
		if ( frame->destroyList ) {
			ri.Free( frame->destroyList );
		}
		VK_DestroyBuffer( &frame->vertexBuffer );
		VK_DestroyBuffer( &frame->stagingBuffer );
		VK_DestroyBuffer( &frame->depthReadBuffer );
		VK_DestroyBuffer( &frame->dlightBuffer );
		VK_DestroyBuffer( &frame->skinBuffer );
		if ( frame->queryPool ) {
			qvkDestroyQueryPool( vk.device, frame->queryPool, NULL );
		}
		if ( frame->fence ) {
			qvkDestroyFence( vk.device, frame->fence, NULL );
		}
		if ( frame->commandPool ) {
			qvkDestroyCommandPool( vk.device, frame->commandPool, NULL );
		}
		Com_Memset( frame, 0, sizeof( *frame ) );
	}

	for ( i = 0; i < VK_MAX_SWAPCHAIN_IMAGES; i++ ) {
		if ( vk.presentSemaphores[i] ) {
			qvkDestroySemaphore( vk.device, vk.presentSemaphores[i], NULL );
			vk.presentSemaphores[i] = VK_NULL_HANDLE;
		}
	}
	for ( i = 0; i < VK_ACQUIRE_SEMAPHORES; i++ ) {
		if ( vk.acquireSemaphores[i] ) {
			qvkDestroySemaphore( vk.device, vk.acquireSemaphores[i], NULL );
			vk.acquireSemaphores[i] = VK_NULL_HANDLE;
		}
	}

	vk.frame = NULL;
}

/*
====================================================================

RENDER TARGETS

====================================================================
*/

static qboolean VK_FormatHasFeatures( VkFormat format, VkFormatFeatureFlags features ) {
	VkFormatProperties props;

	qvkGetPhysicalDeviceFormatProperties( vk.physicalDevice, format, &props );
	return ( props.optimalTilingFeatures & features ) == features;
}

#define VK_TARGET_FORMAT_FEATURES ( VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT \
	| VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT )

static void VK_CreateTargetImage( VkFormat format, VkSampleCountFlagBits samples, VkImageUsageFlags usage, VkImageAspectFlags aspect,
	VkImage *image, VkDeviceMemory *memory, VkImageView *view )
{
	VkImageCreateInfo imageInfo;
	VkImageViewCreateInfo viewInfo;
	VkMemoryRequirements memReqs;

	Com_Memset( &imageInfo, 0, sizeof( imageInfo ) );
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = format;
	imageInfo.extent.width = vk.width;
	imageInfo.extent.height = vk.height;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = samples;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VK_CHECK( qvkCreateImage( vk.device, &imageInfo, NULL, image ) );

	qvkGetImageMemoryRequirements( vk.device, *image, &memReqs );
	*memory = VK_AllocateMemory( memReqs.size, VK_FindMemoryType( memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 ) );
	VK_CHECK( qvkBindImageMemory( vk.device, *image, *memory, 0 ) );

	Com_Memset( &viewInfo, 0, sizeof( viewInfo ) );
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = *image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = aspect;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;
	VK_CHECK( qvkCreateImageView( vk.device, &viewInfo, NULL, view ) );
}

static int VK_DepthBits( VkFormat format ) {
	switch ( format ) {
	case VK_FORMAT_D16_UNORM_S8_UINT: return 16;
	case VK_FORMAT_D24_UNORM_S8_UINT: return 24;
	default: return 32;
	}
}

/*
===============
VK_CreateRenderPass

The render pass without multisampling: the color and depth/stencil
attachments keep their contents across render passes (a render pass ends
for every readback and every upload ordered with draws)
===============
*/
static void VK_CreateRenderPass( VkPipelineStageFlags attachmentStages, VkAccessFlags attachmentAccess ) {
	VkAttachmentDescription attachments[2];
	VkAttachmentReference colorRef, depthRef;
	VkSubpassDescription subpass;
	VkSubpassDependency dependencies[2];
	VkRenderPassCreateInfo renderPassInfo;

	Com_Memset( attachments, 0, sizeof( attachments ) );
	attachments[0].format = vk.colorFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachments[1].format = vk.depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	colorRef.attachment = 0;
	colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	depthRef.attachment = 1;
	depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	Com_Memset( &subpass, 0, sizeof( subpass ) );
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorRef;
	subpass.pDepthStencilAttachment = &depthRef;

	Com_Memset( dependencies, 0, sizeof( dependencies ) );
	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = attachmentStages;
	dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[0].dstStageMask = attachmentStages;
	dependencies[0].dstAccessMask = attachmentAccess;
	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = attachmentStages;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstStageMask = attachmentStages;
	dependencies[1].dstAccessMask = attachmentAccess;

	Com_Memset( &renderPassInfo, 0, sizeof( renderPassInfo ) );
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = ARRAY_LEN( attachments );
	renderPassInfo.pAttachments = attachments;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = ARRAY_LEN( dependencies );
	renderPassInfo.pDependencies = dependencies;
	VK_CHECK( qvkCreateRenderPass( vk.device, &renderPassInfo, NULL, &vk.renderPass ) );
}

/*
===============
VK_ClearTargetImage

Initial contents of a render target image, in the frame being recorded:
black with alpha 1, or depth 1 and stencil 0
===============
*/
static void VK_ClearTargetImage( VkImage image, qboolean color ) {
	VkCommandBuffer cmd = vk.frame->mainCmd;
	VkImageSubresourceRange range;
	VkClearColorValue clearColor;
	VkClearDepthStencilValue clearDepth;
	VkImageAspectFlags aspect;

	aspect = color ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
	Com_Memset( &range, 0, sizeof( range ) );
	range.aspectMask = aspect;
	range.levelCount = 1;
	range.layerCount = 1;

	VK_ImageBarrier( cmd, image, aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT );
	if ( color ) {
		Com_Memset( &clearColor, 0, sizeof( clearColor ) );
		clearColor.float32[3] = 1.0f;
		qvkCmdClearColorImage( cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range );
		VK_ImageBarrier( cmd, image, aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT );
	} else {
		clearDepth.depth = 1.0f;
		clearDepth.stencil = 0;
		qvkCmdClearDepthStencilImage( cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, 1, &range );
		// the depth/stencil resolve of multisampling writes as a color attachment
		VK_ImageBarrier( cmd, image, aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT );
	}
}

/*
===============
VK_SampleCount

The sample count for r_ext_multisample: the largest the device supports up
to the value asked for
===============
*/
static VkSampleCountFlagBits VK_SampleCount( void ) {
	const VkPhysicalDeviceLimits *limits = &vk.deviceProperties.limits;
	VkSampleCountFlags supported;
	int wanted, samples;

	wanted = r_ext_multisample->integer;
	if ( wanted <= 1 ) {
		return VK_SAMPLE_COUNT_1_BIT;
	}
	if ( !vk.depthStencilResolve ) {
		ri.Printf( PRINT_ALL, S_COLOR_YELLOW "WARNING: r_ext_multisample: the device has no %s, no multisampling\n",
			VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME );
		return VK_SAMPLE_COUNT_1_BIT;
	}

	supported = limits->framebufferColorSampleCounts & limits->framebufferDepthSampleCounts & limits->framebufferStencilSampleCounts;
	for ( samples = VK_SAMPLE_COUNT_64_BIT; samples > 1; samples >>= 1 ) {
		if ( samples <= wanted && ( supported & samples ) ) {
			break;
		}
	}
	if ( samples != wanted ) {
		ri.Printf( PRINT_ALL, "r_ext_multisample %d: %d samples\n", wanted, samples );
	}
	return (VkSampleCountFlagBits)samples;
}

/*
===============
VK_CreateMultisampleRenderPasses

With multisampling the draws go to the multisampled color and depth/stencil
attachments, which keep their contents like the single-sampled ones, and
the end of every render pass instance resolves the color into the color
target (average of the samples), which the copy to the swapchain and the
color readbacks use. The depth/stencil buffer is only resolved (sample 0)
into the depth image when it is read back (VK_ResolveDepth), by a render
pass of its own: resolving it with every render pass cost 1.3 ms per frame
at 1920x1080 with 4 samples on a Radeon 780M.
===============
*/
static void VK_MultisampleAttachment( VkAttachmentDescription2 *attachment, VkAttachmentReference2 *ref, int index,
	qboolean color, qboolean multisampled )
{
	Com_Memset( attachment, 0, sizeof( *attachment ) );
	attachment->sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
	attachment->format = color ? vk.colorFormat : vk.depthFormat;
	attachment->samples = multisampled ? vk.samples : VK_SAMPLE_COUNT_1_BIT;
	// the resolve attachments are written over the whole render area
	attachment->loadOp = multisampled ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachment->storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachment->stencilLoadOp = color ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : attachment->loadOp;
	attachment->stencilStoreOp = color ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
	attachment->initialLayout = color ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachment->finalLayout = attachment->initialLayout;

	Com_Memset( ref, 0, sizeof( *ref ) );
	ref->sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
	ref->attachment = index;
	ref->layout = attachment->initialLayout;
	ref->aspectMask = color ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
}

static void VK_CreateMultisampleRenderPasses( VkPipelineStageFlags attachmentStages, VkAccessFlags attachmentAccess ) {
	VkAttachmentDescription2 attachments[3];
	VkAttachmentReference2 refs[3];
	VkSubpassDescriptionDepthStencilResolve depthResolve;
	VkSubpassDescription2 subpass;
	VkSubpassDependency2 dependencies[2];
	VkRenderPassCreateInfo2 info;
	int i;

	// as in the single-sampled render pass; resolves are color attachment
	// writes, for the depth/stencil one too
	Com_Memset( dependencies, 0, sizeof( dependencies ) );
	for ( i = 0; i < 2; i++ ) {
		dependencies[i].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
		dependencies[i].srcSubpass = i ? 0 : VK_SUBPASS_EXTERNAL;
		dependencies[i].dstSubpass = i ? VK_SUBPASS_EXTERNAL : 0;
		dependencies[i].srcStageMask = attachmentStages;
		dependencies[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[i].dstStageMask = attachmentStages;
		dependencies[i].dstAccessMask = attachmentAccess;
	}

	Com_Memset( &info, 0, sizeof( info ) );
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
	info.pAttachments = attachments;
	info.subpassCount = 1;
	info.pSubpasses = &subpass;
	info.dependencyCount = ARRAY_LEN( dependencies );
	info.pDependencies = dependencies;

	// drawing: multisampled color (0) and depth/stencil (1), color resolve (2)
	VK_MultisampleAttachment( &attachments[0], &refs[0], 0, qtrue, qtrue );
	VK_MultisampleAttachment( &attachments[1], &refs[1], 1, qfalse, qtrue );
	VK_MultisampleAttachment( &attachments[2], &refs[2], 2, qtrue, qfalse );
	Com_Memset( &subpass, 0, sizeof( subpass ) );
	subpass.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &refs[0];
	subpass.pResolveAttachments = &refs[2];
	subpass.pDepthStencilAttachment = &refs[1];
	info.attachmentCount = 3;
	VK_CHECK( qvkCreateRenderPass2KHR( vk.device, &info, NULL, &vk.renderPass ) );

	// depth/stencil resolve: multisampled depth/stencil (0) into the depth image (1)
	VK_MultisampleAttachment( &attachments[0], &refs[0], 0, qfalse, qtrue );
	VK_MultisampleAttachment( &attachments[1], &refs[1], 1, qfalse, qfalse );
	Com_Memset( &depthResolve, 0, sizeof( depthResolve ) );
	depthResolve.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE;
	depthResolve.depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;		// always supported
	depthResolve.stencilResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
	depthResolve.pDepthStencilResolveAttachment = &refs[1];
	Com_Memset( &subpass, 0, sizeof( subpass ) );
	subpass.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
	subpass.pNext = &depthResolve;
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.pDepthStencilAttachment = &refs[0];
	info.attachmentCount = 2;
	VK_CHECK( qvkCreateRenderPass2KHR( vk.device, &info, NULL, &vk.depthResolvePass ) );
}

/*
===============
VK_ResolveDepth

With multisampling: resolves the depth/stencil buffer into the depth image
before it is read back; called outside of a render pass
===============
*/
static void VK_ResolveDepth( void ) {
	VkRenderPassBeginInfo info;

	if ( vk.samples == VK_SAMPLE_COUNT_1_BIT ) {
		return;
	}

	Com_Memset( &info, 0, sizeof( info ) );
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	info.renderPass = vk.depthResolvePass;
	info.framebuffer = vk.depthResolveFramebuffer;
	info.renderArea.extent.width = vk.width;
	info.renderArea.extent.height = vk.height;
	qvkCmdBeginRenderPass( vk.frame->mainCmd, &info, VK_SUBPASS_CONTENTS_INLINE );
	qvkCmdEndRenderPass( vk.frame->mainCmd );
}

/*
===============
VK_CreateDepthTarget

The single-sampled depth/stencil image: what the draws use without
multisampling, the target of the depth/stencil resolve before readbacks
with it. Readbacks copy from it, which needs TRANSFER_SRC usage, and RADV
gives such images TC-compatible HTILE, which on GFX9 and later keeps a
depth tile compressed only while it holds at most 4 planes
(DECOMPRESS_ON_N_ZPLANES; 15 otherwise): the forest of m6l1a, with many
small alpha-tested triangles per tile, took 15 % more GPU time to draw
(Radeon 780M). radeonsi's window depth buffer isn't TC-compatible. Without
multisampling the depth image is therefore only readable while depth or
stencil reads are wanted (VK_UpdateDepthTarget).
===============
*/
static void VK_CreateDepthTarget( qboolean readable ) {
	VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	if ( readable ) {
		usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}
	VK_CreateTargetImage( vk.depthFormat, VK_SAMPLE_COUNT_1_BIT, usage,
		VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, &vk.depthImage, &vk.depthMemory, &vk.depthView );
	vk.depthReadable = readable;
}

// a framebuffer of vk.renderPass for a color target: the offscreen color
// image or a swapchain image, the resolve target with multisampling
static VkFramebuffer VK_CreateFramebuffer( VkImageView colorView ) {
	VkFramebufferCreateInfo framebufferInfo;
	VkFramebuffer framebuffer;
	VkImageView views[3];

	Com_Memset( &framebufferInfo, 0, sizeof( framebufferInfo ) );
	framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebufferInfo.renderPass = vk.renderPass;
	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		views[0] = vk.msColorView;
		views[1] = vk.msDepthView;
		views[2] = colorView;
		framebufferInfo.attachmentCount = 3;
	} else {
		views[0] = colorView;
		views[1] = vk.depthView;
		framebufferInfo.attachmentCount = 2;
	}
	framebufferInfo.pAttachments = views;
	framebufferInfo.width = vk.width;
	framebufferInfo.height = vk.height;
	framebufferInfo.layers = 1;
	VK_CHECK( qvkCreateFramebuffer( vk.device, &framebufferInfo, NULL, &framebuffer ) );
	return framebuffer;
}

/*
===============
VK_UpdateDepthTarget

At the start of a frame, before anything is recorded: without
multisampling, replaces the depth image with a readable one when depth or
stencil reads were wanted in the last VK_DEPTH_READ_FRAMES frames (see
vk_want_depth_reads), and with one that draws faster when they no longer
are (see VK_CreateDepthTarget). The new image starts with depth 1 and
stencil 0. The previous contents aren't needed at that point: every 3D view
clears the depth buffer before drawing and 2D drawing doesn't test depth.
The old image and the framebuffers that use it are released once the
frames in flight are done with them.
===============
*/
static void VK_UpdateDepthTarget( void ) {
	qboolean readable;
	uint32_t i;

	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT || !vk.depthImage ) {
		return;
	}
	readable = vk.depthReadSerial && vk.frameSerial - vk.depthReadSerial <= VK_DEPTH_READ_FRAMES;
	if ( readable == vk.depthReadable ) {
		return;
	}

	VK_DeferDestroy( VK_DESTROY_FRAMEBUFFER, (uint64_t)vk.framebuffer, 0 );
	for ( i = 0; i < VK_MAX_SWAPCHAIN_IMAGES; i++ ) {
		VK_DeferDestroy( VK_DESTROY_FRAMEBUFFER, (uint64_t)vk.swapchainFramebuffers[i], 0 );
	}
	VK_DeferDestroy( VK_DESTROY_IMAGE_VIEW, (uint64_t)vk.depthView, 0 );
	VK_DeferDestroy( VK_DESTROY_IMAGE, (uint64_t)vk.depthImage, 0 );
	VK_DeferDestroy( VK_DESTROY_MEMORY, (uint64_t)vk.depthMemory, 0 );

	VK_CreateDepthTarget( readable );
	vk.framebuffer = VK_CreateFramebuffer( vk.colorView );
	for ( i = 0; i < VK_MAX_SWAPCHAIN_IMAGES; i++ ) {
		if ( vk.swapchainFramebuffers[i] ) {
			vk.swapchainFramebuffers[i] = VK_CreateFramebuffer( vk.swapchainViews[i] );
		}
	}
	VK_ClearTargetImage( vk.depthImage, qfalse );
	vk.stats.depthTargets++;
}

static void VK_CreateRenderTargets( void ) {
	static const VkFormat depthFormats[] = {
		VK_FORMAT_D24_UNORM_S8_UINT,	// what GL1 asks for
		VK_FORMAT_D32_SFLOAT_S8_UINT,	// AMD has no D24
		VK_FORMAT_D16_UNORM_S8_UINT
	};
	VkFramebufferCreateInfo framebufferInfo;
	VkImageView views[4];
	VkPipelineStageFlags attachmentStages;
	VkAccessFlags attachmentAccess;
	unsigned int i;

	vk.width = glConfig.vidWidth;
	vk.height = glConfig.vidHeight;
	vk.samples = VK_SampleCount();

	// rendered to, blitted to the swapchain, read back and cleared at
	// creation; the format of the swapchain when it is one of the two 8-bit
	// UNORM ones (B8G8R8A8 for most), so frames can be drawn into it
	vk.colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
	if ( ( vk.swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM || vk.swapchainFormat == VK_FORMAT_R8G8B8A8_UNORM )
		&& VK_FormatHasFeatures( vk.swapchainFormat, VK_TARGET_FORMAT_FEATURES ) ) {
		vk.colorFormat = vk.swapchainFormat;
	}
	if ( !VK_FormatHasFeatures( vk.colorFormat, VK_TARGET_FORMAT_FEATURES ) ) {
		ri.Error( ERR_FATAL, "Vulkan: VK_FORMAT_R8G8B8A8_UNORM can't be rendered to" );
	}

	vk.depthFormat = VK_FORMAT_UNDEFINED;
	for ( i = 0; i < ARRAY_LEN( depthFormats ); i++ ) {
		if ( VK_FormatHasFeatures( depthFormats[i], VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
			| VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT ) ) {
			vk.depthFormat = depthFormats[i];
			break;
		}
	}
	if ( vk.depthFormat == VK_FORMAT_UNDEFINED ) {
		ri.Error( ERR_FATAL, "Vulkan: no depth/stencil format" );
	}

	VK_CreateTargetImage( vk.colorFormat, VK_SAMPLE_COUNT_1_BIT,
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, &vk.colorImage, &vk.colorMemory, &vk.colorView );
	// readable from the start with multisampling: the draws use the
	// multisampled depth image, which is never copied from
	VK_CreateDepthTarget( vk.samples != VK_SAMPLE_COUNT_1_BIT );
	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		VK_CreateTargetImage( vk.colorFormat, vk.samples, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, &vk.msColorImage, &vk.msColorMemory, &vk.msColorView );
		VK_CreateTargetImage( vk.depthFormat, vk.samples, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
			VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, &vk.msDepthImage, &vk.msDepthMemory, &vk.msDepthView );
	}

	// successive render passes read what the previous ones wrote
	attachmentStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		| VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	attachmentAccess = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
		| VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		VK_CreateMultisampleRenderPasses( attachmentStages, attachmentAccess );
	} else {
		VK_CreateRenderPass( attachmentStages, attachmentAccess );
	}

	vk.framebuffer = VK_CreateFramebuffer( vk.colorView );

	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		Com_Memset( &framebufferInfo, 0, sizeof( framebufferInfo ) );
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = vk.depthResolvePass;
		views[0] = vk.msDepthView;
		views[1] = vk.depthView;
		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments = views;
		framebufferInfo.width = vk.width;
		framebufferInfo.height = vk.height;
		framebufferInfo.layers = 1;
		VK_CHECK( qvkCreateFramebuffer( vk.device, &framebufferInfo, NULL, &vk.depthResolveFramebuffer ) );
	}

	// initial contents: black with alpha 1, depth 1, stencil 0
	VK_BeginFrameIfNeeded();
	VK_ClearTargetImage( vk.colorImage, qtrue );
	VK_ClearTargetImage( vk.depthImage, qfalse );
	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		VK_ClearTargetImage( vk.msColorImage, qtrue );
		VK_ClearTargetImage( vk.msDepthImage, qfalse );
	}

	VK_CreateSwapchainTargets();
}

/*
===============
VK_CreateSwapchainTargets

Views and framebuffers of the swapchain images, when they can be the color
target (VK_SelectTarget); called when the swapchain or the render targets
are created
===============
*/
void VK_CreateSwapchainTargets( void ) {
	VkImageViewCreateInfo viewInfo;
	uint32_t i;

	VK_DestroySwapchainTargets();
	// a swapchain of another size than the render target (window resized
	// before the vid_restart) gets the offscreen image's copy
	if ( !vk.swapchain || !vk.renderPass || vk.swapchainFormat != vk.colorFormat
		|| (int)vk.swapchainExtent.width != vk.width || (int)vk.swapchainExtent.height != vk.height
		|| !( vk.swapchainUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT ) || !( vk.swapchainUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT ) ) {
		return;
	}

	for ( i = 0; i < vk.swapchainImageCount; i++ ) {
		Com_Memset( &viewInfo, 0, sizeof( viewInfo ) );
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = vk.swapchainImages[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = vk.swapchainFormat;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.layerCount = 1;
		VK_CHECK( qvkCreateImageView( vk.device, &viewInfo, NULL, &vk.swapchainViews[i] ) );
		vk.swapchainFramebuffers[i] = VK_CreateFramebuffer( vk.swapchainViews[i] );
	}
	vk.swapchainDirect = qtrue;
}

// called when nothing uses them anymore
void VK_DestroySwapchainTargets( void ) {
	int i;

	for ( i = 0; i < VK_MAX_SWAPCHAIN_IMAGES; i++ ) {
		if ( vk.swapchainFramebuffers[i] ) {
			qvkDestroyFramebuffer( vk.device, vk.swapchainFramebuffers[i], NULL );
			vk.swapchainFramebuffers[i] = VK_NULL_HANDLE;
		}
		if ( vk.swapchainViews[i] ) {
			qvkDestroyImageView( vk.device, vk.swapchainViews[i], NULL );
			vk.swapchainViews[i] = VK_NULL_HANDLE;
		}
	}
	vk.swapchainDirect = qfalse;
}

static void VK_DestroyTargetImage( VkImage *image, VkDeviceMemory *memory, VkImageView *view ) {
	if ( *view ) {
		qvkDestroyImageView( vk.device, *view, NULL );
	}
	if ( *image ) {
		qvkDestroyImage( vk.device, *image, NULL );
	}
	VK_FreeMemory( *memory );
	*view = VK_NULL_HANDLE;
	*image = VK_NULL_HANDLE;
	*memory = VK_NULL_HANDLE;
}

static void VK_DestroyRenderTargets( void ) {
	VK_DestroySwapchainTargets();
	if ( vk.framebuffer ) {
		qvkDestroyFramebuffer( vk.device, vk.framebuffer, NULL );
	}
	if ( vk.renderPass ) {
		qvkDestroyRenderPass( vk.device, vk.renderPass, NULL );
	}
	if ( vk.depthResolveFramebuffer ) {
		qvkDestroyFramebuffer( vk.device, vk.depthResolveFramebuffer, NULL );
	}
	if ( vk.depthResolvePass ) {
		qvkDestroyRenderPass( vk.device, vk.depthResolvePass, NULL );
	}
	vk.framebuffer = VK_NULL_HANDLE;
	vk.renderPass = VK_NULL_HANDLE;
	vk.depthResolveFramebuffer = VK_NULL_HANDLE;
	vk.depthResolvePass = VK_NULL_HANDLE;
	VK_DestroyTargetImage( &vk.colorImage, &vk.colorMemory, &vk.colorView );
	VK_DestroyTargetImage( &vk.depthImage, &vk.depthMemory, &vk.depthView );
	VK_DestroyTargetImage( &vk.msColorImage, &vk.msColorMemory, &vk.msColorView );
	VK_DestroyTargetImage( &vk.msDepthImage, &vk.msDepthMemory, &vk.msDepthView );
}

/*
====================================================================

LIFECYCLE

====================================================================
*/

static const char *VK_VendorName( uint32_t vendorID ) {
	switch ( vendorID ) {
	case 0x1002: return "AMD";
	case 0x10DE: return "NVIDIA";
	case 0x8086: return "Intel";
	case 0x13B5: return "ARM";
	case 0x5143: return "Qualcomm";
	case 0x106B: return "Apple";
	case 0x10005: return "Mesa";
	default: return va( "0x%04x", vendorID );
	}
}

/*
===============
VK_ReportCapabilities

Fills glConfig with what the device supports; R_ApplyExtensionCvars then
turns features off from the r_ext_* cvars
===============
*/
static void VK_ReportCapabilities( void ) {
	const VkPhysicalDeviceProperties *props = &vk.deviceProperties;
	char driver[VK_MAX_DRIVER_NAME_SIZE + VK_MAX_DRIVER_INFO_SIZE + 4];

	driver[0] = 0;
	if ( qvkGetPhysicalDeviceProperties2 && props->apiVersion >= VK_API_VERSION_1_2 ) {
		VkPhysicalDeviceProperties2 props2;
		VkPhysicalDeviceDriverProperties driverProps;

		Com_Memset( &driverProps, 0, sizeof( driverProps ) );
		driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
		Com_Memset( &props2, 0, sizeof( props2 ) );
		props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		props2.pNext = &driverProps;
		qvkGetPhysicalDeviceProperties2( vk.physicalDevice, &props2 );
		Com_sprintf( driver, sizeof( driver ), ", %s %s", driverProps.driverName, driverProps.driverInfo );
	}

	Q_strncpyz( glConfig.vendor_string, VK_VendorName( props->vendorID ), sizeof( glConfig.vendor_string ) );
	Q_strncpyz( glConfig.renderer_string, props->deviceName, sizeof( glConfig.renderer_string ) );
	Com_sprintf( glConfig.version_string, sizeof( glConfig.version_string ), "Vulkan %d.%d.%d%s",
		VK_API_VERSION_MAJOR( props->apiVersion ), VK_API_VERSION_MINOR( props->apiVersion ),
		VK_API_VERSION_PATCH( props->apiVersion ), driver );

	glConfig.maxTextureSize = props->limits.maxImageDimension2D;
	glConfig.numTextureUnits = 2;
	glConfig.textureEnvAddAvailable = qtrue;
	glConfig.textureCompression = vk.enabledFeatures.textureCompressionBC ? TC_S3TC_ARB : TC_NONE;

	textureFilterAnisotropic = vk.enabledFeatures.samplerAnisotropy ? qtrue : qfalse;
	maxAnisotropy = textureFilterAnisotropic ? (int)props->limits.maxSamplerAnisotropy : 0;
	haveClampToEdge = qtrue;

	// the bits of the formats actually used; like GL1 the stencil buffer is
	// only reported when r_stencilbits asked for one
	glConfig.colorBits = 24;
	glConfig.depthBits = VK_DepthBits( vk.depthFormat );
	glConfig.stencilBits = r_stencilbits->integer > 0 ? 8 : 0;

	ri.Printf( PRINT_ALL, "Vulkan device: %s (%s)\n", glConfig.renderer_string, glConfig.version_string );
	ri.Printf( PRINT_ALL, "Vulkan present mode: %s, render target %dx%d\n", VK_SwapchainPresentModeName(), vk.width, vk.height );
}

/*
===============
VK_Info_f

vkinfo: the counters of vkStats_t since the last vkinfo (tools/perf.py
reads them), then what exists now
===============
*/
static void VK_Info_f( void ) {
	const vkStats_t *s = &vk.stats, *r = &vk.statsReported;
	VkDeviceSize blockSize, blockUsed;
	int numBlocks;

#define VK_DELTA( field ) ( (unsigned long long)( s->field - r->field ) )
	ri.Printf( PRINT_ALL, "vkinfo: frames=%llu renderPasses=%llu submits=%llu flushes=%llu draws=%llu staticDraws=%llu skinDraws=%llu pipelineBinds=%llu"
		" descriptorBinds=%llu vertexBytes=%llu stagingBytes=%llu mainUploads=%llu textures=%llu pipelines=%llu allocations=%llu depthTargets=%llu"
		" waitUs=%llu submitUs=%llu acquireUs=%llu acquireAheadUs=%llu frameToPresentUs=%llu presentUs=%llu pipelineUs=%llu allocationUs=%llu textureUs=%llu"
		" gpuFrames=%llu gpuMainUs=%llu gpuPresentUs=%llu gpuFrameUs=%llu\n",
		VK_DELTA( frames ), VK_DELTA( renderPasses ), VK_DELTA( submits ), VK_DELTA( flushes ), VK_DELTA( draws ),
		VK_DELTA( staticDraws ), VK_DELTA( skinDraws ), VK_DELTA( pipelineBinds ), VK_DELTA( descriptorBinds ), VK_DELTA( vertexBytes ), VK_DELTA( stagingBytes ), VK_DELTA( mainUploads ),
		VK_DELTA( textures ), VK_DELTA( pipelines ), VK_DELTA( allocations ), VK_DELTA( depthTargets ), VK_DELTA( waitUs ), VK_DELTA( submitUs ),
		VK_DELTA( acquireUs ), VK_DELTA( acquireAheadUs ), VK_DELTA( frameToPresentUs ), VK_DELTA( presentUs ), VK_DELTA( pipelineUs ), VK_DELTA( allocationUs ), VK_DELTA( textureUs ),
		VK_DELTA( gpuFrames ), VK_DELTA( gpuMainUs ), VK_DELTA( gpuPresentUs ), VK_DELTA( gpuFrameUs ) );
#undef VK_DELTA
	VK_MemoryBlockInfo( &numBlocks, &blockSize, &blockUsed );
	ri.Printf( PRINT_ALL, "vkinfo: now %d device memory objects, %d of them texture memory blocks (%.1f MB, %.1f MB used), %d pipelines, depth image %sreadable,"
		" skin data %.1f MB of %d\n",
		vk.liveAllocations, numBlocks, blockSize / ( 1024.0 * 1024.0 ), blockUsed / ( 1024.0 * 1024.0 ), VK_NumPipelines(),
		vk.depthReadable ? "" : "not ", vk_skin.words / ( 256.0 * 1024.0 ), VK_SKIN_DATA_WORDS / ( 256 * 1024 ) );

	vk.statsReported = vk.stats;
}

static void VK_ResetState( void ) {
	int i;

	Com_Memset( &vks, 0, sizeof( vks ) );
	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		vks.texEnv[i] = TEXENV_MODULATE;
	}
	vks.stateBits = GLS_DEPTHTEST_DISABLE | GLS_DEPTHMASK_TRUE;
	vks.colorMask[0] = vks.colorMask[1] = vks.colorMask[2] = vks.colorMask[3] = qtrue;
	vks.depthMax = 1.0f;
	vks.viewport[2] = vks.scissor[2] = glConfig.vidWidth;
	vks.viewport[3] = vks.scissor[3] = glConfig.vidHeight;
	vks.projection[0] = vks.projection[5] = vks.projection[10] = vks.projection[15] = 1.0f;
	vks.modelview[0] = vks.modelview[5] = vks.modelview[10] = vks.modelview[15] = 1.0f;
	vks.lineWidth = 1.0f;
	vks.stencilMask = vks.stencilWriteMask = ~0U;
	vks.fogColor[3] = 1.0f;
	vks.fogEnd = 1.0f;
	vks.lineStipplePattern = 0xFFFF;
	vks.dlightTmu = -1;

	Com_Memset( &vk_lastKey, 0, sizeof( vk_lastKey ) );
	vk_lastPipeline = VK_NULL_HANDLE;
	vk_dirty = VK_DIRTY_ALL;
}

void vk_initialize( void ) {
	r_vkValidation = ri.Cvar_Get( "r_vkValidation", "0", CVAR_LATCH );

	Com_Memset( &vk, 0, sizeof( vk ) );
	VK_ResetState();
	// 0 is the segment of new textures, which no draw has used
	vk.segment = 1;

	VK_CreateDevice();
	VK_CreateFrames();
	VK_CreateSwapchain();
	VK_CreateRenderTargets();
	VK_InitPipelines();
	VK_CreateDlightSets();
	VK_InitTextures();
	VK_ReportCapabilities();

	ri.Cmd_AddCommand( "vkinfo", VK_Info_f );
}

void vk_shutdown( void ) {
	int i;

	ri.Cmd_RemoveCommand( "vkinfo" );

	if ( vk.device ) {
		qvkDeviceWaitIdle( vk.device );
		// before the semaphores go
		VK_ReleaseHeldImage();

		VK_ShutdownTextures();
		for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
			VK_ProcessDestroyList( &vk.frames[i] );
		}
		VK_ShutdownTextureObjects();
		VK_DestroyBuffer( &vk_static.buffer );
		Com_Memset( &vk_static, 0, sizeof( vk_static ) );
		VK_DestroyBuffer( &vk_skin.buffer );
		Com_Memset( &vk_skin, 0, sizeof( vk_skin ) );
		VK_ShutdownPipelines();
		VK_DestroyFrames();
		VK_DestroyRenderTargets();
		VK_DestroyBuffer( &vk.readbackBuffer );
		VK_DestroySwapchain();
	}

	VK_DestroyDevice();

	Com_Memset( &vk, 0, sizeof( vk ) );
}

void vk_begin_frame( void ) {
	if ( !vk.device ) {
		return;
	}
	VK_BeginFrameIfNeeded();
}

void vk_end_frame( void ) {
	vkFrame_t *frame;

	if ( !vk.device ) {
		return;
	}

	VK_BeginFrameIfNeeded();
	frame = vk.frame;

	if ( vk.target == VK_TARGET_SWAPCHAIN ) {
		VK_SubmitFrameCommands( frame->fence, vk.presentSemaphores[vk.targetIndex], qtrue );
		VK_QueuePresent( vk.targetIndex );

		// frame pacing: the image of the next frame is acquired now, so with
		// a vsync present mode the wait for the display happens here, before
		// the game runs the next frame, rather than between the next frame's
		// game logic and its present (one refresh less of input latency,
		// like GL's swap, which waits after queuing the present)
		if ( VK_SwapchainFitsTarget() ) {
			VkResult ahead = VK_AcquireNextImage( &vk.stats.acquireAheadUs );

			if ( ahead == VK_ERROR_OUT_OF_DATE_KHR ) {
				vk.swapchainOutOfDate = qtrue;
			} else {
				VK_CHECK( ahead );
			}
		}
	} else {
		VK_SubmitFrameCommands( VK_NULL_HANDLE, VK_NULL_HANDLE, qtrue );

		// a frame that drew nothing presents nothing: the window keeps the
		// last frame
		if ( vk.target != VK_TARGET_OFFSCREEN || !VK_Present( frame ) ) {
			// the fence still has to signal for the next use of this frame slot
			VK_CHECK( qvkResetFences( vk.device, 1, &frame->fence ) );
			VK_CHECK( qvkQueueSubmit( vk.queue, 0, NULL, frame->fence ) );
			vk.lastSubmitFence = frame->fence;
		}
	}

	vk.target = VK_TARGET_NONE;
	vk.frame = NULL;
	vk.frameIndex = ( vk.frameIndex + 1 ) % VK_MAX_FRAMES;
	vk.stats.frames++;
}

void vk_finish( void ) {
	// like glFinish for what has been submitted; the commands of the frame
	// being recorded are submitted at vk_end_frame
	if ( vk.device && vk.lastSubmitFence ) {
		VK_CHECK( qvkWaitForFences( vk.device, 1, &vk.lastSubmitFence, VK_TRUE, UINT64_MAX ) );
	}
}

/*
====================================================================

RENDER STATE

====================================================================
*/

void vk_set_state( unsigned long stateBits ) {
	vks.stateBits = stateBits;
}

void vk_bind_texture( int tmu, image_t *image ) {
	if ( tmu < 0 || tmu >= NUM_TEXTURE_BUNDLES ) {
		return;
	}
	// the handle, not the storage: a draw uses the storage the image has
	// when the draw is recorded
	vks.texture[tmu] = image ? image->texnum : 0;
}

void vk_set_texenv( int tmu, texEnv_t env ) {
	if ( tmu < 0 || tmu >= NUM_TEXTURE_BUNDLES || env < TEXENV_MODULATE || env > TEXENV_COMBINE4 ) {
		return;
	}
	vks.texEnv[tmu] = env;
}

void vk_set_cull( cullFace_t face ) {
	vks.cull = face;
}

void vk_set_depth_range( float zNear, float zFar ) {
	vks.depthMin = Com_Clamp( 0.0f, 1.0f, zNear );
	vks.depthMax = Com_Clamp( 0.0f, 1.0f, zFar );
	vk_dirty |= VK_DIRTY_VIEWPORT;
}

void vk_set_polygon_offset( qboolean enable, float factor, float units ) {
	vks.polygonOffset = enable;
	vks.offsetFactor = factor;
	vks.offsetUnits = units;
	vk_dirty |= VK_DIRTY_DEPTHBIAS;
}

void vk_set_color_mask( qboolean red, qboolean green, qboolean blue, qboolean alpha ) {
	vks.colorMask[0] = red;
	vks.colorMask[1] = green;
	vks.colorMask[2] = blue;
	vks.colorMask[3] = alpha;
}

void vk_set_fog( qboolean enable ) {
	vks.fog = enable;
}

void vk_set_fog_color( const vec4_t color ) {
	Vector4Copy( color, vks.fogColor );
	vk_dirty |= VK_DIRTY_FOG;
}

void vk_set_fog_range( float start, float end ) {
	vks.fogStart = start;
	vks.fogEnd = end;
	vk_dirty |= VK_DIRTY_FOG;
}

void vk_set_clip_plane( const float *eyePlane ) {
	vks.clipPlaneEnabled = eyePlane != NULL;
	if ( eyePlane ) {
		Vector4Copy( eyePlane, vks.clipPlane );
	}
	// the shaders get the plane in object space
	vk_dirty |= VK_DIRTY_MVP;
}

void vk_set_stencil_test( qboolean enable ) {
	vks.stencilTest = enable;
}

void vk_set_stencil_func( stencilFunc_t func, int ref, unsigned int mask ) {
	vks.stencilFunc = func;
	vks.stencilRef = ref;
	vks.stencilMask = mask;
	vk_dirty |= VK_DIRTY_STENCIL;
}

void vk_set_stencil_op( stencilOp_t stencilFail, stencilOp_t depthFail, stencilOp_t depthPass ) {
	vks.stencilOp[0] = stencilFail;
	vks.stencilOp[1] = depthFail;
	vks.stencilOp[2] = depthPass;
}

void vk_set_stencil_write_mask( unsigned int mask ) {
	vks.stencilWriteMask = mask;
	vk_dirty |= VK_DIRTY_STENCIL;
}

void vk_set_line_width( float width ) {
	vks.lineWidth = width;
	vk_dirty |= VK_DIRTY_LINEWIDTH;
}

void vk_set_line_stipple( int factor, unsigned short pattern ) {
	// GL clamps the factor to [1, 256]; 0 disables stippling here
	vks.lineStippleFactor = factor > 0 ? Q_min( factor, 256 ) : 0;
	vks.lineStipplePattern = pattern;
	vk_dirty |= VK_DIRTY_STIPPLE;
}

int vk_add_dlights( int count, const vkDlight_t *dlights ) {
	vkDlightGPU_t *out;
	int base, i;

	if ( !vk.device || count <= 0 ) {
		return -1;
	}
	VK_BeginFrameIfNeeded();

	base = vk.frame->numDlights;
	if ( base + count > VK_MAX_FRAME_DLIGHTS ) {
		static qboolean warned;

		if ( !warned ) {
			ri.Printf( PRINT_ALL, "Vulkan: more than %d dynamic lights in a frame, some get no light\n", VK_MAX_FRAME_DLIGHTS );
			warned = qtrue;
		}
		return -1;
	}

	out = (vkDlightGPU_t *)vk.frame->dlightBuffer.mapped + base;
	for ( i = 0; i < count; i++ ) {
		VectorCopy( dlights[i].origin, out[i].origin );
		out[i].invRadius = 1.0f / dlights[i].radius;
		VectorScale( dlights[i].color, 1.0f / dlights[i].scale, out[i].color );
		out[i].scale = dlights[i].scale;
	}
	vk.frame->numDlights += count;

	return base;
}

void vk_set_lightmap_dlights( int tmu, int handle, unsigned int mask, qboolean backs ) {
	if ( tmu < 0 || tmu >= NUM_TEXTURE_BUNDLES || handle < 0 || !mask ) {
		vks.dlightTmu = -1;
		return;
	}

	vks.dlightTmu = tmu;
	vks.dlightBacks = backs;
	if ( vks.dlightBase != handle || vks.dlightMask != mask ) {
		vks.dlightBase = handle;
		vks.dlightMask = mask;
		vk_dirty |= VK_DIRTY_DLIGHT;
	}
}

void vk_set_projection( const float *matrix ) {
	Com_Memcpy( vks.projection, matrix, sizeof( vks.projection ) );
	vk_dirty |= VK_DIRTY_MVP;
}

void vk_set_modelview( const float *matrix ) {
	Com_Memcpy( vks.modelview, matrix, sizeof( vks.modelview ) );
	vk_dirty |= VK_DIRTY_MVP;
}

void vk_set_viewport( int x, int y, int width, int height ) {
	vks.viewport[0] = x;
	vks.viewport[1] = y;
	vks.viewport[2] = width;
	vks.viewport[3] = height;
	vk_dirty |= VK_DIRTY_VIEWPORT;
}

void vk_set_scissor( int x, int y, int width, int height ) {
	vks.scissor[0] = x;
	vks.scissor[1] = y;
	vks.scissor[2] = width;
	vks.scissor[3] = height;
	vk_dirty |= VK_DIRTY_SCISSOR;
}

/*
====================================================================

DRAWING

====================================================================
*/

/*
===============
VK_ScissorRect

The scissor rectangle in framebuffer coordinates (origin at the top-left),
clipped to the framebuffer; returns qfalse when it is empty
===============
*/
static qboolean VK_ScissorRect( VkRect2D *rect ) {
	int x0, y0, x1, y1;

	x0 = Q_max( vks.scissor[0], 0 );
	x1 = Q_min( vks.scissor[0] + vks.scissor[2], vk.width );
	y0 = Q_max( vk.height - ( vks.scissor[1] + vks.scissor[3] ), 0 );
	y1 = Q_min( vk.height - vks.scissor[1], vk.height );

	if ( x1 <= x0 || y1 <= y0 ) {
		Com_Memset( rect, 0, sizeof( *rect ) );
		return qfalse;
	}

	rect->offset.x = x0;
	rect->offset.y = y0;
	rect->extent.width = x1 - x0;
	rect->extent.height = y1 - y0;
	return qtrue;
}

/*
===============
VK_ComputeMVP

projection * modelview, with the projection remapped from GL clip space
(z in [-w, w]) to Vulkan's (z in [0, w]): z' = ( z + w ) / 2
===============
*/
static void VK_ComputeMVP( float *mvp ) {
	const float *mv = vks.modelview;
	float p[16];
	int c, r;

	for ( c = 0; c < 4; c++ ) {
		p[c * 4 + 0] = vks.projection[c * 4 + 0];
		p[c * 4 + 1] = vks.projection[c * 4 + 1];
		p[c * 4 + 2] = vk.depthClipControl ? vks.projection[c * 4 + 2] : 0.5f * ( vks.projection[c * 4 + 2] + vks.projection[c * 4 + 3] );
		p[c * 4 + 3] = vks.projection[c * 4 + 3];
	}

	for ( c = 0; c < 4; c++ ) {
		for ( r = 0; r < 4; r++ ) {
			mvp[c * 4 + r] = p[0 * 4 + r] * mv[c * 4 + 0] + p[1 * 4 + r] * mv[c * 4 + 1]
				+ p[2 * 4 + r] * mv[c * 4 + 2] + p[3 * 4 + r] * mv[c * 4 + 3];
		}
	}
}

static void VK_ApplyDynamicState( VkCommandBuffer cmd, qboolean stippled ) {
	if ( vk_dirty & VK_DIRTY_VIEWPORT ) {
		VkViewport viewport;

		// GL window coordinates have the origin at the bottom-left: a
		// negative height flips y
		viewport.x = vks.viewport[0];
		viewport.y = vk.height - vks.viewport[1];
		viewport.width = Q_max( vks.viewport[2], 1 );
		viewport.height = -(float)Q_max( vks.viewport[3], 1 );
		viewport.minDepth = vks.depthMin;
		viewport.maxDepth = vks.depthMax;
		qvkCmdSetViewport( cmd, 0, 1, &viewport );
	}

	if ( vk_dirty & VK_DIRTY_SCISSOR ) {
		VkRect2D scissor;

		VK_ScissorRect( &scissor );
		qvkCmdSetScissor( cmd, 0, 1, &scissor );
	}

	if ( vk_dirty & VK_DIRTY_DEPTHBIAS ) {
		// only used by pipelines with VK_PIPE_DEPTHBIAS
		qvkCmdSetDepthBias( cmd, vks.offsetUnits, 0.0f, vks.offsetFactor );
	}

	if ( vk_dirty & VK_DIRTY_LINEWIDTH ) {
		float width = 1.0f;

		if ( vk.enabledFeatures.wideLines ) {
			width = Com_Clamp( vk.deviceProperties.limits.lineWidthRange[0], vk.deviceProperties.limits.lineWidthRange[1], vks.lineWidth );
		}
		qvkCmdSetLineWidth( cmd, width );
	}

	if ( vk_dirty & VK_DIRTY_STENCIL ) {
		qvkCmdSetStencilCompareMask( cmd, VK_STENCIL_FACE_FRONT_AND_BACK, vks.stencilMask & 0xFF );
		qvkCmdSetStencilWriteMask( cmd, VK_STENCIL_FACE_FRONT_AND_BACK, vks.stencilWriteMask & 0xFF );
		qvkCmdSetStencilReference( cmd, VK_STENCIL_FACE_FRONT_AND_BACK, vks.stencilRef & 0xFF );
	}

	if ( vk_dirty & VK_DIRTY_MVP ) {
		const float *mv = vks.modelview;
		vkPushConstants_t pc;
		int i;

		VK_ComputeMVP( pc.mvp );

		pc.eyeZ[0] = mv[2];
		pc.eyeZ[1] = mv[6];
		pc.eyeZ[2] = mv[10];
		pc.eyeZ[3] = mv[14];

		// the eye space plane P keeps eye positions with P . ( MV v ) >= 0,
		// so the object space plane is P * MV
		for ( i = 0; i < 4; i++ ) {
			pc.clipPlane[i] = vks.clipPlaneEnabled ? vks.clipPlane[0] * mv[i * 4 + 0] + vks.clipPlane[1] * mv[i * 4 + 1]
				+ vks.clipPlane[2] * mv[i * 4 + 2] + vks.clipPlane[3] * mv[i * 4 + 3] : 0.0f;
		}

		qvkCmdPushConstants( cmd, vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, VK_PUSH_VERTEX_SIZE, &pc );
	}

	if ( vk_dirty & VK_DIRTY_FOG ) {
		vkPushConstants_t pc;
		float scale;
		int i;

		for ( i = 0; i < 4; i++ ) {
			pc.fogColor[i] = Com_Clamp( 0.0f, 1.0f, vks.fogColor[i] );
		}
		// computed like Mesa's gl_MesaFogParamsOptimized, so the factor
		// matches GL1 on Mesa drivers to the bit
		scale = ( vks.fogEnd == vks.fogStart ) ? 1.0f : -1.0f / ( vks.fogEnd - vks.fogStart );
		pc.fogParams[0] = scale;
		pc.fogParams[1] = vks.fogEnd * -scale;

		qvkCmdPushConstants( cmd, vk.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, VK_PUSH_VERTEX_SIZE,
			offsetof( vkPushConstants_t, dlightBase ) - VK_PUSH_VERTEX_SIZE, (const byte *)&pc + VK_PUSH_VERTEX_SIZE );
	}

	if ( vk_dirty & VK_DIRTY_DLIGHT ) {
		vkPushConstants_t pc;

		pc.dlightBase = vks.dlightBase;
		pc.dlightMask = vks.dlightMask;
		qvkCmdPushConstants( cmd, vk.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, offsetof( vkPushConstants_t, dlightBase ),
			sizeof( pc.dlightBase ) + sizeof( pc.dlightMask ), &pc.dlightBase );
	}

	if ( stippled ) {
		if ( vk_dirty & VK_DIRTY_STIPPLE ) {
			qvkCmdSetLineStippleEXT( cmd, vks.lineStippleFactor, vks.lineStipplePattern );
		}
		vk_dirty = 0;
	} else {
		// a pipeline with static stipple state invalidates the dynamic one:
		// set it again for the next stippled pipeline
		vk_dirty = VK_DIRTY_STIPPLE;
	}
}

static void VK_BuildPipelineKey( vkPipelineKey_t *key, primType_t prim, const drawGeometry_t *geometry, vkTexture_t **textures ) {
	unsigned long stateBits = vks.stateBits;
	uint64_t bits = 0;
	float alphaRef = 0.0f;
	int alphaFunc = 0;
	int i;

	if ( stateBits & ( GLS_SRCBLEND_BITS | GLS_DSTBLEND_BITS ) ) {
		bits |= (uint64_t)( stateBits & GLS_SRCBLEND_BITS ) << VK_PIPE_SRCBLEND_SHIFT;
		bits |= (uint64_t)( ( stateBits & GLS_DSTBLEND_BITS ) >> 4 ) << VK_PIPE_DSTBLEND_SHIFT;
	}

	// like GL, no depth writes without the depth test
	if ( !( stateBits & GLS_DEPTHTEST_DISABLE ) ) {
		bits |= VK_PIPE_DEPTHTEST;
		if ( stateBits & GLS_DEPTHMASK_TRUE ) {
			bits |= VK_PIPE_DEPTHWRITE;
		}
		if ( stateBits & GLS_DEPTHFUNC_EQUAL ) {
			bits |= (uint64_t)VK_DEPTHFUNC_EQUAL << VK_PIPE_DEPTHFUNC_SHIFT;
		}
	}

	if ( prim == PRIM_LINES ) {
		bits |= VK_PIPE_LINES;
		if ( vks.lineStippleFactor && vk.lineStipple ) {
			bits |= VK_PIPE_STIPPLE;
		}
	} else {
		// culling, polygon mode and polygon offset only apply to polygons
		bits |= (uint64_t)vks.cull << VK_PIPE_CULL_SHIFT;
		if ( stateBits & GLS_POLYMODE_LINE ) {
			if ( vk.enabledFeatures.fillModeNonSolid ) {
				bits |= VK_PIPE_POLYMODE_LINE;
				if ( vks.lineStippleFactor && vk.lineStipple ) {
					bits |= VK_PIPE_STIPPLE;
				}
			}
		} else if ( vks.polygonOffset ) {
			// GL_POLYGON_OFFSET_FILL: filled polygons only
			bits |= VK_PIPE_DEPTHBIAS;
		}
	}

	bits |= (uint64_t)( ( vks.colorMask[0] ? 1 : 0 ) | ( vks.colorMask[1] ? 2 : 0 ) | ( vks.colorMask[2] ? 4 : 0 ) ) << VK_PIPE_COLORMASK_SHIFT;

	switch ( stateBits & GLS_ATEST_BITS ) {
	case GLS_ATEST_GT_0:
		alphaFunc = 1;
		alphaRef = 0.05f;
		break;
	case GLS_ATEST_LT_80:
		alphaFunc = 2;
		alphaRef = 0.5f;
		break;
	case GLS_ATEST_GE_80:
		alphaFunc = 3;
		alphaRef = 0.5f;
		break;
	case GLS_ATEST_LT_FOLIAGE1:
		alphaFunc = 2;
		alphaRef = r_alpha_foliage1->value;
		break;
	case GLS_ATEST_GE_FOLIAGE1:
		alphaFunc = 3;
		alphaRef = r_alpha_foliage1->value;
		break;
	case GLS_ATEST_LT_FOLIAGE2:
		alphaFunc = 2;
		alphaRef = r_alpha_foliage2->value;
		break;
	case GLS_ATEST_GE_FOLIAGE2:
		alphaFunc = 3;
		alphaRef = r_alpha_foliage2->value;
		break;
	}
	bits |= (uint64_t)alphaFunc << VK_PIPE_ATEST_SHIFT;

	if ( geometry->isSkinned ) {
		bits |= VK_PIPE_SKINNED;
	} else if ( geometry->isStatic ? geometry->staticColor >= 0 : geometry->colors != NULL ) {
		bits |= VK_PIPE_COLOR_ARRAY;
	}

	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		if ( textures[i] ) {
			bits |= (uint64_t)vks.texEnv[i] << ( i ? VK_PIPE_TEXENV1_SHIFT : VK_PIPE_TEXENV0_SHIFT );
			// only TEXENV_REPLACE treats RGB textures differently
			if ( textures[i]->rgb && vks.texEnv[i] == TEXENV_REPLACE ) {
				bits |= i ? VK_PIPE_TEX1_RGB : VK_PIPE_TEX0_RGB;
			}
		}
	}

	if ( vks.dlightTmu >= 0 && textures[vks.dlightTmu] ) {
		bits |= (uint64_t)( vks.dlightTmu + 1 ) << VK_PIPE_DLIGHT_SHIFT;
		if ( vks.dlightBacks ) {
			bits |= VK_PIPE_DLIGHT_BACKS;
		}
	}

	if ( vks.fog ) {
		bits |= VK_PIPE_FOG;
	}
	if ( vks.clipPlaneEnabled ) {
		bits |= VK_PIPE_CLIPPLANE;
	}

	if ( vks.stencilTest ) {
		bits |= VK_PIPE_STENCIL;
		bits |= (uint64_t)vks.stencilFunc << VK_PIPE_STENCILFUNC_SHIFT;
		bits |= (uint64_t)vks.stencilOp[0] << VK_PIPE_STENCILOP_SHIFT;
		bits |= (uint64_t)vks.stencilOp[1] << ( VK_PIPE_STENCILOP_SHIFT + 2 );
		bits |= (uint64_t)vks.stencilOp[2] << ( VK_PIPE_STENCILOP_SHIFT + 4 );
	}

	key->bits = bits;
	key->alphaRef = 0;
	if ( alphaFunc ) {
		Com_Memcpy( &key->alphaRef, &alphaRef, sizeof( alphaRef ) );
	}
}

/*
===============
VK_VertexAlloc

Space for size bytes in the vertex buffer of the frame
===============
*/
static byte *VK_VertexAlloc( VkDeviceSize size, VkBuffer *buffer, VkDeviceSize *offset ) {
	vkFrame_t *frame = vk.frame;
	VkDeviceSize start;

	start = PAD( frame->vertexOffset, 16 );
	if ( start + size > frame->vertexBuffer.size ) {
		VkDeviceSize newSize = frame->vertexBuffer.size * 2;

		while ( newSize < size ) {
			newSize *= 2;
		}

		// commands recorded so far still read the old buffer
		VK_DeferDestroy( VK_DESTROY_BUFFER, (uint64_t)frame->vertexBuffer.buffer, 0 );
		VK_DeferDestroy( VK_DESTROY_MEMORY, (uint64_t)frame->vertexBuffer.memory, 0 );
		VK_CreateBuffer( &frame->vertexBuffer, newSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
		ri.Printf( PRINT_DEVELOPER, "Vulkan: vertex buffer grown to %d KB\n", (int)( newSize / 1024 ) );
		start = 0;
	}

	frame->vertexOffset = start + size;
	*buffer = frame->vertexBuffer.buffer;
	*offset = start;
	return frame->vertexBuffer.mapped + start;
}

static void VK_CopyStream( float *dst, const float *src, int count, int numFloats, int stride ) {
	int i;

	if ( stride == (int)sizeof( float ) * numFloats ) {
		Com_Memcpy( dst, src, sizeof( float ) * numFloats * count );
		return;
	}

	for ( i = 0; i < count; i++ ) {
		Com_Memcpy( dst, src, sizeof( float ) * numFloats );
		dst += numFloats;
		src = (const float *)( (const byte *)src + stride );
	}
}

static void VK_BindVertexBuffers( VkCommandBuffer cmd, const VkBuffer *buffers, const VkDeviceSize *offsets ) {
	if ( memcmp( vkb.vertexBuffers, buffers, sizeof( vkb.vertexBuffers ) ) || memcmp( vkb.vertexOffsets, offsets, sizeof( vkb.vertexOffsets ) ) ) {
		qvkCmdBindVertexBuffers( cmd, 0, 4, buffers, offsets );
		Com_Memcpy( vkb.vertexBuffers, buffers, sizeof( vkb.vertexBuffers ) );
		Com_Memcpy( vkb.vertexOffsets, offsets, sizeof( vkb.vertexOffsets ) );
	}
}

static void VK_BindIndexBuffer( VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset ) {
	if ( vkb.indexBuffer != buffer || vkb.indexOffset != offset ) {
		qvkCmdBindIndexBuffer( cmd, buffer, offset, VK_INDEX_TYPE_UINT32 );
		vkb.indexBuffer = buffer;
		vkb.indexOffset = offset;
	}
}

/*
===============
VK_UploadGeometry

Copies the vertex streams and the indexes into the vertex buffer of the
frame and binds them: positions, colors (or the constant color), texture
coordinates of units 0 and 1, each in its own binding
===============
*/
static void VK_UploadGeometry( VkCommandBuffer cmd, const drawGeometry_t *geometry, int numIndexes, const glIndex_t *indexes ) {
	VkBuffer buffer, buffers[4], indexBuffer;
	VkDeviceSize offset, offsets[4], indexOffset;
	VkDeviceSize xyzSize, colorSize, stSize[NUM_TEXTURE_BUNDLES], indexSize, total, pos;
	int numVertexes = geometry->numVertexes;
	qboolean sameXyz, sameIndexes;
	byte *data;
	int i;

	// the passes of a multi-pass shader: same positions, usually the same
	// indexes (not for the dlight pass)
	sameXyz = geometry->version && vk_lastGeometry.epoch == vk_vertexEpoch && vk_lastGeometry.version == geometry->version
		&& vk_lastGeometry.xyz == geometry->xyz && vk_lastGeometry.xyzStride == geometry->xyzStride
		&& vk_lastGeometry.numVertexes == numVertexes;
	sameIndexes = sameXyz && vk_lastGeometry.indexes == indexes && vk_lastGeometry.numIndexes == numIndexes;

	xyzSize = sameXyz ? 0 : PAD( sizeof( float ) * 3 * numVertexes, 16 );
	colorSize = geometry->colors ? PAD( 4 * numVertexes, 16 ) : sizeof( float ) * 4;
	total = xyzSize + colorSize;
	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		stSize[i] = geometry->st[i] ? PAD( sizeof( float ) * 2 * numVertexes, 16 ) : 0;
		total += stSize[i];
	}
	indexSize = sameIndexes ? 0 : sizeof( glIndex_t ) * numIndexes;
	total += indexSize;

	data = VK_VertexAlloc( total, &buffer, &offset );
	vk.stats.vertexBytes += total;
	pos = 0;

	if ( sameXyz ) {
		buffers[0] = vk_lastGeometry.buffer;
		offsets[0] = vk_lastGeometry.xyzOffset;
	} else {
		VK_CopyStream( (float *)( data + pos ), geometry->xyz, numVertexes, 3, geometry->xyzStride );
		buffers[0] = buffer;
		offsets[0] = offset + pos;
		pos += xyzSize;
	}

	if ( geometry->colors ) {
		Com_Memcpy( data + pos, geometry->colors, 4 * numVertexes );
	} else {
		Com_Memcpy( data + pos, geometry->color, sizeof( float ) * 4 );
	}
	buffers[1] = buffer;
	offsets[1] = offset + pos;
	pos += colorSize;

	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		if ( geometry->st[i] ) {
			VK_CopyStream( (float *)( data + pos ), geometry->st[i], numVertexes, 2, geometry->stStride[i] );
			buffers[2 + i] = buffer;
			offsets[2 + i] = offset + pos;
			pos += stSize[i];
		} else {
			// the pipeline reads it with a stride of 0 and ignores it
			buffers[2 + i] = buffers[0];
			offsets[2 + i] = offsets[0];
		}
	}

	if ( sameIndexes ) {
		indexBuffer = vk_lastGeometry.buffer;
		indexOffset = vk_lastGeometry.indexOffset;
	} else {
		Com_Memcpy( data + pos, indexes, indexSize );
		indexBuffer = buffer;
		indexOffset = offset + pos;
	}

	if ( geometry->version ) {
		vk_lastGeometry.epoch = vk_vertexEpoch;
		vk_lastGeometry.version = geometry->version;
		vk_lastGeometry.xyz = geometry->xyz;
		vk_lastGeometry.xyzStride = geometry->xyzStride;
		vk_lastGeometry.numVertexes = numVertexes;
		vk_lastGeometry.buffer = buffers[0];
		vk_lastGeometry.xyzOffset = offsets[0];
		if ( !sameIndexes ) {
			// the indexes of this draw; the dlight pass's hit indexes
			// replace the batch's, which the following passes don't use
			vk_lastGeometry.indexes = indexes;
			vk_lastGeometry.numIndexes = numIndexes;
			vk_lastGeometry.indexOffset = indexOffset;
		}
	}

	VK_BindVertexBuffers( cmd, buffers, offsets );
	VK_BindIndexBuffer( cmd, indexBuffer, indexOffset );
}

/*
===============
VK_ConstantColor

The constant color of a draw in the vertex buffer of the frame (read with a
stride of 0), copied once while it doesn't change
===============
*/
static void VK_ConstantColor( const float *color, VkBuffer *buffer, VkDeviceSize *offset ) {
	if ( vk_lastConstant.epoch != vk_vertexEpoch || memcmp( vk_lastConstant.color, color, sizeof( vk_lastConstant.color ) ) ) {
		byte *data = VK_VertexAlloc( sizeof( float ) * 4, &vk_lastConstant.buffer, &vk_lastConstant.offset );

		Com_Memcpy( data, color, sizeof( float ) * 4 );
		Com_Memcpy( vk_lastConstant.color, color, sizeof( vk_lastConstant.color ) );
		vk_lastConstant.epoch = vk_vertexEpoch;
		vk.stats.vertexBytes += sizeof( float ) * 4;
	}
	*buffer = vk_lastConstant.buffer;
	*offset = vk_lastConstant.offset;
}

/*
===============
VK_BindStaticGeometry

Binds the streams of the static geometry for a draw and its indexes: a
range of the static index stream, or the renderer's indexes copied into the
vertex buffer of the frame (once for the passes of a shader, see
drawGeometry_t::version). Returns the first index of the draw
===============
*/
static uint32_t VK_BindStaticGeometry( VkCommandBuffer cmd, const drawGeometry_t *geometry, int numIndexes, const glIndex_t *indexes ) {
	VkBuffer buffers[4];
	VkDeviceSize offsets[4];
	VkBuffer buffer = vk_static.buffer.buffer;
	int i;

	buffers[0] = buffer;
	offsets[0] = vk_static.xyzOffset + sizeof( float ) * 3 * geometry->staticVertex;

	if ( geometry->staticColor >= 0 ) {
		buffers[1] = buffer;
		offsets[1] = vk_static.colorOffset + 4 * geometry->staticColor;
	} else {
		VK_ConstantColor( geometry->color, &buffers[1], &offsets[1] );
	}

	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		if ( geometry->staticSt[i] == STATIC_ST_NONE ) {
			// the pipeline reads it with a stride of 0 and ignores it
			buffers[2 + i] = buffers[0];
			offsets[2 + i] = offsets[0];
		} else {
			buffers[2 + i] = buffer;
			offsets[2 + i] = vk_static.stOffset[geometry->staticSt[i] == STATIC_ST_LIGHTMAP]
				+ sizeof( float ) * 2 * geometry->staticVertex;
		}
	}

	VK_BindVertexBuffers( cmd, buffers, offsets );

	if ( geometry->staticFirstIndex >= 0 ) {
		VK_BindIndexBuffer( cmd, buffer, vk_static.indexOffset );
		return geometry->staticFirstIndex;
	}

	if ( !geometry->version || vk_lastStaticIndexes.epoch != vk_vertexEpoch || vk_lastStaticIndexes.version != geometry->version
		|| vk_lastStaticIndexes.indexes != indexes || vk_lastStaticIndexes.numIndexes != numIndexes ) {
		VkDeviceSize size = sizeof( glIndex_t ) * numIndexes;
		byte *data = VK_VertexAlloc( size, &vk_lastStaticIndexes.buffer, &vk_lastStaticIndexes.offset );

		Com_Memcpy( data, indexes, size );
		vk.stats.vertexBytes += size;
		vk_lastStaticIndexes.epoch = vk_vertexEpoch;
		vk_lastStaticIndexes.version = geometry->version;
		vk_lastStaticIndexes.indexes = indexes;
		vk_lastStaticIndexes.numIndexes = numIndexes;
	}
	VK_BindIndexBuffer( cmd, vk_lastStaticIndexes.buffer, vk_lastStaticIndexes.offset );
	return 0;
}

void vk_draw( primType_t prim, const drawGeometry_t *geometry, int numIndexes, const glIndex_t *indexes ) {
	vkTexture_t *textures[NUM_TEXTURE_BUNDLES];
	vkPipelineKey_t key;
	VkCommandBuffer cmd;
	VkPipeline pipeline;
	int i;

	if ( !vk.device || numIndexes <= 0 || geometry->numVertexes <= 0 || ( geometry->isSkinned ? !vk_skin.buffer.buffer
		: !( geometry->isStatic ? vk_static.buffer.buffer != VK_NULL_HANDLE : geometry->xyz != NULL ) ) ) {
		return;
	}

	VK_BeginRenderPass();
	cmd = vk.frame->mainCmd;

	// a unit is used when the geometry has texture coordinates for it
	// (skinned draws: unit 0 only)
	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		textures[i] = NULL;
		if ( geometry->isStatic ? geometry->staticSt[i] != STATIC_ST_NONE : geometry->st[i] != NULL && ( !i || !geometry->isSkinned ) ) {
			textures[i] = VK_GetTexture( vks.texture[i] );
			if ( !textures[i] ) {
				textures[i] = VK_DefaultTexture();
			}
			textures[i]->lastUseSegment = vk.segment;
		}
	}

	VK_BuildPipelineKey( &key, prim, geometry, textures );
	if ( key.bits == vk_lastKey.bits && key.alphaRef == vk_lastKey.alphaRef && vk_lastPipeline ) {
		pipeline = vk_lastPipeline;
	} else {
		pipeline = VK_GetPipeline( &key );
		vk_lastKey = key;
		vk_lastPipeline = pipeline;
	}
	if ( pipeline != vkb.pipeline ) {
		qvkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline );
		vkb.pipeline = pipeline;
		vk.stats.pipelineBinds++;
	}

	// every set the shaders declare must be bound, used or not
	for ( i = 0; i < NUM_TEXTURE_BUNDLES; i++ ) {
		VkDescriptorSet set = ( textures[i] ? textures[i] : VK_DefaultTexture() )->descriptorSet;

		if ( set != vkb.sets[i] ) {
			qvkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, i, 1, &set, 0, NULL );
			vkb.sets[i] = set;
			vk.stats.descriptorBinds++;
		}
	}

	// declared by every pipeline (the shader is specialized), so bound for all
	if ( !vkb.dlightSet ) {
		qvkCmdBindDescriptorSets( cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk.pipelineLayout, 2, 1, &vk.frame->dlightSet, 0, NULL );
		vkb.dlightSet = qtrue;
		vk.stats.descriptorBinds++;
	}

	VK_ApplyDynamicState( cmd, ( key.bits & VK_PIPE_STIPPLE ) != 0 );
	if ( geometry->isSkinned ) {
		uint32_t firstIndex;

		if ( geometry->skinFirstIndex >= 0 ) {
			VK_BindIndexBuffer( cmd, vk_skin.buffer.buffer, 0 );
			firstIndex = geometry->skinFirstIndex;
		} else {
			VkBuffer buffer;
			VkDeviceSize offset;
			byte *data = VK_VertexAlloc( sizeof( glIndex_t ) * numIndexes, &buffer, &offset );

			Com_Memcpy( data, indexes, sizeof( glIndex_t ) * numIndexes );
			vk.stats.vertexBytes += sizeof( glIndex_t ) * numIndexes;
			VK_BindIndexBuffer( cmd, buffer, offset );
			firstIndex = 0;
		}
		qvkCmdDrawIndexed( cmd, numIndexes, 1, firstIndex, geometry->skinVertex, geometry->skinDraw );
		vk.stats.skinDraws++;
	} else if ( geometry->isStatic ) {
		qvkCmdDrawIndexed( cmd, numIndexes, 1, VK_BindStaticGeometry( cmd, geometry, numIndexes, indexes ), 0, 0 );
		vk.stats.staticDraws++;
	} else {
		VK_UploadGeometry( cmd, geometry, numIndexes, indexes );
		qvkCmdDrawIndexed( cmd, numIndexes, 1, 0, 0, 0 );
	}
	vk.stats.draws++;
}

/*
===============
vk_set_static_geometry

The streams go into one device local buffer through the staging buffer and
the upload command buffer of the frame, which is submitted before anything
draws from them
===============
*/
void vk_set_static_geometry( const staticGeometry_t *geometry ) {
	VkDeviceSize xyzSize, stSize, colorSize, size;
	VkBufferMemoryBarrier barrier;
	VkBufferCopy region;
	VkCommandBuffer cmd;
	VkBuffer staging;
	VkDeviceSize stagingOffset;
	byte *data;
	int i;

	if ( !vk.device ) {
		return;
	}

	if ( vk_static.buffer.buffer ) {
		// draws already recorded may use it
		VK_DeferDestroy( VK_DESTROY_BUFFER, (uint64_t)vk_static.buffer.buffer, 0 );
		VK_DeferDestroy( VK_DESTROY_MEMORY, (uint64_t)vk_static.buffer.memory, 0 );
		Com_Memset( &vk_static, 0, sizeof( vk_static ) );
	}

	if ( !geometry || geometry->numVertexes <= 0 || geometry->numIndexes <= 0 ) {
		return;
	}

	xyzSize = PAD( sizeof( float ) * 3 * geometry->numVertexes, 16 );
	stSize = PAD( sizeof( float ) * 2 * geometry->numVertexes, 16 );
	colorSize = PAD( 4 * Q_max( geometry->numColors, 1 ), 16 );
	vk_static.xyzOffset = 0;
	vk_static.stOffset[0] = xyzSize;
	vk_static.stOffset[1] = vk_static.stOffset[0] + stSize;
	vk_static.colorOffset = vk_static.stOffset[1] + stSize;
	vk_static.indexOffset = vk_static.colorOffset + colorSize;
	size = vk_static.indexOffset + sizeof( glIndex_t ) * geometry->numIndexes;

	VK_CreateBuffer( &vk_static.buffer, size,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0 );

	data = VK_StagingAlloc( size, 16, &staging, &stagingOffset );
	Com_Memset( data, 0, size );
	Com_Memcpy( data + vk_static.xyzOffset, geometry->xyz, sizeof( float ) * 3 * geometry->numVertexes );
	for ( i = 0; i < 2; i++ ) {
		Com_Memcpy( data + vk_static.stOffset[i], geometry->st[i], sizeof( float ) * 2 * geometry->numVertexes );
	}
	if ( geometry->numColors > 0 ) {
		Com_Memcpy( data + vk_static.colorOffset, geometry->colors, 4 * geometry->numColors );
	}
	Com_Memcpy( data + vk_static.indexOffset, geometry->indexes, sizeof( glIndex_t ) * geometry->numIndexes );

	cmd = VK_GetUploadCommandBuffer();
	region.srcOffset = stagingOffset;
	region.dstOffset = 0;
	region.size = size;
	qvkCmdCopyBuffer( cmd, staging, vk_static.buffer.buffer, 1, &region );

	// the upload command buffer is submitted before the main one
	Com_Memset( &barrier, 0, sizeof( barrier ) );
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = vk_static.buffer.buffer;
	barrier.offset = 0;
	barrier.size = VK_WHOLE_SIZE;
	qvkCmdPipelineBarrier( cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, NULL, 1, &barrier, 0, NULL );

	ri.Printf( PRINT_DEVELOPER, "Vulkan: static geometry %d vertexes, %d colors, %d indexes, %d KB\n",
		geometry->numVertexes, geometry->numColors, geometry->numIndexes, (int)( size / 1024 ) );
}

/*
===============
vk_skin_available
===============
*/
qboolean vk_skin_available( void ) {
	return vk.device && vk_skin.buffer.buffer && vk.skinVertexShader;
}

/*
===============
vk_add_skin_data

Through the staging buffer and the upload command buffer of the frame,
which is submitted before the main one; nothing recorded before uses the
words written
===============
*/
int vk_add_skin_data( const uint32_t *words, int numWords ) {
	VkBufferMemoryBarrier barrier;
	VkBufferCopy region;
	VkCommandBuffer cmd;
	VkBuffer staging;
	VkDeviceSize stagingOffset;
	void *data;
	int first;

	if ( !vk_skin_available() || numWords <= 0 ) {
		return -1;
	}
	first = PAD( vk_skin.words, 8 );
	if ( first + numWords > VK_SKIN_DATA_WORDS || (VkDeviceSize)numWords * 4 > VK_STAGING_BUFFER_SIZE ) {
		if ( !vk_skin.full ) {
			ri.Printf( PRINT_ALL, "Vulkan: skin data full (%d MB), more skeletal models are skinned on the CPU\n",
				VK_SKIN_DATA_WORDS / ( 256 * 1024 ) );
			vk_skin.full = qtrue;
		}
		return -1;
	}
	VK_BeginFrameIfNeeded();

	data = VK_StagingAlloc( (VkDeviceSize)numWords * 4, 16, &staging, &stagingOffset );
	Com_Memcpy( data, words, (size_t)numWords * 4 );

	cmd = VK_GetUploadCommandBuffer();
	region.srcOffset = stagingOffset;
	region.dstOffset = (VkDeviceSize)first * 4;
	region.size = (VkDeviceSize)numWords * 4;
	qvkCmdCopyBuffer( cmd, staging, vk_skin.buffer.buffer, 1, &region );

	Com_Memset( &barrier, 0, sizeof( barrier ) );
	barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = vk_skin.buffer.buffer;
	barrier.offset = region.dstOffset;
	barrier.size = region.size;
	qvkCmdPipelineBarrier( cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
		0, 0, NULL, 1, &barrier, 0, NULL );

	vk_skin.words = first + numWords;
	return first;
}

void vk_clear_skin_data( void ) {
	if ( !vk_skin.words ) {
		return;
	}
	// frames in flight may still read it
	vk_finish();
	vk_skin.words = 0;
	vk_skin.full = qfalse;
}

int vk_alloc_skin_frame( int numWords, uint32_t **data ) {
	int first;

	if ( !vk_skin_available() ) {
		return -1;
	}
	VK_BeginFrameIfNeeded();
	first = PAD( vk.frame->skinWords, 8 );
	if ( first + numWords > VK_SKIN_FRAME_WORDS ) {
		return -1;
	}
	vk.frame->skinWords = first + numWords;
	*data = (uint32_t *)vk.frame->skinBuffer.mapped + first;
	return first;
}

unsigned int vk_frame_serial( void ) {
	if ( !vk.device ) {
		return 0;
	}
	VK_BeginFrameIfNeeded();
	return (unsigned int)vk.frameSerial;
}

/*
===============
VK_ClearComponent

A clear color component rounded to the nearest 8-bit value of the color
target. Drivers convert clear colors with float arithmetic that can round
the other way near a half step (RADV packs 0.7f as 179, GL gives 178, the
correctly rounded value); a color that is exactly representable converts
the same everywhere.
===============
*/
static float VK_ClearComponent( float c ) {
	return (float)( floor( Com_Clamp( 0.0f, 1.0f, c ) * 255.0 + 0.5 ) / 255.0 );
}

/*
===============
VK_ClearColorMasked

glClear of the color buffer with a partial color mask: a quad over the
scissor rectangle with the current mask, since vkCmdClearAttachments
ignores the mask
===============
*/
static void VK_ClearColorMasked( const vec4_t color ) {
	static const float xyz[4][3] = { { -1, -1, 0 }, { 1, -1, 0 }, { 1, 1, 0 }, { -1, 1, 0 } };
	static const glIndex_t indexes[6] = { 0, 1, 2, 0, 2, 3 };
	drawGeometry_t geometry;
	vkState_t saved;
	int i;

	saved = vks;

	vks.stateBits = GLS_DEPTHTEST_DISABLE;
	vks.cull = CULL_NONE;
	vks.polygonOffset = qfalse;
	vks.stencilTest = qfalse;
	vks.fog = qfalse;
	vks.clipPlaneEnabled = qfalse;
	vks.viewport[0] = vks.viewport[1] = 0;
	vks.viewport[2] = vk.width;
	vks.viewport[3] = vk.height;
	Com_Memset( vks.projection, 0, sizeof( vks.projection ) );
	Com_Memset( vks.modelview, 0, sizeof( vks.modelview ) );
	for ( i = 0; i < 4; i++ ) {
		vks.projection[i * 5] = vks.modelview[i * 5] = 1.0f;
	}
	vk_dirty |= VK_DIRTY_VIEWPORT | VK_DIRTY_MVP;

	Com_Memset( &geometry, 0, sizeof( geometry ) );
	geometry.numVertexes = 4;
	geometry.xyz = xyz[0];
	geometry.xyzStride = sizeof( xyz[0] );
	for ( i = 0; i < 4; i++ ) {
		geometry.color[i] = VK_ClearComponent( color[i] );
	}
	vk_draw( PRIM_TRIANGLES, &geometry, 6, indexes );

	vks = saved;
	vk_dirty |= VK_DIRTY_VIEWPORT | VK_DIRTY_MVP;
}

void vk_clear( int buffers, const vec4_t color ) {
	VkClearAttachment attachments[2];
	VkClearRect rect;
	qboolean maskedColor;
	uint32_t numAttachments;

	if ( !vk.device ) {
		return;
	}

	if ( !VK_ScissorRect( &rect.rect ) ) {
		return;
	}
	rect.baseArrayLayer = 0;
	rect.layerCount = 1;

	numAttachments = 0;
	maskedColor = qfalse;

	// alpha isn't stored (no destination alpha, see vk_pipeline.c), so only
	// the RGB mask matters
	if ( ( buffers & CLEAR_COLOR ) && color && ( vks.colorMask[0] || vks.colorMask[1] || vks.colorMask[2] ) ) {
		if ( vks.colorMask[0] && vks.colorMask[1] && vks.colorMask[2] ) {
			attachments[numAttachments].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			attachments[numAttachments].colorAttachment = 0;
			attachments[numAttachments].clearValue.color.float32[0] = VK_ClearComponent( color[0] );
			attachments[numAttachments].clearValue.color.float32[1] = VK_ClearComponent( color[1] );
			attachments[numAttachments].clearValue.color.float32[2] = VK_ClearComponent( color[2] );
			attachments[numAttachments].clearValue.color.float32[3] = 1.0f;
			numAttachments++;
		} else {
			maskedColor = qtrue;
		}
	}

	// like glClear, the depth write mask and the stencil write mask apply
	// (a partial stencil mask clears all bits; the renderer never sets one)
	attachments[numAttachments].aspectMask = 0;
	if ( ( buffers & CLEAR_DEPTH ) && ( vks.stateBits & GLS_DEPTHMASK_TRUE ) ) {
		attachments[numAttachments].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
	}
	if ( ( buffers & CLEAR_STENCIL ) && ( vks.stencilWriteMask & 0xFF ) ) {
		attachments[numAttachments].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	if ( attachments[numAttachments].aspectMask ) {
		attachments[numAttachments].colorAttachment = 0;
		attachments[numAttachments].clearValue.depthStencil.depth = 1.0f;
		attachments[numAttachments].clearValue.depthStencil.stencil = 0;
		numAttachments++;
	}

	if ( !numAttachments && !maskedColor ) {
		return;
	}

	VK_BeginRenderPass();
	if ( numAttachments ) {
		qvkCmdClearAttachments( vk.frame->mainCmd, numAttachments, attachments, 1, &rect );
	}
	if ( maskedColor ) {
		VK_ClearColorMasked( color );
	}
}

/*
===============
VK_DepthImageStages, VK_DepthImageWriteAccess

How the depth/stencil image is written: by draws, or with multisampling by
the resolve at the end of each render pass, which is a color attachment
write even for depth/stencil
===============
*/
static VkPipelineStageFlags VK_DepthImageStages( void ) {
	VkPipelineStageFlags stages = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	}
	return stages;
}

static VkAccessFlags VK_DepthImageWriteAccess( void ) {
	VkAccessFlags access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	if ( vk.samples != VK_SAMPLE_COUNT_1_BIT ) {
		access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	}
	return access;
}

void vk_read_pixels( int x, int y, int width, int height, readFormat_t format, void *buffer ) {
	VkBufferImageCopy region;
	VkImageAspectFlags aspect, barrierAspect;
	VkImageLayout layout;
	VkPipelineStageFlags stages;
	VkAccessFlags access;
	VkCommandBuffer cmd;
	VkImage image;
	VkDeviceSize size;
	int pixelSize, texelSize, red, blue;
	int x0, y0, x1, y1, w, h, row, col;

	switch ( format ) {
	case READ_RGB8:
		pixelSize = 3;
		break;
	case READ_DEPTH_FLOAT:
		pixelSize = sizeof( float );
		break;
	case READ_STENCIL8:
	default:
		pixelSize = 1;
		break;
	}

	if ( width <= 0 || height <= 0 ) {
		return;
	}
	// pixels outside of the framebuffer read as 0
	Com_Memset( buffer, 0, (size_t)pixelSize * width * height );

	x0 = Q_max( x, 0 );
	y0 = Q_max( y, 0 );
	x1 = Q_min( x + width, vk.width );
	y1 = Q_min( y + height, vk.height );
	if ( !vk.device || x1 <= x0 || y1 <= y0 ) {
		return;
	}
	w = x1 - x0;
	h = y1 - y0;

	VK_BeginFrameIfNeeded();
	if ( format != READ_RGB8 ) {
		// zeros until the depth image is readable (vk_want_depth_reads)
		vk.depthReadSerial = vk.frameSerial;
		if ( !vk.depthReadable ) {
			return;
		}
	}
	VK_EndRenderPass();
	cmd = vk.frame->mainCmd;

	if ( format == READ_RGB8 ) {
		// the color target of the frame; before anything is drawn in the
		// frame (no target chosen yet) the offscreen image, whose contents
		// are then stale or undefined, like a GL back buffer's
		image = vk.target != VK_TARGET_NONE ? vk.targetImage : vk.colorImage;
		aspect = barrierAspect = VK_IMAGE_ASPECT_COLOR_BIT;
		layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		texelSize = 4;
	} else {
		VK_ResolveDepth();
		image = vk.depthImage;
		aspect = format == READ_DEPTH_FLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
		barrierAspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
		layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		stages = VK_DepthImageStages();
		access = VK_DepthImageWriteAccess();
		if ( format == READ_STENCIL8 ) {
			texelSize = 1;
		} else if ( vk.depthFormat == VK_FORMAT_D16_UNORM_S8_UINT ) {
			texelSize = 2;
		} else {
			texelSize = 4;
		}
	}

	size = (VkDeviceSize)w * h * texelSize;
	if ( vk.readbackBuffer.size < size ) {
		// readbacks are waited for, nothing uses the old buffer
		VK_DestroyBuffer( &vk.readbackBuffer );
		VK_CreateBuffer( &vk.readbackBuffer, PAD( size, 65536 ), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT );
	}

	VK_ImageBarrier( cmd, image, barrierAspect, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		stages, access, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );

	Com_Memset( &region, 0, sizeof( region ) );
	region.imageSubresource.aspectMask = aspect;
	region.imageSubresource.layerCount = 1;
	region.imageOffset.x = x0;
	region.imageOffset.y = vk.height - y1;
	region.imageExtent.width = w;
	region.imageExtent.height = h;
	region.imageExtent.depth = 1;
	qvkCmdCopyImageToBuffer( cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readbackBuffer.buffer, 1, &region );

	VK_ImageBarrier( cmd, image, barrierAspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0, stages, access | ( format == READ_RGB8 ? VK_ACCESS_COLOR_ATTACHMENT_READ_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT ) );

	VK_HostReadBarrier( cmd, vk.readbackBuffer.buffer );

	VK_FlushAndWait();

	// the copy has the top row first, the caller wants the bottom row first;
	// B8G8R8A8 targets have red third
	red = vk.colorFormat == VK_FORMAT_B8G8R8A8_UNORM ? 2 : 0;
	blue = 2 - red;
	for ( row = 0; row < h; row++ ) {
		const byte *src = vk.readbackBuffer.mapped + (size_t)( h - 1 - row ) * w * texelSize;
		byte *dst = (byte *)buffer + ( (size_t)( y0 - y + row ) * width + ( x0 - x ) ) * pixelSize;

		switch ( format ) {
		case READ_RGB8:
			for ( col = 0; col < w; col++ ) {
				dst[col * 3 + 0] = src[col * 4 + red];
				dst[col * 3 + 1] = src[col * 4 + 1];
				dst[col * 3 + 2] = src[col * 4 + blue];
			}
			break;
		case READ_DEPTH_FLOAT:
			for ( col = 0; col < w; col++ ) {
				float depth = VK_DepthTexelValue( src + col * texelSize );

				Com_Memcpy( dst + col * sizeof( float ), &depth, sizeof( float ) );
			}
			break;
		case READ_STENCIL8:
		default:
			Com_Memcpy( dst, src, w );
			break;
		}
	}
}

/*
===============
vk_queue_depth_reads

Records copies of single depth texels into the frame's depth read buffer,
in order with the draws; the results are collected when the frame slot is
reused (VK_CollectDepthReads), so nothing waits for the GPU. An id is the
frame serial (low 24 bits) and the index of the read in the frame.
===============
*/
void vk_queue_depth_reads( int count, const int *xy, unsigned int *ids ) {
	VkBufferImageCopy regions[VK_MAX_DEPTH_READS];
	VkCommandBuffer cmd;
	vkFrame_t *frame;
	int i, numRegions;

	if ( count <= 0 ) {
		return;
	}
	if ( !vk.device ) {
		Com_Memset( ids, 0, sizeof( *ids ) * count );
		return;
	}

	VK_BeginFrameIfNeeded();
	frame = vk.frame;

	// lost until the depth image is readable (vk_want_depth_reads)
	vk.depthReadSerial = vk.frameSerial;
	if ( !vk.depthReadable ) {
		Com_Memset( ids, 0, sizeof( *ids ) * count );
		return;
	}

	numRegions = 0;
	for ( i = 0; i < count; i++ ) {
		int x = xy[i * 2 + 0];
		int y = xy[i * 2 + 1];
		int index = frame->numDepthReads;

		if ( index == VK_MAX_DEPTH_READS ) {
			ids[i] = 0;
			continue;
		}
		frame->numDepthReads++;
		ids[i] = (unsigned int)( ( frame->serial & 0xFFFFFF ) << 8 ) | index;

		frame->depthReadInside[index] = x >= 0 && y >= 0 && x < vk.width && y < vk.height;
		if ( !frame->depthReadInside[index] ) {
			continue;
		}

		Com_Memset( &regions[numRegions], 0, sizeof( regions[numRegions] ) );
		regions[numRegions].bufferOffset = index * 4;
		regions[numRegions].imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		regions[numRegions].imageSubresource.layerCount = 1;
		regions[numRegions].imageOffset.x = x;
		regions[numRegions].imageOffset.y = vk.height - 1 - y;
		regions[numRegions].imageExtent.width = 1;
		regions[numRegions].imageExtent.height = 1;
		regions[numRegions].imageExtent.depth = 1;
		numRegions++;
	}

	if ( !numRegions ) {
		return;
	}

	VK_EndRenderPass();
	VK_ResolveDepth();
	cmd = frame->mainCmd;

	VK_ImageBarrier( cmd, vk.depthImage, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
		VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_DepthImageStages(), VK_DepthImageWriteAccess(),
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );

	qvkCmdCopyImageToBuffer( cmd, vk.depthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame->depthReadBuffer.buffer, numRegions, regions );

	VK_ImageBarrier( cmd, vk.depthImage, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
		VK_DepthImageStages(), VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_DepthImageWriteAccess() );

	// the host reads the buffer after waiting for the frame fence
	VK_HostReadBarrier( cmd, frame->depthReadBuffer.buffer );
}

depthReadStatus_t vk_get_depth_read( unsigned int id, float *depth ) {
	unsigned int serial = id >> 8;
	int index = id & 255;
	int i;

	if ( !id || !vk.device ) {
		return DEPTHREAD_LOST;
	}

	for ( i = 0; i < VK_MAX_FRAMES; i++ ) {
		const vkFrame_t *frame = &vk.frames[i];

		if ( frame->resultSerial && ( frame->resultSerial & 0xFFFFFF ) == serial && index < frame->numDepthResults ) {
			*depth = frame->depthResults[index];
			return DEPTHREAD_DONE;
		}
		// recorded in a frame that is being recorded or still in flight
		if ( frame->serial && ( frame->serial & 0xFFFFFF ) == serial && index < frame->numDepthReads ) {
			return DEPTHREAD_PENDING;
		}
	}

	return DEPTHREAD_LOST;
}

/*
===============
vk_want_depth_reads

Makes the depth image readable from the start of the next frame, or of the
frame about to start when called between frames (see VK_UpdateDepthTarget)
===============
*/
void vk_want_depth_reads( void ) {
	vk.depthReadSerial = vk.frame ? vk.frameSerial : vk.frameSerial + 1;
}
