
#include "cglm/types.h"
#include "photon/photon_device.h"
#include "photon/photon_log.h"
#include "photon/photon_mesh.h"
#include "photon/photon_pipeline.h"
#include "photon/photon_status.h"
#include "photon/photon_ubo.h"
#include "photon/photon_window.h"
#include "string.h"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "photon/photon.h"

#include "stdio.h"

#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT

static PhStatus _depth_image_create(PhDeviceHandle hDevice, void *userdata, uint32_t frameIndex, void *out)
{
    (void)frameIndex;
    PhExtent2D *pExtent = (PhExtent2D *)userdata;
    PhImage *pImage = (PhImage *)out;

    PhImageCreateInfo info = {
        .width  = pExtent->width,
        .height = pExtent->height,
        .format = DEPTH_FORMAT,
        .usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
    };
    return ph_device_image_create(hDevice, &info, pImage);
}

static void _depth_image_destroy(PhDeviceHandle hDevice, void *resource)
{
    PhImage *pImage = (PhImage *)resource;
    ph_device_image_destroy(hDevice, pImage);
}

static PhStatus _depth_image_recreate(PhDeviceHandle hDevice, void *userdata, void *resource, PhExtent2D newExtent)
{
    (void)userdata;
    PhImage *pImage = (PhImage *)resource;
    ph_device_image_destroy(hDevice, pImage);

    PhImageCreateInfo info = {
        .width  = newExtent.width,
        .height = newExtent.height,
        .format = DEPTH_FORMAT,
        .usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
    };
    return ph_device_image_create(hDevice, &info, pImage);
}

typedef struct MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
} MVP;
typedef struct MVPDescriptorCreateParams {
    PhPipeline *pipeline;
    PhUBOPerFrame *uboPerFrame;
    PhImage *textureImage;
    PhSampler *textureSampler;
} MVPDescriptorCreateParams;

typedef struct MVPData {
    MVP mvp;
    PhUBOPerFrame perFrameMVPUBO;
    PhPerFrameResourceHandle descriptorSetHandle;
    PhImage textureImage;
    PhSampler textureSampler;
} MVPData;

static PhStatus _mvp_descriptor_create(PhDeviceHandle hDevice, void *userdata, uint32_t frameIndex, void *out)
{
    MVPDescriptorCreateParams *params = (MVPDescriptorCreateParams *)userdata;
    PhDescriptorSet *pSet = (PhDescriptorSet *)out;

    PhUBO *pUBO;
    PH_CHECK(PH_LOG_ERROR, ph_device_per_frame_get_at(hDevice, params->uboPerFrame->uboDataHandle, frameIndex, (void **)&pUBO));

    PH_CHECK(PH_LOG_ERROR, ph_device_descriptor_sets_allocate(hDevice, params->pipeline->pSetLayouts, 1, pSet));

    VkDescriptorBufferInfo bufInfo = {
        .buffer = pUBO->ubo.buffer,
        .offset = 0,
        .range  = sizeof(MVP),
    };
    VkDescriptorImageInfo imageInfo = {
        .sampler     = *params->textureSampler,
        .imageView   = params->textureImage->defaultView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    PhDescriptorWrite writes[] = {
        {
            .set          = *pSet,
            .binding      = 0,
            .arrayElement = 0,
            .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .count        = 1,
            .pBufferInfo  = &bufInfo,
        },
        {
            .set          = *pSet,
            .binding      = 1,
            .arrayElement = 0,
            .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .count        = 1,
            .pImageInfo   = &imageInfo,
        },
    };
    PH_CHECK(PH_LOG_ERROR, ph_device_descriptor_sets_write(hDevice, writes, 2));

    return PH_SUCCESS;
}

static void _mvp_descriptor_destroy(PhDeviceHandle hDevice, void *resource)
{
    PhDescriptorSet *pSet = (PhDescriptorSet *)resource;
    ph_device_descriptor_sets_free(hDevice, pSet, 1);
}

static PhStatus initMVP(PhDeviceHandle hDevice, PhPipeline *pipeline, MVPData *pOut)
{
    PhExtent2D screenDimensions;
    PH_CHECK(PH_LOG_ERROR, ph_device_extent_get(hDevice, &screenDimensions));

    glm_mat4_identity(pOut->mvp.model);

    glm_mat4_identity(pOut->mvp.view);
    glm_perspective(glm_rad(0.0f), (float)screenDimensions.width / (float)screenDimensions.height, 0.1f, 1000.0f, pOut->mvp.projection);
    pOut->mvp.projection[1][1] *= -1.0f;

    /* Generate a 256x256 checkerboard texture (RGBA8) */
    enum { TEX_W = 256, TEX_H = 256, TILE = 16 };
    uint8_t *pixels = malloc(TEX_W * TEX_H * 4);
    for (int y = 0; y < TEX_H; y++) {
        for (int x = 0; x < TEX_W; x++) {
            uint8_t c = ((x / TILE) + (y / TILE)) % 2 ? 255 : 64;
            uint8_t *p = &pixels[(y * TEX_W + x) * 4];
            p[0] = c; p[1] = c; p[2] = c; p[3] = 255;
        }
    }

    PhImageCreateInfo imgInfo = {
        .width  = TEX_W,
        .height = TEX_H,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
    };
    PH_CHECK(PH_LOG_ERROR, ph_device_image_create(hDevice, &imgInfo, &pOut->textureImage));
    PH_CHECK(PH_LOG_ERROR, ph_device_image_upload(hDevice, pixels, TEX_W * TEX_H * 4, &pOut->textureImage));
    free(pixels);

    PhSamplerCreateInfo samplerInfo = {
        .magFilter        = VK_FILTER_LINEAR,
        .minFilter        = VK_FILTER_LINEAR,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_TRUE,
    };
    PH_CHECK(PH_LOG_ERROR, ph_device_sampler_create(hDevice, &samplerInfo, &pOut->textureSampler));

    PhUBOCreateInfo createInfo = {
        .size = sizeof(pOut->mvp)
    };
    PH_CHECK(PH_LOG_ERROR, ph_ubo_per_frame_create(hDevice, &createInfo, &pOut->perFrameMVPUBO));

    MVPDescriptorCreateParams descriptorParams = {
        .pipeline       = pipeline,
        .uboPerFrame    = &pOut->perFrameMVPUBO,
        .textureImage   = &pOut->textureImage,
        .textureSampler = &pOut->textureSampler,
    };
    PH_CHECK(PH_LOG_ERROR,
        ph_device_per_frame_register(hDevice,
            sizeof(PhDescriptorSet),
            _mvp_descriptor_create,
            _mvp_descriptor_destroy,
            NULL,
            &pOut->descriptorSetHandle));
    PH_CHECK(PH_LOG_ERROR,
        ph_device_per_frame_create(hDevice, pOut->descriptorSetHandle, &descriptorParams));

    return PH_SUCCESS;
}

static PhStatus updateMVP(PhDeviceHandle hDevice, MVPData *pData, uint32_t frameIdx)
{
    PhExtent2D screenDimensions;
    PH_CHECK(PH_LOG_ERROR, ph_device_extent_get(hDevice, &screenDimensions));

    glm_mat4_identity(pData->mvp.model);

    glm_perspective(glm_rad(90.0f), (float)screenDimensions.width / (float)screenDimensions.height, 0.1f, 1000.0f, pData->mvp.projection);
    pData->mvp.projection[1][1] *= -1.0f;

    PhUBO *pUBO;
    PH_CHECK(PH_LOG_ERROR, ph_ubo_per_frame_get(hDevice, &pData->perFrameMVPUBO, &pUBO));
    memcpy(pUBO->ubo.hostPtr, &pData->mvp, sizeof(pData->mvp));

    return PH_SUCCESS;
}

PhStatus renderTriangle(PhDeviceHandle device, PhPipeline *pipeline, PhMesh *mesh, PhImage *image, PhImage *depthImage, PhSemaphore *pWait, PhSemaphore *pSignal, MVPData *pMVP)
{
    PhCommandBuffer buffer = { 0 };
    uint32_t imageIndex = 0;
    PhSemaphore finishedSemaphore = { 0 };

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    PH_CHECK(PH_LOG_ERROR, ph_device_semaphore_create(device, &finishedSemaphore));
    PH_CHECK(PH_LOG_ERROR, ph_device_command_buffer_create(device, PH_QUEUE_TYPE_GRAPHICS_BIT, 1, &buffer));
    PH_VK_CHECK(PH_LOG_ERROR, vkBeginCommandBuffer(buffer, &beginInfo));

    VkImageMemoryBarrier barriers[] = {
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask       = 0,
            .dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image->image,
            .subresourceRange    = {
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        },
        {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask       = 0,
            .dstAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = depthImage->image,
            .subresourceRange    = {
                .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        },
    };
    vkCmdPipelineBarrier(buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0, 0, NULL, 0, NULL, 2, barriers);

    VkRenderingAttachmentInfo colorAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = image->defaultView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
    };
    VkRenderingAttachmentInfo depthAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = depthImage->defaultView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .clearValue  = { .depthStencil = { .depth = 1.0f, .stencil = 0 } },
    };
    VkRenderingInfo renderingInfo = {
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = image->extent },
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthAttachment,
    };

    vkCmdBeginRendering(buffer, &renderingInfo);

    vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

    PhDescriptorSet *pMvpSet;
    PH_CHECK(PH_LOG_ERROR, ph_device_per_frame_get(device, pMVP->descriptorSetHandle, (void **)&pMvpSet));
    vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->layout, 0, 1, pMvpSet, 0, NULL);

    VkViewport viewport = {
        .x        = 0.0f,
        .y        = 0.0f,
       .width    = (float)image->extent.width,
        .height   = (float)image->extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(buffer, 0, 1, &viewport);

    VkRect2D scissor = { .offset = { 0, 0 }, .extent = image->extent };
    vkCmdSetScissor(buffer, 0, 1, &scissor);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(buffer, 0, 1, &mesh->gpuVertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(buffer, mesh->gpuIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(buffer, mesh->indices.len, 1, 0, 0, 0);

    vkCmdEndRendering(buffer);
    PH_VK_CHECK(PH_LOG_ERROR, vkEndCommandBuffer(buffer));


    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkSubmitInfo submitInfo = {
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = NULL,
        .waitSemaphoreCount   = 1,
        .pWaitSemaphores      = pWait,
        .pWaitDstStageMask    = &waitStage,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = &finishedSemaphore,
    };

    PH_CHECK(PH_LOG_ERROR, 
        ph_device_queue_submit(device, PH_QUEUE_TYPE_GRAPHICS_BIT, &submitInfo));
    
    *pSignal = finishedSemaphore;
    return PH_SUCCESS;
}

int main(void) {
    PhInstanceHandle hInstance                  = { 0 };
    PhWindowHandle hWindow                      = { 0 };
    PhCapability deviceCaps                     = { 0 };
    PhDeviceInfoSpan deviceInfos                = { 0 };
    PhPresentOptions presentOptions             = { 0 };
    PhDeviceHandle chosenDevice                 = { 0 };
    PhShaderModule triangleShader               = { 0 };
    PhGraphicsPipelineOptions pipelineOptions   = PH_PIPELINE_OPTIONS_DEFAULT;
    PhPipeline pipeline                         = { 0 };
    PhImage presentImage                        = { 0 };
    PhPerFrameResourceHandle depthImageHandle   = 0;
    PhMesh  mesh                                = { 0 };
    MVPData mvp                                 = { 0 };
    size_t frameIdx                             = 0;
    size_t frameCount                           = 0;
    double lastTime                             = 0;

    PhInstanceSettings instanceSettings = {
        .appName = "Farcaster",
        .appVersion = 1,
        .enableDebug = true,
    };
    PH_CHECK(PH_LOG_ERROR, ph_create_instance(&instanceSettings, &hInstance));

    PhWindowSettings windowSettings = {
        .height = 1080,
        .width = 1920,
        .resizable = true,
        .title = "Farcaster",
        .hInstance = hInstance
    };
    PH_CHECK(PH_LOG_ERROR, ph_create_window(&windowSettings, &hWindow));

    deviceCaps = (PhCapability) {
#ifndef __APPLE__
        .asyncComputeQueue = true,
        .dedicatedTransfer = true,
        .rtCapable = true,
        .discrete = true,
#endif
        .swapchain = true,
        .graphicsQueue = true,
        .samplerAnisotropy = true,
        .minimumImageDimensions = {
            .height = 1080,
            .width = 1920,
        },
        .timelineSemaphore = true,
        .synchronization2 = true,
        .descriptorIndexing = true,
        .dynamicRendering = true,
    };

    PH_CHECK(PH_LOG_ERROR, ph_devices_enumerate(hInstance, deviceCaps, &deviceInfos));

    presentOptions = (PhPresentOptions) {
        .format = {
            .format = VK_FORMAT_B8G8R8A8_SRGB,
            .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        },
        .mode = VK_PRESENT_MODE_IMMEDIATE_KHR
    };
    chosenDevice = deviceInfos.ptr[0].handle;
    PH_CHECK(PH_LOG_ERROR, ph_device_window_attach(chosenDevice, hWindow, presentOptions));
    PH_CHECK(PH_LOG_ERROR, ph_create_shader_module(deviceInfos.ptr[0].handle, SHADER_DIR "/triangle.spv", &triangleShader));

    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    PhColorBlendAttachmentOptions blendAttachment = PH_COLOR_BLEND_ATTACHMENT_OPTIONS_DEFAULT;

    VkVertexInputAttributeDescription vertexAttributes[] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PhVertex, position) },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PhVertex, normal)   },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(PhVertex, texCoord) },
    };

    VkVertexInputBindingDescription vertexBinding = {
        .binding = 0,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        .stride = sizeof(PhVertex),
    };

    pipelineOptions.inputStateInfo.pVertexAttributeDescriptions = vertexAttributes;
    pipelineOptions.inputStateInfo.vertexAttributeDescriptionCount = 3;
    pipelineOptions.inputStateInfo.pVertexBindingDescriptions = &vertexBinding;
    pipelineOptions.inputStateInfo.vertexBindingDescriptionCount = 1;

    pipelineOptions.rasterStateInfo.cullMode           = VK_CULL_MODE_FRONT_BIT;
    pipelineOptions.stencilStateInfo.depthTestEnable   = VK_TRUE;
    pipelineOptions.stencilStateInfo.depthWriteEnable  = VK_TRUE;
    pipelineOptions.stencilStateInfo.depthCompareOp    = VK_COMPARE_OP_LESS;
    pipelineOptions.depthAttachmentFormat              = DEPTH_FORMAT;
    pipelineOptions.pShaders                = &triangleShader;
    pipelineOptions.shaderCount             = 1UL;
    pipelineOptions.pColorAttachmentFormats = &colorFormat;
    pipelineOptions.colorAttachmentCount    = 1;
    pipelineOptions.pColorBlendAttachments  = &blendAttachment;

    PH_CHECK(PH_LOG_ERROR, ph_create_graphics_pipeline(chosenDevice, pipelineOptions, &pipeline));
    PH_CHECK(PH_LOG_ERROR, ph_device_create_staging_buffer(chosenDevice, 1024*1024*256));
    PH_CHECK(PH_LOG_ERROR, ph_mesh_create_from_file(chosenDevice, ASSET_DIR "/sponza/main_sponza/NewSponza_Main_glTF_003.gltf", &mesh));

    PhExtent2D extent;
    PH_CHECK(PH_LOG_ERROR, ph_device_extent_get(chosenDevice, &extent));
    PH_CHECK(PH_LOG_ERROR,
        ph_device_per_frame_register(chosenDevice, sizeof(PhImage),
            _depth_image_create, _depth_image_destroy, _depth_image_recreate, &depthImageHandle));
    PH_CHECK(PH_LOG_ERROR, ph_device_per_frame_create(chosenDevice, depthImageHandle, &extent));

    PH_CHECK(PH_LOG_ERROR, initMVP(chosenDevice, &pipeline, &mvp));


    while(!ph_window_should_close(hWindow))
    {
        PhSemaphore renderSemaphore;
        PH_CHECK(PH_LOG_ERROR, ph_device_frame_index_get(chosenDevice, &frameIdx));
        PhStatus presentStatus = ph_device_present_image_get_next(chosenDevice, &presentImage);
        while (presentStatus == PH_ERR_SWAPCHAIN_OUT_OF_DATE || presentStatus == PH_ERR_SWAPCHAIN_SUBOPTIMAL)
        {
            PH_CHECK(PH_LOG_ERROR, ph_device_swapchain_recreate(chosenDevice));
            presentStatus = ph_device_present_image_get_next(chosenDevice, &presentImage);
        }
        PhImage *pDepthImage;
        PH_CHECK(PH_LOG_ERROR, ph_device_per_frame_get(chosenDevice, depthImageHandle, (void **)&pDepthImage));
        PH_CHECK(PH_LOG_ERROR, ph_device_camera_view_get(chosenDevice, mvp.mvp.view));
        PH_CHECK(PH_LOG_ERROR, updateMVP(chosenDevice, &mvp, frameIdx));
        PH_CHECK(PH_LOG_ERROR, renderTriangle(chosenDevice, &pipeline, &mesh, &presentImage, pDepthImage, &presentImage.readySemaphore, &renderSemaphore, &mvp));
        PH_CHECK(PH_LOG_ERROR, ph_device_present(chosenDevice, &renderSemaphore, 1UL));
        ph_window_poll_events(hWindow);

        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 1.0) {
            printf("FPS: %lu\n", frameCount);
            frameCount = 0;
            lastTime = now;
        }
    }

    return 0;
}
