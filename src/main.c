
#include "cglm/types.h"
#include "photon/photon_device.h"
#include "photon/photon_log.h"
#include "photon/photon_mesh.h"
#include "photon/photon_pipeline.h"
#include "photon/photon_status.h"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "photon/photon.h"

#include "stdio.h"

PhStatus renderTriangle(PhDeviceHandle device, PhPipeline *pipeline, PhMesh *mesh, PhImage *image, PhSemaphore *pWait, PhSemaphore *pSignal)
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


    VkImageMemoryBarrier toRender = {
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
    };
    vkCmdPipelineBarrier(buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, NULL, 0, NULL, 1, &toRender);

    VkRenderingAttachmentInfo colorAttachment = {
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = image->defaultView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
    };
    VkRenderingInfo renderingInfo = {
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = image->extent },
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
    };

    vkCmdBeginRendering(buffer, &renderingInfo);

    vkCmdBindPipeline(buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);

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
    vkCmdBindIndexBuffer(buffer, mesh->gpuIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
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
    PhSurfaceHandle hSurface                    = { 0 };
    PhDeviceHandle chosenDevice                 = { 0 };
    PhShaderModule triangleShader               = { 0 };
    PhGraphicsPipelineOptions pipelineOptions   = PH_PIPELINE_OPTIONS_DEFAULT;
    PhPipeline pipeline                         = { 0 };
    PhImage presentImage                        = { 0 };
    PhMesh  mesh                                = { 0 };

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
        .mode = VK_PRESENT_MODE_FIFO_KHR
    };
    PH_CHECK(PH_LOG_ERROR, ph_window_get_surface(hWindow, &hSurface));
    chosenDevice = deviceInfos.ptr[0].handle;
    PH_CHECK(PH_LOG_ERROR, ph_device_configure_for_present(chosenDevice, hSurface, presentOptions));
    PH_CHECK(PH_LOG_ERROR, ph_create_shader_module(deviceInfos.ptr[0].handle, SHADER_DIR "/triangle.spv", &triangleShader));

    static PhVertex vertices[] = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
    };
    static uint16_t indices[] = {
        0, 1, 2, 2, 3, 0
    };

    VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
    PhColorBlendAttachmentOptions blendAttachment = PH_COLOR_BLEND_ATTACHMENT_OPTIONS_DEFAULT;

    VkVertexInputAttributeDescription vertexAttributes[] = {
        {
            0,
            0,
            VK_FORMAT_R32G32_SFLOAT,
            offsetof(PhVertex, position)
        },
        {
            1,
            0,
            VK_FORMAT_R32G32B32_SFLOAT,
            offsetof(PhVertex, color)
        },
    };
    PhVertexSpan vertexSpan = PhVertexSpan_from(vertices, PH_NUM_ELEMS(vertices));
    PhIndexSpan indexSpan = PhIndexSpan_from(indices, PH_NUM_ELEMS(indices));

    VkVertexInputBindingDescription vertexBinding = {
        .binding = 0,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        .stride = sizeof(PhVertex),
    };

    pipelineOptions.inputStateInfo.pVertexAttributeDescriptions = vertexAttributes;
    pipelineOptions.inputStateInfo.vertexAttributeDescriptionCount = 2;
    pipelineOptions.inputStateInfo.pVertexBindingDescriptions = &vertexBinding;
    pipelineOptions.inputStateInfo.vertexBindingDescriptionCount = 1;

    pipelineOptions.pShaders                = &triangleShader;
    pipelineOptions.shaderCount             = 1UL;
    pipelineOptions.pColorAttachmentFormats = &colorFormat;
    pipelineOptions.colorAttachmentCount    = 1;
    pipelineOptions.pColorBlendAttachments  = &blendAttachment;
    PH_CHECK(PH_LOG_ERROR, ph_create_graphics_pipeline(chosenDevice, pipelineOptions, &pipeline));
    PH_CHECK(PH_LOG_ERROR, ph_device_create_staging_buffer(chosenDevice, 1024*1024*64));
    PH_CHECK(PH_LOG_ERROR, ph_mesh_create(chosenDevice, vertexSpan, indexSpan, &mesh));

    while(!ph_window_should_close(hWindow))
    {
        PhSemaphore renderSemaphore;
        PH_CHECK(PH_LOG_ERROR, ph_device_present_image_get_next(chosenDevice, &presentImage));
        PH_CHECK(PH_LOG_ERROR, renderTriangle(chosenDevice, &pipeline, &mesh, &presentImage, &presentImage.readySemaphore, &renderSemaphore));
        PH_CHECK(PH_LOG_ERROR, ph_device_present(chosenDevice, &renderSemaphore, 1UL));
        ph_window_poll_events(hWindow);
    }

    return 0;
}
