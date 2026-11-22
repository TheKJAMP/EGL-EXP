// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

struct EndRenderPassBarriers {
    std::array<VkImageMemoryBarrier, 9> barriers{};
    VkPipelineStageFlags src_stages{};
    u32 count{};
};

/// Builds post-render-pass image barriers scoped to each attachment's actual aspect mask
/// (color vs depth/stencil) so depth-only passes do not serialize color stages or accesses.
[[nodiscard]] inline EndRenderPassBarriers BuildEndRenderPassBarriers(
    u32 num_images, const std::array<VkImage, 9>& images,
    const std::array<VkImageSubresourceRange, 9>& ranges) noexcept {
    EndRenderPassBarriers result{};
    result.count = (std::min)(num_images, static_cast<u32>(result.barriers.size()));
    for (u32 i = 0; i < result.count; ++i) {
        const VkImageSubresourceRange& range = ranges[i];
        const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
        const bool is_depth_stencil =
            (range.aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;

        VkAccessFlags src_access = 0;
        VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT;

        if (is_color) {
            result.src_stages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            dst_access |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        } else if (is_depth_stencil) {
            result.src_stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            src_access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dst_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        } else {
            result.src_stages |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            src_access = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            dst_access |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        }

        result.barriers[i] = VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src_access,
            .dstAccessMask = dst_access,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = images[i],
            .subresourceRange = range,
        };
    }
    return result;
}

} // namespace Vulkan
