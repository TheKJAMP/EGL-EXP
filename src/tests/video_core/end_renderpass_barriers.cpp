// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdint>
#include <type_traits>
#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/end_renderpass_barriers.h"

namespace {
template <typename Handle>
Handle FakeHandle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(value);
    } else {
        return static_cast<Handle>(value);
    }
}

VkImageSubresourceRange MakeRange(VkImageAspectFlags aspect) {
    return VkImageSubresourceRange{
        .aspectMask = aspect,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}
} // namespace

TEST_CASE("EndRenderPass barriers handle empty attachment list", "[vulkan][end_renderpass_barriers]") {
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> ranges{};
    const auto built = Vulkan::BuildEndRenderPassBarriers(0, images, ranges);
    REQUIRE(built.count == 0);
    REQUIRE(built.src_stages == 0);
}

TEST_CASE("EndRenderPass barriers scope color-only MRTs without depth-stencil stages",
          "[vulkan][end_renderpass_barriers]") {
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> ranges{};
    images[0] = FakeHandle<VkImage>(1);
    images[1] = FakeHandle<VkImage>(2);
    ranges[0] = MakeRange(VK_IMAGE_ASPECT_COLOR_BIT);
    ranges[1] = MakeRange(VK_IMAGE_ASPECT_COLOR_BIT);

    const auto built = Vulkan::BuildEndRenderPassBarriers(2, images, ranges);
    REQUIRE(built.count == 2);
    REQUIRE(built.src_stages == VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    for (u32 i = 0; i < 2; ++i) {
        REQUIRE(built.barriers[i].image == images[i]);
        REQUIRE(built.barriers[i].srcAccessMask == VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        REQUIRE(built.barriers[i].dstAccessMask ==
                (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
        REQUIRE((built.barriers[i].dstAccessMask &
                 (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)) == 0);
    }
}

TEST_CASE("EndRenderPass barriers scope depth-only and depth-stencil passes without color stages",
          "[vulkan][end_renderpass_barriers]") {
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> ranges{};
    images[0] = FakeHandle<VkImage>(3);
    images[1] = FakeHandle<VkImage>(4);
    ranges[0] = MakeRange(VK_IMAGE_ASPECT_DEPTH_BIT);
    ranges[1] = MakeRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    const auto built = Vulkan::BuildEndRenderPassBarriers(2, images, ranges);
    REQUIRE(built.count == 2);
    REQUIRE(built.src_stages == (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT));

    for (u32 i = 0; i < 2; ++i) {
        REQUIRE(built.barriers[i].image == images[i]);
        REQUIRE(built.barriers[i].srcAccessMask == VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        REQUIRE(built.barriers[i].dstAccessMask ==
                (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
        REQUIRE((built.barriers[i].dstAccessMask &
                 (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                  VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)) == 0);
    }
}

TEST_CASE("EndRenderPass barriers keep per-attachment access masks for mixed color+depth passes",
          "[vulkan][end_renderpass_barriers]") {
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> ranges{};
    images[0] = FakeHandle<VkImage>(5);
    images[1] = FakeHandle<VkImage>(6);
    ranges[0] = MakeRange(VK_IMAGE_ASPECT_COLOR_BIT);
    ranges[1] = MakeRange(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    const auto built = Vulkan::BuildEndRenderPassBarriers(2, images, ranges);
    REQUIRE(built.count == 2);
    REQUIRE(built.src_stages == (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT));

    REQUIRE(built.barriers[0].srcAccessMask == VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    REQUIRE((built.barriers[0].dstAccessMask & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) != 0);
    REQUIRE((built.barriers[0].dstAccessMask & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) == 0);

    REQUIRE(built.barriers[1].srcAccessMask == VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    REQUIRE((built.barriers[1].dstAccessMask & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) != 0);
    REQUIRE((built.barriers[1].dstAccessMask & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) == 0);
}

TEST_CASE("EndRenderPass barriers fall back conservatively on unknown aspect masks",
          "[vulkan][end_renderpass_barriers]") {
    std::array<VkImage, 9> images{};
    std::array<VkImageSubresourceRange, 9> ranges{};
    images[0] = FakeHandle<VkImage>(7);
    ranges[0] = MakeRange(0);

    const auto built = Vulkan::BuildEndRenderPassBarriers(1, images, ranges);
    REQUIRE(built.count == 1);
    REQUIRE(built.src_stages == (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT));
    REQUIRE(built.barriers[0].srcAccessMask ==
            (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
    REQUIRE(built.barriers[0].dstAccessMask ==
            (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
}
