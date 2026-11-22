// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "common/common_types.h"
#include "common/literals.h"
#include "common/range_sets.inc"
#include "video_core/buffer_cache/download_coalesce.h"

namespace {

using namespace Common::Literals;

TEST_CASE("DownloadCoalesce: single 4 KiB range stays 4 KiB", "[download_coalesce]") {
    Common::RangeSet<DAddr> gpu_modified;
    gpu_modified.Add(0x1000'0000, 4_KiB);

    const DAddr result = VideoCommon::CoalesceContiguousDownloadEnd(
        0x1000'0000, 0x1000'1000, 0x1010'0000, 4_KiB, 256_KiB, gpu_modified);
    CHECK(result == 0x1000'1000);
}

TEST_CASE("DownloadCoalesce: contiguous modified pages coalesce up to cap", "[download_coalesce]") {
    Common::RangeSet<DAddr> gpu_modified;
    gpu_modified.Add(0x1000'0000, 128_KiB);

    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x1000'0000, 0x1000'1000, 0x1010'0000, 4_KiB, 256_KiB, gpu_modified) ==
          0x1000'0000 + 128_KiB);

    // Adjacent unaligned sub-page writes across consecutive pages still coalesce.
    Common::RangeSet<DAddr> subpage;
    subpage.Add(0x2000'0100, 0x200);
    subpage.Add(0x2000'1200, 0x100);
    subpage.Add(0x2000'2800, 0x400);
    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x2000'0000, 0x2000'1000, 0x2010'0000, 4_KiB, 256_KiB, subpage) ==
          0x2000'3000);
}

TEST_CASE("DownloadCoalesce: stops at page gaps and caps at max_coalesce_bytes and buffer_end",
          "[download_coalesce]") {
    Common::RangeSet<DAddr> gapped;
    gapped.Add(0x1000'0000, 16_KiB);
    gapped.Add(0x1000'5000, 16_KiB); // 4 KiB gap at 0x1000'4000..0x1000'5000

    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x1000'0000, 0x1000'1000, 0x1010'0000, 4_KiB, 256_KiB, gapped) ==
          0x1000'4000);

    Common::RangeSet<DAddr> large;
    large.Add(0x1000'0000, 1_MiB);
    // Capped at 256 KiB.
    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x1000'0000, 0x1000'1000, 0x1020'0000, 4_KiB, 256_KiB, large) ==
          0x1000'0000 + 256_KiB);

    // Clipped to buffer_end.
    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x1000'0000, 0x1000'1000, 0x1002'0000, 4_KiB, 256_KiB, large) ==
          0x1002'0000);

    // Request already >= cap returns end untouched (clipped only to buffer_end).
    CHECK(VideoCommon::CoalesceContiguousDownloadEnd(
              0x1000'0000, 0x1004'0000, 0x1020'0000, 4_KiB, 256_KiB, large) ==
          0x1004'0000);
}

} // namespace
