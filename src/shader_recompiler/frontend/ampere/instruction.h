// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Shader::Ampere {

struct Instruction {
    // Trivial so UnpackNativeAmpereCode can memcpy packed words into it.
    std::array<u32, 4> words;

    [[nodiscard]] constexpr u32 Bits(u32 offset, u32 count) const noexcept {
        u64 bits = words[offset / 32];
        if (offset / 32 + 1 < words.size()) {
            bits |= u64{words[offset / 32 + 1]} << 32;
        }
        return static_cast<u32>((bits >> (offset % 32)) & ((u64{1} << count) - 1));
    }
};
static_assert(sizeof(Instruction) == 16);

struct LoweringOptions {
    bool geometry{};
    bool tensor{};
    bool compact_tensor{};
    u64 native_code_address{};
};

/// Non-owning view of a native Ampere program ready for direct translation.
struct NativeAmpereView {
    std::span<const Instruction> code;
    LoweringOptions options{};
    bool uses_async_copy{};
    const std::vector<u8>* branch_targets{};
};

// Byte uniform loads, ULEA carries to UP3, two-source F2IP and I2F.U8 of a UR byte. Without them
// five of Ocarina of Time's compute programs are rejected on every dispatch (LOD vegetation).
inline bool LowerExtended() {
    const char* value = std::getenv("EDEN_NX2_LOWER_EXTENDED");
    return value == nullptr || value[0] != '0';
}

// EDEN_NX2_STRONG_GLOBAL_FENCE=1: LDG/STG .STRONG.GPU fenced with MEMBAR.GL on both sides
// (Hookshot 2bbe7e183d). Off by default; part of the NVN translation key (TranslationSwitches).
inline bool StrongGlobalFence() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_STRONG_GLOBAL_FENCE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

inline u64 NativeProgramKey(std::span<const Instruction> source) {
    u64 hash = 14695981039346656037ULL;
    for (const auto& instruction : source) {
        for (const u32 word : instruction.words) {
            for (u32 byte = 0; byte < 4; ++byte) {
                hash = (hash ^ ((word >> (byte * 8)) & 255)) * 1099511628211ULL;
            }
        }
    }
    return hash;
}

constexpr u64 NativeAmpereCodeMagic = 0x455245504d41584eULL; // "NXAMPERE"

[[nodiscard]] inline bool HasNativeAmpereCode(std::span<const u64> code) noexcept {
    return code.size() >= 14 && code.back() == NativeAmpereCodeMagic;
}

[[nodiscard]] inline std::vector<u64> PackNativeAmpereCode(std::span<const u64> header,
                                                           std::span<const Instruction> native,
                                                           const LoweringOptions& options,
                                                           bool uses_async_copy) {
    std::vector<u64> packed;
    packed.reserve(header.size() + native.size() * 2 + 4);
    packed.insert(packed.end(), header.begin(), header.end());
    if (!native.empty()) {
        const size_t base = packed.size();
        packed.resize(base + native.size() * 2);
        std::memcpy(packed.data() + base, native.data(), native.size_bytes());
    }
    const u64 flags = (u64{options.geometry} << 0) | (u64{options.tensor} << 1) |
                      (u64{options.compact_tensor} << 2) | (u64{uses_async_copy} << 3);
    packed.push_back(flags);
    packed.push_back(options.native_code_address);
    packed.push_back(static_cast<u64>(native.size()));
    packed.push_back(NativeAmpereCodeMagic);
    return packed;
}

struct UnpackedNativeAmpere {
    std::vector<Instruction> instructions;
    LoweringOptions options{};
    bool uses_async_copy{};
};

[[nodiscard]] inline std::optional<UnpackedNativeAmpere> UnpackNativeAmpereCode(
    std::span<const u64> code, size_t header_words = 10) {
    if (!HasNativeAmpereCode(code)) {
        return std::nullopt;
    }
    const u64 count = code[code.size() - 2];
    if (count == 0 || count > 65536 || header_words + count * 2 + 4 != code.size()) {
        return std::nullopt;
    }
    const u64 flags = code[code.size() - 4];
    UnpackedNativeAmpere out;
    out.options.geometry = (flags & (u64{1} << 0)) != 0;
    out.options.tensor = (flags & (u64{1} << 1)) != 0;
    out.options.compact_tensor = (flags & (u64{1} << 2)) != 0;
    out.options.native_code_address = code[code.size() - 3];
    out.uses_async_copy = (flags & (u64{1} << 3)) != 0;
    out.instructions.resize(static_cast<size_t>(count));
    std::memcpy(out.instructions.data(), code.data() + header_words,
                out.instructions.size() * sizeof(Instruction));
    return out;
}

} // namespace Shader::Ampere
