// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

[[nodiscard]] std::optional<std::string_view> ResolveTensorMode(
    const LoweringOptions& options) noexcept {
    if (const char* const forced_mode = std::getenv("EDEN_NX2_TENSOR"); forced_mode != nullptr) {
        return std::string_view{forced_mode};
    }
    if (!options.tensor) {
        return std::nullopt;
    }
    return options.compact_tensor ? std::string_view{"compact"} : std::string_view{"1"};
}

/// Rounds an F32 accumulator to IEEE-754 binary16 and converts back to F32 using explicit bit
/// operations matching `emit_spirv_tensor.cpp`, preventing GPU compilers from eliminating an
/// `unpackHalf2x16(packHalf2x16(x))` round-trip.
[[nodiscard]] IR::F32 RoundHalfF32(IR::IREmitter& ir, const IR::F32& value) {
    const IR::U32 bits = ir.BitCast<IR::U32>(value);
    const IR::U32 magnitude = ir.BitwiseAnd(bits, ir.Imm32(0x7fffffffU));
    const IR::U32 tie =
        ir.BitwiseAnd(ir.ShiftRightLogical(magnitude, ir.Imm32(13U)), ir.Imm32(1U));
    const IR::U32 rounded = ir.BitwiseAnd(
        ir.IAdd(ir.IAdd(magnitude, ir.Imm32(0xfffU)), tie), ir.Imm32(~0x1fffU));
    const IR::U32 normal = IR::U32{ir.Select(
        ir.IGreaterThanEqual(rounded, ir.Imm32(0x47800000U), false), ir.Imm32(0x7f800000U),
        rounded)};
    const IR::F32 scaled = ir.FPMul(ir.BitCast<IR::F32>(magnitude), ir.Imm32(16777216.0f));
    const IR::F32 rounded_subnormal = ir.FPMul(ir.FPRoundEven(scaled), ir.Imm32(0x1p-24f));
    const IR::U32 subnormal = ir.BitCast<IR::U32>(rounded_subnormal);
    const IR::U32 finite = IR::U32{ir.Select(
        ir.IGreaterThanEqual(magnitude, ir.Imm32(0x38800000U), false), normal, subnormal)};
    const IR::U32 mag_res = IR::U32{ir.Select(
        ir.IGreaterThanEqual(magnitude, ir.Imm32(0x7f800000U), false), magnitude, finite)};
    return ir.BitCast<IR::F32>(ir.BitwiseOr(mag_res, ir.BitwiseAnd(bits, ir.Imm32(0x80000000U))));
}

} // Anonymous namespace

void TranslatorVisitor::MOVM(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::U32 src = R(insn.Bits(24, 8));
    const IR::U32 lane = ir.LaneId();

    // Lane t holds row t/4, columns 2*(t%4)..+1; fetch from 8*(t%4) + t/8 (+4) at half (t/4)%2.
    const IR::U32 base_lane = ir.IAdd(
        ir.ShiftLeftLogical(ir.BitwiseAnd(lane, ir.Imm32(3U)), ir.Imm32(3U)),
        ir.ShiftRightLogical(lane, ir.Imm32(3U)));
    const IR::U32 pair0 = ir.ShuffleIndex(src, base_lane, ir.Imm32(31U), ir.Imm32(0U));
    const IR::U32 pair1 =
        ir.ShuffleIndex(src, ir.IAdd(base_lane, ir.Imm32(4U)), ir.Imm32(31U), ir.Imm32(0U));
    const IR::U32 bit_shift = ir.ShiftLeftLogical(
        ir.BitwiseAnd(ir.ShiftRightLogical(lane, ir.Imm32(2U)), ir.Imm32(1U)), ir.Imm32(4U));
    const IR::U32 lo = ir.BitwiseAnd(ir.ShiftRightLogical(pair0, bit_shift), ir.Imm32(0xffffU));
    const IR::U32 hi = ir.ShiftRightLogical(pair1, bit_shift);
    R(dest, ir.BitFieldInsert(lo, hi, ir.Imm32(16U), ir.Imm32(16U)));
}

void TranslatorVisitor::LDSM(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 count = 1U << insn.Bits(72, 2);
    const bool transpose = insn.Bits(78, 1) != 0U;
    const bool uniform_offset = insn.Bits(91, 1) != 0U;
    const u32 ra = insn.Bits(24, 8);
    const s32 signed_off = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;

    IR::U32 row_addr = R(ra);
    if (uniform_offset) {
        const u32 ureg = insn.Bits(32, 8);
        if (ureg != 63U) {
            row_addr = ir.IAdd(row_addr, UR(ureg));
        }
    }

    const IR::U32 lane = ir.LaneId();
    const auto with_offset = [&](const IR::U32& base, const IR::U32& col) {
        const IR::U32 combined = ir.IAdd(base, col);
        return signed_off == 0
                   ? combined
                   : IR::U32{ir.IAdd(combined, ir.Imm32(static_cast<u32>(signed_off)))};
    };

    if (transpose) {
        const IR::U32 row_base =
            ir.ShiftLeftLogical(ir.BitwiseAnd(lane, ir.Imm32(3U)), ir.Imm32(1U));
        const IR::U32 col_byte =
            ir.ShiftLeftLogical(ir.ShiftRightLogical(lane, ir.Imm32(2U)), ir.Imm32(1U));
        for (u32 m = 0U; m < count; ++m) {
            const IR::U32 base0 = ir.ShuffleIndex(row_addr, ir.IAdd(row_base, ir.Imm32(m * 8U)),
                                                  ir.Imm32(31U), ir.Imm32(0U));
            const IR::U32 base1 =
                ir.ShuffleIndex(row_addr, ir.IAdd(row_base, ir.Imm32(m * 8U + 1U)),
                                ir.Imm32(31U), ir.Imm32(0U));
            const IR::U32 h0{ir.LoadShared(16, false, with_offset(base0, col_byte))};
            const IR::U32 h1{ir.LoadShared(16, false, with_offset(base1, col_byte))};
            R(dest + m, ir.BitFieldInsert(h0, h1, ir.Imm32(16U), ir.Imm32(16U)));
        }
        return;
    }

    const IR::U32 row = ir.ShiftRightLogical(lane, ir.Imm32(2U));
    const IR::U32 col_bytes =
        ir.ShiftLeftLogical(ir.BitwiseAnd(lane, ir.Imm32(3U)), ir.Imm32(2U));
    for (u32 m = 0U; m < count; ++m) {
        const IR::U32 src_lane = m == 0U ? row : IR::U32{ir.IAdd(row, ir.Imm32(m * 8U))};
        const IR::U32 base = ir.ShuffleIndex(row_addr, src_lane, ir.Imm32(31U), ir.Imm32(0U));
        R(dest + m, IR::U32{ir.LoadShared(32, false, with_offset(base, col_bytes))});
    }
}

void TranslatorVisitor::HMMA(const Instruction& insn) {
    const std::optional<std::string_view> mode = ResolveTensorMode(options);
    if (!mode) {
        throw NotImplementedException("HMMA encountered without EDEN_NX2_TENSOR enabled");
    }
    const u32 dest = insn.Bits(16, 8);
    const u32 a = insn.Bits(24, 8);
    const u32 b = insn.Bits(32, 8);
    const u32 c = insn.Bits(64, 8);
    const u32 depth = (insn.words[2] & ~0xffU) == 0U ? 8U : 16U;
    const bool short_depth = depth == 8U;

    const IR::U32 a0 = R(a);
    const IR::U32 a1 = R(a + 1U);
    const IR::U32 a2 = short_depth ? ir.Imm32(0U) : R(a + 2U);
    const IR::U32 a3 = short_depth ? ir.Imm32(0U) : R(a + 3U);
    const IR::U32 b0 = R(b);
    const IR::U32 b1 = short_depth ? ir.Imm32(0U) : R(b + 1U);
    const IR::U32 c0 = c == 255U ? ir.Imm32(0U) : R(c);
    const IR::U32 c1 = c == 255U ? ir.Imm32(0U) : R(c + 1U);

    if (*mode == "compact" && insn.Bits(12, 4) == 7U) {
        const IR::Value result =
            ir.TensorMulAdd(ir.CompositeConstruct(a0, a1, a2, a3), ir.CompositeConstruct(b0, b1),
                            ir.CompositeConstruct(c0, c1), depth);
        R(dest, IR::U32{ir.CompositeExtract(result, 0U)});
        R(dest + 1U, IR::U32{ir.CompositeExtract(result, 1U)});
        return;
    }

    if (*mode == "stub") {
        R(dest, c0);
        R(dest + 1U, c1);
        return;
    }

    const IR::Value c0_unpacked = ir.UnpackHalf2x16(c0);
    const IR::Value c1_unpacked = ir.UnpackHalf2x16(c1);
    std::array<IR::F32, 4> sums{
        IR::F32{ir.CompositeExtract(c0_unpacked, 0U)},
        IR::F32{ir.CompositeExtract(c0_unpacked, 1U)},
        IR::F32{ir.CompositeExtract(c1_unpacked, 0U)},
        IR::F32{ir.CompositeExtract(c1_unpacked, 1U)},
    };

    const IR::U32 lane = ir.LaneId();
    const IR::U32 a_base = ir.BitwiseAnd(lane, ir.Imm32(0x1cU));
    const IR::U32 b_base = ir.ShiftLeftLogical(ir.BitwiseAnd(lane, ir.Imm32(3U)), ir.Imm32(3U));
    const std::array<IR::U32, 4> a_regs{a0, a1, a2, a3};
    const std::array<IR::U32, 2> b_regs{b0, b1};
    const IR::FpControl fma_control{.no_contraction = true};

    for (u32 pair = 0U; pair < depth / 2U; ++pair) {
        const IR::U32 offset = ir.Imm32(pair & 3U);
        const IR::U32 a_lane = ir.IAdd(a_base, offset);
        const IR::U32 b_lane0 = ir.IAdd(b_base, offset);
        const IR::U32 b_lane1 = ir.IAdd(b_lane0, ir.Imm32(4U));

        const u32 a_base_idx = pair >= 4U ? 2U : 0U;
        const u32 b_idx = pair / 4U;

        const IR::Value a_row0 = ir.UnpackHalf2x16(
            ir.ShuffleIndex(a_regs[a_base_idx], a_lane, ir.Imm32(31U), ir.Imm32(0U)));
        const IR::Value a_row1 = ir.UnpackHalf2x16(
            ir.ShuffleIndex(a_regs[a_base_idx + 1U], a_lane, ir.Imm32(31U), ir.Imm32(0U)));
        const IR::Value b_col0 = ir.UnpackHalf2x16(
            ir.ShuffleIndex(b_regs[b_idx], b_lane0, ir.Imm32(31U), ir.Imm32(0U)));
        const IR::Value b_col1 = ir.UnpackHalf2x16(
            ir.ShuffleIndex(b_regs[b_idx], b_lane1, ir.Imm32(31U), ir.Imm32(0U)));

        for (u32 half = 0U; half < 2U; ++half) {
            const IR::F32 bv0{ir.CompositeExtract(b_col0, half)};
            const IR::F32 bv1{ir.CompositeExtract(b_col1, half)};
            const IR::F32 av0{ir.CompositeExtract(a_row0, half)};
            const IR::F32 av1{ir.CompositeExtract(a_row1, half)};
            sums[0] = ir.FPFma(av0, bv0, sums[0], fma_control);
            sums[1] = ir.FPFma(av0, bv1, sums[1], fma_control);
            sums[2] = ir.FPFma(av1, bv0, sums[2], fma_control);
            sums[3] = ir.FPFma(av1, bv1, sums[3], fma_control);
        }

        if ((pair & 3U) == 3U) {
            for (IR::F32& sum : sums) {
                sum = RoundHalfF32(ir, sum);
            }
        }
    }

    R(dest, ir.PackHalf2x16(ir.CompositeConstruct(sums[0], sums[1])));
    R(dest + 1U, ir.PackHalf2x16(ir.CompositeConstruct(sums[2], sums[3])));
}

} // namespace Shader::Ampere
