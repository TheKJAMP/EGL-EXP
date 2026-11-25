// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <utility>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_encoding.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Ampere {
namespace {

void PromoteHalfPair(IR::IREmitter& ir, IR::F16F32F64& lhs, IR::F16F32F64& rhs) {
    if (lhs.Type() == IR::Type::F16) {
        lhs = ir.FPConvert(32, lhs);
        rhs = ir.FPConvert(32, rhs);
    }
}

[[nodiscard]] IR::F32 RoundFloat32(IR::IREmitter& ir, const IR::F32& value, u32 round_mode,
                                   IR::FmzMode fmz) {
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = fmz,
    };
    switch (round_mode) {
    case 0U:
        return IR::F32{ir.FPRoundEven(value, control)};
    case 1U:
        return IR::F32{ir.FPFloor(value, control)};
    case 2U:
        return IR::F32{ir.FPCeil(value, control)};
    case 3U:
        return IR::F32{ir.FPTrunc(value, control)};
    default:
        throw NotImplementedException("Invalid float rounding mode {}", round_mode);
    }
}

[[nodiscard]] IR::U32 ClampHalfReluLane(IR::IREmitter& ir, const IR::U32& h) {
    const IR::U32 zero = ir.Imm32(0U);
    const IR::U32 mag = ir.BitwiseAnd(h, ir.Imm32(0x7fffU));
    const IR::U1 is_nan = ir.IGreaterThan(mag, ir.Imm32(0x7c00U), false);
    const IR::U1 is_neg = ir.INotEqual(ir.BitwiseAnd(h, ir.Imm32(0x8000U)), zero);
    const IR::U32 clamped = IR::U32{ir.Select(is_neg, zero, h)};
    return IR::U32{ir.Select(is_nan, ir.Imm32(0x7fffU), clamped)};
}

[[nodiscard]] IR::F32 ConvertI32ToF32Rounded(IR::IREmitter& ir, const IR::U32& src, bool is_signed,
                                             u32 round_mode) {
    if (round_mode == 0U) {
        const IR::FpControl control{
            .no_contraction = false,
            .rounding = IR::FpRounding::RN,
            .fmz_mode = IR::FmzMode::DontCare,
        };
        return IR::F32{ir.ConvertIToF(32, 32, is_signed, src, control)};
    }
    const IR::U1 neg = is_signed ? ir.ILessThan(src, ir.Imm32(0U), true) : ir.Imm1(false);
    const IR::U32 mag = is_signed ? IR::U32{ir.Select(neg, ir.INeg(src), src)} : src;
    const IR::U1 round_up =
        round_mode == 1U ? neg : (round_mode == 2U ? ir.LogicalNot(neg) : ir.Imm1(false));
    const IR::U1 is_zero = ir.IEqual(mag, ir.Imm32(0U));
    const IR::U32 msb{ir.Select(is_zero, ir.Imm32(0U), ir.FindUMsb(mag))};
    const IR::U1 large = ir.IGreaterThan(msb, ir.Imm32(23U), false);
    const IR::U32 shift_r{ir.Select(large, ir.ISub(msb, ir.Imm32(23U)), ir.Imm32(0U))};
    const IR::U32 shift_l{ir.Select(large, ir.Imm32(0U), ir.ISub(ir.Imm32(23U), msb))};
    const IR::U32 mant{
        ir.Select(large, ir.ShiftRightLogical(mag, shift_r), ir.ShiftLeftLogical(mag, shift_l))};
    const IR::U32 rem_mask{ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1U), shift_r), ir.Imm32(1U))};
    const IR::U1 inexact = ir.INotEqual(ir.BitwiseAnd(mag, rem_mask), ir.Imm32(0U));
    const IR::U32 inc{ir.Select(ir.LogicalAnd(inexact, round_up), ir.Imm32(1U), ir.Imm32(0U))};
    const IR::U32 mant_r{ir.IAdd(mant, inc)};
    const IR::U32 exp_base{ir.ShiftLeftLogical(ir.IAdd(msb, ir.Imm32(126U)), ir.Imm32(23U))};
    const IR::U32 bits{ir.IAdd(exp_base, mant_r)};
    const IR::U32 signed_bits{
        ir.BitwiseOr(bits, IR::U32{ir.Select(neg, ir.Imm32(0x80000000U), ir.Imm32(0U))})};
    return ir.BitCast<IR::F32>(IR::U32{ir.Select(is_zero, ir.Imm32(0U), signed_bits)});
}

[[nodiscard]] IR::F32 ConvertI64ToF32Rounded(IR::IREmitter& ir, const IR::U32& lo, const IR::U32& hi,
                                             bool is_signed, u32 round_mode) {
    if (round_mode == 0U) {
        const IR::U64 packed = ir.PackUint2x32(ir.CompositeConstruct(lo, hi));
        const IR::FpControl control{
            .no_contraction = false,
            .rounding = IR::FpRounding::RN,
            .fmz_mode = IR::FmzMode::DontCare,
        };
        return IR::F32{ir.ConvertIToF(32, 64, is_signed, packed, control)};
    }
    const IR::U1 neg = is_signed
                           ? ir.INotEqual(ir.BitwiseAnd(hi, ir.Imm32(0x80000000U)), ir.Imm32(0U))
                           : ir.Imm1(false);
    const IR::U32 neg_lo{ir.INeg(lo)};
    const IR::U32 neg_hi{
        ir.IAdd(ir.BitwiseNot(hi),
                IR::U32{ir.Select(ir.INotEqual(lo, ir.Imm32(0U)), ir.Imm32(0U), ir.Imm32(1U))})};
    const IR::U32 mag_lo = is_signed ? IR::U32{ir.Select(neg, neg_lo, lo)} : lo;
    const IR::U32 mag_hi = is_signed ? IR::U32{ir.Select(neg, neg_hi, hi)} : hi;
    const IR::U1 round_up =
        round_mode == 1U ? neg : (round_mode == 2U ? ir.LogicalNot(neg) : ir.Imm1(false));
    const IR::U1 hi_nz = ir.INotEqual(mag_hi, ir.Imm32(0U));
    const IR::U1 lo_nz = ir.INotEqual(mag_lo, ir.Imm32(0U));
    const IR::U1 is_zero = ir.LogicalNot(ir.LogicalOr(hi_nz, lo_nz));
    const IR::U32 msb_lo{ir.Select(lo_nz, ir.FindUMsb(mag_lo), ir.Imm32(0U))};
    const IR::U32 msb{ir.Select(hi_nz, ir.IAdd(ir.FindUMsb(mag_hi), ir.Imm32(32U)), msb_lo)};
    const IR::U1 case_a = ir.ILessThan(msb, ir.Imm32(24U), false);
    const IR::U1 case_c = ir.IGreaterThanEqual(msb, ir.Imm32(55U), false);
    const IR::U32 shift_a{ir.Select(case_a, ir.ISub(ir.Imm32(23U), msb), ir.Imm32(0U))};
    const IR::U32 mant_a{ir.ShiftLeftLogical(mag_lo, shift_a)};
    const IR::U32 s_b{ir.Select(ir.LogicalOr(case_a, case_c), ir.Imm32(1U),
                                ir.ISub(msb, ir.Imm32(23U)))};
    const IR::U32 inv_s_b{ir.ISub(ir.Imm32(32U), s_b)};
    const IR::U32 mant_b{
        ir.BitwiseOr(ir.ShiftRightLogical(mag_lo, s_b), ir.ShiftLeftLogical(mag_hi, inv_s_b))};
    const IR::U1 inexact_b = ir.INotEqual(ir.ShiftLeftLogical(mag_lo, inv_s_b), ir.Imm32(0U));
    const IR::U32 s_c{ir.Select(case_c, ir.ISub(msb, ir.Imm32(55U)), ir.Imm32(0U))};
    const IR::U32 mant_c{ir.ShiftRightLogical(mag_hi, s_c)};
    const IR::U32 rem_mask_c{ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1U), s_c), ir.Imm32(1U))};
    const IR::U1 inexact_c =
        ir.LogicalOr(lo_nz, ir.INotEqual(ir.BitwiseAnd(mag_hi, rem_mask_c), ir.Imm32(0U)));
    const IR::U32 mant{
        ir.Select(case_a, mant_a, IR::U32{ir.Select(case_c, mant_c, mant_b)})};
    const IR::U1 inexact{
        ir.Select(case_a, ir.Imm1(false), IR::U1{ir.Select(case_c, inexact_c, inexact_b)})};
    const IR::U32 inc{ir.Select(ir.LogicalAnd(inexact, round_up), ir.Imm32(1U), ir.Imm32(0U))};
    const IR::U32 mant_r{ir.IAdd(mant, inc)};
    const IR::U32 exp_base{ir.ShiftLeftLogical(ir.IAdd(msb, ir.Imm32(126U)), ir.Imm32(23U))};
    const IR::U32 bits{ir.IAdd(exp_base, mant_r)};
    const IR::U32 signed_bits{
        ir.BitwiseOr(bits, IR::U32{ir.Select(neg, ir.Imm32(0x80000000U), ir.Imm32(0U))})};
    return ir.BitCast<IR::F32>(IR::U32{ir.Select(is_zero, ir.Imm32(0U), signed_bits)});
}

} // Anonymous namespace

TranslatorVisitor::HalfOperand TranslatorVisitor::HalfSource(const Instruction& insn, u32 which) {
    if (which == 0U) {
        return {R(insn.Bits(24, 8)), insn.Bits(74, 2), insn.Bits(73, 1) != 0U,
                insn.Bits(72, 1) != 0U};
    }
    const u32 form = insn.Bits(9, 3);
    const bool wide = which == 1U ? (form == 4U || form == 5U) : (form == 2U || form == 3U);
    if (wide) {
        if (form == 2U || form == 4U) {
            return {ir.Imm32(insn.words[1]), 0U, false, false};
        }
        return {ReadCbuf32(insn), insn.Bits(60, 2), insn.Bits(62, 1) != 0U,
                insn.Bits(63, 1) != 0U};
    }
    const bool upper = which == 2U || form == 2U || form == 3U;
    return {R(insn.Bits(upper ? 64U : 32U, 8)), insn.Bits(upper ? 81U : 60U, 2),
            insn.Bits(upper ? 83U : 62U, 1) != 0U, insn.Bits(upper ? 84U : 63U, 1) != 0U};
}

std::pair<IR::U1, IR::U1> TranslatorVisitor::EvalHalfCompare(const Instruction& insn, u32 b_which) {
    const HalfOperand a = HalfSource(insn, 0U);
    const HalfOperand b = HalfSource(insn, b_which);

    auto [lhs_a, rhs_a] = Maxwell::Extract(ir, a.value, static_cast<Maxwell::Swizzle>(a.swizzle));
    auto [lhs_b, rhs_b] = Maxwell::Extract(ir, b.value, static_cast<Maxwell::Swizzle>(b.swizzle));
    if (lhs_a.Type() != lhs_b.Type()) {
        PromoteHalfPair(ir, lhs_a, rhs_a);
        PromoteHalfPair(ir, lhs_b, rhs_b);
    }
    lhs_a = ir.FPAbsNeg(lhs_a, a.abs, a.neg);
    rhs_a = ir.FPAbsNeg(rhs_a, a.abs, a.neg);
    lhs_b = ir.FPAbsNeg(lhs_b, b.abs, b.neg);
    rhs_b = ir.FPAbsNeg(rhs_b, b.abs, b.neg);

    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };
    const auto compare_op = static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4));
    const auto combine = static_cast<Maxwell::BooleanOp>(insn.Bits(69, 2));
    const IR::U1 pred = P(insn.Bits(87, 4));
    const IR::U1 cmp_lhs = Maxwell::FloatingPointCompare(ir, lhs_a, lhs_b, compare_op, control);
    const IR::U1 cmp_rhs = Maxwell::FloatingPointCompare(ir, rhs_a, rhs_b, compare_op, control);
    return {Maxwell::PredicateCombine(ir, cmp_lhs, pred, combine),
            Maxwell::PredicateCombine(ir, cmp_rhs, pred, combine)};
}

void TranslatorVisitor::F2F(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 src = IR::F32{ir.FPAbsNeg(F(insn.Bits(32, 8)), insn.Bits(62, 1) != 0U,
                                            insn.Bits(63, 1) != 0U)};
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::RN,
        .fmz_mode = IR::FmzMode::None,
    };
    const IR::F16 converted{ir.FPConvert(16, src, control)};
    const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
    R(dest, ir.PackFloat2x16(ir.CompositeConstruct(converted, zero)));
}

void TranslatorVisitor::F2FP(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const bool merge = insn.Bits(78, 1) != 0U;
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::RN,
        .fmz_mode = IR::FmzMode::None,
    };
    const IR::F32 b = ir.BitCast<IR::F32>(AluSourceU32(insn, 1U));
    const IR::F16 lo_h{ir.FPConvert(16, b, control)};
    const IR::F16 zero_h{ir.FPConvert(16, ir.Imm32(0.0f))};
    if (merge) {
        const IR::U32 c = AluSourceU32(insn, 2U);
        const IR::U32 low_packed =
            ir.BitwiseAnd(ir.PackFloat2x16(ir.CompositeConstruct(lo_h, zero_h)), ir.Imm32(0xffffU));
        R(dest, ir.BitwiseOr(ir.ShiftLeftLogical(c, ir.Imm32(16U)), low_packed));
    } else {
        const IR::F32 a = ir.BitCast<IR::F32>(AluSourceU32(insn, 0U));
        const IR::F16 hi_h{ir.FPConvert(16, a, control)};
        R(dest, ir.PackFloat2x16(ir.CompositeConstruct(lo_h, hi_h)));
    }
}

void TranslatorVisitor::HMUL2(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const HalfOperand a = HalfSource(insn, 0U);
    const HalfOperand b = HalfSource(insn, 1U);

    auto [lhs_a, rhs_a] = Maxwell::Extract(ir, a.value, static_cast<Maxwell::Swizzle>(a.swizzle));
    auto [lhs_b, rhs_b] = Maxwell::Extract(ir, b.value, static_cast<Maxwell::Swizzle>(b.swizzle));
    const bool promotion = lhs_a.Type() != lhs_b.Type();
    if (promotion) {
        PromoteHalfPair(ir, lhs_a, rhs_a);
        PromoteHalfPair(ir, lhs_b, rhs_b);
    }

    const bool combined_neg_b = a.neg != b.neg;
    lhs_a = ir.FPAbsNeg(lhs_a, a.abs, false);
    rhs_a = ir.FPAbsNeg(rhs_a, a.abs, false);
    lhs_b = ir.FPAbsNeg(lhs_b, b.abs, combined_neg_b);
    rhs_b = ir.FPAbsNeg(rhs_b, b.abs, combined_neg_b);

    const bool fmz = insn.Bits(76, 1) != 0U;
    const bool sat = insn.Bits(77, 1) != 0U;
    const IR::FmzMode fmz_mode =
        fmz ? IR::FmzMode::FMZ : (insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None);
    const IR::FpControl fp_control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = fmz_mode,
    };

    IR::F16F32F64 lhs = ir.FPMul(lhs_a, lhs_b, fp_control);
    IR::F16F32F64 rhs = ir.FPMul(rhs_a, rhs_b, fp_control);
    if (fmz && !sat) {
        const IR::F16F32F64 zero =
            lhs_a.Type() == IR::Type::F16 ? IR::F16F32F64{ir.FPConvert(16, ir.Imm32(0.0f))}
                                          : IR::F16F32F64{ir.Imm32(0.0f)};
        const IR::U1 lhs_zero = ir.LogicalOr(ir.FPEqual(lhs_a, zero), ir.FPEqual(lhs_b, zero));
        const IR::U1 rhs_zero = ir.LogicalOr(ir.FPEqual(rhs_a, zero), ir.FPEqual(rhs_b, zero));
        lhs = IR::F16F32F64{ir.Select(lhs_zero, zero, lhs)};
        rhs = IR::F16F32F64{ir.Select(rhs_zero, zero, rhs)};
    }
    if (sat) {
        lhs = ir.FPSaturate(lhs);
        rhs = ir.FPSaturate(rhs);
    }
    if (promotion) {
        lhs = ir.FPConvert(16, lhs);
        rhs = ir.FPConvert(16, rhs);
    }
    R(dest, ir.PackFloat2x16(ir.CompositeConstruct(lhs, rhs)));
}

void TranslatorVisitor::HADD2(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const HalfOperand a = HalfSource(insn, 0U);
    const HalfOperand b = HalfSource(insn, opcode == 0x230U ? 1U : 2U);

    if (insn.Bits(78, 1) != 0U) {
        const IR::Value unpacked = ir.UnpackFloat2x16(b.value);
        const IR::F16 lane{ir.CompositeExtract(unpacked, b.swizzle == 3U ? 1U : 0U)};
        F(dest, IR::F32{ir.FPAbsNeg(ir.FPConvert(32, lane), b.abs, b.neg)});
        return;
    }

    auto [lhs_a, rhs_a] = Maxwell::Extract(ir, a.value, static_cast<Maxwell::Swizzle>(a.swizzle));
    auto [lhs_b, rhs_b] = Maxwell::Extract(ir, b.value, static_cast<Maxwell::Swizzle>(b.swizzle));
    const bool promotion = lhs_a.Type() != lhs_b.Type();
    if (promotion) {
        PromoteHalfPair(ir, lhs_a, rhs_a);
        PromoteHalfPair(ir, lhs_b, rhs_b);
    }

    lhs_a = ir.FPAbsNeg(lhs_a, a.abs, a.neg);
    rhs_a = ir.FPAbsNeg(rhs_a, a.abs, a.neg);
    lhs_b = ir.FPAbsNeg(lhs_b, b.abs, b.neg);
    rhs_b = ir.FPAbsNeg(rhs_b, b.abs, b.neg);

    const IR::FpControl fp_control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };

    IR::F16F32F64 lhs = ir.FPAdd(lhs_a, lhs_b, fp_control);
    IR::F16F32F64 rhs = ir.FPAdd(rhs_a, rhs_b, fp_control);
    if (insn.Bits(77, 1) != 0U) {
        lhs = ir.FPSaturate(lhs);
        rhs = ir.FPSaturate(rhs);
    }
    if (promotion) {
        lhs = ir.FPConvert(16, lhs);
        rhs = ir.FPConvert(16, rhs);
    }
    R(dest, ir.PackFloat2x16(ir.CompositeConstruct(lhs, rhs)));
}

void TranslatorVisitor::HFMA2(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const HalfOperand a = HalfSource(insn, 0U);
    HalfOperand b = HalfSource(insn, 1U);
    const HalfOperand c = HalfSource(insn, 2U);

    auto [lhs_a, rhs_a] = Maxwell::Extract(ir, a.value, static_cast<Maxwell::Swizzle>(a.swizzle));
    auto [lhs_b, rhs_b] = Maxwell::Extract(ir, b.value, static_cast<Maxwell::Swizzle>(b.swizzle));
    auto [lhs_c, rhs_c] = Maxwell::Extract(ir, c.value, static_cast<Maxwell::Swizzle>(c.swizzle));
    const bool promotion = lhs_a.Type() != lhs_b.Type() || lhs_a.Type() != lhs_c.Type();
    if (promotion) {
        PromoteHalfPair(ir, lhs_a, rhs_a);
        PromoteHalfPair(ir, lhs_b, rhs_b);
        PromoteHalfPair(ir, lhs_c, rhs_c);
    }

    lhs_a = ir.FPAbsNeg(lhs_a, a.abs, false);
    rhs_a = ir.FPAbsNeg(rhs_a, a.abs, false);
    if (insn.Bits(86, 1) != 0U) {
        const IR::F16F32F64 new_lhs_b = ir.FPNeg(rhs_b);
        const IR::F16F32F64 new_rhs_b = lhs_b;
        lhs_b = new_lhs_b;
        rhs_b = new_rhs_b;
    }
    const bool combined_neg_b = a.neg != b.neg;
    lhs_b = ir.FPAbsNeg(lhs_b, false, combined_neg_b);
    rhs_b = ir.FPAbsNeg(rhs_b, false, combined_neg_b);
    lhs_c = ir.FPAbsNeg(lhs_c, false, c.neg);
    rhs_c = ir.FPAbsNeg(rhs_c, false, c.neg);

    const bool fmz = insn.Bits(76, 1) != 0U;
    const bool sat = insn.Bits(77, 1) != 0U;
    const bool relu = insn.Bits(79, 1) != 0U;
    const IR::FmzMode fmz_mode =
        fmz ? IR::FmzMode::FMZ : (insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None);
    const IR::FpControl fp_control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = fmz_mode,
    };

    IR::F16F32F64 lhs = ir.FPFma(lhs_a, lhs_b, lhs_c, fp_control);
    IR::F16F32F64 rhs = ir.FPFma(rhs_a, rhs_b, rhs_c, fp_control);
    if (fmz && !sat) {
        const IR::F16F32F64 zero =
            lhs_a.Type() == IR::Type::F16 ? IR::F16F32F64{ir.FPConvert(16, ir.Imm32(0.0f))}
                                          : IR::F16F32F64{ir.Imm32(0.0f)};
        const IR::U1 lhs_zero = ir.LogicalOr(ir.FPEqual(lhs_a, zero), ir.FPEqual(lhs_b, zero));
        const IR::U1 rhs_zero = ir.LogicalOr(ir.FPEqual(rhs_a, zero), ir.FPEqual(rhs_b, zero));
        lhs = IR::F16F32F64{ir.Select(lhs_zero, lhs_c, lhs)};
        rhs = IR::F16F32F64{ir.Select(rhs_zero, rhs_c, rhs)};
    }
    if (sat) {
        lhs = ir.FPSaturate(lhs);
        rhs = ir.FPSaturate(rhs);
    }
    if (promotion) {
        lhs = ir.FPConvert(16, lhs);
        rhs = ir.FPConvert(16, rhs);
    }
    IR::U32 packed = ir.PackFloat2x16(ir.CompositeConstruct(lhs, rhs));
    if (relu) {
        const IR::U32 lo = ClampHalfReluLane(ir, ir.BitwiseAnd(packed, ir.Imm32(0xffffU)));
        const IR::U32 hi = ClampHalfReluLane(ir, ir.ShiftRightLogical(packed, ir.Imm32(16U)));
        const IR::U32 relu_packed = ir.BitwiseOr(lo, ir.ShiftLeftLogical(hi, ir.Imm32(16U)));
        const u32 clamp_pred = insn.Bits(87, 4);
        packed = clamp_pred != 7U ? IR::U32{ir.Select(P(clamp_pred), relu_packed, packed)}
                                  : relu_packed;
    }
    R(dest, packed);
}

void TranslatorVisitor::HSET2(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const auto [res_lhs, res_rhs] = EvalHalfCompare(insn, opcode == 0x233U ? 1U : 2U);

    const u32 true_val = insn.Bits(71, 1) != 0U ? 0x3c00U : 0xffffU;
    const IR::U32 out_lhs{ir.Select(res_lhs, ir.Imm32(true_val), ir.Imm32(0U))};
    const IR::U32 out_rhs{ir.Select(res_rhs, ir.Imm32(true_val << 16), ir.Imm32(0U))};
    R(dest, ir.BitwiseOr(out_lhs, out_rhs));
}

void TranslatorVisitor::HSETP2(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const auto [res_lhs, res_rhs] = EvalHalfCompare(insn, opcode == 0x234U ? 1U : 2U);

    if (insn.Bits(71, 1) != 0U) {
        const IR::U1 combined = ir.LogicalAnd(res_lhs, res_rhs);
        SetP(insn.Bits(81, 3), combined);
        SetP(insn.Bits(84, 3), ir.LogicalNot(combined));
    } else {
        SetP(insn.Bits(81, 3), res_lhs);
        SetP(insn.Bits(84, 3), res_rhs);
    }
}

void TranslatorVisitor::HMNMX2(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const HalfOperand a = HalfSource(insn, 0U);
    const HalfOperand b = opcode == 0x840U ? HalfOperand{ir.Imm32(insn.words[1]), 0U, false, false}
                                           : HalfSource(insn, 1U);

    auto [lhs_a, rhs_a] = Maxwell::Extract(ir, a.value, static_cast<Maxwell::Swizzle>(a.swizzle));
    auto [lhs_b, rhs_b] = Maxwell::Extract(ir, b.value, static_cast<Maxwell::Swizzle>(b.swizzle));
    PromoteHalfPair(ir, lhs_a, rhs_a);
    PromoteHalfPair(ir, lhs_b, rhs_b);
    lhs_a = ir.FPAbsNeg(lhs_a, a.abs, a.neg);
    rhs_a = ir.FPAbsNeg(rhs_a, a.abs, a.neg);
    lhs_b = ir.FPAbsNeg(lhs_b, b.abs, b.neg);
    rhs_b = ir.FPAbsNeg(rhs_b, b.abs, b.neg);

    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = IR::FmzMode::None,
    };
    const u32 pred_bits = insn.Bits(87, 4);
    const auto select_minmax = [&](const IR::F16F32F64& x, const IR::F16F32F64& y) {
        if (pred_bits == 7U) {
            return IR::F16{ir.FPConvert(16, ir.FPMin(x, y, control))};
        }
        if (pred_bits == 15U) {
            return IR::F16{ir.FPConvert(16, ir.FPMax(x, y, control))};
        }
        return IR::F16{ir.FPConvert(
            16, IR::F16F32F64{ir.Select(P(pred_bits), ir.FPMin(x, y, control),
                                        ir.FPMax(x, y, control))})};
    };
    R(dest, ir.PackFloat2x16(
                ir.CompositeConstruct(select_minmax(lhs_a, lhs_b), select_minmax(rhs_a, rhs_b))));
}

void TranslatorVisitor::FSWZADD(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 swizzle = insn.words[1];
    // Maxwell's lane codes 1 and 2 are swapped compared to Ampere (nvdisasm PP/PN/NP/ZP).
    const u32 ir_swizzle = ((swizzle & 0x55U) << 1) | ((swizzle & 0xaaU) >> 1);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = IR::FmzMode::None,
    };
    F(dest, ir.FSwizzleAdd(F(insn.Bits(24, 8)), F(insn.Bits(64, 8)), ir.Imm32(ir_swizzle),
                           control));
}

void TranslatorVisitor::F2I_64(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 src = F(insn.Bits(32, 8));
    const IR::FpControl control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = IR::FmzMode::DontCare,
    };
    const IR::F32 truncated{ir.FPTrunc(src, control)};
    const IR::F32 max_u64_f = ir.Imm32(std::bit_cast<f32>(0x5f7fffffU));
    const IR::U1 overflow =
        ir.FPGreaterThanEqual(truncated, ir.Imm32(std::bit_cast<f32>(0x5f800000U)));
    const IR::F32 clamped{ir.FPClamp(truncated, ir.Imm32(0.0f), max_u64_f)};
    IR::U64 result{ir.Select(overflow, ir.Imm64(~u64{0}), ir.ConvertFToI(64, false, clamped))};
    result = IR::U64{
        ir.Select(ir.FPIsNan(src), ir.Imm64(u64{0x8000'0000'0000'0000ULL}), result)};
    const IR::Value parts = ir.UnpackUint2x32(result);
    R(dest, IR::U32{ir.CompositeExtract(parts, 0U)});
    R(dest + 1U, IR::U32{ir.CompositeExtract(parts, 1U)});
}

void TranslatorVisitor::I2F_64(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 src = insn.Bits(32, 8);
    F(dest, ConvertI64ToF32Rounded(ir, R(src), R(src + 1U), insn.Bits(74, 1) != 0U,
                                   insn.Bits(78, 2)));
}

void TranslatorVisitor::F2IP(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 ra = insn.Bits(24, 8);
    const u32 rb = insn.Bits(32, 8);
    const u32 rc = insn.Bits(64, 8);
    const u32 rounding = insn.Bits(78, 2);
    const bool relu = insn.Bits(75, 1) != 0U;
    const bool is_signed = insn.Bits(76, 1) != 0U;
    const f32 low = (is_signed && !relu) ? -128.0f : 0.0f;
    const f32 high = is_signed ? 127.0f : 255.0f;

    const auto convert_byte = [&](u32 reg) {
        const IR::F32 x = F(reg);
        const IR::F32 clean{ir.Select(ir.FPIsNan(x), ir.Imm32(0.0f), x)};
        const IR::F32 rounded = RoundFloat32(ir, clean, rounding, IR::FmzMode::DontCare);
        const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(low), ir.Imm32(high))};
        return IR::U32{
            ir.BitwiseAnd(IR::U32{ir.ConvertFToI(32, is_signed && !relu, clamped)},
                          ir.Imm32(0xffU))};
    };

    IR::U32 result = convert_byte(rb);
    if (ra != 255U) {
        result = ir.BitwiseOr(result, ir.ShiftLeftLogical(convert_byte(ra), ir.Imm32(8U)));
    }
    if (rc != 255U) {
        result = ir.BitwiseOr(result, ir.ShiftLeftLogical(R(rc), ir.Imm32(16U)));
    }
    R(dest, result);
}

void TranslatorVisitor::F2I(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 mode = insn.words[2] & ~(opcode == 0xd05U ? 0x08000000U : 0U);
    const bool half = opcode == 0x305U && (mode & ~0x100U) == 0x0010f000U;
    const bool small = mode == 0x00212800U;
    const bool floor = (mode & ~0x10100U) == 0x00207000U;
    const bool nearest = (mode & ~0x10100U) == 0x00203000U;
    const bool round_up = (mode & ~0x10100U) == 0x0020b000U;
    const bool signed_byte = opcode == 0x305U && mode == 0x0021e100U;
    const bool short_trunc = opcode == 0x305U && (mode & ~0x100U) == 0x0021e800U;
    const bool unsigned_byte = opcode == 0x305U && (mode & ~0x10000U) == 0x0020e000U;
    const bool is_signed = insn.Bits(72, 1) != 0U;
    const bool ftz = insn.Bits(80, 1) != 0U;

    IR::F32 base;
    IR::F32 op_a;
    bool abs = false;
    bool neg = false;
    if (half) {
        const IR::Value unpacked = ir.UnpackFloat2x16(R(insn.Bits(32, 8)));
        const IR::F16 lane{ir.CompositeExtract(unpacked, insn.Bits(60, 1))};
        op_a = IR::F32{ir.FPConvert(32, lane)};
    } else {
        base = opcode == 0xb05U
                   ? ReadCbufF32(insn)
                   : (opcode == 0xd05U ? ir.BitCast<IR::F32>(UR(insn.Bits(32, 8)))
                                       : F(insn.Bits(32, 8)));
        abs = opcode != 0xd05U && insn.Bits(62, 1) != 0U;
        neg = opcode != 0xd05U && insn.Bits(63, 1) != 0U;
        op_a = IR::F32{ir.FPAbsNeg(base, abs, neg)};
    }

    const u32 round_mode = (small || nearest) ? 0U : (floor ? 1U : (round_up ? 2U : 3U));
    if (!half) {
        const IR::U32 base_bits = ir.BitCast<IR::U32>(base);
        const IR::U1 exp_zero =
            ir.IEqual(ir.BitwiseAnd(base_bits, ir.Imm32(0x7f800000U)), ir.Imm32(0U));
        const IR::U1 is_neg =
            abs ? ir.Imm1(neg)
                : (neg ? ir.IEqual(ir.BitwiseAnd(base_bits, ir.Imm32(0x80000000U)), ir.Imm32(0U))
                       : ir.INotEqual(ir.BitwiseAnd(base_bits, ir.Imm32(0x80000000U)),
                                      ir.Imm32(0U)));
        const IR::U32 op_a_bits = ir.BitCast<IR::U32>(op_a);
        if (ftz) {
            const IR::U32 zero_signed{ir.Select(is_neg, ir.Imm32(0x80000000U), ir.Imm32(0U))};
            op_a = ir.BitCast<IR::F32>(IR::U32{ir.Select(exp_zero, zero_signed, op_a_bits)});
        } else if (round_mode == 1U || round_mode == 2U) {
            const IR::U1 mant_nz =
                ir.INotEqual(ir.BitwiseAnd(base_bits, ir.Imm32(0x007fffffU)), ir.Imm32(0U));
            const IR::U1 is_subnormal_nz = ir.LogicalAnd(exp_zero, mant_nz);
            const IR::U32 sub_val{round_mode == 1U
                                      ? ir.Select(is_neg, ir.Imm32(0xbf800000U), ir.Imm32(0U))
                                      : ir.Select(is_neg, ir.Imm32(0U), ir.Imm32(0x3f800000U))};
            op_a = ir.BitCast<IR::F32>(IR::U32{ir.Select(is_subnormal_nz, sub_val, op_a_bits)});
        }
    }

    const IR::FmzMode fmz = IR::FmzMode::DontCare;
    IR::U32 result;
    if (unsigned_byte) {
        const IR::F32 clean{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0.0f), op_a)};
        const IR::F32 clamped{ir.FPClamp(clean, ir.Imm32(0.0f), ir.Imm32(255.0f))};
        const IR::F32 rounded = RoundFloat32(ir, clamped, round_mode, fmz);
        result = IR::U32{ir.ConvertFToI(32, false, rounded)};
    } else if (signed_byte) {
        const IR::F32 rounded = RoundFloat32(ir, op_a, round_mode, fmz);
        const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(-128.0f), ir.Imm32(127.0f))};
        result = IR::U32{
            ir.Select(ir.FPIsNan(op_a), ir.Imm32(0U), ir.ConvertFToI(32, true, clamped))};
    } else if (!is_signed) {
        const IR::F32 clean{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0.0f), op_a)};
        if (small || short_trunc) {
            const IR::F32 rounded = RoundFloat32(ir, clean, round_mode, fmz);
            const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(0.0f), ir.Imm32(65535.0f))};
            result = IR::U32{ir.ConvertFToI(32, false, clamped)};
        } else {
            const IR::F32 max_u32_f = ir.Imm32(std::bit_cast<f32>(0x4f7fffffU));
            const IR::U1 overflow =
                ir.FPGreaterThanEqual(clean, ir.Imm32(std::bit_cast<f32>(0x4f800000U)));
            const IR::F32 capped{ir.FPMin(clean, max_u32_f)};
            const IR::F32 rounded = RoundFloat32(ir, capped, round_mode, fmz);
            const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(0.0f), max_u32_f)};
            result = IR::U32{
                ir.Select(overflow, ir.Imm32(0xffffffffU), ir.ConvertFToI(32, false, clamped))};
        }
    } else {
        const IR::F32 rounded = RoundFloat32(ir, op_a, round_mode, fmz);
        if (short_trunc) {
            const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(-32768.0f), ir.Imm32(32767.0f))};
            result = IR::U32{
                ir.Select(ir.FPIsNan(op_a), ir.Imm32(0U), ir.ConvertFToI(32, true, clamped))};
        } else {
            const IR::U1 overflow =
                ir.FPGreaterThanEqual(rounded, ir.Imm32(std::bit_cast<f32>(0x4f000000U)));
            const IR::F32 min_s32_f = ir.Imm32(std::bit_cast<f32>(0xcf000000U));
            const IR::F32 max_s32_f = ir.Imm32(std::bit_cast<f32>(0x4effffffU));
            const IR::F32 clamped{ir.FPClamp(rounded, min_s32_f, max_s32_f)};
            const IR::U32 conv{
                ir.Select(overflow, ir.Imm32(0x7fffffffU), ir.ConvertFToI(32, true, clamped))};
            result = IR::U32{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0U), conv)};
        }
    }
    R(dest, result);
}

void TranslatorVisitor::FRND(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 base = opcode == 0xb07U ? ReadCbufF32(insn) : F(insn.Bits(32, 8));
    const IR::F32 src{ir.FPAbsNeg(base, insn.Bits(62, 1) != 0U, insn.Bits(63, 1) != 0U)};
    const IR::FmzMode fmz = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None;
    F(dest, RoundFloat32(ir, src, insn.Bits(78, 2), fmz));
}

void TranslatorVisitor::I2F(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const bool is_signed = insn.Bits(74, 1) != 0U;

    if (opcode == 0x906U) {
        F(dest, ConvertI32ToF32Rounded(ir, ir.Imm32(insn.words[1]), is_signed, insn.Bits(78, 2)));
        return;
    }

    const bool uniform = opcode == 0xd06U;
    const IR::U32 raw = uniform ? UR(insn.Bits(32, 8)) : R(insn.Bits(32, 8));
    if ((insn.words[2] & ~0xc400U) == (uniform ? 0x08201000U : 0x00201000U)) {
        F(dest, ConvertI32ToF32Rounded(ir, raw, is_signed, insn.Bits(78, 2)));
        return;
    }

    const bool is_16bit = (insn.words[2] & ~0x400U) == (uniform ? 0x08101000U : 0x00101000U);
    const u32 sel = insn.Bits(60, 2);
    const IR::U32 extracted = ir.BitFieldExtract(
        raw, ir.Imm32(sel * (is_16bit ? 16U : 8U)), ir.Imm32(is_16bit ? 16U : 8U), is_signed);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::RN,
        .fmz_mode = IR::FmzMode::DontCare,
    };
    F(dest, IR::F32{ir.ConvertIToF(32, 32, is_signed, extracted, control)});
}

void TranslatorVisitor::I2FP(const Instruction& insn) {
    F(insn.Bits(16, 8), ConvertI32ToF32Rounded(ir, AluSourceU32(insn, 1U), insn.Bits(74, 1) != 0U,
                                               insn.Bits(78, 2)));
}

} // namespace Shader::Ampere
