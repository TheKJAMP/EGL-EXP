// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"

namespace Shader::Ampere {
namespace {

[[nodiscard]] IR::U1 EvalPredLUT2(IR::IREmitter& ir, const IR::U1& a, const IR::U1& b, u32 t4) {
    switch (t4 & 0xfU) {
    case 0x0:
        return ir.Imm1(false);
    case 0x1:
        return ir.LogicalNot(ir.LogicalOr(a, b));
    case 0x2:
        return ir.LogicalAnd(ir.LogicalNot(a), b);
    case 0x3:
        return ir.LogicalNot(a);
    case 0x4:
        return ir.LogicalAnd(a, ir.LogicalNot(b));
    case 0x5:
        return ir.LogicalNot(b);
    case 0x6:
        return ir.LogicalXor(a, b);
    case 0x7:
        return ir.LogicalNot(ir.LogicalAnd(a, b));
    case 0x8:
        return ir.LogicalAnd(a, b);
    case 0x9:
        return ir.LogicalNot(ir.LogicalXor(a, b));
    case 0xa:
        return b;
    case 0xb:
        return ir.LogicalOr(ir.LogicalNot(a), b);
    case 0xc:
        return a;
    case 0xd:
        return ir.LogicalOr(a, ir.LogicalNot(b));
    case 0xe:
        return ir.LogicalOr(a, b);
    case 0xf:
        return ir.Imm1(true);
    default:
        return ir.Imm1(false);
    }
}

[[nodiscard]] IR::U1 EvalPredLUT3(IR::IREmitter& ir, const IR::U1& a, const IR::U1& b,
                                  const IR::U1& c, u32 lut) {
    // Bit index is (a << 2) | (b << 1) | c
    if (((lut ^ (lut >> 1)) & 0x55U) == 0U) {
        const u32 t4 = ((lut >> 0) & 1U) | (((lut >> 2) & 1U) << 1) | (((lut >> 4) & 1U) << 2) |
                       (((lut >> 6) & 1U) << 3);
        return EvalPredLUT2(ir, a, b, t4);
    }
    if (((lut ^ (lut >> 2)) & 0x33U) == 0U) {
        const u32 t4 = ((lut >> 0) & 1U) | (((lut >> 1) & 1U) << 1) | (((lut >> 4) & 1U) << 2) |
                       (((lut >> 5) & 1U) << 3);
        return EvalPredLUT2(ir, a, c, t4);
    }
    if (((lut ^ (lut >> 4)) & 0x0fU) == 0U) {
        return EvalPredLUT2(ir, b, c, lut & 0x0fU);
    }
    const u32 lo = lut & 0x0fU;
    const u32 hi = (lut >> 4) & 0x0fU;
    const IR::U1 hi_res = EvalPredLUT2(ir, b, c, hi);
    const IR::U1 lo_res = EvalPredLUT2(ir, b, c, lo);
    if (lo == 0U) {
        return ir.LogicalAnd(a, hi_res);
    }
    if (hi == 0U) {
        return ir.LogicalAnd(ir.LogicalNot(a), lo_res);
    }
    if (lo == 0x0fU) {
        return ir.LogicalOr(ir.LogicalNot(a), hi_res);
    }
    if (hi == 0x0fU) {
        return ir.LogicalOr(a, lo_res);
    }
    return IR::U1{ir.Select(a, hi_res, lo_res)};
}

} // Anonymous namespace

void TranslatorVisitor::ISETP(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x28c || opcode == 0x88c;
    const bool is_signed = insn.Bits(73, 1) != 0;
    const auto combine = static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2));
    const auto cmp_op = static_cast<Maxwell::CompareOp>(insn.Bits(76, 3));
    const u32 dst_a = insn.Bits(81, 3);
    const u32 dst_b = insn.Bits(84, 3);
    const u32 accum = insn.Bits(87, 4);

    const IR::U32 a = is_uniform ? UR(insn.Bits(24, 8)) : R(insn.Bits(24, 8));
    const IR::U32 b = opcode == 0x88c  ? ir.Imm32(insn.words[1])
                      : opcode == 0x28c ? UR(insn.Bits(32, 8))
                                        : AluSourceU32(insn, 1);

    IR::U1 comparison;
    if (!is_uniform && insn.Bits(72, 1)) {
        // 64-bit extended integer compare: low-word result comes in bits 68..71.
        const IR::U1 low_pred = P(insn.Bits(68, 4));
        const IR::U1 eq_hi = ir.IEqual(a, b);
        switch (cmp_op) {
        case Maxwell::CompareOp::Equal:
            comparison = ir.LogicalAnd(eq_hi, low_pred);
            break;
        case Maxwell::CompareOp::NotEqual:
            comparison = ir.LogicalOr(ir.INotEqual(a, b), low_pred);
            break;
        case Maxwell::CompareOp::LessThan:
        case Maxwell::CompareOp::LessThanEqual:
            comparison =
                ir.LogicalOr(ir.ILessThan(a, b, is_signed), ir.LogicalAnd(eq_hi, low_pred));
            break;
        case Maxwell::CompareOp::GreaterThan:
        case Maxwell::CompareOp::GreaterThanEqual:
            comparison =
                ir.LogicalOr(ir.IGreaterThan(a, b, is_signed), ir.LogicalAnd(eq_hi, low_pred));
            break;
        default:
            throw NotImplementedException("Unsupported ISETP.EX compare {}", static_cast<u32>(cmp_op));
        }
    } else {
        comparison = Maxwell::IntegerCompare(ir, a, b, cmp_op, is_signed);
    }

    const IR::U1 accum_pred = is_uniform ? UP(accum) : P(accum);
    const IR::U1 res_a = Maxwell::PredicateCombine(ir, comparison, accum_pred, combine);
    const IR::U1 res_b =
        Maxwell::PredicateCombine(ir, ir.LogicalNot(comparison), accum_pred, combine);

    if (is_uniform) {
        SetUP(dst_a, res_a);
        SetUP(dst_b, res_b);
    } else {
        SetP(dst_a, res_a);
        SetP(dst_b, res_b);
    }
}

void TranslatorVisitor::SEL(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x287 || opcode == 0x887;
    const u32 dest = insn.Bits(16, 8);
    const u32 pred_bits = insn.Bits(87, 4);

    const IR::U32 a = is_uniform ? UR(insn.Bits(24, 8)) : R(insn.Bits(24, 8));
    const IR::U32 b = opcode == 0x887   ? ir.Imm32(insn.words[1])
                      : is_uniform       ? UR(insn.Bits(32, 8))
                                         : AluSourceU32(insn, 1);
    const IR::U1 cond = is_uniform ? UP(pred_bits) : P(pred_bits);
    const IR::U32 res{ir.Select(cond, a, b)};
    is_uniform ? UR(dest, res) : R(dest, res);
}

void TranslatorVisitor::PLOP3(const Instruction& insn) {
    const bool is_uniform = insn.Bits(0, 12) == 0x89c;
    const u32 target1 = insn.Bits(81, 3);
    const u32 target2 = is_uniform ? 7U : insn.Bits(84, 3);
    if (target1 == 7U && target2 == 7U) {
        return;
    }

    const u32 lut1 = insn.Bits(64, 3) | (insn.Bits(72, 5) << 3);
    const u32 lut2 = insn.Bits(16, 8);
    const IR::U1 pa = is_uniform ? UP(insn.Bits(87, 4)) : P(insn.Bits(87, 4));
    const IR::U1 pb = is_uniform ? UP(insn.Bits(77, 4)) : P(insn.Bits(77, 4));
    const IR::U1 pc =
        (is_uniform || insn.Bits(67, 1) != 0) ? UP(insn.Bits(68, 4)) : P(insn.Bits(68, 4));

    const IR::U1 res1 =
        target1 != 7U ? EvalPredLUT3(ir, pa, pb, pc, lut1) : ir.Imm1(false);
    const IR::U1 res2 =
        target2 != 7U ? EvalPredLUT3(ir, pa, pb, pc, lut2) : ir.Imm1(false);

    if (is_uniform) {
        SetUP(target1, res1);
    } else {
        SetP(target1, res1);
        SetP(target2, res2);
    }
}

void TranslatorVisitor::P2R(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    if (dest == 255U) {
        return;
    }
    const IR::U32 src = R(insn.Bits(24, 8));
    const u32 mask = insn.words[1] & 0x7fU;
    const u32 offset = insn.Bits(76, 2) * 8U;

    IR::U32 insert = ir.Imm32(0U);
    for (u32 k = 0; k < 7U; ++k) {
        if (((mask >> k) & 1U) == 0U) {
            continue;
        }
        insert = ir.BitwiseOr(
            insert, IR::U32{ir.Select(P(k), ir.Imm32(1U << (k + offset)), ir.Imm32(0U))});
    }
    const IR::U32 masked = ir.BitwiseAnd(src, ir.Imm32(~(mask << offset)));
    R(dest, ir.BitwiseOr(masked, insert));
}

void TranslatorVisitor::R2P(const Instruction& insn) {
    const IR::U32 src = R(insn.Bits(24, 8));
    const u32 mask = insn.words[1] & 0x7fU;
    const u32 offset = insn.Bits(76, 2) * 8U;
    for (u32 k = 0; k < 7U; ++k) {
        if (((mask >> k) & 1U) == 0U) {
            continue;
        }
        const IR::U32 bit = ir.BitFieldExtract(src, ir.Imm32(offset + k), ir.Imm32(1U), false);
        SetP(k, ir.INotEqual(bit, ir.Imm32(0U)));
    }
}

void TranslatorVisitor::UP2UR(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    if (dest == 63U) {
        return;
    }
    const u32 mask = insn.words[1] & 0x7fU;
    IR::U32 acc = ir.Imm32(0U);
    for (u32 k = 0; k < 7U; ++k) {
        if (((mask >> k) & 1U) == 0U) {
            continue;
        }
        acc = ir.BitwiseOr(acc, IR::U32{ir.Select(UP(k), ir.Imm32(1U << k), ir.Imm32(0U))});
    }
    UR(dest, acc);
}

void TranslatorVisitor::CSMTEST(const Instruction& insn) {
    const IR::U1 always_false = ir.Imm1(false);
    SetP(insn.Bits(81, 3), always_false);
    SetP(insn.Bits(84, 3), always_false);
}

} // namespace Shader::Ampere
