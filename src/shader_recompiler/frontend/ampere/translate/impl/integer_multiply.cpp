// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {

void TranslatorVisitor::IMAD(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x2a4 || opcode == 0x4a4 || opcode == 0x8a4;
    const u32 zero_reg = is_uniform ? 63U : 255U;
    const u32 dest = insn.Bits(16, 8);
    const u32 a_reg = insn.Bits(24, 8);

    const auto read_reg = [&](u32 reg) { return is_uniform ? UR(reg) : R(reg); };
    const auto write_reg = [&](u32 reg, const IR::U32& val) {
        is_uniform ? UR(reg, val) : R(reg, val);
    };
    const auto read_pred = [&](u32 pred4) { return is_uniform ? UP(pred4) : P(pred4); };

    const bool extended = insn.Bits(74, 1) != 0;
    const bool negate_c =
        (opcode == 0xe24 && insn.Bits(63, 1) != 0) || (opcode != 0xe24 && insn.Bits(75, 1) != 0);

    const IR::U32 a_val = read_reg(a_reg);
    IR::U32 b_val;
    IR::U32 c_val;
    if (is_uniform) {
        b_val = opcode == 0x2a4 ? UR(insn.Bits(32, 8))
                : opcode == 0x4a4 ? UR(insn.Bits(64, 8))
                                  : ir.Imm32(insn.words[1]);
        c_val = opcode == 0x4a4 ? ir.Imm32(insn.words[1]) : UR(insn.Bits(64, 8));
    } else {
        b_val = AluSourceU32(insn, 1);
        c_val = AluSourceU32(insn, 2);
    }

    if (extended) {
        // IMAD.X Rd, Ra, 1, Rc, Pn or IMAD.X RZ, RZ, c, Pn
        const u32 cin_bits = insn.Bits(87, 4);
        const IR::U32 cin =
            cin_bits == 7U
                ? ir.Imm32(1U)
                : (cin_bits == 15U
                       ? ir.Imm32(0U)
                       : IR::U32{ir.Select(read_pred(cin_bits), ir.Imm32(1U), ir.Imm32(0U))});
        const IR::U32 adjusted_c = negate_c ? IR::U32{ir.BitwiseNot(c_val)} : c_val;
        const bool has_a = a_reg != zero_reg;
        const IR::U32 sum = has_a ? IR::U32{ir.IAdd(a_val, adjusted_c)} : adjusted_c;
        write_reg(dest, ir.IAdd(sum, cin));
        return;
    }

    const u32 form = insn.Bits(9, 3);
    const bool b_is_zero_reg =
        is_uniform ? (opcode == 0x2a4 && insn.Bits(32, 8) == 63U) ||
                         (opcode == 0x4a4 && insn.Bits(64, 8) == 63U) ||
                         (opcode == 0x8a4 && insn.words[1] == 0U)
                   : (form == 1U && insn.Bits(32, 8) == 255U) ||
                         ((form == 2U || form == 3U || form == 7U) && insn.Bits(64, 8) == 255U) ||
                         (form == 4U && insn.words[1] == 0U) ||
                         (form == 6U && insn.Bits(32, 8) == 63U);
    if (a_reg == zero_reg || b_is_zero_reg) {
        write_reg(dest, negate_c ? IR::U32{ir.INeg(c_val)} : c_val);
        return;
    }

    const bool c_is_zero_reg =
        is_uniform ? (opcode == 0x4a4 ? insn.words[1] == 0U : insn.Bits(64, 8) == 63U)
                   : ((form == 1U || form == 4U || form == 5U || form == 6U) &&
                      insn.Bits(64, 8) == 255U) ||
                         (form == 2U && insn.words[1] == 0U) ||
                         (form == 7U && insn.Bits(32, 8) == 63U);

    const IR::U32 product = ir.IMul(a_val, b_val);
    if (c_is_zero_reg) {
        write_reg(dest, product);
    } else {
        write_reg(dest, negate_c ? ir.ISub(product, c_val) : ir.IAdd(product, c_val));
    }
}

void TranslatorVisitor::IMAD_WIDE(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x2a5 || opcode == 0x8a5;
    const bool wide = opcode == 0x225 || opcode == 0x625 || opcode == 0x825 || is_uniform;
    const bool signed_product = !is_uniform && insn.Bits(73, 1) != 0;
    const bool imm_b = opcode == 0x825 || opcode == 0x827 || opcode == 0x8a5;
    const u32 zero_reg = is_uniform ? 63U : 255U;
    const u32 dest = insn.Bits(16, 8);
    const u32 carry_out = is_uniform ? 7U : insn.Bits(81, 3);

    const auto read_reg = [&](u32 reg) { return is_uniform ? UR(reg) : R(reg); };
    const auto write_reg = [&](u32 reg, const IR::U32& val) {
        is_uniform ? UR(reg, val) : R(reg, val);
    };

    const IR::U32 a = read_reg(insn.Bits(24, 8));
    IR::U32 b;
    if (imm_b) {
        b = ir.Imm32(insn.words[1]);
    } else if (opcode == 0xa27) {
        b = ReadCbuf32(insn);
    } else {
        b = read_reg(insn.Bits(opcode == 0x625 ? 64U : 32U, 8));
    }

    // Full 32x32 -> 64 product via 4 16-bit half-products (4 IMul32s total).
    const IR::U32 mask16 = ir.Imm32(0xffffU);
    const IR::U32 shift16 = ir.Imm32(16U);
    const IR::U32 a_lo = ir.BitwiseAnd(a, mask16);
    const IR::U32 a_hi = ir.ShiftRightLogical(a, shift16);
    const IR::U32 b_lo = ir.BitwiseAnd(b, mask16);
    const IR::U32 b_hi = ir.ShiftRightLogical(b, shift16);

    const IR::U32 ll = ir.IMul(a_lo, b_lo);
    const IR::U32 lh = ir.IMul(a_lo, b_hi);
    const IR::U32 hl = ir.IMul(a_hi, b_lo);
    const IR::U32 hh = ir.IMul(a_hi, b_hi);

    const IR::U32 mid = ir.IAdd(ir.ShiftRightLogical(ll, shift16),
                                ir.IAdd(ir.BitwiseAnd(lh, mask16), ir.BitwiseAnd(hl, mask16)));
    const IR::U32 prod_lo = ir.IMul(a, b);
    IR::U32 prod_hi =
        ir.IAdd(hh, ir.IAdd(ir.ShiftRightLogical(lh, shift16),
                            ir.IAdd(ir.ShiftRightLogical(hl, shift16),
                                    ir.ShiftRightLogical(mid, shift16))));
    if (signed_product) {
        const IR::U32 sign_a = ir.ShiftRightArithmetic(a, ir.Imm32(31U));
        const IR::U32 sign_b = ir.ShiftRightArithmetic(b, ir.Imm32(31U));
        const IR::U32 adj = ir.IAdd(ir.BitwiseAnd(sign_a, b), ir.BitwiseAnd(sign_b, a));
        prod_hi = ir.ISub(prod_hi, adj);
    }

    const u32 c_reg = opcode == 0x625 ? zero_reg : insn.Bits(64, 8);
    if (!wide && opcode != 0x625 && c_reg == zero_reg && carry_out == 7U) {
        write_reg(dest, prod_hi);
        return;
    }

    IR::U32 c_lo;
    IR::U32 c_hi;
    if (opcode == 0x625) {
        const IR::U32 bank = ir.Imm32(CbufBank(insn));
        const u32 off = insn.Bits(38, 16);
        c_lo = ir.GetCbuf(bank, ir.Imm32(off));
        c_hi = ir.GetCbuf(bank, ir.Imm32(off + 4U));
    } else if (c_reg == zero_reg) {
        c_lo = ir.Imm32(0U);
        c_hi = ir.Imm32(0U);
    } else {
        c_lo = read_reg(c_reg);
        c_hi = read_reg(c_reg + 1U);
    }

    IR::U32 res_lo = prod_lo;
    IR::U32 res_hi = prod_hi;
    if (opcode == 0x625 || c_reg != zero_reg) {
        res_lo = ir.IAdd(prod_lo, c_lo);
        const IR::U1 lo_carry = ir.GetCarryFromOp(res_lo);
        const IR::U32 sum_hi = ir.IAdd(prod_hi, c_hi);
        const IR::U1 hi_carry_1 = ir.GetCarryFromOp(sum_hi);
        res_hi = ir.IAdd(sum_hi, IR::U32{ir.Select(lo_carry, ir.Imm32(1U), ir.Imm32(0U))});
        if (carry_out != 7U) {
            const IR::U1 hi_carry_2 = ir.GetCarryFromOp(res_hi);
            SetP(carry_out, ir.LogicalOr(hi_carry_1, hi_carry_2));
        }
    } else if (carry_out != 7U) {
        SetP(carry_out, ir.Imm1(false));
    }

    if (wide) {
        write_reg(dest, res_lo);
        write_reg(dest + 1U, res_hi);
    } else {
        write_reg(dest, res_hi);
    }
}

void TranslatorVisitor::IDP2A(const Instruction& insn) {
    const IR::U32 a = R(insn.Bits(24, 8));
    const IR::U32 b = R(insn.Bits(32, 8));
    const IR::U32 c = R(insn.Bits(64, 8));
    const IR::U32 a0 = ir.BitFieldExtract(a, ir.Imm32(0U), ir.Imm32(16U), false);
    const IR::U32 a1 = ir.BitFieldExtract(a, ir.Imm32(16U), ir.Imm32(16U), false);
    const IR::U32 b0 = ir.BitFieldExtract(b, ir.Imm32(0U), ir.Imm32(8U), false);
    const IR::U32 b1 = ir.BitFieldExtract(b, ir.Imm32(8U), ir.Imm32(8U), false);
    const IR::U32 dot = ir.IAdd(ir.IMul(a0, b0), ir.IMul(a1, b1));
    R(insn.Bits(16, 8), ir.IAdd(dot, c));
}

} // namespace Shader::Ampere
