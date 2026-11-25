// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>

#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {

void TranslatorVisitor::IADD3(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x290 || opcode == 0x890;
    const u32 zero_reg = is_uniform ? 63U : 255U;
    const u32 dest = insn.Bits(16, 8);
    const u32 a_reg = insn.Bits(24, 8);
    const u32 c_reg = insn.Bits(64, 8);

    const auto read_reg = [&](u32 reg) { return is_uniform ? UR(reg) : R(reg); };
    const auto write_reg = [&](u32 reg, const IR::U32& val) {
        is_uniform ? UR(reg, val) : R(reg, val);
    };
    const auto read_pred = [&](u32 pred4) { return is_uniform ? UP(pred4) : P(pred4); };
    const auto write_pred = [&](u32 idx, const IR::U1& val) {
        is_uniform ? SetUP(idx, val) : SetP(idx, val);
    };

    const bool imm_b = opcode == 0x810 || opcode == 0x890;
    const IR::U32 a_val = read_reg(a_reg);
    const IR::U32 b_val = imm_b      ? ir.Imm32(insn.words[1])
                          : is_uniform ? UR(insn.Bits(32, 8))
                                       : AluSourceU32(insn, 1);
    const IR::U32 c_val = read_reg(c_reg);

    const bool extended = insn.Bits(74, 1) != 0;
    const bool neg_a = insn.Bits(72, 1) != 0;
    const bool neg_b = !imm_b && insn.Bits(63, 1) != 0;
    const bool neg_c = insn.Bits(75, 1) != 0;
    const u32 out1 = insn.Bits(81, 3);
    const u32 out2 = insn.Bits(84, 3);
    const u32 in1 = insn.Bits(87, 4);
    const u32 in2 = insn.Bits(77, 4);

    // Fast path: modulo-2^32 addition/subtraction with no carry-in and no carry-out.
    if (!extended && out1 == 7U && out2 == 7U) {
        std::optional<IR::U32> pos;
        std::optional<IR::U32> neg;
        const auto add_signed = [&](const IR::U32& val, bool present, bool is_neg) {
            if (!present) {
                return;
            }
            if (is_neg) {
                neg = neg ? IR::U32{ir.IAdd(*neg, val)} : val;
            } else {
                pos = pos ? IR::U32{ir.IAdd(*pos, val)} : val;
            }
        };
        add_signed(a_val, a_reg != zero_reg, neg_a);
        add_signed(b_val, imm_b || insn.Bits(32, 8) != zero_reg || opcode == 0xa10 || opcode == 0xc10, neg_b);
        add_signed(c_val, c_reg != zero_reg, neg_c);

        IR::U32 res = ir.Imm32(0U);
        if (pos && neg) {
            res = ir.ISub(*pos, *neg);
        } else if (pos) {
            res = *pos;
        } else if (neg) {
            res = ir.INeg(*neg);
        }
        write_reg(dest, res);
        return;
    }

    // Carry-producing / extended (.X) path: compute exact 34-bit unsigned sum over inverted
    // inputs + carry-in (or +ones when not extended).
    const bool need_carry = out1 != 7U || out2 != 7U;
    std::optional<IR::U32> sum;
    std::optional<IR::U1> single_carry;
    std::optional<IR::U32> carry_count;
    u32 add_ops = 0;

    const auto accumulate = [&](const IR::U32& term) {
        if (!sum) {
            sum = term;
            return;
        }
        const IR::U32 next_sum = ir.IAdd(*sum, term);
        ++add_ops;
        if (need_carry) {
            const IR::U1 carry = ir.GetCarryFromOp(next_sum);
            if (!single_carry) {
                single_carry = carry;
            } else {
                if (!carry_count) {
                    carry_count = IR::U32{ir.Select(*single_carry, ir.Imm32(1U), ir.Imm32(0U))};
                }
                carry_count =
                    ir.IAdd(*carry_count, IR::U32{ir.Select(carry, ir.Imm32(1U), ir.Imm32(0U))});
            }
        }
        sum = next_sum;
    };

    if (a_reg != zero_reg || neg_a) {
        accumulate(neg_a ? ir.BitwiseNot(a_val) : a_val);
    }
    const bool b_present =
        imm_b || opcode == 0xa10 || opcode == 0xc10 || insn.Bits(32, 8) != zero_reg || neg_b;
    if (b_present) {
        accumulate(neg_b ? ir.BitwiseNot(b_val) : b_val);
    }
    if (c_reg != zero_reg || neg_c) {
        accumulate(neg_c ? ir.BitwiseNot(c_val) : c_val);
    }
    if (extended) {
        for (const u32 cin : {in1, in2}) {
            if (cin == 15U) {
                continue;
            }
            const IR::U32 cin_u32 =
                cin == 7U ? ir.Imm32(1U)
                          : IR::U32{ir.Select(read_pred(cin), ir.Imm32(1U), ir.Imm32(0U))};
            accumulate(cin_u32);
        }
    } else if (const u32 ones = u32{neg_a} + u32{neg_b} + u32{neg_c}; ones != 0U) {
        accumulate(ir.Imm32(ones));
    }

    const IR::U32 final_sum = sum.value_or(ir.Imm32(0U));
    IR::U1 res_out1 = ir.Imm1(false);
    IR::U1 res_out2 = ir.Imm1(false);
    if (need_carry) {
        if (add_ops == 0U) {
            res_out1 = ir.Imm1(false);
            res_out2 = ir.Imm1(false);
        } else if (add_ops == 1U) {
            res_out1 = *single_carry;
            res_out2 = ir.Imm1(false);
        } else {
            const IR::U32 total =
                carry_count.value_or(IR::U32{ir.Select(*single_carry, ir.Imm32(1U), ir.Imm32(0U))});
            if (out1 != 7U) {
                res_out1 = ir.IGreaterThanEqual(total, ir.Imm32(1U), false);
            }
            if (out2 != 7U) {
                res_out2 = ir.IGreaterThanEqual(total, ir.Imm32(2U), false);
            }
        }
    }

    write_reg(dest, final_sum);
    write_pred(out1, res_out1);
    write_pred(out2, res_out2);
}

void TranslatorVisitor::LEA(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool is_uniform = opcode == 0x291 || opcode == 0x891;
    const u32 zero_reg = is_uniform ? 63U : 255U;
    const u32 dest = insn.Bits(16, 8);
    const u32 a_reg = insn.Bits(24, 8);
    const u32 c_reg = insn.Bits(64, 8);
    const u32 shift = insn.Bits(75, 5);
    const bool high_mode = insn.Bits(80, 1) != 0;
    const bool sx32 = insn.Bits(73, 1) != 0;
    const bool extended = insn.Bits(74, 1) != 0;
    const bool imm_b = opcode == 0x811 || opcode == 0x891;
    const bool neg_b = (opcode == 0x211 || opcode == 0xc11 || opcode == 0x291) && insn.Bits(63, 1);

    const auto read_reg = [&](u32 reg) { return is_uniform ? UR(reg) : R(reg); };
    const auto write_reg = [&](u32 reg, const IR::U32& val) {
        is_uniform ? UR(reg, val) : R(reg, val);
    };
    const auto read_pred = [&](u32 pred4) { return is_uniform ? UP(pred4) : P(pred4); };
    const auto write_pred = [&](u32 idx, const IR::U1& val) {
        is_uniform ? SetUP(idx, val) : SetP(idx, val);
    };

    const IR::U32 a_val = read_reg(a_reg);
    const bool b_is_zero =
        (imm_b && insn.words[1] == 0U) || (!imm_b && opcode != 0xa11 && opcode != 0xc11 && insn.Bits(32, 8) == zero_reg);
    const IR::U32 b_val = imm_b      ? ir.Imm32(insn.words[1])
                          : is_uniform ? UR(insn.Bits(32, 8))
                                       : AluSourceU32(insn, 1);

    if (high_mode) {
        IR::U32 high_word;
        if (shift == 0U) {
            if (c_reg == zero_reg) {
                high_word =
                    sx32 ? IR::U32{ir.ShiftRightArithmetic(a_val, ir.Imm32(31U))} : ir.Imm32(0U);
            } else {
                high_word = read_reg(c_reg);
            }
        } else if (c_reg == zero_reg) {
            high_word = sx32 ? IR::U32{ir.ShiftRightArithmetic(a_val, ir.Imm32(32U - shift))}
                             : IR::U32{ir.ShiftRightLogical(a_val, ir.Imm32(32U - shift))};
        } else {
            const IR::U32 lo_part = ir.ShiftRightLogical(a_val, ir.Imm32(32U - shift));
            const IR::U32 hi_part = ir.ShiftLeftLogical(read_reg(c_reg), ir.Imm32(shift));
            high_word = ir.BitwiseOr(hi_part, lo_part);
        }

        IR::U32 result =
            b_is_zero ? high_word
                      : (neg_b ? IR::U32{ir.ISub(high_word, b_val)} : IR::U32{ir.IAdd(high_word, b_val)});
        if (extended) {
            const u32 carry_pred = insn.Bits(87, 4);
            if (carry_pred != 15U) {
                const IR::U32 cin =
                    carry_pred == 7U
                        ? ir.Imm32(1U)
                        : IR::U32{ir.Select(read_pred(carry_pred), ir.Imm32(1U), ir.Imm32(0U))};
                result = ir.IAdd(result, cin);
            }
        }
        write_reg(dest, result);
        return;
    }

    const IR::U32 scaled_a =
        shift == 0U ? a_val : IR::U32{ir.ShiftLeftLogical(a_val, ir.Imm32(shift))};
    const u32 carry_out = insn.Bits(81, 3);
    if (carry_out != 7U) {
        const IR::U32 result = ir.IAdd(scaled_a, b_val);
        write_pred(carry_out, ir.GetCarryFromOp(result));
        write_reg(dest, result);
        return;
    }
    const IR::U32 result =
        b_is_zero ? scaled_a
                  : (neg_b ? IR::U32{ir.ISub(scaled_a, b_val)} : IR::U32{ir.IAdd(scaled_a, b_val)});
    write_reg(dest, result);
}

void TranslatorVisitor::IABS(const Instruction& insn) {
    R(insn.Bits(16, 8), ir.IAbs(AluSourceU32(insn, 1)));
}

void TranslatorVisitor::IMNMX(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const IR::U32 op_a = R(insn.Bits(24, 8));
    IR::U32 op_b;
    if (opcode == 0x817) {
        op_b = ir.Imm32(insn.words[1]);
    } else if (opcode == 0xc17) {
        op_b = UR(insn.words[1]);
    } else if (opcode == 0xa17) {
        op_b = AluSourceU32(insn, 1);
    } else {
        op_b = R(insn.words[1]);
    }
    const bool is_signed = insn.Bits(73, 1) != 0;
    const IR::U32 min_v = is_signed ? ir.SMin(op_a, op_b) : ir.UMin(op_a, op_b);
    const IR::U32 max_v = is_signed ? ir.SMax(op_a, op_b) : ir.UMax(op_a, op_b);
    const u32 pred_bits = insn.Bits(87, 4);
    if (pred_bits == 7U) {
        R(dest, min_v);
    } else if (pred_bits == 15U) {
        R(dest, max_v);
    } else {
        R(dest, IR::U32{ir.Select(P(pred_bits), min_v, max_v)});
    }
}

} // namespace Shader::Ampere
