// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {

void TranslatorVisitor::VOTE(const Instruction& insn) {
    const bool is_uniform = insn.Bits(0, 12) == 0x886;
    const u32 dest = insn.Bits(16, 8);
    const u32 pred_target = insn.Bits(81, 3);
    const u32 mode = insn.Bits(72, 2);
    const IR::U1 pred = P(insn.Bits(87, 4));

    const IR::U1 vote_res = mode == 0U   ? ir.VoteAll(pred)
                            : mode == 1U ? ir.VoteAny(pred)
                                         : ir.VoteEqual(pred);
    const IR::U32 ballot = ir.SubgroupBallot(pred);
    if (is_uniform) {
        SetUP(pred_target, vote_res);
        UR(dest, ballot);
    } else {
        SetP(pred_target, vote_res);
        R(dest, ballot);
    }
}

void TranslatorVisitor::R2UR(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 guard = insn.Bits(12, 4);
    if (guard == 15U || dest == 63U) {
        return;
    }
    const IR::U1 pred = P(guard);
    const IR::U32 ballot = ir.SubgroupBallot(pred);
    const IR::U32 lowest_bit = ir.BitwiseAnd(ballot, ir.INeg(ballot));
    const IR::U32 first_lane = ir.FindUMsb(lowest_bit);
    const IR::U32 broadcast =
        ir.ShuffleIndex(R(insn.Bits(24, 8)), first_lane, ir.Imm32(31U), ir.Imm32(0U));
    if (guard == 7U) {
        // Keep unconditional provenance intact for pointer/handle tracking passes.
        UR(dest, broadcast);
    } else {
        UR(dest, IR::U32{ir.Select(ir.VoteAny(pred), broadcast, UR(dest))});
    }
}

void TranslatorVisitor::REDUX(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    if (dest == 63U) {
        return;
    }
    const IR::U32 src = R(insn.Bits(24, 8));
    const u32 mode = insn.words[2] & ~0x200U;
    const bool is_max = mode == 0x14000U;
    const bool is_min = mode == 0x10000U;
    const bool is_signed = insn.Bits(73, 1) != 0;
    const bool is_or = insn.words[2] == 0x4000U;
    const bool is_xor = insn.words[2] == 0x8000U;

    const IR::U32 ballot = ir.SubgroupBallot(ir.Imm1(true));
    const IR::U32 identity =
        is_min ? ir.Imm32(is_signed ? 0x7fffffffU : 0xffffffffU)
        : is_max ? ir.Imm32(is_signed ? 0x80000000U : 0U)
                 : ir.Imm32(0U);

    IR::U32 acc = identity;
    for (u32 lane = 0; lane < 32U; ++lane) {
        const IR::U32 shuffled =
            ir.ShuffleIndex(src, ir.Imm32(lane), ir.Imm32(31U), ir.Imm32(0U));
        const IR::U1 active = ir.INotEqual(ir.BitwiseAnd(ballot, ir.Imm32(1U << lane)), ir.Imm32(0U));
        const IR::U32 contrib{ir.Select(active, shuffled, identity)};
        if (is_min) {
            acc = is_signed ? ir.SMin(acc, contrib) : ir.UMin(acc, contrib);
        } else if (is_max) {
            acc = is_signed ? ir.SMax(acc, contrib) : ir.UMax(acc, contrib);
        } else if (is_or) {
            acc = ir.BitwiseOr(acc, contrib);
        } else if (is_xor) {
            acc = ir.BitwiseXor(acc, contrib);
        } else {
            acc = ir.IAdd(acc, contrib);
        }
    }
    UR(dest, acc);
}

} // namespace Shader::Ampere
