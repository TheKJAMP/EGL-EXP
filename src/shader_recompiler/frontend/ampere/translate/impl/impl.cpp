// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/program_header.h"

namespace Shader::Ampere {

TranslatorVisitor::TranslatorVisitor(Environment& env_, IR::Block& block,
                                     std::span<const Instruction> code_,
                                     const LoweringOptions& options_, bool uses_async_copy_,
                                     u32& required_texture_buffer_,
                                     const std::vector<u8>* branch_targets_)
    : ir{block}, env{env_}, code{code_}, options{options_}, uses_async_copy{uses_async_copy_},
      required_texture_buffer{required_texture_buffer_}, branch_targets{branch_targets_} {}

IR::U32 TranslatorVisitor::R(u32 reg) {
    return reg == 255U ? ir.Imm32(0U) : ir.GetReg(Gpr(reg));
}

void TranslatorVisitor::R(u32 reg, const IR::U32& value) {
    if (reg == 255U) {
        return;
    }
    if (active_guard) {
        const IR::U32 prev = ir.GetReg(Gpr(reg));
        ir.SetReg(Gpr(reg), IR::U32{ir.Select(*active_guard, value, prev)});
        return;
    }
    ir.SetReg(Gpr(reg), value);
}

IR::U32 TranslatorVisitor::UR(u32 reg) {
    return reg == 63U ? ir.Imm32(0U) : ir.GetReg(UniformReg(reg));
}

void TranslatorVisitor::UR(u32 reg, const IR::U32& value) {
    if (reg == 63U) {
        return;
    }
    if (active_guard) {
        const IR::U32 prev = ir.GetReg(UniformReg(reg));
        ir.SetReg(UniformReg(reg), IR::U32{ir.Select(*active_guard, value, prev)});
        return;
    }
    ir.SetReg(UniformReg(reg), value);
}

IR::F32 TranslatorVisitor::F(u32 reg) {
    return reg == 255U ? ir.Imm32(0.0f) : ir.BitCast<IR::F32>(R(reg));
}

void TranslatorVisitor::F(u32 reg, const IR::F32& value) {
    if (reg != 255U) {
        R(reg, ir.BitCast<IR::U32>(value));
    }
}

IR::U1 TranslatorVisitor::P(u32 pred4) {
    const u32 idx = pred4 & 7U;
    const bool negated = (pred4 & 8U) != 0;
    if (idx == 7U) {
        return ir.Imm1(!negated);
    }
    return ir.GetPred(ThreadPred(idx), negated);
}

IR::U1 TranslatorVisitor::UP(u32 pred4) {
    const u32 idx = pred4 & 7U;
    const bool negated = (pred4 & 8U) != 0;
    if (idx == 7U) {
        return ir.Imm1(!negated);
    }
    return ir.GetPred(UniformPred(idx), negated);
}

void TranslatorVisitor::SetP(u32 idx, const IR::U1& value) {
    if (idx >= 7U) {
        return;
    }
    if (active_guard) {
        const IR::U1 prev = ir.GetPred(ThreadPred(idx));
        ir.SetPred(ThreadPred(idx), IR::U1{ir.Select(*active_guard, value, prev)});
        return;
    }
    ir.SetPred(ThreadPred(idx), value);
}

void TranslatorVisitor::SetUP(u32 idx, const IR::U1& value) {
    if (idx >= 7U) {
        return;
    }
    if (active_guard) {
        const IR::U1 prev = ir.GetPred(UniformPred(idx));
        ir.SetPred(UniformPred(idx), IR::U1{ir.Select(*active_guard, value, prev)});
        return;
    }
    ir.SetPred(UniformPred(idx), value);
}

IR::U1 TranslatorVisitor::EvalGuard(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 pred_bits = insn.Bits(12, 4);
    return UsesUniformGuardPredicate(opcode) ? UP(pred_bits) : P(pred_bits);
}

u32 TranslatorVisitor::CbufBank(const Instruction& insn) const noexcept {
    const u32 value = insn.Bits(54, 5);
    return value == 24U ? 17U : value;
}

IR::U32 TranslatorVisitor::ReadCbuf32(const Instruction& insn) {
    const u32 bank = CbufBank(insn);
    const u32 offset = insn.Bits(38, 16);
    if (insn.Bits(91, 1) || bank >= 18U || (offset % 4U) != 0U) {
        throw RuntimeError("Invalid native constant buffer address");
    }
    return ir.GetCbuf(ir.Imm32(bank), ir.Imm32(offset));
}

IR::U32 TranslatorVisitor::AluSourceU32(const Instruction& insn, u32 which) {
    if (which == 0U) {
        return R(insn.Bits(24, 8));
    }
    const u32 form = insn.Bits(9, 3);
    if (form < 1U || form > 7U) {
        throw RuntimeError("Invalid native ALU source form {}", form);
    }
    const bool wide = which == 1U ? (form == 4U || form == 5U || form == 6U)
                                  : (form == 2U || form == 3U || form == 7U);
    if (wide) {
        if (form == 6U || form == 7U) {
            const u32 reg = insn.Bits(32, 8);
            if (reg > 63U) {
                throw RuntimeError("Invalid native uniform source register UR{}", reg);
            }
            return UR(reg);
        }
        if (form == 4U || form == 2U) {
            return ir.Imm32(insn.words[1]);
        }
        return ReadCbuf32(insn);
    }
    const bool upper = which == 1U ? (form == 2U || form == 3U || form == 7U) : true;
    return R(insn.Bits(upper ? 64U : 32U, 8));
}

IR::F32 TranslatorVisitor::ReadCbufF32(const Instruction& insn) {
    const u32 bank = CbufBank(insn);
    const u32 offset = insn.Bits(38, 16);
    if (insn.Bits(91, 1) || bank >= 18U || (offset % 4U) != 0U) {
        throw RuntimeError("Invalid native constant buffer address");
    }
    return ir.GetFloatCbuf(ir.Imm32(bank), ir.Imm32(offset));
}

IR::F32 TranslatorVisitor::AluSourceF32(const Instruction& insn, u32 which) {
    if (which == 0U) {
        const u32 reg = insn.Bits(24, 8);
        const bool abs = insn.Bits(73, 1) != 0U;
        const bool neg = insn.Bits(72, 1) != 0U;
        return ir.FPAbsNeg(F(reg), abs, neg);
    }
    const u32 form = insn.Bits(9, 3);
    if (form < 1U || form > 7U) {
        throw RuntimeError("Invalid native ALU source form {}", form);
    }
    const bool wide = which == 1U ? (form == 4U || form == 5U || form == 6U)
                                  : (form == 2U || form == 3U || form == 7U);
    if (wide) {
        if (form == 6U || form == 7U) {
            const u32 reg = insn.Bits(32, 8);
            if (reg > 63U) {
                throw RuntimeError("Invalid native uniform source register UR{}", reg);
            }
            const bool abs = insn.Bits(62, 1) != 0U;
            const bool neg = insn.Bits(63, 1) != 0U;
            return ir.FPAbsNeg(ir.BitCast<IR::F32>(UR(reg)), abs, neg);
        }
        if (form == 4U || form == 2U) {
            return ir.Imm32(std::bit_cast<f32>(insn.words[1]));
        }
        const bool abs = insn.Bits(62, 1) != 0U;
        const bool neg = insn.Bits(63, 1) != 0U;
        return ir.FPAbsNeg(ReadCbufF32(insn), abs, neg);
    }
    const bool upper = which == 1U ? (form == 2U || form == 3U || form == 7U) : true;
    const u32 reg = insn.Bits(upper ? 64U : 32U, 8);
    const bool abs = insn.Bits(upper ? 74U : 62U, 1) != 0U;
    const bool neg = insn.Bits(upper ? 75U : 63U, 1) != 0U;
    return ir.FPAbsNeg(F(reg), abs, neg);
}

IR::Value TranslatorVisitor::TextureDescriptorHandle(u32 binding, u32 word_offset) {
    const u32 byte_offset = word_offset * 4U;
    if (binding == env.TextureBoundBuffer()) {
        return ir.Imm32(byte_offset);
    }
    return ir.GetCbuf(ir.Imm32(binding), ir.Imm32(byte_offset));
}

void TranslatorVisitor::EXIT() {
    if (env.ShaderStage() != Stage::Fragment) {
        return;
    }
    const ProgramHeader sph{env.SPH()};
    u32 src_reg{0U};
    for (u32 render_target = 0; render_target < 8U; ++render_target) {
        if (!sph.ps.HasOutputComponents(render_target)) {
            continue;
        }
        const std::array<bool, 4> mask{sph.ps.EnabledOutputComponents(render_target)};
        for (u32 component = 0; component < 4U; ++component) {
            if (!mask[component]) {
                ++src_reg;
                continue;
            }
            ir.SetFragColor(render_target, component, ir.BitCast<IR::F32>(R(src_reg)));
            ++src_reg;
        }
    }
    if (sph.ps.omap.sample_mask != 0) {
        ir.SetSampleMask(R(src_reg));
    }
    if (sph.ps.omap.depth != 0) {
        ir.SetFragDepth(ir.BitCast<IR::F32>(R(src_reg + 1U)));
    }
}

} // namespace Shader::Ampere
