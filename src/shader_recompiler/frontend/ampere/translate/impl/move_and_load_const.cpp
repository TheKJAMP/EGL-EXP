// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

template <typename WriteWordFn>
void EmitConstantLoad(IR::IREmitter& ir, const IR::U32& cbuf_idx, const IR::U32& cbuf_off,
                      u32 width, WriteWordFn&& write_word) {
    switch (width) {
    case 0:
        write_word(0U, IR::U32{ir.GetCbuf(cbuf_idx, cbuf_off, 8, false)});
        break;
    case 1:
        write_word(0U, IR::U32{ir.GetCbuf(cbuf_idx, cbuf_off, 8, true)});
        break;
    case 2:
        write_word(0U, IR::U32{ir.GetCbuf(cbuf_idx, cbuf_off, 16, false)});
        break;
    case 3:
        write_word(0U, IR::U32{ir.GetCbuf(cbuf_idx, cbuf_off, 16, true)});
        break;
    case 4:
        write_word(0U, IR::U32{ir.GetCbuf(cbuf_idx, cbuf_off, 32, false)});
        break;
    case 5: {
        const IR::Value vector{ir.GetCbuf(cbuf_idx, cbuf_off, 64, false)};
        write_word(0U, IR::U32{ir.CompositeExtract(vector, 0)});
        write_word(1U, IR::U32{ir.CompositeExtract(vector, 1)});
        break;
    }
    default:
        throw NotImplementedException("Ampere constant load width {}", width);
    }
}

[[nodiscard]] IR::U32 ReadSpecialRegister(IR::IREmitter& ir, u32 special) {
    switch (special) {
    case 0: // SR_LANEID
        return ir.LaneId();
    case 17: // SR_INVOCATION_ID
        return ir.InvocationId();
    case 18: // SR_Y_DIRECTION
        return ir.BitCast<IR::U32>(ir.YDirection());
    case 19: // SR_THREAD_KILL
        return IR::U32{ir.Select(ir.IsHelperInvocation(), ir.Imm32(-1), ir.Imm32(0))};
    case 29: // SR_INVOCATION_INFO
        return ir.InvocationInfo();
    case 30: // SR_WSCALEFACTOR_XY
        return ir.SR_WScaleFactorXY();
    case 31: // SR_WSCALEFACTOR_Z
        return ir.SR_WScaleFactorZ();
    case 32: { // SR_TID
        const IR::Value tid{ir.LocalInvocationId()};
        return ir.BitFieldInsert(
            ir.BitFieldInsert(IR::U32{ir.CompositeExtract(tid, 0)},
                              IR::U32{ir.CompositeExtract(tid, 1)}, ir.Imm32(16), ir.Imm32(8)),
            IR::U32{ir.CompositeExtract(tid, 2)}, ir.Imm32(26), ir.Imm32(6));
    }
    case 33: // SR_TID_X
        return ir.LocalInvocationIdX();
    case 34: // SR_TID_Y
        return ir.LocalInvocationIdY();
    case 35: // SR_TID_Z
        return ir.LocalInvocationIdZ();
    case 37: // SR_CTAID_X
        return ir.WorkgroupIdX();
    case 38: // SR_CTAID_Y
        return ir.WorkgroupIdY();
    case 39: // SR_CTAID_Z
        return ir.WorkgroupIdZ();
    case 56: // SR_EQMASK
        return ir.SubgroupEqMask();
    case 57: // SR_LTMASK
        return ir.SubgroupLtMask();
    case 58: // SR_LEMASK
        return ir.SubgroupLeMask();
    case 59: // SR_GTMASK
        return ir.SubgroupGtMask();
    case 60: // SR_GEMASK
        return ir.SubgroupGeMask();
    case 132: // SR_VARIABLE_RATE (1x1 shading rate)
        return ir.Imm32(0x00010100U);
    default:
        throw NotImplementedException("Ampere special register {}", special);
    }
}

} // Anonymous namespace

void TranslatorVisitor::MOV(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 lane_mask = insn.Bits(72, 4);
    if (opcode == 0x202 && std::popcount(lane_mask) == 1 && !insn.Bits(76, 16) &&
        !insn.Bits(40, 24)) {
        const u32 target_lane = static_cast<u32>(std::countr_zero(lane_mask));
        const IR::U1 match{
            ir.IEqual(ir.BitwiseAnd(ir.LaneId(), ir.Imm32(3U)), ir.Imm32(target_lane))};
        const IR::U32 src{R(insn.Bits(32, 8))};
        R(dest, IR::U32{ir.Select(match, src, R(dest))});
        return;
    }
    if (opcode == 0x802) {
        R(dest, ir.Imm32(insn.words[1]));
    } else if (opcode == 0xc02) {
        R(dest, UR(insn.words[1]));
    } else if (opcode == 0x202) {
        R(dest, R(insn.Bits(32, 8)));
    } else {
        R(dest, ReadCbuf32(insn));
    }
}

void TranslatorVisitor::UMOV(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    if (opcode == 0x882) {
        UR(dest, ir.Imm32(insn.words[1]));
    } else {
        UR(dest, UR(insn.words[1]));
    }
}

void TranslatorVisitor::LDC(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 width = insn.Bits(73, 3);
    const u32 mode = insn.Bits(78, 2);
    const IR::U32 imm_bank{ir.Imm32(CbufBank(insn))};
    const IR::U32 address{ir.IAdd(R(insn.Bits(24, 8)), ir.Imm32(insn.Bits(38, 16)))};

    IR::U32 index{imm_bank};
    IR::U32 offset{address};
    if (mode == 2U) { // Mode::IS segmented constant space
        index = ir.IAdd(imm_bank, ir.BitFieldExtract(address, ir.Imm32(16U), ir.Imm32(16U)));
        offset = ir.BitFieldExtract(address, ir.Imm32(0U), ir.Imm32(16U));
    }
    EmitConstantLoad(ir, index, offset, width, [&](u32 component, const IR::U32& value) {
        R(dest + component, value);
    });
}

void TranslatorVisitor::ULDC(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 width = insn.Bits(73, 3);
    const u32 raw_offset = insn.Bits(38, 16);
    const IR::U32 index{ir.Imm32(CbufBank(insn))};
    const IR::U32 offset{opcode == 0xabb ? IR::U32{ir.IAdd(UR(insn.Bits(24, 8)), ir.Imm32(raw_offset))}
                                         : ir.Imm32(raw_offset)};
    EmitConstantLoad(ir, index, offset, width, [&](u32 component, const IR::U32& value) {
        UR(dest + component, value);
    });
}

void TranslatorVisitor::S2R(const Instruction& insn) {
    R(insn.Bits(16, 8), ReadSpecialRegister(ir, insn.Bits(72, 8)));
}

void TranslatorVisitor::S2UR(const Instruction& insn) {
    UR(insn.Bits(16, 8), ReadSpecialRegister(ir, insn.Bits(72, 8)));
}

void TranslatorVisitor::CS2R(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::U32 zero{ir.Imm32(0U)};
    R(dest, zero);
    R(dest + 1U, zero);
}

void TranslatorVisitor::LEPC(const Instruction& insn, u32 index) {
    const u32 dest = insn.Bits(16, 8);
    if (dest == 255U) {
        return;
    }
    const u64 address = options.native_code_address + u64{index} * sizeof(Instruction);
    R(dest, ir.Imm32(static_cast<u32>(address)));
    R(dest + 1U, ir.Imm32(static_cast<u32>(address >> 32)));
}

} // namespace Shader::Ampere
