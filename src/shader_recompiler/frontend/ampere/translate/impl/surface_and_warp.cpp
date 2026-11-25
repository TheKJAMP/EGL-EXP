// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

namespace Shader::Ampere {
namespace {

enum class ShuffleMode : u32 {
    IDX = 0,
    UP = 1,
    DOWN = 2,
    BFLY = 3,
};

[[nodiscard]] Shader::TextureType SurfaceDimensionToType(u32 dim) {
    switch (dim) {
    case 0U:
        return Shader::TextureType::Color1D;
    case 1U:
        return Shader::TextureType::Buffer;
    case 2U:
        return Shader::TextureType::ColorArray1D;
    case 3U:
        return Shader::TextureType::Color2D;
    case 4U:
        return Shader::TextureType::ColorArray2D;
    case 5U:
        return Shader::TextureType::Color3D;
    default:
        throw NotImplementedException("Invalid Ampere surface dimension {}", dim);
    }
}

[[nodiscard]] IR::Value MakeSurfaceCoords(TranslatorVisitor& v, u32 reg, u32 dim) {
    const auto array = [&](u32 idx) -> IR::U32 {
        return v.ir.BitFieldExtract(v.R(reg == 255U ? 255U : reg + idx), v.ir.Imm32(0U),
                                    v.ir.Imm32(16U));
    };
    const auto r = [&](u32 idx) -> IR::U32 {
        return v.R(reg == 255U ? 255U : reg + idx);
    };
    switch (dim) {
    case 0U:
    case 1U:
        return r(0U);
    case 2U:
        return v.ir.CompositeConstruct(r(0U), array(1U));
    case 3U:
        return v.ir.CompositeConstruct(r(0U), r(1U));
    case 4U:
        return v.ir.CompositeConstruct(r(0U), r(1U), array(2U));
    case 5U:
        return v.ir.CompositeConstruct(r(0U), r(1U), r(2U));
    default:
        throw NotImplementedException("Invalid Ampere surface dimension {}", dim);
    }
}

[[nodiscard]] IR::U32 RunShuffleOp(IR::IREmitter& ir, const IR::U32& value, const IR::U32& index,
                                   const IR::U32& mask, ShuffleMode mode) {
    const IR::U32 clamp = ir.BitFieldExtract(mask, ir.Imm32(0U), ir.Imm32(5U));
    const IR::U32 seg_mask = ir.BitFieldExtract(mask, ir.Imm32(8U), ir.Imm32(5U));
    switch (mode) {
    case ShuffleMode::IDX:
        return ir.ShuffleIndex(value, index, clamp, seg_mask);
    case ShuffleMode::UP:
        return ir.ShuffleUp(value, index, clamp, seg_mask);
    case ShuffleMode::DOWN:
        return ir.ShuffleDown(value, index, clamp, seg_mask);
    case ShuffleMode::BFLY:
        return ir.ShuffleButterfly(value, index, clamp, seg_mask);
    default:
        throw NotImplementedException("Invalid Ampere SHFL mode {}", static_cast<u32>(mode));
    }
}

} // Anonymous namespace

void TranslatorVisitor::SULD(const Instruction& insn) {
    const u32 channels = insn.Bits(72, 4);
    const u32 dimension = insn.Bits(61, 3);
    const IR::Value coords = MakeSurfaceCoords(*this, insn.Bits(24, 8), dimension);
    const IR::Value handle = TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    IR::TextureInstInfo info{};
    info.type.Assign(SurfaceDimensionToType(dimension));
    info.image_format.Assign(ImageFormat::Typeless);
    const IR::Value result = ir.ImageRead(handle, coords, info);

    u32 dest = insn.Bits(16, 8);
    for (u32 component = 0U; component < 4U; ++component) {
        if (((channels >> component) & 1U) == 0U) {
            continue;
        }
        R(dest++, IR::U32{ir.CompositeExtract(result, component)});
    }
}

void TranslatorVisitor::SUST(const Instruction& insn) {
    const u32 channels = insn.Bits(72, 4);
    const u32 dimension = insn.Bits(61, 3);
    const u32 data = insn.Bits(32, 8);
    const bool strong = (insn.words[2] & ~0xf00U) == 0x0010a000U;

    const IR::Value coords = MakeSurfaceCoords(*this, insn.Bits(24, 8), dimension);
    const IR::Value handle = TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    IR::TextureInstInfo info{};
    info.type.Assign(SurfaceDimensionToType(dimension));
    info.image_format.Assign(ImageFormat::Typeless);

    if (strong) {
        ir.DeviceMemoryBarrier();
    }
    const auto data_at = [&](u32 idx) -> IR::U32 {
        return data == 255U ? ir.Imm32(0U) : R(data + idx);
    };
    IR::Value color;
    if (channels == 0xfU) {
        color = ir.CompositeConstruct(data_at(0U), data_at(1U), data_at(2U), data_at(3U));
    } else {
        const IR::Value texel = ir.ImageRead(handle, coords, info);
        std::array<IR::U32, 4> components;
        u32 src_idx = 0U;
        for (u32 component = 0U; component < 4U; ++component) {
            components[component] =
                ((channels >> component) & 1U) != 0U
                    ? data_at(src_idx++)
                    : IR::U32{ir.CompositeExtract(texel, component)};
        }
        color = ir.CompositeConstruct(components[0], components[1], components[2], components[3]);
    }
    ir.ImageWrite(handle, coords, color, info);
    if (strong) {
        ir.DeviceMemoryBarrier();
    }
}

void TranslatorVisitor::SUATOM(const Instruction& insn) {
    const u32 dimension = insn.Bits(61, 3);
    const u32 operation = insn.Bits(87, 3);
    const IR::Value coords = MakeSurfaceCoords(*this, insn.Bits(24, 8), dimension);
    const IR::Value handle = TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    IR::TextureInstInfo info{};
    info.type.Assign(SurfaceDimensionToType(dimension));
    info.image_format.Assign(ImageFormat::R32_UINT);

    const IR::U32 operand = R(insn.Bits(32, 8));
    IR::Value result;
    if (operation == 0U) {
        result = ir.ImageAtomicIAdd(handle, coords, operand, info);
    } else if (operation == 6U) {
        result = ir.ImageAtomicOr(handle, coords, operand, info);
    } else {
        throw NotImplementedException("Invalid Ampere SUATOM operation {}", operation);
    }
    R(insn.Bits(16, 8), IR::U32{result});
}

void TranslatorVisitor::SURED(const Instruction& insn) {
    const u32 dimension = insn.Bits(61, 3);
    const bool is_signed = insn.Bits(73, 3) == 1U;
    const u32 operation = insn.Bits(87, 4);
    const IR::Value coords = MakeSurfaceCoords(*this, insn.Bits(24, 8), dimension);
    const IR::Value handle = TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    IR::TextureInstInfo info{};
    info.type.Assign(SurfaceDimensionToType(dimension));
    info.image_format.Assign(ImageFormat::R32_UINT);

    const IR::U32 operand = R(insn.Bits(32, 8));
    switch (operation) {
    case 0U:
        (void)ir.ImageAtomicIAdd(handle, coords, operand, info);
        break;
    case 1U:
        (void)ir.ImageAtomicIMin(handle, coords, operand, is_signed, info);
        break;
    case 2U:
        (void)ir.ImageAtomicIMax(handle, coords, operand, is_signed, info);
        break;
    case 5U:
        (void)ir.ImageAtomicAnd(handle, coords, operand, info);
        break;
    case 6U:
        (void)ir.ImageAtomicOr(handle, coords, operand, info);
        break;
    case 7U:
        (void)ir.ImageAtomicXor(handle, coords, operand, info);
        break;
    default:
        throw NotImplementedException("Invalid Ampere SURED operation {}", operation);
    }
}

void TranslatorVisitor::SHFL(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const IR::U32 src = R(insn.Bits(24, 8));
    const ShuffleMode mode = static_cast<ShuffleMode>(insn.Bits(58, 2));
    const u32 pred = opcode == 0xf89U ? 7U : insn.Bits(81, 3);

    const bool index_is_imm = opcode != 0x589U;
    const bool mask_is_imm = opcode != 0x989U;
    const u32 index_imm = insn.Bits(53, 5);
    const u32 mask_imm = insn.Bits(40, 13);

    const IR::U32 index = index_is_imm ? ir.Imm32(index_imm) : R(insn.Bits(32, 8));
    const IR::U32 mask = mask_is_imm ? ir.Imm32(mask_imm) : R(insn.Bits(64, 8));

    static constexpr u32 QuadMask = (28U << 8) | 3U;
    if (mask_is_imm && mask_imm == QuadMask && index_is_imm &&
        env.ShaderStage() == Stage::Fragment) {
        if (mode == ShuffleMode::IDX && index_imm <= 3U) {
            R(dest, ir.QuadBroadcast(src, ir.Imm32(index_imm)));
            SetP(pred, ir.Imm1(true));
            return;
        }
        if (mode == ShuffleMode::BFLY && index_imm >= 1U && index_imm <= 3U) {
            R(dest, ir.QuadSwap(src, ir.Imm32(index_imm - 1U)));
            SetP(pred, ir.Imm1(true));
            return;
        }
    }

    const IR::U32 result = RunShuffleOp(ir, src, index, mask, mode);
    SetP(pred, ir.GetInBoundsFromOp(result));
    R(dest, result);
}

void TranslatorVisitor::MATCH_ALL(const Instruction& insn) {
    const u32 source = insn.Bits(24, 8);
    const u32 target = insn.Bits(81, 3);
    const bool is_u64 = insn.Bits(73, 1) != 0U;

    const IR::U32 active = ir.SubgroupBallot(ir.Imm1(true));
    const IR::U32 lowest_bit = ir.BitwiseAnd(active, ir.INeg(active));
    const IR::U32 lowest_lane = ir.FindUMsb(lowest_bit);

    const IR::U32 w0 = R(source);
    const IR::U32 ref0 = ir.ShuffleIndex(w0, lowest_lane, ir.Imm32(31U), ir.Imm32(0U));
    IR::U1 eq = ir.IEqual(w0, ref0);
    if (is_u64) {
        const IR::U32 w1 = R(source + 1U);
        const IR::U32 ref1 = ir.ShuffleIndex(w1, lowest_lane, ir.Imm32(31U), ir.Imm32(0U));
        eq = ir.LogicalAnd(eq, ir.IEqual(w1, ref1));
    }
    SetP(target, ir.VoteAll(eq));
}

} // namespace Shader::Ampere
