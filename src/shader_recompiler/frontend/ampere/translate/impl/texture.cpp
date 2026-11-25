// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>
#include <string_view>

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

namespace Shader::Ampere {
namespace {

[[nodiscard]] Shader::TextureType TextureDimensionToType(u32 dim) {
    switch (dim) {
    case 0U:
        return Shader::TextureType::Color1D;
    case 1U:
        return Shader::TextureType::Color2D;
    case 2U:
        return Shader::TextureType::Color3D;
    case 3U:
        return Shader::TextureType::ColorCube;
    case 4U:
        return Shader::TextureType::ColorArray1D;
    case 5U:
        return Shader::TextureType::ColorArray2D;
    case 7U:
        return Shader::TextureType::ColorArrayCube;
    default:
        throw NotImplementedException("Invalid Ampere texture dimension {}", dim);
    }
}

[[nodiscard]] IR::Value MakeSignedOffsetVector(IR::IREmitter& ir, const IR::U32& packed,
                                                 u32 dimension, u32 width_bits,
                                                 u32 component_stride_bits) {
    const IR::U32 width = ir.Imm32(width_bits);
    const auto extract = [&](u32 comp) {
        return ir.BitFieldExtract(packed, ir.Imm32(comp * component_stride_bits), width, true);
    };
    switch (dimension) {
    case 0U:
    case 4U:
        return extract(0U);
    case 1U:
    case 5U:
        return ir.CompositeConstruct(extract(0U), extract(1U));
    case 2U:
        return ir.CompositeConstruct(extract(0U), extract(1U), extract(2U));
    default:
        throw NotImplementedException("Illegal offset on texture dimension {}", dimension);
    }
}

template <typename RegAt>
[[nodiscard]] IR::Value BuildFloatTextureCoords(TranslatorVisitor& v, u32 dim,
                                                RegAt&& coord_reg_at) {
    const auto read_array = [&]() -> IR::F32 {
        return v.ir.ConvertUToF(32, 16, v.R(coord_reg_at(0U)));
    };
    const auto f = [&](u32 i) -> IR::F32 { return v.F(coord_reg_at(i)); };
    switch (dim) {
    case 0U:
        return f(0U);
    case 4U:
        return v.ir.CompositeConstruct(f(1U), read_array());
    case 1U:
        return v.ir.CompositeConstruct(f(0U), f(1U));
    case 5U:
        return v.ir.CompositeConstruct(f(1U), f(2U), read_array());
    case 2U:
        return v.ir.CompositeConstruct(f(0U), f(1U), f(2U));
    case 3U:
        return v.ir.CompositeConstruct(f(0U), f(1U), f(2U));
    case 7U:
        return v.ir.CompositeConstruct(f(1U), f(2U), f(3U), read_array());
    default:
        throw NotImplementedException("Invalid Ampere texture dimension {}", dim);
    }
}

void WriteTextureResultsF32(TranslatorVisitor& v, u32 dest, u32 second_dest, u32 mask,
                            const IR::Value& sample, bool is_depth) {
    const bool split = second_dest != 255U;
    u32 out_idx = 0U;
    for (u32 element = 0U; element < 4U; ++element) {
        if (((mask >> element) & 1U) == 0U) {
            continue;
        }
        IR::F32 val;
        if (is_depth) {
            val = element < 3U ? IR::F32{sample} : v.ir.Imm32(1.0f);
        } else {
            val = IR::F32{v.ir.CompositeExtract(sample, element)};
        }
        const u32 target = (!split || out_idx < 2U)
                               ? (dest == 255U ? 255U : dest + out_idx)
                               : (second_dest + (out_idx - 2U));
        if (target != 255U) {
            v.F(target, val);
        }
        ++out_idx;
    }
}

void WriteTextureResultsF16(TranslatorVisitor& v, u32 dest, u32 second_dest, u32 mask,
                            const IR::Value& sample, bool is_depth,
                            bool allow_consecutive_pair2) {
    std::array<IR::F32, 4> vals;
    u32 count = 0U;
    for (u32 element = 0U; element < 4U; ++element) {
        if (((mask >> element) & 1U) == 0U) {
            continue;
        }
        vals[count++] =
            is_depth ? (element < 3U ? IR::F32{sample} : v.ir.Imm32(1.0f))
                     : IR::F32{v.ir.CompositeExtract(sample, element)};
    }
    const IR::F32 zero = v.ir.Imm32(0.0f);
    const auto pack_pair = [&](u32 target, u32 i0, u32 i1) {
        if (target == 255U || i0 >= count) {
            return;
        }
        const IR::F32 hi = i1 < count ? vals[i1] : zero;
        v.R(target, v.ir.PackHalf2x16(v.ir.CompositeConstruct(vals[i0], hi)));
    };
    pack_pair(dest, 0U, 1U);
    if (count > 2U) {
        const u32 second_target =
            second_dest != 255U
                ? second_dest
                : (allow_consecutive_pair2 && dest != 255U ? dest + 1U : 255U);
        pack_pair(second_target, 2U, 3U);
    }
}

void WriteTextureResultsU32(TranslatorVisitor& v, u32 dest, u32 second_dest, u32 mask,
                            const IR::Value& result) {
    const bool split = second_dest != 255U;
    u32 out_idx = 0U;
    for (u32 element = 0U; element < 4U; ++element) {
        if (((mask >> element) & 1U) == 0U) {
            continue;
        }
        const u32 target = (!split || out_idx < 2U)
                               ? (dest == 255U ? 255U : dest + out_idx)
                               : (second_dest + (out_idx - 2U));
        if (target != 255U) {
            v.R(target, IR::U32{v.ir.CompositeExtract(result, element)});
        }
        ++out_idx;
    }
}

} // Anonymous namespace

void TranslatorVisitor::TEX(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 mask = insn.Bits(72, 4);
    const u32 dest = insn.Bits(16, 8);
    const u32 second_dest = insn.Bits(64, 8);
    const bool f16 = insn.Bits(79, 1) != 0U;
    const bool depth_compare = insn.Bits(78, 1) != 0U;
    const bool ndv = insn.Bits(77, 1) != 0U;
    const bool offsets = insn.Bits(76, 1) != 0U;
    const u32 dimension = insn.Bits(61, 3);
    const u32 lod = insn.Bits(87, 3);
    const u32 source_a = insn.Bits(24, 8);
    const u32 source_b = insn.Bits(32, 8);

    constexpr std::array<u32, 8> CoordinateCount{1, 2, 3, 3, 2, 3, 0, 4};
    const u32 coordinates = CoordinateCount[dimension];
    const u32 handle_slot = opcode == 0x361U ? 1U : 0U;
    const bool has_lod = lod == 2U || lod == 3U || lod == 6U || lod == 7U;
    const u32 count = coordinates + handle_slot + (has_lod ? 1U : 0U) + (offsets ? 1U : 0U) +
                      (depth_compare ? 1U : 0U);
    const u32 in_a = opcode == 0x361U && !insn.Bits(60, 1) ? coordinates
                     : count <= 4U                         ? (count + 1U) / 2U
                     : opcode == 0x361U                    ? coordinates
                                                           : 4U;
    const auto reg_at = [&](u32 i) -> u32 {
        if (i < in_a) {
            return source_a == 255U ? 255U : source_a + i;
        }
        return source_b == 255U ? 255U : source_b + (i - in_a);
    };

    const IR::Value coords = BuildFloatTextureCoords(*this, dimension, reg_at);
    const IR::Value handle = [&]() -> IR::Value {
        if (opcode == 0xb60U) {
            return TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));
        }
        if (opcode == 0xf60U) {
            return UR(insn.Bits(40, 6));
        }
        return R(reg_at(coordinates));
    }();

    u32 next = coordinates + handle_slot;
    const IR::F32 lod_arg = has_lod ? F(reg_at(next++)) : ir.Imm32(0.0f);
    IR::Value offset;
    if (offsets) {
        offset = MakeSignedOffsetVector(ir, R(reg_at(next++)), dimension, 4U, 4U);
    }
    IR::F32 dref;
    if (depth_compare) {
        dref = F(reg_at(next++));
    }

    IR::TextureInstInfo info{};
    info.type.Assign(TextureDimensionToType(dimension));
    info.is_depth.Assign(depth_compare ? 1U : 0U);
    info.has_bias.Assign((lod == 2U || lod == 6U) ? 1U : 0U);
    info.ndv_is_active.Assign(ndv ? 1U : 0U);
    if (f16) {
        info.relaxed_precision.Assign(1U);
    }

    const bool explicit_lod = lod == 1U || lod == 3U || lod == 7U;
    const IR::Value sample = [&]() -> IR::Value {
        if (!depth_compare) {
            if (explicit_lod) {
                return ir.ImageSampleExplicitLod(handle, coords, lod_arg, offset, info);
            }
            return ir.ImageSampleImplicitLod(handle, coords, lod_arg, offset, {}, info);
        }
        if (explicit_lod) {
            return ir.ImageSampleDrefExplicitLod(handle, coords, dref, lod_arg, offset, info);
        }
        return ir.ImageSampleDrefImplicitLod(handle, coords, dref, lod_arg, offset, {}, info);
    }();

    if (f16) {
        WriteTextureResultsF16(*this, dest, second_dest, mask, sample, depth_compare, true);
    } else {
        WriteTextureResultsF32(*this, dest, second_dest, mask, sample, depth_compare);
    }
}

void TranslatorVisitor::TLD4(const Instruction& insn, u32 index) {
    const u32 opcode = insn.Bits(0, 12);
    const bool bindless = opcode == 0xf63U;
    const bool array = insn.Bits(61, 3) == 5U;
    const bool f16 = !bindless && insn.Bits(79, 2) == 1U;
    const u32 mask = insn.Bits(72, 4);
    const u32 dest = insn.Bits(16, 8);
    const u32 second_dest = insn.Bits(64, 8);
    const u32 source_a = insn.Bits(24, 8);
    const u32 source_b = insn.Bits(32, 8);
    const u32 offset_mode = insn.Bits(76, 2);
    const bool depth = insn.Bits(78, 1) != 0U;

    const u32 coordinates = array ? 3U : 2U;
    const u32 meta = (offset_mode == 1U ? 1U : offset_mode == 2U ? 2U : 0U) + (depth ? 1U : 0U);
    const u32 in_a = meta != 0U ? coordinates : (coordinates + 1U) / 2U;
    const auto coord_reg_at = [&](u32 i) -> u32 {
        if (i < in_a) {
            return source_a == 255U ? 255U : source_a + i;
        }
        return source_b == 255U ? 255U : source_b + (i - in_a);
    };
    const IR::Value coords = BuildFloatTextureCoords(*this, array ? 5U : 1U, coord_reg_at);

    const IR::Value handle = bindless
                                 ? IR::Value{UR(insn.Bits(40, 6))}
                                 : TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    u32 meta_idx = 0U;
    const auto meta_reg = [&](u32 i) -> u32 { return source_b == 255U ? 255U : source_b + i; };
    IR::Value offset;
    IR::Value offset2;
    if (offset_mode == 1U) {
        offset = MakeSignedOffsetVector(ir, R(meta_reg(meta_idx++)), array ? 5U : 1U, 6U, 8U);
    } else if (offset_mode == 2U) {
        const auto ptp_vec = [&](u32 reg) {
            std::string_view ptp_error;
            const std::optional<u32> imm = FindGatherPtpOffsetImmediate(
                code, index, reg, insn.Bits(12, 4), branch_targets, ptp_error);
            if (!ptp_error.empty() &&
                (f16 || ptp_error != "native per-tap gather nonconstant offsets")) {
                throw NotImplementedException("Invalid TLD4 PTP offset: {}", ptp_error);
            }
            const IR::U32 value = imm ? ir.Imm32(*imm) : R(reg);
            const IR::U32 width = ir.Imm32(6U);
            return ir.CompositeConstruct(
                ir.BitFieldExtract(value, ir.Imm32(0U), width, true),
                ir.BitFieldExtract(value, ir.Imm32(8U), width, true),
                ir.BitFieldExtract(value, ir.Imm32(16U), width, true),
                ir.BitFieldExtract(value, ir.Imm32(24U), width, true));
        };
        offset = ptp_vec(meta_reg(meta_idx++));
        offset2 = ptp_vec(meta_reg(meta_idx++));
    }

    IR::F32 dref;
    if (depth) {
        dref = F(meta_reg(meta_idx++));
    }

    IR::TextureInstInfo info{};
    info.type.Assign(array ? Shader::TextureType::ColorArray2D : Shader::TextureType::Color2D);
    info.is_depth.Assign(depth ? 1U : 0U);
    info.gather_component.Assign(insn.Bits(87, 2));
    if (f16) {
        info.relaxed_precision.Assign(1U);
    }

    const IR::Value sample =
        depth ? ir.ImageGatherDref(handle, coords, offset, offset2, dref, info)
              : ir.ImageGather(handle, coords, offset, offset2, info);
    if (f16) {
        WriteTextureResultsF16(*this, dest, second_dest, mask, sample, false, false);
    } else {
        WriteTextureResultsF32(*this, dest, second_dest, mask, sample, false);
    }
}

void TranslatorVisitor::TLD(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 mask = insn.Bits(72, 4);
    const u32 dest = insn.Bits(16, 8);
    const u32 second_dest = insn.Bits(64, 8);
    const bool f16 = opcode == 0xb66U && insn.Bits(79, 2) == 1U;
    const u32 dimension = insn.Bits(61, 3);
    const bool array = dimension == 5U;
    const bool volume = dimension == 2U;
    const u32 source_a = insn.Bits(24, 8);
    const u32 source_b = insn.Bits(32, 8);

    const auto read_array = [&](u32 reg) {
        return ir.BitFieldExtract(R(reg), ir.Imm32(0U), ir.Imm32(16U));
    };

    IR::Value handle;
    IR::Value coords;
    IR::Value offset;
    IR::U32 lod_val = ir.Imm32(0U);
    IR::U32 ms;

    if (opcode == 0x367U) {
        constexpr std::array<u32, 8> CoordinateCount{1, 2, 3, 0, 0, 3, 0, 0};
        const u32 coordinates = CoordinateCount[dimension];
        const u32 count = coordinates + 1U;
        const u32 in_a = (count + 1U) / 2U;
        const auto reg_at = [&](u32 i) -> u32 {
            return i < in_a ? source_a + i : source_b + (i - in_a);
        };
        if (dimension == 0U) {
            coords = R(reg_at(0U));
        } else if (dimension == 1U) {
            coords = ir.CompositeConstruct(R(reg_at(0U)), R(reg_at(1U)));
        } else if (volume) {
            coords = ir.CompositeConstruct(R(reg_at(0U)), R(reg_at(1U)), R(reg_at(2U)));
        } else {
            coords = ir.CompositeConstruct(R(reg_at(1U)), R(reg_at(2U)), read_array(reg_at(0U)));
        }
        handle = R(reg_at(coordinates));
    } else if (opcode == 0xf66U) {
        handle = UR(insn.Bits(40, 6));
        if (dimension == 0U) {
            coords = R(source_a);
        } else if (dimension == 1U) {
            coords = ir.CompositeConstruct(R(source_a), R(source_b));
        } else if (volume) {
            coords = ir.CompositeConstruct(R(source_a), R(source_a + 1U), R(source_b));
        } else {
            coords = ir.CompositeConstruct(R(source_a + 1U), R(source_b), read_array(source_a));
        }
    } else {
        handle = TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));
        const bool level = insn.Bits(87, 3) == 3U;
        const bool offsets = insn.Bits(76, 1) != 0U;
        const bool multisample = insn.Bits(78, 1) != 0U;
        if (array && level) {
            coords = ir.CompositeConstruct(R(source_a + 1U), R(source_b), read_array(source_a));
            lod_val = R(source_b + 1U);
        } else if (array) {
            coords = ir.CompositeConstruct(R(source_a + 1U), R(source_b), read_array(source_a));
        } else if (volume) {
            coords = ir.CompositeConstruct(R(source_a), R(source_a + 1U), R(source_b));
        } else if (dimension == 0U) {
            coords = R(source_a);
            if (level) {
                lod_val = R(source_b);
            }
        } else if (offsets) {
            coords = ir.CompositeConstruct(R(source_a), R(source_a + 1U));
            if (level) {
                lod_val = R(source_b);
                offset = MakeSignedOffsetVector(ir, R(source_b + 1U), 1U, 4U, 4U);
            } else {
                offset = MakeSignedOffsetVector(ir, R(source_b), 1U, 4U, 4U);
            }
        } else if (level) {
            coords = ir.CompositeConstruct(R(source_a), R(source_a + 1U));
            lod_val = R(source_b);
        } else if (multisample) {
            coords = ir.CompositeConstruct(R(source_a), R(source_a + 1U));
            ms = R(source_b);
        } else {
            coords = ir.CompositeConstruct(R(source_a), R(source_b));
        }
    }

    IR::TextureInstInfo info{};
    info.type.Assign(TextureDimensionToType(dimension));
    if (f16) {
        info.relaxed_precision.Assign(1U);
    }
    const IR::Value sample = ir.ImageFetch(handle, coords, offset, lod_val, ms, info);
    if (f16) {
        WriteTextureResultsF16(*this, dest, second_dest, mask, sample, false, false);
    } else {
        WriteTextureResultsF32(*this, dest, second_dest, mask, sample, false);
    }
}

void TranslatorVisitor::TMML(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 mask = insn.Bits(72, 4);
    if ((mask & 0b1100U) != 0U) {
        throw NotImplementedException("TMML BA results are not implemented");
    }
    const u32 dest = insn.Bits(16, 8);
    const u32 second_dest = insn.Bits(64, 8);
    const u32 source_a = insn.Bits(24, 8);
    const u32 source_b = insn.Bits(32, 8);
    const u32 dimension = insn.Bits(61, 3);

    constexpr std::array<u32, 4> CoordinateCount{1, 2, 3, 3};
    const u32 coordinates = CoordinateCount[dimension];
    const u32 in_a =
        (source_b == 255U || opcode == 0x36aU) ? coordinates : (coordinates + 1U) / 2U;
    const auto reg_at = [&](u32 i) -> u32 {
        if (i < in_a) {
            return source_a == 255U ? 255U : source_a + i;
        }
        return source_b == 255U ? 255U : source_b + (i - in_a);
    };

    const IR::Value coords = BuildFloatTextureCoords(*this, dimension, reg_at);
    const IR::Value handle = [&]() -> IR::Value {
        if (opcode == 0xf69U) {
            return UR(insn.Bits(40, 6));
        }
        if (opcode == 0x36aU) {
            return R(source_b);
        }
        return TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));
    }();

    IR::TextureInstInfo info{};
    info.type.Assign(TextureDimensionToType(dimension));
    const IR::Value sample = ir.ImageQueryLod(handle, coords, info);

    const bool split = second_dest != 255U;
    u32 out_idx = 0U;
    for (u32 element = 0U; element < 4U; ++element) {
        if (((mask >> element) & 1U) == 0U) {
            continue;
        }
        const IR::F32 value{ir.CompositeExtract(sample, element)};
        const IR::U32 casted = ir.ConvertFToU(32, value);
        const IR::U32 fixed_8_8 = ir.ShiftLeftLogical(casted, ir.Imm32(8U));
        const u32 target = (!split || out_idx < 2U)
                               ? (dest == 255U ? 255U : dest + out_idx)
                               : (second_dest + (out_idx - 2U));
        if (target != 255U) {
            R(target, fixed_8_8);
        }
        ++out_idx;
    }
}

void TranslatorVisitor::TXD(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool bindless = opcode == 0xf6cU;
    const u32 dimension = insn.Bits(61, 3);
    const bool array = dimension == 5U;
    const u32 coords_base = insn.Bits(24, 8);
    const u32 deriv_base = insn.Bits(32, 8);
    const u32 dest = insn.Bits(16, 8);
    const u32 high = insn.Bits(64, 8);
    const u32 mask = insn.Bits(72, 4);

    const IR::Value handle = bindless
                                 ? IR::Value{UR(insn.Bits(40, 6))}
                                 : TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));

    const IR::Value coords = [&]() -> IR::Value {
        if (!array) {
            return ir.CompositeConstruct(F(coords_base), F(coords_base + 1U));
        }
        const IR::U32 layer_idx = ir.BitFieldExtract(R(coords_base + 2U), ir.Imm32(0U), ir.Imm32(16U));
        const IR::F32 layer_f = ir.ConvertUToF(32, 16, layer_idx);
        return ir.CompositeConstruct(F(coords_base), F(coords_base + 1U), layer_f);
    }();
    const IR::Value derivatives = ir.CompositeConstruct(
        F(deriv_base), F(deriv_base + 1U), F(deriv_base + 2U), F(deriv_base + 3U));

    IR::TextureInstInfo info{};
    info.type.Assign(TextureDimensionToType(dimension));
    info.num_derivatives.Assign(2U);
    const IR::Value sample = ir.ImageGradient(handle, coords, derivatives, {}, {}, info);

    WriteTextureResultsF32(*this, dest, high, mask, sample, false);
}

void TranslatorVisitor::TXQ(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool bindless = opcode == 0x370U;
    const u32 dest = insn.Bits(16, 8);
    const u32 second_dest = insn.Bits(64, 8);
    const u32 ra = insn.Bits(24, 8);
    const u32 mask = insn.Bits(72, 4);

    const IR::Value handle = bindless
                                 ? IR::Value{R(ra)}
                                 : TextureDescriptorHandle(insn.Bits(54, 5), insn.Bits(40, 14));
    const IR::U32 lod = R(bindless ? ra + 1U : ra);
    const IR::U1 skip_mips = ir.Imm1(((mask >> 3) & 1U) == 0U);
    const IR::Value query = ir.ImageQueryDimension(handle, lod, skip_mips);

    WriteTextureResultsU32(*this, dest, second_dest, mask, query);
}

} // namespace Shader::Ampere
