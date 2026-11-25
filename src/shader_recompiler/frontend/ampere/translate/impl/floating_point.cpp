// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cstdlib>

#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/program_header.h"

namespace Shader::Ampere {
namespace {

[[nodiscard]] bool StrictFpContraction() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_STRICT_FP_CONTRACT");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

[[nodiscard]] IR::FpControl MakeFpControl(u32 rnd_bits, bool ftz, bool allow_contract) {
    IR::FpRounding rounding = IR::FpRounding::RN;
    switch (rnd_bits & 3U) {
    case 0U:
        rounding = IR::FpRounding::RN;
        break;
    case 1U:
        rounding = IR::FpRounding::RM;
        break;
    case 2U:
        rounding = IR::FpRounding::RP;
        break;
    case 3U:
        rounding = IR::FpRounding::RZ;
        break;
    }
    const bool no_contract = !allow_contract || (rnd_bits & 3U) != 0U || StrictFpContraction();
    return IR::FpControl{
        .no_contraction = no_contract,
        .rounding = rounding,
        .fmz_mode = ftz ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };
}

} // Anonymous namespace

void TranslatorVisitor::FFMA(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);
    const IR::F32 c = AluSourceF32(insn, 2U);
    const IR::FpControl control = MakeFpControl(insn.Bits(78, 2), insn.Bits(80, 1) != 0U, true);

    IR::F32 result = ir.FPFma(a, b, c, control);
    if (insn.Bits(77, 1) != 0U) {
        result = ir.FPSaturate(result);
    }
    F(dest, result);
}

void TranslatorVisitor::FMUL(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 scale = insn.Bits(84, 3);
    IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);

    if (scale != 4U) {
        static constexpr std::array<f32, 8> ScaleFactors{
            1.0f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f,
        };
        a = ir.FPMul(a, ir.Imm32(ScaleFactors[scale]));
    }

    const IR::FpControl control = MakeFpControl(insn.Bits(78, 2), insn.Bits(80, 1) != 0U, true);
    IR::F32 result = ir.FPMul(a, b, control);
    if (insn.Bits(77, 1) != 0U) {
        result = ir.FPSaturate(result);
    }
    F(dest, result);
}

void TranslatorVisitor::FADD(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 form = insn.Bits(9, 3);
    const u32 b_which = (form == 2U || form == 3U || form == 7U) ? 2U : 1U;
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, b_which);
    const IR::FpControl control = MakeFpControl(insn.Bits(78, 2), insn.Bits(80, 1) != 0U, true);

    IR::F32 result = ir.FPAdd(a, b, control);
    if (insn.Bits(77, 1) != 0U) {
        result = ir.FPSaturate(result);
    }
    F(dest, result);
}

void TranslatorVisitor::FMNMX(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };

    const u32 pred_bits = insn.Bits(87, 4);
    IR::F32 result;
    if (pred_bits == 7U) {
        result = ir.FPMin(a, b, control);
    } else if (pred_bits == 15U) {
        result = ir.FPMax(a, b, control);
    } else {
        result = IR::F32{ir.Select(P(pred_bits), ir.FPMin(a, b, control), ir.FPMax(a, b, control))};
    }

    if (insn.Bits(81, 1) != 0U) {
        const IR::U1 unordered = ir.FPUnordered(a, b);
        R(dest, IR::U32{ir.Select(unordered, ir.Imm32(0x7fffffffU), ir.BitCast<IR::U32>(result))});
    } else {
        F(dest, result);
    }
}

void TranslatorVisitor::MUFU(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 op = insn.Bits(74, 6);
    const bool is_fp16 = insn.Bits(73, 1) != 0U;

    IR::F32 src = [&] {
        if (!is_fp16) {
            return AluSourceF32(insn, 1U);
        }
        const IR::Value unpacked = ir.UnpackFloat2x16(R(insn.Bits(32, 8)));
        const IR::F16 half{ir.CompositeExtract(unpacked, insn.Bits(60, 1))};
        const IR::F32 promoted{ir.FPConvert(32, half)};
        return IR::F32{ir.FPAbsNeg(promoted, insn.Bits(62, 1) != 0U, insn.Bits(63, 1) != 0U)};
    }();

    if (op == 0U || op == 1U) {
        // SM70+ MUFU.COS/SIN accept angles in turns (1/2pi * radians).
        src = ir.FPMul(src, ir.Imm32(std::bit_cast<f32>(0x40c90fdbU)));
    }

    IR::F32 value;
    switch (op) {
    case 0U:
        value = ir.FPCos(src);
        break;
    case 1U:
        value = ir.FPSin(src);
        break;
    case 2U:
        value = ir.FPExp2(src);
        break;
    case 3U:
        value = ir.FPLog2(src);
        break;
    case 4U:
        value = ir.FPRecip(src);
        break;
    case 5U:
        value = ir.FPRecipSqrt(src);
        break;
    case 8U:
        value = ir.FPSqrt(src);
        break;
    case 9U:
        value = ir.FPTanh(src);
        break;
    default:
        throw NotImplementedException("Invalid Ampere MUFU operation {}", op);
    }
    if (is_fp16) {
        const IR::F16 low{ir.FPConvert(16, value)};
        const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
        R(dest, ir.PackFloat2x16(ir.CompositeConstruct(low, zero)));
    } else {
        F(dest, value);
    }
}

void TranslatorVisitor::IPA(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 attr_index_raw = insn.Bits(64, 8);
    const auto attribute = static_cast<IR::Attribute>(attr_index_raw);
    const bool is_indexed = opcode == 0x526U && insn.Bits(24, 8) != 63U;

    IR::F32 value =
        is_indexed
            ? ir.GetAttributeIndexed(ir.IAdd(UR(insn.Bits(24, 8)), ir.Imm32(attr_index_raw * 4U)))
            : ir.GetAttribute(attribute);

    const bool is_legacy = IR::IsLegacyAttribute(attribute);
    if (IR::IsGeneric(attribute) || is_legacy) {
        bool is_perspective = is_legacy && insn.Bits(78, 1) == 0U;
        if (!is_legacy) {
            const ProgramHeader sph{env.SPH()};
            const u32 generic_idx = IR::GenericAttributeIndex(attribute);
            const std::array input_map{sph.ps.GenericInputMap(generic_idx)};
            PixelImap effective_imap{PixelImap::Unused};
            for (const PixelImap component : input_map) {
                if (component != PixelImap::Unused) {
                    effective_imap = component;
                    break;
                }
            }
            is_perspective =
                effective_imap == PixelImap::Perspective || effective_imap == PixelImap::Unused;
        }
        if (is_perspective) {
            value = ir.FPMul(value, ir.GetAttribute(IR::Attribute::PositionW));
        }
    }

    F(dest, value);
}

void TranslatorVisitor::FSEL(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);
    const IR::U1 cond = P(insn.Bits(87, 4));
    F(dest, IR::F32{ir.Select(cond, a, b)});
}

void TranslatorVisitor::FSET(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };

    const auto compare_op = static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4));
    const auto combine = static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2));
    const IR::U1 cmp_result = Maxwell::FloatingPointCompare(ir, a, b, compare_op, control);
    const IR::U1 combined = Maxwell::PredicateCombine(ir, cmp_result, P(insn.Bits(87, 4)), combine);

    R(dest, IR::U32{ir.Select(combined, ir.Imm32(0x3f800000U), ir.Imm32(0U))});
}

void TranslatorVisitor::FSETP(const Instruction& insn) {
    const IR::F32 a = AluSourceF32(insn, 0U);
    const IR::F32 b = AluSourceF32(insn, 1U);
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = insn.Bits(80, 1) != 0U ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };

    const auto compare_op = static_cast<Maxwell::FPCompareOp>(insn.Bits(76, 4));
    const auto combine = static_cast<Maxwell::BooleanOp>(insn.Bits(74, 2));
    const IR::U1 cmp_result = Maxwell::FloatingPointCompare(ir, a, b, compare_op, control);
    const IR::U1 accum = P(insn.Bits(87, 4));

    const IR::U1 res_a = Maxwell::PredicateCombine(ir, cmp_result, accum, combine);
    const IR::U1 res_b = Maxwell::PredicateCombine(ir, ir.LogicalNot(cmp_result), accum, combine);
    SetP(insn.Bits(81, 3), res_a);
    SetP(insn.Bits(84, 3), res_b);
}

} // namespace Shader::Ampere
