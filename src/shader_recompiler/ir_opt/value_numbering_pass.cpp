// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {

struct KnownBits {
    u32 zeros{};
    u32 ones{};
};

struct ValueKey {
    bool is_inst{};
    IR::Type type{IR::Type::Void};
    u64 payload{};

    auto operator<=>(const ValueKey&) const noexcept = default;
};

[[nodiscard]] bool IsKeyableValue(const IR::Value& value) noexcept {
    if (value.IsEmpty()) {
        return true;
    }
    if (!value.IsImmediate()) {
        return true;
    }
    return value.Type() != IR::Type::F16;
}

[[nodiscard]] ValueKey MakeValueKey(const IR::Value& value) {
    if (value.IsEmpty()) {
        return {.is_inst = false, .type = IR::Type::Void, .payload = 0ULL};
    }
    if (!value.IsImmediate()) {
        return {
            .is_inst = true,
            .type = IR::Type::Opaque,
            .payload = static_cast<u64>(reinterpret_cast<std::uintptr_t>(value.Inst())),
        };
    }
    const IR::Type t = value.Type();
    u64 payload = 0ULL;
    switch (t) {
    case IR::Type::U1:
        payload = value.U1() ? 1ULL : 0ULL;
        break;
    case IR::Type::U8:
        payload = value.U8();
        break;
    case IR::Type::U16:
        payload = value.U16();
        break;
    case IR::Type::U32:
        payload = value.U32();
        break;
    case IR::Type::F32:
        payload = std::bit_cast<u32>(value.F32());
        break;
    case IR::Type::U64:
        payload = value.U64();
        break;
    case IR::Type::F64:
        payload = std::bit_cast<u64>(value.F64());
        break;
    case IR::Type::Reg:
        payload = static_cast<u64>(value.Reg());
        break;
    case IR::Type::Pred:
        payload = static_cast<u64>(value.Pred());
        break;
    case IR::Type::Attribute:
        payload = static_cast<u64>(value.Attribute());
        break;
    case IR::Type::Patch:
        payload = static_cast<u64>(value.Patch());
        break;
    default:
        break;
    }
    return {.is_inst = false, .type = t, .payload = payload};
}

struct ExprKey {
    IR::Opcode op{};
    u32 flags{};
    u8 num_args{};
    std::array<ValueKey, 5> args{};

    bool operator==(const ExprKey&) const noexcept = default;
};

struct ExprKeyHash {
    [[nodiscard]] size_t operator()(const ExprKey& key) const noexcept {
        u64 h = (static_cast<u64>(key.op) * 0x9e3779b97f4a7c15ULL) ^
                (static_cast<u64>(key.flags) << 32) ^ key.num_args;
        for (u8 i = 0; i < key.num_args; ++i) {
            const auto& a = key.args[i];
            const u64 v = (a.payload ^ (static_cast<u64>(a.type) << 1) ^ (a.is_inst ? 1ULL : 0ULL));
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<size_t>(h);
    }
};

[[nodiscard]] bool HasLivePseudoOp(IR::Inst& inst) {
    return inst.HasAssociatedPseudoOperation();
}

[[nodiscard]] bool IsPureValueOpcode(IR::Opcode op) noexcept {
    switch (op) {
    case IR::Opcode::GetCbufU8:
    case IR::Opcode::GetCbufS8:
    case IR::Opcode::GetCbufU16:
    case IR::Opcode::GetCbufS16:
    case IR::Opcode::GetCbufU32:
    case IR::Opcode::GetCbufF32:
    case IR::Opcode::GetCbufU32x2:
    case IR::Opcode::WorkgroupId:
    case IR::Opcode::LocalInvocationId:
    case IR::Opcode::InvocationId:
    case IR::Opcode::InvocationInfo:
    case IR::Opcode::SampleId:
    case IR::Opcode::SR_WScaleFactorXY:
    case IR::Opcode::SR_WScaleFactorZ:
    case IR::Opcode::YDirection:
    case IR::Opcode::ResolutionDownFactor:
    case IR::Opcode::RenderArea:
    case IR::Opcode::LaneId:
    case IR::Opcode::SubgroupEqMask:
    case IR::Opcode::SubgroupLtMask:
    case IR::Opcode::SubgroupLeMask:
    case IR::Opcode::SubgroupGtMask:
    case IR::Opcode::SubgroupGeMask:
    case IR::Opcode::CompositeConstructU32x2:
    case IR::Opcode::CompositeConstructU32x3:
    case IR::Opcode::CompositeConstructU32x4:
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x3:
    case IR::Opcode::CompositeExtractU32x4:
    case IR::Opcode::CompositeInsertU32x2:
    case IR::Opcode::CompositeInsertU32x3:
    case IR::Opcode::CompositeInsertU32x4:
    case IR::Opcode::CompositeConstructF16x2:
    case IR::Opcode::CompositeConstructF16x3:
    case IR::Opcode::CompositeConstructF16x4:
    case IR::Opcode::CompositeExtractF16x2:
    case IR::Opcode::CompositeExtractF16x3:
    case IR::Opcode::CompositeExtractF16x4:
    case IR::Opcode::CompositeInsertF16x2:
    case IR::Opcode::CompositeInsertF16x3:
    case IR::Opcode::CompositeInsertF16x4:
    case IR::Opcode::CompositeConstructF32x2:
    case IR::Opcode::CompositeConstructF32x3:
    case IR::Opcode::CompositeConstructF32x4:
    case IR::Opcode::CompositeExtractF32x2:
    case IR::Opcode::CompositeExtractF32x3:
    case IR::Opcode::CompositeExtractF32x4:
    case IR::Opcode::CompositeInsertF32x2:
    case IR::Opcode::CompositeInsertF32x3:
    case IR::Opcode::CompositeInsertF32x4:
    case IR::Opcode::CompositeConstructF64x2:
    case IR::Opcode::CompositeConstructF64x3:
    case IR::Opcode::CompositeConstructF64x4:
    case IR::Opcode::CompositeExtractF64x2:
    case IR::Opcode::CompositeExtractF64x3:
    case IR::Opcode::CompositeExtractF64x4:
    case IR::Opcode::CompositeInsertF64x2:
    case IR::Opcode::CompositeInsertF64x3:
    case IR::Opcode::CompositeInsertF64x4:
    case IR::Opcode::SelectU1:
    case IR::Opcode::SelectU8:
    case IR::Opcode::SelectU16:
    case IR::Opcode::SelectU32:
    case IR::Opcode::SelectU64:
    case IR::Opcode::SelectF16:
    case IR::Opcode::SelectF32:
    case IR::Opcode::SelectF64:
    case IR::Opcode::BitCastU16F16:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastU64F64:
    case IR::Opcode::BitCastF16U16:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::BitCastF64U64:
    case IR::Opcode::PackUint2x32:
    case IR::Opcode::UnpackUint2x32:
    case IR::Opcode::PackFloat2x16:
    case IR::Opcode::UnpackFloat2x16:
    case IR::Opcode::PackHalf2x16:
    case IR::Opcode::UnpackHalf2x16:
    case IR::Opcode::PackDouble2x32:
    case IR::Opcode::UnpackDouble2x32:
    case IR::Opcode::FPAbs16:
    case IR::Opcode::FPAbs32:
    case IR::Opcode::FPAbs64:
    case IR::Opcode::FPAdd16:
    case IR::Opcode::FPAdd32:
    case IR::Opcode::FPAdd64:
    case IR::Opcode::FPFma16:
    case IR::Opcode::FPFma32:
    case IR::Opcode::FPFma64:
    case IR::Opcode::FPMax32:
    case IR::Opcode::FPMax64:
    case IR::Opcode::FPMin32:
    case IR::Opcode::FPMin64:
    case IR::Opcode::FPMul16:
    case IR::Opcode::FPMul32:
    case IR::Opcode::FPMul64:
    case IR::Opcode::FPNeg16:
    case IR::Opcode::FPNeg32:
    case IR::Opcode::FPNeg64:
    case IR::Opcode::FPRecip32:
    case IR::Opcode::FPRecip64:
    case IR::Opcode::FPRecipSqrt32:
    case IR::Opcode::FPRecipSqrt64:
    case IR::Opcode::FPSqrt:
    case IR::Opcode::FPSin:
    case IR::Opcode::FPExp2:
    case IR::Opcode::FPTanh:
    case IR::Opcode::FPCos:
    case IR::Opcode::FPLog2:
    case IR::Opcode::FPSaturate16:
    case IR::Opcode::FPSaturate32:
    case IR::Opcode::FPSaturate64:
    case IR::Opcode::FPClamp16:
    case IR::Opcode::FPClamp32:
    case IR::Opcode::FPClamp64:
    case IR::Opcode::FPRoundEven16:
    case IR::Opcode::FPRoundEven32:
    case IR::Opcode::FPRoundEven64:
    case IR::Opcode::FPFloor16:
    case IR::Opcode::FPFloor32:
    case IR::Opcode::FPFloor64:
    case IR::Opcode::FPCeil16:
    case IR::Opcode::FPCeil32:
    case IR::Opcode::FPCeil64:
    case IR::Opcode::FPTrunc16:
    case IR::Opcode::FPTrunc32:
    case IR::Opcode::FPTrunc64:
    case IR::Opcode::FPOrdEqual16:
    case IR::Opcode::FPOrdEqual32:
    case IR::Opcode::FPOrdEqual64:
    case IR::Opcode::FPUnordEqual16:
    case IR::Opcode::FPUnordEqual32:
    case IR::Opcode::FPUnordEqual64:
    case IR::Opcode::FPOrdNotEqual16:
    case IR::Opcode::FPOrdNotEqual32:
    case IR::Opcode::FPOrdNotEqual64:
    case IR::Opcode::FPUnordNotEqual16:
    case IR::Opcode::FPUnordNotEqual32:
    case IR::Opcode::FPUnordNotEqual64:
    case IR::Opcode::FPOrdLessThan16:
    case IR::Opcode::FPOrdLessThan32:
    case IR::Opcode::FPOrdLessThan64:
    case IR::Opcode::FPUnordLessThan16:
    case IR::Opcode::FPUnordLessThan32:
    case IR::Opcode::FPUnordLessThan64:
    case IR::Opcode::FPOrdGreaterThan16:
    case IR::Opcode::FPOrdGreaterThan32:
    case IR::Opcode::FPOrdGreaterThan64:
    case IR::Opcode::FPUnordGreaterThan16:
    case IR::Opcode::FPUnordGreaterThan32:
    case IR::Opcode::FPUnordGreaterThan64:
    case IR::Opcode::FPOrdLessThanEqual16:
    case IR::Opcode::FPOrdLessThanEqual32:
    case IR::Opcode::FPOrdLessThanEqual64:
    case IR::Opcode::FPUnordLessThanEqual16:
    case IR::Opcode::FPUnordLessThanEqual32:
    case IR::Opcode::FPUnordLessThanEqual64:
    case IR::Opcode::FPOrdGreaterThanEqual16:
    case IR::Opcode::FPOrdGreaterThanEqual32:
    case IR::Opcode::FPOrdGreaterThanEqual64:
    case IR::Opcode::FPUnordGreaterThanEqual16:
    case IR::Opcode::FPUnordGreaterThanEqual32:
    case IR::Opcode::FPUnordGreaterThanEqual64:
    case IR::Opcode::FPIsNan16:
    case IR::Opcode::FPIsNan32:
    case IR::Opcode::FPIsNan64:
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
    case IR::Opcode::ISub32:
    case IR::Opcode::ISub64:
    case IR::Opcode::IMul32:
    case IR::Opcode::SDiv32:
    case IR::Opcode::UDiv32:
    case IR::Opcode::INeg32:
    case IR::Opcode::INeg64:
    case IR::Opcode::IAbs32:
    case IR::Opcode::IAbs64:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftLeftLogical64:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::ShiftRightLogical64:
    case IR::Opcode::ShiftRightArithmetic32:
    case IR::Opcode::ShiftRightArithmetic64:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitFieldInsert:
    case IR::Opcode::BitFieldSExtract:
    case IR::Opcode::BitFieldUExtract:
    case IR::Opcode::BitReverse32:
    case IR::Opcode::BitCount32:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::FindSMsb32:
    case IR::Opcode::FindUMsb32:
    case IR::Opcode::SMin32:
    case IR::Opcode::UMin32:
    case IR::Opcode::SMax32:
    case IR::Opcode::UMax32:
    case IR::Opcode::SClamp32:
    case IR::Opcode::UClamp32:
    case IR::Opcode::SLessThan:
    case IR::Opcode::ULessThan:
    case IR::Opcode::IEqual:
    case IR::Opcode::SLessThanEqual:
    case IR::Opcode::ULessThanEqual:
    case IR::Opcode::SGreaterThan:
    case IR::Opcode::UGreaterThan:
    case IR::Opcode::INotEqual:
    case IR::Opcode::SGreaterThanEqual:
    case IR::Opcode::UGreaterThanEqual:
    case IR::Opcode::LogicalOr:
    case IR::Opcode::LogicalAnd:
    case IR::Opcode::LogicalXor:
    case IR::Opcode::LogicalNot:
    case IR::Opcode::ConvertS16F16:
    case IR::Opcode::ConvertS16F32:
    case IR::Opcode::ConvertS16F64:
    case IR::Opcode::ConvertS32F16:
    case IR::Opcode::ConvertS32F32:
    case IR::Opcode::ConvertS32F64:
    case IR::Opcode::ConvertS64F16:
    case IR::Opcode::ConvertS64F32:
    case IR::Opcode::ConvertS64F64:
    case IR::Opcode::ConvertU16F16:
    case IR::Opcode::ConvertU16F32:
    case IR::Opcode::ConvertU16F64:
    case IR::Opcode::ConvertU32F16:
    case IR::Opcode::ConvertU32F32:
    case IR::Opcode::ConvertU32F64:
    case IR::Opcode::ConvertU64F16:
    case IR::Opcode::ConvertU64F32:
    case IR::Opcode::ConvertU64F64:
    case IR::Opcode::ConvertU64U32:
    case IR::Opcode::ConvertU32U64:
    case IR::Opcode::ConvertF16F32:
    case IR::Opcode::ConvertF32F16:
    case IR::Opcode::ConvertF32F64:
    case IR::Opcode::ConvertF64F32:
    case IR::Opcode::ConvertF16S8:
    case IR::Opcode::ConvertF16S16:
    case IR::Opcode::ConvertF16S32:
    case IR::Opcode::ConvertF16S64:
    case IR::Opcode::ConvertF16U8:
    case IR::Opcode::ConvertF16U16:
    case IR::Opcode::ConvertF16U32:
    case IR::Opcode::ConvertF16U64:
    case IR::Opcode::ConvertF32S8:
    case IR::Opcode::ConvertF32S16:
    case IR::Opcode::ConvertF32S32:
    case IR::Opcode::ConvertF32S64:
    case IR::Opcode::ConvertF32U8:
    case IR::Opcode::ConvertF32U16:
    case IR::Opcode::ConvertF32U32:
    case IR::Opcode::ConvertF32U64:
    case IR::Opcode::ConvertF64S8:
    case IR::Opcode::ConvertF64S16:
    case IR::Opcode::ConvertF64S32:
    case IR::Opcode::ConvertF64S64:
    case IR::Opcode::ConvertF64U8:
    case IR::Opcode::ConvertF64U16:
    case IR::Opcode::ConvertF64U32:
    case IR::Opcode::ConvertF64U64:
    case IR::Opcode::ConvertU16U32:
    case IR::Opcode::ConvertU32U16:
    case IR::Opcode::ConvertU8U32:
    case IR::Opcode::ConvertU32U8:
    case IR::Opcode::ConvertS32S8:
    case IR::Opcode::ConvertS32S16:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsCommutativeBinary(IR::Opcode op) noexcept {
    switch (op) {
    case IR::Opcode::IAdd32:
    case IR::Opcode::IAdd64:
    case IR::Opcode::IMul32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::SMin32:
    case IR::Opcode::UMin32:
    case IR::Opcode::SMax32:
    case IR::Opcode::UMax32:
    case IR::Opcode::IEqual:
    case IR::Opcode::INotEqual:
    case IR::Opcode::LogicalOr:
    case IR::Opcode::LogicalAnd:
    case IR::Opcode::LogicalXor:
    case IR::Opcode::FPAdd16:
    case IR::Opcode::FPAdd32:
    case IR::Opcode::FPAdd64:
    case IR::Opcode::FPMul16:
    case IR::Opcode::FPMul32:
    case IR::Opcode::FPMul64:
    case IR::Opcode::FPMin32:
    case IR::Opcode::FPMin64:
    case IR::Opcode::FPMax32:
    case IR::Opcode::FPMax64:
    case IR::Opcode::FPOrdEqual16:
    case IR::Opcode::FPOrdEqual32:
    case IR::Opcode::FPOrdEqual64:
    case IR::Opcode::FPUnordEqual16:
    case IR::Opcode::FPUnordEqual32:
    case IR::Opcode::FPUnordEqual64:
    case IR::Opcode::FPOrdNotEqual16:
    case IR::Opcode::FPOrdNotEqual32:
    case IR::Opcode::FPOrdNotEqual64:
    case IR::Opcode::FPUnordNotEqual16:
    case IR::Opcode::FPUnordNotEqual32:
    case IR::Opcode::FPUnordNotEqual64:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool IsCommutativeFma(IR::Opcode op) noexcept {
    return op == IR::Opcode::FPFma16 || op == IR::Opcode::FPFma32 || op == IR::Opcode::FPFma64;
}

class ValueNumbering {
public:
    explicit ValueNumbering(IR::Program& program_) : program{program_} {}

    void Run() {
        if (program.post_order_blocks.empty()) {
            return;
        }
        BuildDominatorTree();
        VisitDominatorSubtree(0U);
    }

private:
    void BuildDominatorTree() {
        const size_t n = program.post_order_blocks.size();
        rpo.resize(n);
        rpo_index.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            IR::Block* const blk = program.post_order_blocks[n - 1U - i];
            rpo[i] = blk;
            rpo_index.emplace(blk, i);
        }

        constexpr size_t Unset = ~size_t{0};
        std::vector<size_t> idom(n, Unset);
        idom[0] = 0U;

        const auto intersect = [&](size_t b1, size_t b2) {
            size_t f1 = b1;
            size_t f2 = b2;
            while (f1 != f2) {
                while (f1 > f2) {
                    f1 = idom[f1];
                }
                while (f2 > f1) {
                    f2 = idom[f2];
                }
            }
            return f1;
        };

        std::vector<std::vector<size_t>> preds(n);
        for (size_t i = 1; i < n; ++i) {
            for (IR::Block* const p : rpo[i]->ImmPredecessors()) {
                if (const auto it = rpo_index.find(p); it != rpo_index.end()) {
                    preds[i].push_back(it->second);
                }
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t b = 1; b < n; ++b) {
                size_t new_idom = Unset;
                for (const size_t p : preds[b]) {
                    if (idom[p] == Unset) {
                        continue;
                    }
                    if (new_idom == Unset) {
                        new_idom = p;
                    } else {
                        new_idom = intersect(p, new_idom);
                    }
                }
                if (new_idom != Unset && idom[b] != new_idom) {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        dom_children.assign(n, {});
        for (size_t b = 1; b < n; ++b) {
            if (idom[b] != Unset) {
                dom_children[idom[b]].push_back(b);
            }
        }
    }

    [[nodiscard]] KnownBits GetKnownBits(const IR::Value& value) const {
        const IR::Value resolved = value.Resolve();
        if (resolved.IsImmediate()) {
            if (resolved.Type() == IR::Type::U32) {
                const u32 imm = resolved.U32();
                return {.zeros = ~imm, .ones = imm};
            }
            return {};
        }
        if (resolved.IsEmpty()) {
            return {};
        }
        const auto it = known_bits.find(resolved.Inst());
        return it != known_bits.end() ? it->second : KnownBits{};
    }

    [[nodiscard]] KnownBits ComputeKnownBits(const IR::Inst& inst) const {
        switch (inst.GetOpcode()) {
        case IR::Opcode::GetCbufU8:
            return {.zeros = 0xffffff00U, .ones = 0U};
        case IR::Opcode::GetCbufU16:
            return {.zeros = 0xffff0000U, .ones = 0U};
        case IR::Opcode::BitwiseAnd32: {
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            const KnownBits kb = GetKnownBits(inst.Arg(1));
            return {.zeros = ka.zeros | kb.zeros, .ones = ka.ones & kb.ones};
        }
        case IR::Opcode::BitwiseOr32: {
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            const KnownBits kb = GetKnownBits(inst.Arg(1));
            return {.zeros = ka.zeros & kb.zeros, .ones = ka.ones | kb.ones};
        }
        case IR::Opcode::BitwiseXor32: {
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            const KnownBits kb = GetKnownBits(inst.Arg(1));
            return {
                .zeros = (ka.zeros & kb.zeros) | (ka.ones & kb.ones),
                .ones = (ka.zeros & kb.ones) | (ka.ones & kb.zeros),
            };
        }
        case IR::Opcode::BitwiseNot32: {
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            return {.zeros = ka.ones, .ones = ka.zeros};
        }
        case IR::Opcode::ShiftLeftLogical32: {
            const IR::Value shift = inst.Arg(1).Resolve();
            if (!shift.IsImmediate() || shift.U32() >= 32U) {
                return {};
            }
            const u32 s = shift.U32();
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            const u32 low_zeros = s == 0U ? 0U : ((1U << s) - 1U);
            return {.zeros = (ka.zeros << s) | low_zeros, .ones = ka.ones << s};
        }
        case IR::Opcode::ShiftRightLogical32: {
            const IR::Value shift = inst.Arg(1).Resolve();
            if (!shift.IsImmediate() || shift.U32() >= 32U) {
                return {};
            }
            const u32 s = shift.U32();
            const KnownBits ka = GetKnownBits(inst.Arg(0));
            const u32 top_zeros = s == 0U ? 0U : (~0U << (32U - s));
            return {.zeros = (ka.zeros >> s) | top_zeros, .ones = ka.ones >> s};
        }
        case IR::Opcode::BitFieldUExtract: {
            const IR::Value off_v = inst.Arg(1).Resolve();
            const IR::Value cnt_v = inst.Arg(2).Resolve();
            if (!off_v.IsImmediate() || !cnt_v.IsImmediate()) {
                return {};
            }
            const u32 off = off_v.U32();
            const u32 cnt = cnt_v.U32();
            if (off > 32U || cnt > 32U - off) {
                return {};
            }
            if (cnt == 0U) {
                return {.zeros = ~0U, .ones = 0U};
            }
            const KnownBits kbase = GetKnownBits(inst.Arg(0));
            if (cnt == 32U) {
                return kbase;
            }
            const u32 keep = (1U << cnt) - 1U;
            return {
                .zeros = ((kbase.zeros >> off) & keep) | ~keep,
                .ones = (kbase.ones >> off) & keep,
            };
        }
        case IR::Opcode::BitFieldInsert: {
            const IR::Value off_v = inst.Arg(2).Resolve();
            const IR::Value cnt_v = inst.Arg(3).Resolve();
            if (!off_v.IsImmediate() || !cnt_v.IsImmediate()) {
                return {};
            }
            const u32 off = off_v.U32();
            const u32 cnt = cnt_v.U32();
            if (off > 32U || cnt > 32U - off) {
                return {};
            }
            const KnownBits kbase = GetKnownBits(inst.Arg(0));
            if (cnt == 0U) {
                return kbase;
            }
            const KnownBits kins = GetKnownBits(inst.Arg(1));
            const u32 mask = cnt == 32U ? ~0U : (((1U << cnt) - 1U) << off);
            return {
                .zeros = (kbase.zeros & ~mask) | ((kins.zeros << off) & mask),
                .ones = (kbase.ones & ~mask) | ((kins.ones << off) & mask),
            };
        }
        case IR::Opcode::SelectU32: {
            const KnownBits kt = GetKnownBits(inst.Arg(1));
            const KnownBits kf = GetKnownBits(inst.Arg(2));
            return {.zeros = kt.zeros & kf.zeros, .ones = kt.ones & kf.ones};
        }
        case IR::Opcode::Phi: {
            if (inst.Type() != IR::Type::U32 || inst.NumArgs() == 0U) {
                return {};
            }
            KnownBits acc{.zeros = ~0U, .ones = ~0U};
            for (size_t i = 0; i < inst.NumArgs(); ++i) {
                const IR::Value op = inst.Arg(i).Resolve();
                if (op.IsImmediate()) {
                    if (op.Type() != IR::Type::U32) {
                        return {};
                    }
                    const u32 imm = op.U32();
                    acc.zeros &= ~imm;
                    acc.ones &= imm;
                } else if (!op.IsEmpty()) {
                    const auto it = known_bits.find(op.Inst());
                    if (it == known_bits.end()) {
                        return {};
                    }
                    acc.zeros &= it->second.zeros;
                    acc.ones &= it->second.ones;
                } else {
                    return {};
                }
            }
            return acc;
        }
        default:
            return {};
        }
    }

    [[nodiscard]] std::optional<IR::Value> TryFold(IR::Block& block, IR::Inst& inst) {
        const IR::Opcode op = inst.GetOpcode();
        switch (op) {
        case IR::Opcode::BitCastU32F32: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<u32>(x.F32())};
            }
            if (x.Inst()->GetOpcode() == IR::Opcode::BitCastF32U32) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::BitCastF32U32: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<f32>(x.U32())};
            }
            IR::Inst* const inner = x.Inst();
            if (inner->GetOpcode() == IR::Opcode::BitCastU32F32) {
                return inner->Arg(0).Resolve();
            }
            if (inner->GetOpcode() == IR::Opcode::BitwiseXor32) {
                const IR::Value a0 = inner->Arg(0).Resolve();
                const IR::Value a1 = inner->Arg(1).Resolve();
                const IR::Value* src = nullptr;
                if (a1.IsImmediate() && a1.U32() == 0x80000000U && !a0.IsImmediate()) {
                    src = &a0;
                } else if (a0.IsImmediate() && a0.U32() == 0x80000000U && !a1.IsImmediate()) {
                    src = &a1;
                }
                if (src && src->Inst()->GetOpcode() == IR::Opcode::BitCastU32F32) {
                    const IR::F32 fp_val{src->Inst()->Arg(0).Resolve()};
                    const auto next_it = std::next(IR::Block::InstructionList::s_iterator_to(inst));
                    IR::IREmitter ir{block, next_it};
                    return IR::Value{ir.FPNeg(fp_val)};
                }
            }
            if (inner->GetOpcode() == IR::Opcode::BitwiseAnd32) {
                const IR::Value a0 = inner->Arg(0).Resolve();
                const IR::Value a1 = inner->Arg(1).Resolve();
                const IR::Value* src = nullptr;
                if (a1.IsImmediate() && a1.U32() == 0x7fffffffU && !a0.IsImmediate()) {
                    src = &a0;
                } else if (a0.IsImmediate() && a0.U32() == 0x7fffffffU && !a1.IsImmediate()) {
                    src = &a1;
                }
                if (src && src->Inst()->GetOpcode() == IR::Opcode::BitCastU32F32) {
                    const IR::F32 fp_val{src->Inst()->Arg(0).Resolve()};
                    const auto next_it = std::next(IR::Block::InstructionList::s_iterator_to(inst));
                    IR::IREmitter ir{block, next_it};
                    return IR::Value{ir.FPAbs(fp_val)};
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitCastU16F16: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (!x.IsImmediate() && x.Inst()->GetOpcode() == IR::Opcode::BitCastF16U16) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::BitCastF16U16: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (!x.IsImmediate() && x.Inst()->GetOpcode() == IR::Opcode::BitCastU16F16) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::BitCastU64F64: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<u64>(x.F64())};
            }
            if (x.Inst()->GetOpcode() == IR::Opcode::BitCastF64U64) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::BitCastF64U64: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<f64>(x.U64())};
            }
            if (x.Inst()->GetOpcode() == IR::Opcode::BitCastU64F64) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::FPNeg32: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<f32>(std::bit_cast<u32>(x.F32()) ^ 0x80000000U)};
            }
            if (x.Inst()->GetOpcode() == IR::Opcode::FPNeg32) {
                return x.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::FPAbs32: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                return IR::Value{std::bit_cast<f32>(std::bit_cast<u32>(x.F32()) & 0x7fffffffU)};
            }
            const IR::Opcode inner_op = x.Inst()->GetOpcode();
            if (inner_op == IR::Opcode::FPAbs32) {
                return x;
            }
            if (inner_op == IR::Opcode::FPNeg32) {
                inst.SetArg(0, x.Inst()->Arg(0).Resolve());
                return std::nullopt;
            }
            return std::nullopt;
        }
        case IR::Opcode::FPSaturate32: {
            const IR::Value x = inst.Arg(0).Resolve();
            if (x.IsImmediate()) {
                const f32 v = x.F32();
                if (!std::isnan(v)) {
                    return IR::Value{std::clamp(v, 0.0f, 1.0f)};
                }
                return std::nullopt;
            }
            if (x.Inst()->GetOpcode() == IR::Opcode::FPSaturate32) {
                return x;
            }
            return std::nullopt;
        }
        case IR::Opcode::IAdd32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() + b.U32()};
            }
            if (a.IsImmediate() && a.U32() == 0U) {
                return b;
            }
            if (b.IsImmediate() && b.U32() == 0U) {
                return a;
            }
            return std::nullopt;
        }
        case IR::Opcode::ISub32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() - b.U32()};
            }
            if (b.IsImmediate() && b.U32() == 0U) {
                return a;
            }
            if (a == b) {
                return IR::Value{u32{0U}};
            }
            return std::nullopt;
        }
        case IR::Opcode::IMul32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() * b.U32()};
            }
            if ((a.IsImmediate() && a.U32() == 0U) || (b.IsImmediate() && b.U32() == 0U)) {
                return IR::Value{u32{0U}};
            }
            if (a.IsImmediate() && a.U32() == 1U) {
                return b;
            }
            if (b.IsImmediate() && b.U32() == 1U) {
                return a;
            }
            return std::nullopt;
        }
        case IR::Opcode::BitwiseAnd32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() & b.U32()};
            }
            if ((a.IsImmediate() && a.U32() == 0U) || (b.IsImmediate() && b.U32() == 0U)) {
                return IR::Value{u32{0U}};
            }
            if (a.IsImmediate() && a.U32() == ~0U) {
                return b;
            }
            if (b.IsImmediate() && b.U32() == ~0U) {
                return a;
            }
            if (a == b) {
                return a;
            }
            if (b.IsImmediate()) {
                const u32 mask = b.U32();
                const KnownBits ka = GetKnownBits(a);
                if ((ka.zeros & mask) == mask) {
                    return IR::Value{u32{0U}};
                }
                if ((ka.zeros & ~mask) == ~mask) {
                    return a;
                }
            }
            if (a.IsImmediate()) {
                const u32 mask = a.U32();
                const KnownBits kb = GetKnownBits(b);
                if ((kb.zeros & mask) == mask) {
                    return IR::Value{u32{0U}};
                }
                if ((kb.zeros & ~mask) == ~mask) {
                    return b;
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitwiseOr32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() | b.U32()};
            }
            if ((a.IsImmediate() && a.U32() == ~0U) || (b.IsImmediate() && b.U32() == ~0U)) {
                return IR::Value{~u32{0U}};
            }
            if (a.IsImmediate() && a.U32() == 0U) {
                return b;
            }
            if (b.IsImmediate() && b.U32() == 0U) {
                return a;
            }
            if (a == b) {
                return a;
            }
            if (b.IsImmediate()) {
                const u32 mask = b.U32();
                const KnownBits ka = GetKnownBits(a);
                if ((ka.ones & mask) == mask) {
                    return a;
                }
            }
            if (a.IsImmediate()) {
                const u32 mask = a.U32();
                const KnownBits kb = GetKnownBits(b);
                if ((kb.ones & mask) == mask) {
                    return b;
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitwiseXor32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && b.IsImmediate()) {
                return IR::Value{a.U32() ^ b.U32()};
            }
            if (a.IsImmediate() && a.U32() == 0U) {
                return b;
            }
            if (b.IsImmediate() && b.U32() == 0U) {
                return a;
            }
            if (a == b) {
                return IR::Value{u32{0U}};
            }
            return std::nullopt;
        }
        case IR::Opcode::BitwiseNot32: {
            const IR::Value a = inst.Arg(0).Resolve();
            if (a.IsImmediate()) {
                return IR::Value{~a.U32()};
            }
            if (a.Inst()->GetOpcode() == IR::Opcode::BitwiseNot32) {
                return a.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::ShiftLeftLogical32:
        case IR::Opcode::ShiftRightLogical32:
        case IR::Opcode::ShiftRightArithmetic32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value s = inst.Arg(1).Resolve();
            if (s.IsImmediate() && s.U32() == 0U) {
                return a;
            }
            if (a.IsImmediate() && s.IsImmediate() && s.U32() < 32U) {
                if (op == IR::Opcode::ShiftLeftLogical32) {
                    return IR::Value{a.U32() << s.U32()};
                }
                if (op == IR::Opcode::ShiftRightLogical32) {
                    return IR::Value{a.U32() >> s.U32()};
                }
                return IR::Value{static_cast<u32>(static_cast<s32>(a.U32()) >> s.U32())};
            }
            return std::nullopt;
        }
        case IR::Opcode::BitFieldUExtract: {
            const IR::Value base = inst.Arg(0).Resolve();
            const IR::Value off = inst.Arg(1).Resolve();
            const IR::Value cnt = inst.Arg(2).Resolve();
            if (off.IsImmediate() && cnt.IsImmediate()) {
                const u32 o = off.U32();
                const u32 c = cnt.U32();
                if (c == 0U) {
                    return IR::Value{u32{0U}};
                }
                if (o == 0U && c == 32U) {
                    return base;
                }
                if (o == 0U && c < 32U) {
                    const KnownBits kb = GetKnownBits(base);
                    const u32 upper = ~0U << c;
                    if ((kb.zeros & upper) == upper) {
                        return base;
                    }
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitFieldInsert: {
            const IR::Value base = inst.Arg(0).Resolve();
            const IR::Value ins = inst.Arg(1).Resolve();
            const IR::Value off = inst.Arg(2).Resolve();
            const IR::Value cnt = inst.Arg(3).Resolve();
            if (off.IsImmediate() && cnt.IsImmediate()) {
                const u32 o = off.U32();
                const u32 c = cnt.U32();
                if (c == 0U) {
                    return base;
                }
                if (o == 0U && c == 32U) {
                    return ins;
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::SMin32:
        case IR::Opcode::UMin32:
        case IR::Opcode::SMax32:
        case IR::Opcode::UMax32: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a == b) {
                return a;
            }
            return std::nullopt;
        }
        case IR::Opcode::SelectU1:
        case IR::Opcode::SelectU8:
        case IR::Opcode::SelectU16:
        case IR::Opcode::SelectU32:
        case IR::Opcode::SelectU64:
        case IR::Opcode::SelectF16:
        case IR::Opcode::SelectF32:
        case IR::Opcode::SelectF64: {
            const IR::Value cond = inst.Arg(0).Resolve();
            const IR::Value t = inst.Arg(1).Resolve();
            const IR::Value f = inst.Arg(2).Resolve();
            if (cond.IsImmediate()) {
                return cond.U1() ? t : f;
            }
            if (t == f) {
                return t;
            }
            if (op == IR::Opcode::SelectU1 && t.IsImmediate() && f.IsImmediate()) {
                if (t.U1() && !f.U1()) {
                    return cond;
                }
                if (!t.U1() && f.U1()) {
                    const auto next_it = std::next(IR::Block::InstructionList::s_iterator_to(inst));
                    IR::IREmitter ir{block, next_it};
                    return IR::Value{ir.LogicalNot(IR::U1{cond})};
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::LogicalNot: {
            const IR::Value a = inst.Arg(0).Resolve();
            if (a.IsImmediate()) {
                return IR::Value{!a.U1()};
            }
            if (a.Inst()->GetOpcode() == IR::Opcode::LogicalNot) {
                return a.Inst()->Arg(0).Resolve();
            }
            return std::nullopt;
        }
        case IR::Opcode::LogicalAnd: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if ((a.IsImmediate() && !a.U1()) || (b.IsImmediate() && !b.U1())) {
                return IR::Value{false};
            }
            if (a.IsImmediate() && a.U1()) {
                return b;
            }
            if (b.IsImmediate() && b.U1()) {
                return a;
            }
            if (a == b) {
                return a;
            }
            if ((!a.IsImmediate() && a.Inst()->GetOpcode() == IR::Opcode::LogicalNot &&
                 a.Inst()->Arg(0).Resolve() == b) ||
                (!b.IsImmediate() && b.Inst()->GetOpcode() == IR::Opcode::LogicalNot &&
                 b.Inst()->Arg(0).Resolve() == a)) {
                return IR::Value{false};
            }
            return std::nullopt;
        }
        case IR::Opcode::LogicalOr: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if ((a.IsImmediate() && a.U1()) || (b.IsImmediate() && b.U1())) {
                return IR::Value{true};
            }
            if (a.IsImmediate() && !a.U1()) {
                return b;
            }
            if (b.IsImmediate() && !b.U1()) {
                return a;
            }
            if (a == b) {
                return a;
            }
            if ((!a.IsImmediate() && a.Inst()->GetOpcode() == IR::Opcode::LogicalNot &&
                 a.Inst()->Arg(0).Resolve() == b) ||
                (!b.IsImmediate() && b.Inst()->GetOpcode() == IR::Opcode::LogicalNot &&
                 b.Inst()->Arg(0).Resolve() == a)) {
                return IR::Value{true};
            }
            return std::nullopt;
        }
        case IR::Opcode::LogicalXor: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a.IsImmediate() && !a.U1()) {
                return b;
            }
            if (b.IsImmediate() && !b.U1()) {
                return a;
            }
            if (a == b) {
                return IR::Value{false};
            }
            return std::nullopt;
        }
        case IR::Opcode::IEqual: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a == b) {
                return IR::Value{true};
            }
            if (a.IsImmediate() && b.IsImmediate()) {
                if (a.Type() == IR::Type::U32 && b.Type() == IR::Type::U32) {
                    return IR::Value{a.U32() == b.U32()};
                }
                if (a.Type() == IR::Type::U64 && b.Type() == IR::Type::U64) {
                    return IR::Value{a.U64() == b.U64()};
                }
            }
            if (a.Type() == IR::Type::U32 && b.Type() == IR::Type::U32) {
                const KnownBits ka = GetKnownBits(a);
                const KnownBits kb = GetKnownBits(b);
                if (((ka.zeros & kb.ones) != 0U) || ((ka.ones & kb.zeros) != 0U)) {
                    return IR::Value{false};
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::INotEqual: {
            const IR::Value a = inst.Arg(0).Resolve();
            const IR::Value b = inst.Arg(1).Resolve();
            if (a == b) {
                return IR::Value{false};
            }
            if (a.IsImmediate() && b.IsImmediate()) {
                if (a.Type() == IR::Type::U32 && b.Type() == IR::Type::U32) {
                    return IR::Value{a.U32() != b.U32()};
                }
            }
            if (a.Type() == IR::Type::U32 && b.Type() == IR::Type::U32) {
                const KnownBits ka = GetKnownBits(a);
                const KnownBits kb = GetKnownBits(b);
                if (((ka.zeros & kb.ones) != 0U) || ((ka.ones & kb.zeros) != 0U)) {
                    return IR::Value{true};
                }
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<ExprKey> MakeExprKey(const IR::Inst& inst) const {
        const IR::Opcode op = inst.GetOpcode();
        if (!IsPureValueOpcode(op)) {
            return std::nullopt;
        }
        const size_t n = inst.NumArgs();
        if (n > 5U) {
            return std::nullopt;
        }
        ExprKey key{
            .op = op,
            .flags = inst.Flags<u32>(),
            .num_args = static_cast<u8>(n),
        };
        for (size_t i = 0; i < n; ++i) {
            const IR::Value v = inst.Arg(i).Resolve();
            if (!IsKeyableValue(v)) {
                return std::nullopt;
            }
            key.args[i] = MakeValueKey(v);
        }
        if (IsCommutativeBinary(op) && n == 2U && key.args[1] < key.args[0]) {
            std::swap(key.args[0], key.args[1]);
        } else if (IsCommutativeFma(op) && n == 3U && key.args[1] < key.args[0]) {
            std::swap(key.args[0], key.args[1]);
        }
        return key;
    }

    void VisitDominatorSubtree(size_t rpo_node) {
        IR::Block* const block = rpo[rpo_node];
        const size_t scope_mark = gvn_undo.size();

        for (IR::Inst& inst : block->Instructions()) {
            const IR::Opcode op = inst.GetOpcode();
            if (op == IR::Opcode::Void || op == IR::Opcode::Identity) {
                continue;
            }
            if (op == IR::Opcode::Phi) {
                const KnownBits phi_kb = ComputeKnownBits(inst);
                if (phi_kb.zeros != 0U || phi_kb.ones != 0U) {
                    known_bits.emplace(&inst, phi_kb);
                }
                continue;
            }
            if (inst.IsPseudoInstruction() || inst.MayHaveSideEffects() || HasLivePseudoOp(inst)) {
                continue;
            }

            const size_t num_args = inst.NumArgs();
            for (size_t i = 0; i < num_args; ++i) {
                const IR::Value arg = inst.Arg(i);
                const IR::Value resolved = arg.Resolve();
                if (resolved != arg) {
                    inst.SetArg(i, resolved);
                }
            }

            if (const auto folded = TryFold(*block, inst)) {
                inst.ReplaceUsesWith(*folded);
                continue;
            }

            const KnownBits kb = ComputeKnownBits(inst);
            if ((kb.zeros | kb.ones) == ~0U && inst.Type() == IR::Type::U32) {
                inst.ReplaceUsesWith(IR::Value{kb.ones});
                continue;
            }
            if (kb.zeros != 0U || kb.ones != 0U) {
                known_bits.emplace(&inst, kb);
            }

            if (const auto key = MakeExprKey(inst)) {
                const auto [it, inserted] = gvn_table.emplace(*key, IR::Value{&inst});
                if (!inserted) {
                    inst.ReplaceUsesWith(it->second);
                    continue;
                }
                gvn_undo.push_back(*key);
            }
        }

        for (const size_t child : dom_children[rpo_node]) {
            VisitDominatorSubtree(child);
        }

        while (gvn_undo.size() > scope_mark) {
            gvn_table.erase(gvn_undo.back());
            gvn_undo.pop_back();
        }
    }

    IR::Program& program;
    std::vector<IR::Block*> rpo;
    std::unordered_map<const IR::Block*, size_t> rpo_index;
    std::vector<std::vector<size_t>> dom_children;
    std::unordered_map<const IR::Inst*, KnownBits> known_bits;
    std::unordered_map<ExprKey, IR::Value, ExprKeyHash> gvn_table;
    std::vector<ExprKey> gvn_undo;
};

} // Anonymous namespace

void ValueNumberingPass(IR::Program& program) {
    ValueNumbering pass{program};
    pass.Run();
    IdentityRemovalPass(program);
}

} // namespace Shader::Optimization
