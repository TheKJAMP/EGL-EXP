// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ir/condition.h"
#include "shader_recompiler/frontend/ir/pred.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/location.h"
#include "shader_recompiler/object_pool.h"

namespace Shader::Ampere {

/// Non-zero base offset for raw native Ampere CFG locations so Virtual() at instruction 0 never
/// wraps below 0 (`Virtual()` subtracts 4 from the raw byte offset).
constexpr u32 AmpereLocationBase = 16;

[[nodiscard]] constexpr Maxwell::Location LocationOf(u32 index) noexcept {
    return Maxwell::Location::Raw(AmpereLocationBase + index * static_cast<u32>(sizeof(Instruction)));
}

[[nodiscard]] constexpr u32 IndexOf(Maxwell::Location location) noexcept {
    return (location.Offset() - AmpereLocationBase) / static_cast<u32>(sizeof(Instruction));
}

[[nodiscard]] constexpr Maxwell::Location NextLocation(Maxwell::Location location) noexcept {
    return Maxwell::Location::Raw(location.Offset() + static_cast<u32>(sizeof(Instruction)));
}

/// True if `opcode` reads a uniform guard predicate (`@UP0..@UP6`) in bits 12..15.
/// `R2UR` (`0x3c2`) intentionally returns false: it reads a thread predicate and runs warp-wide.
[[nodiscard]] bool UsesUniformGuardPredicate(u32 opcode) noexcept;

/// Builds the IR guard condition for a native Ampere instruction's 4-bit predicate field.
[[nodiscard]] IR::Condition GuardCondition(u32 opcode, u32 pred_bits) noexcept;

/// True for opcodes natively translated by the direct Ampere -> IR frontend (Tier 1).
[[nodiscard]] bool IsDirectTier1Opcode(u32 opcode) noexcept;

/// True for opcodes natively translated by the direct Ampere -> IR frontend (Tier 2).
[[nodiscard]] bool IsDirectTier2Opcode(u32 opcode) noexcept;

/// True if a predicated native Ampere instruction is pure and side-effect-free,
/// allowing inline SSA Select if-conversion instead of splitting basic blocks.
[[nodiscard]] bool CanIfConvertPredicated(const Instruction& insn) noexcept;

/// Scans backward within the current basic block for a constant 4x8-bit per-tap offset immediate
/// feeding a `TLD4` (`0xb63`) PTP gather instruction. Returns std::nullopt when no constant is
/// found or sets `error` on clobber / nonconstant / out-of-range offsets.
[[nodiscard]] std::optional<u32> FindGatherPtpOffsetImmediate(
    std::span<const Instruction> code, u32 index, u32 reg, u32 guard_pred,
    const std::vector<u8>* branch_targets, std::string_view& error) noexcept;

struct ProgramAnalysis {
    u32 source_size{};
    u32 instruction_count{};
    u32 required_texture_buffer{~0U};
    u32 generic_attributes_read{};
    bool uses_async_copy{};
    std::vector<u8> branch_targets;
    u32 error_offset{};
    std::string_view error;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error.empty() && source_size != 0;
    }
};

/// Validates and analyzes a native Ampere shader program without lowering it to Maxwell.
[[nodiscard]] ProgramAnalysis AnalyzeProgram(std::span<const Instruction> code,
                                             const LoweringOptions& options = {});

namespace Flow {

using Block = Maxwell::Flow::Block;
using EndClass = Maxwell::Flow::EndClass;
using Function = Maxwell::Flow::Function;
using FunctionId = Maxwell::Flow::FunctionId;
using IndirectBranch = Maxwell::Flow::IndirectBranch;
using Label = Maxwell::Flow::Label;
using Stack = Maxwell::Flow::Stack;

class CFG {
public:
    explicit CFG(Environment& env, ObjectPool<Block>& block_pool,
                 std::span<const Instruction> code, bool exits_to_dispatcher = false);
    CFG(const CFG&) = delete;
    CFG(CFG&&) = delete;

    [[nodiscard]] std::span<const Function> Functions() const noexcept {
        return std::span(functions.data(), functions.size());
    }
    [[nodiscard]] std::span<Function> Functions() noexcept {
        return std::span(functions.data(), functions.size());
    }

    [[nodiscard]] bool ExitsToDispatcher() const noexcept {
        return exits_to_dispatcher;
    }

private:
    enum class AnalysisState {
        Continue,
        Branch,
    };

    void AnalyzeLabel(FunctionId function_id, Label& label);
    bool InspectVisitedBlocks(FunctionId function_id, const Label& label);
    AnalysisState AnalyzeInst(Block* block, FunctionId function_id, Maxwell::Location pc);

    void AnalyzeCondInst(Block* block, FunctionId function_id, Maxwell::Location pc,
                         EndClass insn_end_class, IR::Condition cond);

    AnalysisState AnalyzeBRA(Block* block, FunctionId function_id, Maxwell::Location pc,
                             u32 index, const Instruction& insn);
    AnalysisState AnalyzeEXIT(Block* block, FunctionId function_id, Maxwell::Location pc,
                              u32 index, const Instruction& insn);
    AnalysisState AnalyzeKILL(Block* block, FunctionId function_id, Maxwell::Location pc,
                              const Instruction& insn);
    AnalysisState AnalyzeTRAP(Block* block, FunctionId function_id, Maxwell::Location pc,
                              const Instruction& insn);
    AnalysisState AnalyzeCALL(Block* block, FunctionId function_id, Maxwell::Location pc,
                              u32 index, const Instruction& insn);
    AnalysisState AnalyzeRET(Block* block, Maxwell::Location pc, u32 index,
                             const Instruction& insn);
    AnalysisState AnalyzeBRX(Block* block, FunctionId function_id, Maxwell::Location pc,
                             u32 index, const Instruction& insn);

    Block* AddLabel(Block* block, Stack stack, Maxwell::Location pc, FunctionId function_id);

    Environment& env;
    ObjectPool<Block>& block_pool;
    std::span<const Instruction> code;
    boost::container::small_vector<Function, 1> functions;
    bool exits_to_dispatcher{};
    Block* dispatch_block{};
    // Call targets discovered during CFG traversal, used to validate RET return links.
    struct CallSiteInfo {
        u32 target_index{};
        u32 link_reg{};
    };
    std::vector<CallSiteInfo> call_sites;
};

} // namespace Flow
} // namespace Shader::Ampere
