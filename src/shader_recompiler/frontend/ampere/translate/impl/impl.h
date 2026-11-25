// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/pred.h"
#include "shader_recompiler/frontend/ir/reg.h"

namespace Shader::Ampere {

class TranslatorVisitor {
public:
    explicit TranslatorVisitor(Environment& env, IR::Block& block,
                               std::span<const Instruction> code, const LoweringOptions& options,
                               bool uses_async_copy, u32& required_texture_buffer,
                               const std::vector<u8>* branch_targets);

    // Moves, constant loads, special registers, no-ops
    void MOV(const Instruction& insn);
    void UMOV(const Instruction& insn);
    void LDC(const Instruction& insn);
    void ULDC(const Instruction& insn);
    void S2R(const Instruction& insn);
    void S2UR(const Instruction& insn);
    void CS2R(const Instruction& insn);
    void LEPC(const Instruction& insn, u32 index);
    void EXIT();

    // Integer add, effective address, abs, min/max
    void IADD3(const Instruction& insn);
    void LEA(const Instruction& insn);
    void IABS(const Instruction& insn);
    void IMNMX(const Instruction& insn);

    // Integer multiply & dot product
    void IMAD(const Instruction& insn);
    void IMAD_WIDE(const Instruction& insn);
    void IDP2A(const Instruction& insn);

    // Logic & bit manipulation
    void LOP3(const Instruction& insn);
    void SHF(const Instruction& insn);
    void SGXT(const Instruction& insn);
    void PRMT(const Instruction& insn);
    void FLO(const Instruction& insn);
    void POPC(const Instruction& insn);
    void BREV(const Instruction& insn);

    // Predicate & select
    void ISETP(const Instruction& insn);
    void SEL(const Instruction& insn);
    void PLOP3(const Instruction& insn);
    void P2R(const Instruction& insn);
    void R2P(const Instruction& insn);
    void UP2UR(const Instruction& insn);
    void CSMTEST(const Instruction& insn);

    // Warp & uniform instructions
    void VOTE(const Instruction& insn);
    void R2UR(const Instruction& insn);
    void REDUX(const Instruction& insn);

    // Floating-point ALU & fragment interpolation (Tier 2)
    void FFMA(const Instruction& insn);
    void FMUL(const Instruction& insn);
    void FADD(const Instruction& insn);
    void FMNMX(const Instruction& insn);
    void MUFU(const Instruction& insn);
    void IPA(const Instruction& insn);
    void FSEL(const Instruction& insn);
    void FSET(const Instruction& insn);
    void FSETP(const Instruction& insn);

    // FP16, scalar/packed conversions, and swizzled FP add
    void F2F(const Instruction& insn);
    void F2FP(const Instruction& insn);
    void HMUL2(const Instruction& insn);
    void HADD2(const Instruction& insn);
    void HFMA2(const Instruction& insn);
    void HSET2(const Instruction& insn);
    void HSETP2(const Instruction& insn);
    void HMNMX2(const Instruction& insn);
    void FSWZADD(const Instruction& insn);
    void F2I_64(const Instruction& insn);
    void I2F_64(const Instruction& insn);
    void F2IP(const Instruction& insn);
    void F2I(const Instruction& insn);
    void FRND(const Instruction& insn);
    void I2F(const Instruction& insn);
    void I2FP(const Instruction& insn);

    // Memory, atomics, barriers, attributes, geometry OUT, stage buffers
    void LDG(const Instruction& insn);
    void STG(const Instruction& insn);
    void ATOMG(const Instruction& insn);
    void RED(const Instruction& insn);
    void LDS_STS(const Instruction& insn);
    void ATOMS(const Instruction& insn);
    void ATOMS_POPC_INC(const Instruction& insn);
    void LDGSTS(const Instruction& insn, bool load_global);
    void LDL_STL(const Instruction& insn);
    void DEPBAR(const Instruction& insn);
    void MEMBAR(const Instruction& insn);
    void ERRBAR();
    void LDGDEPBAR();
    void BAR_SYNC();
    void ALD_AST(const Instruction& insn);
    void OUT(const Instruction& insn);
    void ISBE(const Instruction& insn);

    // Texture operations
    void TEX(const Instruction& insn);
    void TLD4(const Instruction& insn, u32 index);
    void TLD(const Instruction& insn);
    void TMML(const Instruction& insn);
    void TXD(const Instruction& insn);
    void TXQ(const Instruction& insn);

    // Surface and warp operations
    void SULD(const Instruction& insn);
    void SUST(const Instruction& insn);
    void SUATOM(const Instruction& insn);
    void SURED(const Instruction& insn);
    void SHFL(const Instruction& insn);
    void MATCH_ALL(const Instruction& insn);

    // Tensor / matrix operations
    void MOVM(const Instruction& insn);
    void HMMA(const Instruction& insn);
    void LDSM(const Instruction& insn);

    void SetActiveGuard(std::optional<IR::U1> guard) noexcept {
        active_guard = std::move(guard);
    }
    [[nodiscard]] IR::U1 EvalGuard(const Instruction& insn);

    IR::IREmitter ir;
    Environment& env;

    [[nodiscard]] IR::U32 R(u32 reg);
    void R(u32 reg, const IR::U32& value);
    [[nodiscard]] IR::U32 UR(u32 reg);
    void UR(u32 reg, const IR::U32& value);
    [[nodiscard]] IR::F32 F(u32 reg);
    void F(u32 reg, const IR::F32& value);

    [[nodiscard]] IR::U1 P(u32 pred4);
    [[nodiscard]] IR::U1 UP(u32 pred4);
    void SetP(u32 idx, const IR::U1& value);
    void SetUP(u32 idx, const IR::U1& value);

private:
    [[nodiscard]] constexpr IR::Reg Gpr(u32 reg) const noexcept {
        return reg == 255U ? IR::Reg::RZ : static_cast<IR::Reg>(reg);
    }
    [[nodiscard]] constexpr IR::Reg UniformReg(u32 reg) const noexcept {
        return reg == 63U ? IR::Reg::RZ : static_cast<IR::Reg>(256U + reg);
    }
    [[nodiscard]] constexpr IR::Pred ThreadPred(u32 pred) const noexcept {
        return static_cast<IR::Pred>(pred & 7U);
    }
    [[nodiscard]] constexpr IR::Pred UniformPred(u32 pred) const noexcept {
        const u32 slot = pred & 7U;
        return slot == 7U ? IR::Pred::PT : static_cast<IR::Pred>(8U + slot);
    }

    [[nodiscard]] u32 CbufBank(const Instruction& insn) const noexcept;
    [[nodiscard]] IR::U32 ReadCbuf32(const Instruction& insn);
    [[nodiscard]] IR::U32 AluSourceU32(const Instruction& insn, u32 which);
    [[nodiscard]] IR::F32 ReadCbufF32(const Instruction& insn);
    [[nodiscard]] IR::F32 AluSourceF32(const Instruction& insn, u32 which);

    struct HalfOperand {
        IR::U32 value;
        u32 swizzle{};
        bool abs{};
        bool neg{};
    };
    [[nodiscard]] HalfOperand HalfSource(const Instruction& insn, u32 which);
    [[nodiscard]] std::pair<IR::U1, IR::U1> EvalHalfCompare(const Instruction& insn, u32 b_which);
    [[nodiscard]] IR::U64 PackU64Gpr(u32 reg);
    [[nodiscard]] IR::U64 PackU64Ur(u32 ureg);
    [[nodiscard]] IR::U64 GlobalAddress(const Instruction& insn, bool is_uniform_opcode,
                                        u32 ureg_bit, bool is_descriptor);
    [[nodiscard]] IR::Value TextureDescriptorHandle(u32 binding, u32 word_offset);

    std::span<const Instruction> code;
    const LoweringOptions& options;
    bool uses_async_copy{};
    u32& required_texture_buffer;
    const std::vector<u8>* branch_targets{};
    std::optional<IR::U1> active_guard;
};

} // namespace Shader::Ampere
