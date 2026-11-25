// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/program_header.h"

namespace {

using Shader::Ampere::AnalyzeProgram;
using Shader::Ampere::Instruction;

class VoteEnv final : public Shader::Environment {
public:
    VoteEnv() {
        stage = Shader::Stage::Compute;
        is_proprietary_driver = true;
        start_address = sizeof(Shader::ProgramHeader);
    }
    u64 ReadInstruction(u32) override { return 0; }
    bool IsNativeProgram() const override { return true; }
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override {
        return 0x101 + cbuf_index * 16 + cbuf_offset;
    }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 2; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

} // namespace

TEST_CASE("Ampere uniform vote writes uniform registers and validates bounds",
          "[video_core][ampere]") {
    constexpr u32 Sched = 0x000fe200;
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const Instruction mov_u_ptr0{{0xff027624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction mov_u_ptr1{{0xff037624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r_lane{{0x00087919, 0, 0U << 8, Sched}};                  // S2R R8, SR_LANEID
    const Instruction p_nonzero{{0x0800780c, 0, 0x03f06070U | (5U << 17), Sched}}; // ISETP.NE.AND P0, PT, R8, 0, PT

    for (const u32 target : {15U, 16U, 21U, 22U, 27U, 28U, 62U, 63U}) {
        const Instruction vote{{0x00007886U | (target << 16), 0, 0x038e0100, 0}};
        const Instruction copy_ur{{0x00047c02U, target, 0x08000f00U, Sched}}; // MOV R4, URt
        const Instruction stg{{0x02007386U, 4U, 0x0010a900U, Sched}};         // STG [R2], R4
        const std::array code{mov_u_ptr0, mov_u_ptr1, s2r_lane, p_nonzero,
                              vote,       copy_ur,    stg,      exit};
        const auto analysis = AnalyzeProgram(code);
        INFO(analysis.error);
        REQUIRE(static_cast<bool>(analysis));

        VoteEnv env;
        Shader::ObjectPool<Shader::IR::Inst> inst_pool;
        Shader::ObjectPool<Shader::IR::Block> block_pool;
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        Shader::HostTranslateInfo host{};
        host.min_ssbo_alignment = 16;
        auto program = Shader::Ampere::TranslateProgram(inst_pool, block_pool, flow_pool, env, code,
                                                        {}, analysis.uses_async_copy,
                                                        &analysis.branch_targets, host);
        Shader::Optimization::VerificationPass(program);

        size_t ballots = 0;
        for (const auto* block : program.blocks) {
            for (const auto& inst : block->Instructions()) {
                if (inst.GetOpcode() == Shader::IR::Opcode::SubgroupBallot) {
                    ++ballots;
                }
            }
        }
        CHECK(ballots == (target == 63U ? 0U : 1U));
    }

    // Invalid destination UR64 (bits 16..23 == 64), invalid words[1], or invalid vote mode (3).
    const Instruction out_of_range{{0x00407886, 0, 0x038e0100, 0}};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{out_of_range, exit})));
    const Instruction bad_w1{{0x00007886, 1, 0x038e0100, 0}};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{bad_w1, exit})));
    const Instruction bad_mode{{0x00007886, 0, 0x038e0300, 0}};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{bad_mode, exit})));
}
