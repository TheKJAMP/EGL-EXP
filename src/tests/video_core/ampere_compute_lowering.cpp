// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
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

constexpr u32 Sched = 0x000fe200;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

class ComputeLoweringEnv final : public Shader::Environment {
public:
    explicit ComputeLoweringEnv(u32 tex_bound = 2) : texture_bound{tex_bound} {
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
    u32 TextureBoundBuffer() const override { return texture_bound; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    u32 texture_bound;
};

struct IrPools {
    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    const u32 tex_bound =
        analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2U;
    ComputeLoweringEnv env{tex_bound};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    auto program = Shader::Ampere::TranslateProgram(pools.inst, pools.block, flow_pool, env, code,
                                                    {}, analysis.uses_async_copy,
                                                    &analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(program);
    return program;
}

bool HasOpcode(const Shader::IR::Program& program, Shader::IR::Opcode op) {
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() == op) {
                return true;
            }
        }
    }
    return false;
}

TEST_CASE("Ampere Ocarina high effective address negates the register base",
          "[video_core][ampere]") {
    const Instruction capture{{0x01017211, 0x80000000, 0x078ff0ff, 0x000fca00}};
    for (const u32 guard : {1U, 7U, 9U}) {
        auto instruction = capture;
        instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
        const std::array code{instruction, Exit};
        const auto analysis = AnalyzeProgram(code);
        INFO(analysis.error);
        REQUIRE(static_cast<bool>(analysis));
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }
    for (const u32 bit : {62U, 74U, 81U}) {
        auto invalid = capture;
        invalid.words[bit / 32] ^= 1U << (bit % 32);
        CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    }
}

TEST_CASE("Ampere surface reductions preserve every U32 atomic operation",
          "[video_core][ampere]") {
    constexpr std::array<Shader::IR::Opcode, 8> atomic_ops{
        Shader::IR::Opcode::ImageAtomicIAdd32,  Shader::IR::Opcode::ImageAtomicUMin32,
        Shader::IR::Opcode::ImageAtomicUMax32,  Shader::IR::Opcode::ImageAtomicIAdd32,
        Shader::IR::Opcode::ImageAtomicIAdd32,  Shader::IR::Opcode::ImageAtomicAnd32,
        Shader::IR::Opcode::ImageAtomicOr32,   Shader::IR::Opcode::ImageAtomicXor32,
    };
    for (u32 op = 0; op < 8; ++op) {
        if (op == 3 || op == 4) {
            continue;
        }
        for (const u32 dimension : {1U, 3U}) {
            for (const u32 guard : {1U, 7U, 9U}) {
                const Instruction instruction{
                    {0x0400059fU | (guard << 12), 0x00008400U | (dimension << 29),
                     0x0010e000U | (op << 23), 0}};
                const std::array code{instruction, Exit};
                const auto analysis = AnalyzeProgram(code);
                INFO(analysis.error);
                REQUIRE(static_cast<bool>(analysis));
                REQUIRE(analysis.required_texture_buffer == 0);
                IrPools pools;
                CHECK(HasOpcode(TranslateDirect(pools, code), atomic_ops[op]));
                auto invalid = instruction;
                invalid.words[2] |= 2U << 9;
                CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
            }
        }
    }
}

TEST_CASE("Half per-tap gathers prove offsets across an unrelated LEA.HI",
          "[video_core][ampere][gather]") {
    const Instruction lea{{0x08087211, 0x00000007, 0x078ff8ff, 0x000fc800}};
    const Instruction gather{{0x0a0c7b63, 0x30003000, 0x001eaf0d, 0x000fc200}};
    auto code = std::array{
        Instruction{{0x00007802, 0xff00ff00, 0xf00, 0}},
        Instruction{{0x00017802, 0x01000001, 0xf00, 0}},
        lea,
        gather,
        Exit,
    };
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    IrPools pools;
    (void)TranslateDirect(pools, code);
    code[2].words[0] = 0x08017211;
    CHECK(AnalyzeProgram(code).error == "native per-tap gather nonconstant offsets");
}

TEST_CASE("Ampere LEA.HI with a constant base and no carry lowers", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x00117919, 0, 33U << 8, Sched}};
    // LEA.HI R1, R17, c[0][0x0], RZ, 0x1e from Pokemon Pokopia's compute shaders.
    const Instruction lea_hi{{0x11017a11, 0x00c00000, 0x078ff0ff, 0x000fe400}};
    const Instruction stg{{0x10007386, 1U, 0x0010a900U, Sched}};
    const std::array code{ptr0, ptr1, s2r, lea_hi, stg, Exit};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(HasOpcode(program, Shader::IR::Opcode::ShiftRightLogical32));
    CHECK(HasOpcode(program, Shader::IR::Opcode::IAdd32));
}

TEST_CASE("Ampere LEA.HI with a constant base still rejects a carry out", "[video_core][ampere]") {
    Instruction carry{{0x11017a11, 0x00c00000, 0x078ff0ff, 0x000fe400}};
    carry.words[2] &= ~(7U << 17);
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{carry, Exit})));
}

TEST_CASE("Ampere LEA with a constant base and no carry out emits shift-add",
          "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x00007919, 0, 33U << 8, Sched}};
    const Instruction lea{{0x00007a11, 0x01002e00, 0x078e40ff, 0x000fe200}};
    const Instruction stg{{0x10007386, 0U, 0x0010a900U, Sched}};
    IrPools pools;
    const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, s2r, lea, stg, Exit});
    CHECK(HasOpcode(program, Shader::IR::Opcode::ShiftLeftLogical32));
    CHECK(HasOpcode(program, Shader::IR::Opcode::IAdd32));
}

TEST_CASE("Ampere SHF.L.U32.HI immediate without .W lowers to a clamping shift",
          "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x000e7919, 0, 33U << 8, Sched}};
    const Instruction shf{{0xff107419, 0x000000ff, 0x0001060e, 0x080fe400}};
    const Instruction stg{{0x10007386, 16U, 0x0010a900U, Sched}};
    IrPools pools;
    const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, s2r, shf, stg, Exit});
    CHECK(HasOpcode(program, Shader::IR::Opcode::ShiftLeftLogical32));
}

TEST_CASE("Ampere guarded UIADD3 translates uniformly", "[video_core][ampere]") {
    const std::array variants{
        Instruction{{0x0e068890, 0x1c, 0x0fffe03f, 0x000fcc00}},
        Instruction{{0x06060290, 0x7, 0x0fffe03f, 0x000fc800}},
        Instruction{{0x09069290, 0x8, 0x0fffe006, 0x000fc800}},
    };
    for (const Instruction& insn : variants) {
        const std::array code{insn, Exit};
        const auto analysis = AnalyzeProgram(code);
        INFO(analysis.error);
        REQUIRE(static_cast<bool>(analysis));
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }
}

TEST_CASE("Ampere DEPBAR.LE with any scoreboard and wait list is a no-op without LDGSTS",
          "[video_core][ampere]") {
    for (const u32 word : {0x0000d040U, 0x0000800eU}) {
        const std::array<Instruction, 2> code{{{{0x0000791a, word, 0, 0x000fce00}}, Exit}};
        IrPools pools;
        const auto program = TranslateDirect(pools, code);
        CHECK_FALSE(HasOpcode(program, Shader::IR::Opcode::WorkgroupMemoryBarrier));
    }
}

TEST_CASE("Ampere guarded USHF.R.U32.HI translates natively", "[video_core][ampere]") {
    const std::array<Instruction, 2> guarded{{{{0x3f070899, 0x2, 0x08011606, 0x000fcc00}}, Exit}};
    const auto analysis = AnalyzeProgram(guarded);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    IrPools pools;
    (void)TranslateDirect(pools, guarded);
}
} // namespace
