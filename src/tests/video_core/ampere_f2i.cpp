// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <vector>

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

class F2iEnv final : public Shader::Environment {
public:
    F2iEnv() {
        stage = Shader::Stage::Compute;
        is_proprietary_driver = false;
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

struct IrPools {
    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    F2iEnv env;
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

} // namespace

TEST_CASE("Ampere F2I round up preserves conversion mode", "[video_core][ampere]") {
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction load_f{{0x00097919, 0, 33U << 8, Sched}}; // S2R R9 (non-constant bits -> float)
    const Instruction stg{{0x10007386, 6U, 0x0010a900U, Sched}};

    for (const bool is_signed : {false, true}) {
        const Instruction convert{
            {0x00067305, 0x00000009, is_signed ? 0x0021b100U : 0x0020b000U, 0}};
        IrPools pools;
        const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, load_f, convert, stg, exit});
        CHECK(HasOpcode(program, Shader::IR::Opcode::FPCeil32));
        CHECK(HasOpcode(program, is_signed ? Shader::IR::Opcode::ConvertS32F32
                                           : Shader::IR::Opcode::ConvertU32F32));
    }

    const Instruction unsupported{{0x00067305, 0x00000009, 0x0020b001, 0}};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit})));
}

TEST_CASE("Ampere F2I signed byte clamps after conversion", "[video_core][ampere]") {
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction load_f{{0x00057919, 0, 33U << 8, Sched}};
    const Instruction convert{{0x00097305, 0x00000005, 0x0021e100, 0}};
    const Instruction stg{{0x10007386, 9U, 0x0010a900U, Sched}};

    IrPools pools;
    const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, load_f, convert, stg, exit});
    CHECK(HasOpcode(program, Shader::IR::Opcode::FPTrunc32));
    CHECK(HasOpcode(program, Shader::IR::Opcode::ConvertS32F32));
    CHECK(HasOpcode(program, Shader::IR::Opcode::FPClamp32));
}

TEST_CASE("Ampere F2I applies abs and neg to constant-buffer sources", "[video_core][ampere]") {
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction stg{{0x10007386, 21U, 0x0010a900U, Sched}};

    // nvdisasm prints bit 62 as |c[..]| and bit 63 as -c[..] (U16 and truncating word forms).
    for (const u32 mode : {0x00212800U, 0x0021f100U}) {
        for (const u32 modifiers : {0U, 1U, 2U, 3U}) {
            const Instruction f2i{
                {0x00157b05, 0x00400400 | (modifiers << 30), mode, 0x0001e400}};
            IrPools pools;
            const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, f2i, stg, exit});
            CHECK(HasOpcode(program, Shader::IR::Opcode::GetCbufF32));
            CHECK(HasOpcode(program, Shader::IR::Opcode::FPAbs32) == ((modifiers & 1) != 0));
            CHECK(HasOpcode(program, Shader::IR::Opcode::FPNeg32) == ((modifiers & 2) != 0));
        }
    }
}
