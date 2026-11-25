// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
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
using Shader::Ampere::LoweringOptions;

constexpr u32 Sched = 0x000fe200;
constexpr Instruction exit_inst{{0x0000794d, 0, 0x03800000, 0}};

constexpr std::array<Instruction, 5> geometry_program{{
    {{0x0004734e, 0, 0, 0}},         // LEPC R4
    {{0x00017923, 0, 0, 0}},         // ISBERD R1, [R0]
    {{0x00027324, 0xff, 0x4000, 0}}, // OUT.EMIT R2, R0, RZ
    {{0x02037924, 0, 0x8000, 0}},    // OUT.CUT R3, R2, 0
    exit_inst,
}};

class GeomEnv final : public Shader::Environment {
public:
    explicit GeomEnv(Shader::Stage s = Shader::Stage::Geometry) {
        stage = s;
        is_proprietary_driver = true;
        start_address = (s == Shader::Stage::Compute) ? sizeof(Shader::ProgramHeader) : 0;
        std::memset(&sph, 0, sizeof(sph));
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
    std::array<u32, 3> WorkgroupSize() const override { return {1, 1, 1}; }
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

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code,
                                    const LoweringOptions& options = {},
                                    Shader::Stage stage = Shader::Stage::Geometry) {
    const auto analysis = AnalyzeProgram(code, options);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    GeomEnv env{stage};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::HostTranslateInfo host{};
    host.min_ssbo_alignment = 16;
    auto program = Shader::Ampere::TranslateProgram(pools.inst, pools.block, flow_pool, env, code,
                                                    options, analysis.uses_async_copy,
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

TEST_CASE("Ampere geometry program counter uses the native address", "[video_core][ampere]") {
    const Instruction nop{{0x00007918, 0, 0, 0}};
    const Instruction lepc{{0x0014734e, 0, 0, 0}};
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction stg20{{0x10007386, 20U, 0x0010a900U, Sched}};
    const Instruction stg21{{0x10007386, 21U | (4U << 8), 0x0010a900U, Sched}};
    const std::array code{nop, lepc, ptr0, ptr1, stg20, stg21, exit_inst};

    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{nop, lepc, exit_inst})));
    IrPools pools;
    const auto program = TranslateDirect(
        pools, code, LoweringOptions{.geometry = true, .native_code_address = 0x00000012fffffff0ULL});

    std::vector<u32> stored;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() == Shader::IR::Opcode::WriteGlobal32) {
                stored.push_back(inst.Arg(1).Resolve().U32());
            } else if (inst.GetOpcode() == Shader::IR::Opcode::WriteStorage32) {
                stored.push_back(inst.Arg(2).Resolve().U32());
            }
        }
    }
    REQUIRE(stored.size() == 2);
    CHECK(stored[0] == 0x0U);
    CHECK(stored[1] == 0x13U);

    Instruction unsupported = lepc;
    unsupported.words[1] = 1;
    REQUIRE_FALSE(
        static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit_inst}, {.geometry = true})));
}

TEST_CASE("Ampere geometry stage map and exit setup are stage specific", "[video_core][ampere]") {
    const Instruction map_read{{0x05127923, 0, 0, 0}};
    const Instruction exit_setup{{0x03007357, 0, 0, 0}};
    const std::array code{map_read, exit_setup, exit_inst};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(code)));
    IrPools pools;
    const auto program = TranslateDirect(pools, code, {.geometry = true});
    CHECK_FALSE(program.blocks.empty());

    Instruction unsupported = exit_setup;
    unsupported.words[1] = 1;
    REQUIRE_FALSE(
        static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit_inst}, {.geometry = true})));
}

TEST_CASE("Ampere F2I supports 16-bit truncation", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction load_f{{0x00097919, 0, 33U << 8, Sched}};
    const Instruction stg{{0x10007386, 6U, 0x0010a900U, Sched}};
    for (const bool is_signed : {false, true}) {
        const Instruction convert{{0x00067305, 9, 0x0021e800U | (is_signed ? 0x100U : 0), 0}};
        IrPools pools;
        const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, load_f, convert, stg, exit_inst},
                                             {}, Shader::Stage::Compute);
        CHECK(HasOpcode(program, Shader::IR::Opcode::FPTrunc32));
        CHECK(HasOpcode(program, is_signed ? Shader::IR::Opcode::ConvertS32F32
                                           : Shader::IR::Opcode::ConvertU32F32));
    }
    const Instruction unsupported{{0x00067305, 9, 0x0021e801, 0}};
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit_inst})));
}

TEST_CASE("Ampere geometry programs translate their output instructions", "[video_core][ampere]") {
    IrPools pools;
    const auto program = TranslateDirect(pools, geometry_program, {.geometry = true});
    CHECK(HasOpcode(program, Shader::IR::Opcode::EmitVertex));
    CHECK(HasOpcode(program, Shader::IR::Opcode::EndPrimitive));
}

TEST_CASE("Ampere output instructions stay rejected outside geometry programs",
          "[video_core][ampere]") {
    const auto analysis = AnalyzeProgram(geometry_program);
    REQUIRE(!static_cast<bool>(analysis));
    REQUIRE(analysis.error == "unsupported native opcode");
}
