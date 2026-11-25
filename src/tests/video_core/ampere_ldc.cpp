// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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

class LdcEnv final : public Shader::Environment {
public:
    LdcEnv() {
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

struct IrPools {
    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    LdcEnv env;
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

TEST_CASE("Ampere indexed constant loads retain narrow sizes", "[video_core][ampere]") {
    constexpr u32 Sched = 0x000fe200;
    const Instruction load{{0x02147b82, 0, 0, 0}}; // LDC R20, c[0x0][R2]
    const Instruction ptr0{{0xff047624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff057624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction idx_from_tid{{0x00027919, 0, 33U << 8, Sched}}; // S2R R2, SR_TID_X
    const Instruction stg20{{0x04007386, 20U, 0x0010a900U, Sched}};   // STG [R4], R20
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};

    REQUIRE(static_cast<bool>(AnalyzeProgram(std::array{load, exit})));

    constexpr std::array<Shader::IR::Opcode, 6> expected_ops{
        Shader::IR::Opcode::GetCbufU8,  Shader::IR::Opcode::GetCbufS8,
        Shader::IR::Opcode::GetCbufU16, Shader::IR::Opcode::GetCbufS16,
        Shader::IR::Opcode::GetCbufU32, Shader::IR::Opcode::GetCbufU32x2,
    };

    for (const u32 mode : {0U, 2U}) {
        for (u32 width = 0; width <= 5; ++width) {
            auto variant = load;
            variant.words[2] = (width << 9) | (mode << 14);
            IrPools pools;
            const auto program =
                TranslateDirect(pools, std::array{ptr0, ptr1, idx_from_tid, variant, stg20, exit});
            CHECK(HasOpcode(program, expected_ops[width]));
        }
    }

    auto unsupported = load;
    unsupported.words[2] = 6U << 9;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit})));
    unsupported.words[0] = 0x02157b82;
    unsupported.words[2] = 5U << 9;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit})));
    unsupported.words[2] = 1U << 14;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{unsupported, exit})));
}

TEST_CASE("Ampere uniform constant loads retain narrow sizes", "[video_core][ampere]") {
    constexpr u32 Sched = 0x000fe200;
    const Instruction ptr0{{0xff027624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff037624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction mov_r8_ur4{{0x00087c02, 4U, 0x08000f00U, Sched}};
    const Instruction stg8{{0x02007386, 8U, 0x0010a900U, Sched}};
    const Instruction exit{{0x0000794d, 0, 0x03800000, 0}};

    // Star Fox: ULDC.U16 UR4, c[6][0x20].
    Instruction load{{0x00047ab9, 0x01800800, 0x00000400, 0x000fe200}};
    constexpr std::array<Shader::IR::Opcode, 6> expected_ops{
        Shader::IR::Opcode::GetCbufU8,  Shader::IR::Opcode::GetCbufS8,
        Shader::IR::Opcode::GetCbufU16, Shader::IR::Opcode::GetCbufS16,
        Shader::IR::Opcode::GetCbufU32, Shader::IR::Opcode::GetCbufU32x2,
    };

    for (const u32 width : {0U, 1U, 2U, 3U, 4U, 5U}) {
        load.words[2] = width << 9;
        IrPools pools;
        const auto program =
            TranslateDirect(pools, std::array{ptr0, ptr1, load, mov_r8_ur4, stg8, exit});
        CHECK(HasOpcode(program, expected_ops[width]));
    }
    load.words[2] = 6U << 9;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{load, exit})));
    load.words[2] = 2U << 9;
    load.words[1] |= 1U << 6;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{load, exit})));
}
