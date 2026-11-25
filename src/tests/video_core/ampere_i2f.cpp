// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction PTR0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
constexpr Instruction PTR1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
constexpr Instruction TID_R1{{0x00017919, 0, 33U << 8, Sched}};
constexpr Instruction TID_R3{{0x00037919, 0, 33U << 8, Sched}};
constexpr Instruction R2UR_4{{0x010473c2, 0, 0x000e0000, Sched}};

class I2fEnv final : public Shader::Environment {
public:
    I2fEnv() {
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
    I2fEnv env;
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

constexpr Instruction StoreAt(u32 reg, u32 offset = 0) {
    return Instruction{{0x10007386, reg | (offset << 8), 0x0010a900, Sched}};
}

TEST_CASE("Ampere I2F.F32.S16 emits signed 16-bit conversion", "[video_core][ampere]") {
    const std::array<Instruction, 8> code{{
        PTR0,
        PTR1,
        TID_R1,
        TID_R3,
        {{0x00017306, 0x00000001, 0x00101400, 0x020ea200}},
        {{0x00037306, 0x00000003, 0x00101400, 0x000e2400}},
        StoreAt(1, 0),
        StoreAt(3, 4),
    }};
    std::vector<Instruction> with_exit(code.begin(), code.end());
    with_exit.push_back(EXIT);
    IrPools pools;
    const auto program = TranslateDirect(pools, with_exit);
    CHECK(HasOpcode(program, Shader::IR::Opcode::BitFieldSExtract));
    CHECK(HasOpcode(program, Shader::IR::Opcode::ConvertF32S32));
}

TEST_CASE("Ampere I2F keeps unsigned 16-bit sources unsigned", "[video_core][ampere]") {
    const std::array<Instruction, 6> code{{
        PTR0,
        PTR1,
        TID_R1,
        {{0x00017306, 0x00000001, 0x00101000, 0}},
        StoreAt(1, 0),
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(HasOpcode(program, Shader::IR::Opcode::BitFieldUExtract));
    CHECK(HasOpcode(program, Shader::IR::Opcode::ConvertF32U32));
}

TEST_CASE("Ampere I2F converts 16-bit uniform sources", "[video_core][ampere]") {
    // Star Fox's graphics shader: I2F.F32.U16 R2, UR4.
    std::array<Instruction, 7> code{{
        PTR0,
        PTR1,
        TID_R1,
        R2UR_4,
        {{0x00027d06, 0x00000004, 0x08101000, 0x000e6200}},
        StoreAt(2, 0),
        EXIT,
    }};
    IrPools lower_pools;
    const auto unsigned_lower = TranslateDirect(lower_pools, code);
    CHECK(HasOpcode(unsigned_lower, Shader::IR::Opcode::BitFieldUExtract));
    CHECK(HasOpcode(unsigned_lower, Shader::IR::Opcode::ConvertF32U32));

    // Upper half selector (bits 60..61 == 1)
    Instruction upper_inst{{0x00047d06, 0x10000004, 0x08101000, 0x000ef000}};
    code[4] = upper_inst;
    code[5] = StoreAt(4, 0);
    IrPools upper_pools;
    const auto upper_half = TranslateDirect(upper_pools, code);
    CHECK(HasOpcode(upper_half, Shader::IR::Opcode::BitFieldUExtract));
    CHECK(HasOpcode(upper_half, Shader::IR::Opcode::ConvertF32U32));

    // Selector 2 out of range for 16-bit source
    upper_inst.words[1] = 0x20000004;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{upper_inst, EXIT})));

    // Signed 16-bit uniform source
    Instruction signed_inst{{0x00027d06, 0x00000004, 0x08101400, 0x000e6200}};
    code[4] = signed_inst;
    code[5] = StoreAt(2, 0);
    IrPools signed_pools;
    const auto signed_prog = TranslateDirect(signed_pools, code);
    CHECK(HasOpcode(signed_prog, Shader::IR::Opcode::BitFieldSExtract));
    CHECK(HasOpcode(signed_prog, Shader::IR::Opcode::ConvertF32S32));

    // UR64 out of range
    signed_inst.words[1] = 64;
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{signed_inst, EXIT})));
}

} // namespace
