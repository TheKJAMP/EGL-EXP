// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/program_header.h"

namespace {
using Shader::Ampere::AnalyzeProgram;
using Shader::Ampere::Instruction;

constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};

class UleaEnv final : public Shader::Environment {
public:
    UleaEnv() {
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

TEST_CASE("Ampere guarded ULEA translates natively with and without carry out",
          "[video_core][ampere]") {
    // @!UP5 ULEA UR4, UR6, 0x1, 0x2 and the same on UR5, from Pokemon Pokopia's fullscreen
    // pixel shaders (the black sky).
    for (const Instruction lea : {Instruction{{0x0604d891, 0x1, 0x0f8e103f, 0x000fca00}},
                                  Instruction{{0x0405d891, 0x1, 0x0f8e103f, 0x000fe200}}}) {
        auto with_carry = lea;
        with_carry.words[2] &= ~(7U << 17); // write carry out predicate directly in SSA IR
        for (const Instruction variant : {lea, with_carry}) {
            const std::array code{variant, Exit};
            const auto analysis = AnalyzeProgram(code);
            INFO(analysis.error);
            REQUIRE(static_cast<bool>(analysis));

            UleaEnv env;
            Shader::ObjectPool<Shader::IR::Inst> inst_pool;
            Shader::ObjectPool<Shader::IR::Block> block_pool;
            Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
            Shader::HostTranslateInfo host{};
            host.min_ssbo_alignment = 16;
            auto program = Shader::Ampere::TranslateProgram(
                inst_pool, block_pool, flow_pool, env, code, {}, analysis.uses_async_copy,
                &analysis.branch_targets, host);
            Shader::Optimization::VerificationPass(program);
        }
    }
}
} // namespace
