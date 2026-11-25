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

constexpr u32 Sched = 0x000fe200;
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};

class GradientEnv final : public Shader::Environment {
public:
    explicit GradientEnv(u32 tex_bound = 2, Shader::TextureType tex_type = Shader::TextureType::Color2D)
        : texture_bound{tex_bound}, texture_type{tex_type} {
        stage = Shader::Stage::Compute;
        is_proprietary_driver = true;
        start_address = sizeof(Shader::ProgramHeader);
    }
    u64 ReadInstruction(u32) override { return 0; }
    bool IsNativeProgram() const override { return true; }
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override {
        return 0x101 + cbuf_index * 16 + cbuf_offset;
    }
    Shader::TextureType ReadTextureType(u32) override { return texture_type; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::R32_UINT;
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
    Shader::TextureType texture_type;
};

struct IrPools {
    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code,
                                    Shader::TextureType tex_type = Shader::TextureType::Color2D) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    const u32 tex_bound =
        analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2U;
    GradientEnv env{tex_bound, tex_type};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::HostTranslateInfo host{};
    host.support_float16 = true;
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

TEST_CASE("Ampere surface reductions preserve 3D coordinates", "[video_core][ampere]") {
    // Star Fox: atomic unsigned add of R3 to the 3D image at c[0][0x88], xyz in R0-R2.
    Instruction sured{{0x0000759f, 0xa0002203, 0x0010e000, 0x0011e600}};
    for (const u32 dimension : {3U, 5U}) {
        sured.words[1] = (sured.words[1] & 0x1fffffffU) | (dimension << 29);
        const std::array code{sured, EXIT};
        const auto analysis = AnalyzeProgram(code);
        REQUIRE(static_cast<bool>(analysis));
        REQUIRE(analysis.required_texture_buffer == 0);
        IrPools pools;
        const auto program = TranslateDirect(
            pools, code,
            dimension == 3U ? Shader::TextureType::Color3D : Shader::TextureType::ColorArray2D);
        CHECK(HasOpcode(program, Shader::IR::Opcode::ImageAtomicIAdd32));
    }
    sured.words[0] = 0xfd00759f; // R253-R255 would cross into the zero register in 3D
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{sured, EXIT})));

    sured.words[0] = 0x0a00859f; // xyz in R10-R12, guarded by !P0
    {
        IrPools pools;
        REQUIRE(HasOpcode(
            TranslateDirect(pools, std::array{sured, EXIT}, Shader::TextureType::ColorArray2D),
            Shader::IR::Opcode::ImageAtomicIAdd32));
    }

    sured = {{0x0000759f, 0xa0002403, 0x0310e000, 0x000fe200}};
    {
        IrPools pools;
        REQUIRE(HasOpcode(
            TranslateDirect(pools, std::array{sured, EXIT}, Shader::TextureType::ColorArray2D),
            Shader::IR::Opcode::ImageAtomicOr32));
    }

    sured.words[2] = 0x0010e000 | (3U << 23); // unimplemented atomic operation must still be rejected
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{sured, EXIT})));
}

TEST_CASE("Ampere surface atomic add returns the old 3D texel value", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    Instruction atomic{{0x00087593, 0xa0002407, 0x001ee000, 0x000ea400}};
    const Instruction stg{{0x10007386, 8U, 0x0010a900U, Sched}};
    const std::array code{ptr0, ptr1, atomic, stg, EXIT};
    const auto analysis = AnalyzeProgram(code);
    REQUIRE(static_cast<bool>(analysis));
    REQUIRE(analysis.required_texture_buffer == 0);
    IrPools pools;
    const auto program = TranslateDirect(pools, code, Shader::TextureType::ColorArray2D);
    CHECK(HasOpcode(program, Shader::IR::Opcode::ImageAtomicIAdd32));
    CHECK(HasOpcode(program, Shader::IR::Opcode::WriteStorage32));

    atomic.words[2] &= ~(7U << 17); // writing a fault predicate is not yet supported
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{atomic, EXIT})));
}

TEST_CASE("Ampere gradient textures accept uniform array handles", "[video_core][ampere]") {
    const Instruction ptr0{{0xff207624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff217624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction uldc4{{0x00047ab9, 0x00800800, 0x00000800, Sched}};
    // Star Fox's material shader: uniform handle UR4, coordinates R20, derivatives R16.
    Instruction txd{{0x14127f6c, 0xa0000410, 0x081e0f10, 0x001f4200}};
    const Instruction stg{{0x20007386, 18U, 0x0010a900U, Sched}};

    const std::array code{ptr0, ptr1, uldc4, txd, stg, EXIT};
    const auto analysis = AnalyzeProgram(code);
    REQUIRE(static_cast<bool>(analysis));
    REQUIRE(analysis.required_texture_buffer == ~0U);
    IrPools pools;
    const auto program = TranslateDirect(pools, code, Shader::TextureType::ColorArray2D);
    CHECK(HasOpcode(program, Shader::IR::Opcode::ImageGradient));

    txd = {{0x18320f6c, 0xa000041c, 0x0c1e0f34, 0x00414200}}; // @P0 guarded, UR12 handle
    const Instruction uldc12{{0x000c7ab9, 0x00800800, 0x00000800, Sched}};
    const Instruction lane{{0x00017919, 0, 0U << 8, Sched}};
    const Instruction p0{{0x0100780c, 0, 0x03f02070, Sched}};
    const Instruction stg_guarded{{0x20007386, 50U, 0x0010a900U, Sched}};
    IrPools guarded_pools;
    const auto guarded = TranslateDirect(
        guarded_pools, std::array{ptr0, ptr1, lane, p0, uldc12, txd, stg_guarded, EXIT},
        Shader::TextureType::ColorArray2D);
    CHECK(HasOpcode(guarded, Shader::IR::Opcode::ImageGradient));

    txd.words[1] = 0xa0004020; // invalid uniform handle
    REQUIRE_FALSE(static_cast<bool>(AnalyzeProgram(std::array{txd, EXIT})));
}

TEST_CASE("Ampere array gathers preserve the layer and translate natively", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction stg{{0x10007386, 44U, 0x0010a900U, Sched}};
    const std::array<Instruction, 5> code{{
        ptr0,
        ptr1,
        {{0x082c7b63, 0xb0801004, 0x001e0f2a, 0x00014200}},
        stg,
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code, Shader::TextureType::ColorArray2D);
    CHECK(HasOpcode(program, Shader::IR::Opcode::ImageGather));
}

TEST_CASE("Ampere gathers support high-register coordinate pairs natively", "[video_core][ampere]") {
    struct Mode {
        Instruction instruction;
        bool pair;
    };
    const std::array modes{
        Mode{{{0x201c7b63, 0xb0000a2b, 0x041e0f16, 0x000f4200}}, true},
        Mode{{{0xd61c7f63, 0xb000002b, 0x0c1e0f16, 0x000f4200}}, true},
        Mode{{{0x201c7b63, 0x30000a2b, 0x041e1f16, 0x000f4200}}, true},
        Mode{{{0x201c7b63, 0x30000a2b, 0x041e0f16, 0x000f4200}}, false},
    };
    for (const auto& mode : modes) {
        for (const u32 source : {214U, 215U, 242U, 243U, 253U, 254U, 255U}) {
            CAPTURE(mode.pair, source);
            auto gather = mode.instruction;
            gather.words[0] = (gather.words[0] & 0x00ffffffU) | (source << 24);
            const std::array code{gather, EXIT};
            const auto analysis = AnalyzeProgram(code);
            if (mode.pair && source == 254U) {
                REQUIRE_FALSE(static_cast<bool>(analysis));
                continue;
            }
            INFO(analysis.error);
            REQUIRE(static_cast<bool>(analysis));
            IrPools pools;
            (void)TranslateDirect(pools, code, Shader::TextureType::ColorArray2D);
        }
    }
}

TEST_CASE("Ampere signed high-word shifts keep sign extension", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x00257919, 0, 33U << 8, Sched}};
    const Instruction shf{{0xff247219, 0x00000025, 0x00011444, 0x000fc600}};
    const Instruction stg{{0x10007386, 36U, 0x0010a900U, Sched}};
    IrPools pools;
    const auto program = TranslateDirect(pools, std::array{ptr0, ptr1, s2r, shf, stg, EXIT});
    CHECK(HasOpcode(program, Shader::IR::Opcode::ShiftRightArithmetic32));
}

TEST_CASE("Ampere uniform multiply accepts an immediate addend", "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x00017919, 0, 33U << 8, Sched}};
    const Instruction r2ur{{0x010573c2, 0, 0x000e0000, Sched}}; // UR5 = R1
    const Instruction uimad{{0x040474a4, 0x00000003, 0x0f8e0205, 0x000fe200}};
    const Instruction copy{{0x00087c02, 4U, 0x08000f00U, Sched}};
    const Instruction stg{{0x10007386, 8U, 0x0010a900U, Sched}};
    IrPools pools;
    const auto program =
        TranslateDirect(pools, std::array{ptr0, ptr1, s2r, r2ur, uimad, copy, stg, EXIT});
    CHECK(HasOpcode(program, Shader::IR::Opcode::IMul32));
}

TEST_CASE("Ampere dual predicate LUTs translate natively and preserve their guard",
          "[video_core][ampere]") {
    const Instruction ptr0{{0xff107624, 0x20U << 6, 0x078e00ff, Sched}};
    const Instruction ptr1{{0xff117624, 0x24U << 6, 0x078e00ff, Sched}};
    const Instruction s2r{{0x00017919, 0, 0U << 8, Sched}};
    const Instruction p0{{0x0100780c, 0, 0x03f02070, Sched}};
    const Instruction p1{{0x0100780c, 1, 0x03f02070 | (1U << 17), Sched}};
    const Instruction sel0{{0x00047807, 1, 0x00000000U, Sched}};       // SEL R4, R0, 1, P0
    const Instruction sel1{{0x00057807, 2, 0x00800000U, Sched}};       // SEL R5, R0, 2, P1
    const Instruction add{{0x04047210, 5, 0x07ffe0ff, Sched}};         // IADD3 R4, R4, R5, RZ
    const Instruction stg{{0x10007386, 4U, 0x0010a900U, Sched}};
    for (const u32 guard : {0U, 7U, 8U}) {
        const Instruction plop3{{0x00c8081c | (guard << 12), 0, 0x0010f010, 0x000fc400}};
        IrPools pools;
        const auto program = TranslateDirect(
            pools, std::array{ptr0, ptr1, s2r, p0, p1, plop3, sel0, sel1, add, stg, EXIT});
        CHECK(HasOpcode(program, Shader::IR::Opcode::WriteStorage32));
    }
}
