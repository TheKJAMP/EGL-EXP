// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/program_header.h"

namespace {
using Shader::Ampere::AnalyzeProgram;
using Shader::Ampere::Instruction;
namespace IR = Shader::IR;

constexpr u32 Sched = 0x000fe200;
constexpr Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction LoggedF2I{{0x00067b05, 0x40c03400, 0x00212800, 0x000e2200}};
constexpr std::array<Instruction, 2> LoggedShifts{{
    {{0xff107419, 0x000000ff, 0x0001060e, 0x080fe400}},
    {{0xff0c7419, 0x000000ff, 0x0001060b, 0x080fe400}},
}};

class LoggedEnvironment final : public Shader::Environment {
public:
    explicit LoggedEnvironment(Shader::Stage s = Shader::Stage::Compute) {
        stage = s;
        is_proprietary_driver = false;
        start_address = (s == Shader::Stage::Compute) ? sizeof(Shader::ProgramHeader) : 0;
        std::memset(&sph, 0, sizeof(sph));
    }

    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0x3f800000U; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 4096; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    bool IsNativeProgram() const override { return true; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

u32 EvaluateShift(IR::Value value) {
    value = value.Resolve();
    if (value.IsImmediate()) {
        if (value.Type() == IR::Type::F32) {
            return std::bit_cast<u32>(value.F32());
        }
        return value.Type() == IR::Type::U1 ? u32{value.U1()} : value.U32();
    }
    const auto& inst = *value.Inst();
    const auto arg = [&](size_t index) { return EvaluateShift(inst.Arg(index)); };
    switch (inst.GetOpcode()) {
    case IR::Opcode::Identity:
        return arg(0);
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::BitCastF32U32:
        return arg(0);
    case IR::Opcode::FPAdd32:
        return std::bit_cast<u32>(std::bit_cast<float>(arg(0)) + std::bit_cast<float>(arg(1)));
    case IR::Opcode::FPMul32:
        return std::bit_cast<u32>(std::bit_cast<float>(arg(0)) * std::bit_cast<float>(arg(1)));
    case IR::Opcode::FPFma32:
        return std::bit_cast<u32>(std::fma(std::bit_cast<float>(arg(0)),
                                           std::bit_cast<float>(arg(1)),
                                           std::bit_cast<float>(arg(2))));
    case IR::Opcode::BitwiseAnd32:
        return arg(0) & arg(1);
    case IR::Opcode::ULessThan:
        return arg(0) < arg(1);
    case IR::Opcode::SelectU32:
        return arg(0) ? arg(1) : arg(2);
    case IR::Opcode::ShiftLeftLogical32: {
        const u32 shift = arg(1);
        if (shift >= 32) {
            throw std::domain_error("Unmasked out-of-range shift reaches the result");
        }
        return arg(0) << shift;
    }
    default:
        throw std::logic_error(std::string{IR::NameOf(inst.GetOpcode())});
    }
}

struct IrPools {
    Shader::ObjectPool<IR::Inst> inst;
    Shader::ObjectPool<IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code,
                                    Shader::Stage stage = Shader::Stage::Compute) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    LoggedEnvironment env{stage};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::HostTranslateInfo host{};
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_float64 = true;
    host.min_ssbo_alignment = 16;
    auto program = Shader::Ampere::TranslateProgram(pools.inst, pools.block, flow_pool, env, code,
                                                    {}, analysis.uses_async_copy,
                                                    &analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(program);
    return program;
}

std::vector<u32> TranslateAndEmit(std::span<const Instruction> code,
                                  std::optional<u32> expected = std::nullopt) {
    IrPools pools;
    auto program = TranslateDirect(pools, code);
    u32 stores = 0;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::WriteSharedU32) {
                continue;
            }
            ++stores;
            if (expected) {
                INFO(IR::DumpProgram(program));
                CHECK(EvaluateShift(inst.Arg(1)) == *expected);
            }
        }
    }
    REQUIRE(stores == 1);
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010400;
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, program);
    REQUIRE(spirv.size() > 5);
    CHECK(spirv.front() == 0x07230203);
    return spirv;
}

Instruction MovImm(u32 dest, u32 imm) {
    return {{0x00007802U | (dest << 16), imm, 0x00000f00U, Sched}};
}

Instruction StoreShared(u32 source) {
    return {{0xff007388U, source, 0x800U, Sched}};
}
} // namespace

TEST_CASE("Logged Pokopia absolute constant F2I retains U16 rounding and reaches SPIRV",
          "[video_core][ampere][logged_numeric]") {
    TranslateAndEmit(std::array{LoggedF2I, StoreShared(6), Exit});
}

TEST_CASE("Logged Pokopia F2I preserves guards and rejects unrelated modifiers",
          "[video_core][ampere][logged_numeric]") {
    for (const u32 guard : {0U, 6U, 7U, 8U, 14U}) {
        auto instruction = LoggedF2I;
        instruction.words[0] = (instruction.words[0] & ~0xf000U) | (guard << 12);
        const auto analysis = AnalyzeProgram(std::array{instruction, Exit});
        REQUIRE(static_cast<bool>(analysis));
        IrPools pools;
        (void)TranslateDirect(pools, std::array{instruction, StoreShared(6), Exit});
    }
    for (const u32 source : {0x44c03400U, 0x40c03440U, 0x48c03400U, 0x50c03400U, 0x60c03400U}) {
        auto invalid = LoggedF2I;
        invalid.words[1] = source;
        CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    }
    for (const u32 bit : {0U, 8U, 30U, 31U}) {
        auto invalid = LoggedF2I;
        invalid.words[3] ^= 1U << bit;
        CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    }
    auto invalid = LoggedF2I;
    invalid.words[0] |= 1U << 24;
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    invalid = LoggedF2I;
    invalid.words[2] |= 1U << 27;
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    invalid.words = {0x00157b05, 0x00400440, 0x0021e100, 0x0001e400};
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
}

TEST_CASE("Logged Pokopia SHF clamps instead of wrapping through production translation",
          "[video_core][ampere][logged_numeric]") {
    for (const auto instruction : LoggedShifts) {
        for (const bool wrap : {false, true}) {
            auto variant = instruction;
            variant.words[2] |= u32{wrap} << 11;
            const u32 shift_reg = instruction.Bits(64, 8);
            const u32 dest_reg = instruction.Bits(16, 8);
            for (const u32 shift : {0U, 1U, 7U, 24U, 31U, 32U, 33U, 63U, 0xffffffffU}) {
                const u32 effective = wrap ? shift & 31 : std::min(shift, 32U);
                const u32 expected = effective == 32 ? 0U : 255U << effective;
                CAPTURE(shift, wrap);
                TranslateAndEmit(
                    std::array{MovImm(shift_reg, shift), variant, StoreShared(dest_reg), Exit},
                    expected);
            }
        }
    }
    for (const u32 bit : {8U, 12U, 17U, 27U}) {
        auto invalid = LoggedShifts[0];
        invalid.words[2] ^= 1U << bit;
        CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
    }
    auto invalid = LoggedShifts[0];
    invalid.words[0] &= ~0xff000000U;
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{invalid, Exit})));
}

TEST_CASE("TXD high-register coordinates and destinations validate natively",
          "[video_core][ampere][logged_numeric]") {
    const auto txd = [](u32 dest, u32 coordinates, u32 dimension, u32 mask) {
        return Instruction{{0xb6c | (dest << 16) | (coordinates << 24), (dimension << 29),
                            0x001e0000U | 255U | (mask << 8), 0}};
    };
    CHECK(static_cast<bool>(AnalyzeProgram(std::array{txd(247, 0, 1, 0x1), Exit})));
    CHECK(static_cast<bool>(AnalyzeProgram(std::array{txd(247, 0, 1, 0x3), Exit})));
    CHECK(static_cast<bool>(AnalyzeProgram(std::array{txd(0, 218, 5, 0x3), Exit})));
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{txd(255, 0, 1, 0x1), Exit})));
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{txd(0, 253, 5, 0x3), Exit})));
}

TEST_CASE("Captured Kirby high-register ALU and memory operands validate natively",
          "[video_core][ampere][logged_numeric][kirby]") {
    const std::array fixtures{
        Instruction{{0x8ee87221, 0x0000005a, 0x00010000, 0x000fe200}},
        Instruction{{0x21db7820, 0x42fe0000, 0x00410000, 0x000fc600}},
        Instruction{{0xdb507421, 0x3f800000, 0x00010000, 0x042fe200}},
        Instruction{{0xde4f7421, 0x3f800000, 0x00010000, 0x042fe200}},
        Instruction{{0x00f97919, 0x00000000, 0x00002100, 0x000e2200}},
        Instruction{{0xe3bd7823, 0x3c8efa35, 0x000100b1, 0x000fe400}},
        Instruction{{0xdc00780b, 0x42fe0000, 0x03f5e000, 0x000fc800}},
        Instruction{{0xdce57223, 0x0000008f, 0x0001004e, 0x000fe200}},
        Instruction{{0x21217208, 0x000000dc, 0x05000000, 0x000fe200}},
        Instruction{{0xfb2ec981, 0x00000006, 0x081ea900, 0x00012200}},
        Instruction{{0xe3e37981, 0x0000000c, 0x081e8900, 0x000f2200}},
        Instruction{{0xe4e47817, 0x00000001, 0x07800200, 0x000fe200}},
        Instruction{{0xfa00920b, 0x000000ff, 0x03f1c000, 0x000fe400}},
        Instruction{{0x00307245, 0x000000e4, 0x00201000, 0x000fe200}},
        Instruction{{0x39007986, 0x000000fa, 0x0810a906, 0x0001e200}},
        Instruction{{0x86000386, 0x000000fa, 0x0010a900, 0x0005e200}},
    };
    for (const auto& instruction : fixtures) {
        CAPTURE(instruction.words);
        const std::array code{instruction, Exit};
        const auto analysis = AnalyzeProgram(code);
        INFO(analysis.error);
        REQUIRE(static_cast<bool>(analysis));
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }
}

TEST_CASE("Captured Kirby high-register floating arithmetic preserves values",
          "[video_core][ampere][logged_numeric][kirby]") {
    struct Fixture {
        Instruction instruction;
        std::vector<u32> sources;
        float expected;
    };
    const std::array fixtures{
        Fixture{{{0x8ee87221, 0x0000005a, 0x00010000, 0x000fe200}}, {142, 90}, 4.0f},
        Fixture{{{0x21db7820, 0x42fe0000, 0x00410000, 0x000fc600}}, {33}, 254.0f},
        Fixture{{{0xdb507421, 0x3f800000, 0x00010000, 0x042fe200}}, {219}, 3.0f},
        Fixture{{{0xdce57223, 0x0000008f, 0x0001004e, 0x000fe200}}, {220, 143, 78}, 6.0f},
        Fixture{{{0xe3bd7823, 0x3c8efa35, 0x000100b1, 0x000fe400}}, {227, 177},
                2.0f * std::bit_cast<float>(0x3c8efa35U) + 2.0f},
    };
    for (const auto& fixture : fixtures) {
        CAPTURE(fixture.instruction.words);
        std::vector<Instruction> code;
        for (const u32 source : fixture.sources) {
            code.push_back(MovImm(source, 0x40000000U)); // 2.0f
        }
        code.push_back(fixture.instruction);
        code.push_back(StoreShared(fixture.instruction.Bits(16, 8)));
        code.push_back(Exit);
        TranslateAndEmit(code, std::bit_cast<u32>(fixture.expected));
    }
}

TEST_CASE("Logged Pokopia masked dependency barrier fences shared memory and rejects unknown modes",
          "[video_core][ampere][logged_numeric]") {
    constexpr Instruction wait{{0x0000791a, 0x0000800e, 0, 0x000fce00}};
    constexpr Instruction copy{{0x02067fae, 0, 0x0f901c08, 0}};
    for (const u32 mask : {0x800eU, 0x8000U}) {
        auto supported = wait;
        supported.words[1] = mask;
        IrPools pools;
        const auto copying = TranslateDirect(pools, std::array{copy, supported, Exit});
        bool has_barrier = false;
        for (const auto* b : copying.blocks) {
            for (const auto& inst : b->Instructions()) {
                has_barrier |= inst.GetOpcode() == IR::Opcode::WorkgroupMemoryBarrier;
            }
        }
        CHECK(has_barrier);
    }
    for (u32 bit = 15; bit < 32; ++bit) {
        auto invalid = wait;
        invalid.words[1] ^= 1U << bit;
        const auto rejected = AnalyzeProgram(std::array{invalid, Exit});
        REQUIRE_FALSE(static_cast<bool>(rejected));
        CHECK(rejected.error == "native dependency barrier mode");
        CHECK(rejected.error_offset == 0);
    }
    auto invalid = wait;
    invalid.words[2] = 1;
    const auto rejected = AnalyzeProgram(std::array{invalid, Exit});
    REQUIRE_FALSE(static_cast<bool>(rejected));
    CHECK(rejected.error == "native dependency barrier mode");
}

TEST_CASE("Ocarina mesh attribute stores read high registers directly",
          "[video_core][ampere][logged_numeric][oot]") {
    // Ocarina mesh 8159e91e0097876a at 0x83c0: ISBEWR.32.ATTR.SKEW.O [R197 + 0x780], R240.
    const Instruction store{{0xc5007927, 0x000780f0, 0x0000f800, 0x000fe200}};
    IrPools pools;
    const auto program = TranslateDirect(pools, std::array{store, Exit}, Shader::Stage::Mesh);
    bool has_isbe_write = false;
    for (const auto* b : program.blocks) {
        for (const auto& inst : b->Instructions()) {
            has_isbe_write |= inst.GetOpcode() == IR::Opcode::IsbeWriteAttribute;
        }
    }
    CHECK(has_isbe_write);
}
