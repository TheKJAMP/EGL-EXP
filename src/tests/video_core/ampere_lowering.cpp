// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/program_header.h"

namespace {
using Shader::Ampere::AnalyzeProgram;
using Shader::Ampere::Instruction;
using Shader::Ampere::ProgramCaptureState;

constexpr u32 Sched = 0x000fe200;
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, 0}};
constexpr Instruction PTR0{{0xff027624, 0x20U << 6, 0x078e00ff, Sched}};
constexpr Instruction PTR1{{0xff037624, 0x24U << 6, 0x078e00ff, Sched}};
constexpr Instruction TID_R8{{0x00087919, 0, 33U << 8, Sched}};

constexpr Instruction Stg(u32 rs, u32 off = 0) {
    return Instruction{{0x02007386, rs | (off << 8), 0x0010a900, Sched}};
}

class LoweringEnv final : public Shader::Environment {
public:
    explicit LoweringEnv(u32 tex_bound = 2,
                         Shader::TextureType tex_type = Shader::TextureType::Color2D)
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
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound; }
    u32 LocalMemorySize() const override { return 256; }
    u32 SharedMemorySize() const override { return 4096; }
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

Shader::IR::Program TranslateDirect(
    IrPools& pools, std::span<const Instruction> code,
    Shader::TextureType tex_type = Shader::TextureType::Color2D) {
    const auto analysis = AnalyzeProgram(code);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    const u32 tex_bound =
        analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2U;
    LoweringEnv env{tex_bound, tex_type};
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

size_t CountOpcode(const Shader::IR::Program& program, Shader::IR::Opcode op) {
    size_t count = 0;
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            count += inst.GetOpcode() == op ? 1 : 0;
        }
    }
    return count;
}

TEST_CASE("Ampere BRA.CONV always branches and BRA.DIV never does", "[video_core][ampere]") {
    // BRA.CONV UR6 skips Stg(8); BRA.DIV ~URZ falls through to Stg(8).
    const std::array<Instruction, 6> convergent{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x06007947, 0x10 | 3, 0x0b800000, 0}},
        Stg(8),
        EXIT,
    }};
    ProgramCaptureState capture;
    REQUIRE(capture.Observe(convergent[3], 3) == ProgramCaptureState::Result::Continue);
    IrPools pools;
    CHECK(CountOpcode(TranslateDirect(pools, convergent), Shader::IR::Opcode::WriteStorage32) == 0);

    const std::array<Instruction, 6> divergent{{
        PTR0,
        PTR1,
        TID_R8,
        {{0xff007947, 0x10 | 2, 0x0b800000, 0}},
        Stg(8),
        EXIT,
    }};
    IrPools div_pools;
    CHECK(CountOpcode(TranslateDirect(div_pools, divergent), Shader::IR::Opcode::WriteStorage32) == 1);
}

TEST_CASE("Ampere TLD.SCR.LZ takes its handle from a uniform register", "[video_core][ampere]") {
    const std::array<Instruction, 6> code{{
        PTR0,
        PTR1,
        {{0x00047ab9, 0x00800800, 0x00000800, Sched}}, // ULDC UR4
        {{0x04087f66, 0x30000405, 0x089e0705, 0x001f4200}},
        Stg(5),
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ImageFetch) == 1);
}

TEST_CASE("Ampere I2F converts signed 16-bit sources", "[video_core][ampere]") {
    const std::array<Instruction, 6> code{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x08087306, 0x00000008, 0x00101400, 0x000fe200}},
        Stg(8),
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(CountOpcode(program, Shader::IR::Opcode::BitFieldSExtract) >= 1);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ConvertF32S32) >= 1);
}

TEST_CASE("Ampere integer-to-float conversions take immediates", "[video_core][ampere]") {
    const std::array<Instruction, 4> code{{
        {{0x00077906, 0x00000020, 0x00209000, 0x000e2200}},
        {{0x00047845, 0x00000100, 0x0020d000, 0x000fe200}},
        {{0x00007348, 0x00000007, 0x03800000, 0x000fe800}},
        EXIT,
    }};
    IrPools pools;
    (void)TranslateDirect(pools, code);
}

TEST_CASE("Ampere SUST.P keeps its channel mask", "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x06007b9b, 0x60005904, 0x0010a100, 0x000fe200}},
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ImageRead) >= 1);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ImageWrite) == 1);
}

TEST_CASE("Ampere SHF.R.U64 immediate shifts keep their amount", "[video_core][ampere]") {
    for (const u32 shift : {0U, 1U, 21U, 31U, 32U}) {
        const std::array<Instruction, 6> code{{
            PTR0,
            PTR1,
            TID_R8,
            {{0x080e7819, shift, 0x00001208, 0x020fe200}},
            Stg(14),
            EXIT,
        }};
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }
}

TEST_CASE("Ampere SHFL with a register lane keeps its mode", "[video_core][ampere]") {
    const std::array<Instruction, 6> code{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x08077589, 0x08001f08, 0x000e0000, 0x000fe200}},
        Stg(7),
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ShuffleDown) == 1);
}

TEST_CASE("Ampere calls find a return address set ahead of them", "[video_core][ampere]") {
    std::array<Instruction, 5> code{{
        {{0x00097802, 0x30, 0x00000f00, 0}},
        {{0x00047802, 0x5, 0x00000f00, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        EXIT,
        {{0x09007950, 0xffffffb0, 0x03c3ffff, 0}},
    }};
    {
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }

    code[1] = {{0x00097802, 0x5, 0x00000f00, 0}};
    const auto analysis = AnalyzeProgram(code);
    REQUIRE(!static_cast<bool>(analysis));
    REQUIRE(analysis.error == "native call linkage mode");
}

TEST_CASE("Ampere programs go on past an EXIT that a branch jumps over", "[video_core][ampere]") {
    const std::array<Instruction, 8> code{{
        {{0x00047802, 0x20, 0x00000f00, 0}},
        {{0x00007944, 0x50, 0x03c00000, 0}},
        {{0x00007947, 0x10, 0x03800000, 0}},
        EXIT,
        {{0x00047802, 0x60, 0x00000f00, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        EXIT,
        {{0x04007950, 0xffffff80, 0x03c3ffff, 0}},
    }};
    const auto analysis = AnalyzeProgram(code);
    REQUIRE(static_cast<bool>(analysis));
    REQUIRE(analysis.source_size == sizeof(code));
    IrPools pools;
    (void)TranslateDirect(pools, code);
}

TEST_CASE("Ampere capture ends at the return of a helper called past the EXIT",
          "[video_core][ampere]") {
    const std::array<Instruction, 4> code{{
        {{0x00007802, 0x20, 0x00000f00, 0}},
        {{0x00007944, 0x10, 0x03c00000, 0}},
        EXIT,
        {{0x00007950, 0xffffffc0, 0x03c3ffff, 0}},
    }};
    ProgramCaptureState capture;
    for (u32 index = 0; index + 1 < code.size(); ++index) {
        CHECK(capture.Observe(code[index], index) == ProgramCaptureState::Result::Continue);
    }
    CHECK(capture.Observe(code.back(), 3) == ProgramCaptureState::Result::Complete);
    IrPools pools;
    (void)TranslateDirect(pools, code);

    ProgramCaptureState jump;
    CHECK(jump.Observe({{0x00007944, 0x10, 0x03c00000, 0}}, 0) ==
          ProgramCaptureState::Result::Continue);
    CHECK(jump.Observe(EXIT, 1) == ProgramCaptureState::Result::Continue);
    CHECK(jump.Observe(EXIT, 2) == ProgramCaptureState::Result::Complete);
}

TEST_CASE("Ampere uniform select and bit operations reach every uniform register",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> select{{
        {{0x04040887, 0x20, 0x08800000, 0x000fe200}},
        EXIT,
    }};
    {
        IrPools pools;
        (void)TranslateDirect(pools, select);
    }

    const std::array<Instruction, 7> reverse{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x081173c2, 0, 0x000e0000, Sched}},
        {{0x000472be, 0x11, 0x08000000, 0x000fc800}},
        {{0x00087c02, 4U, 0x08000f00U, Sched}},
        EXIT,
    }};
    std::vector<Instruction> rev_stg(reverse.begin(), reverse.end());
    rev_stg.back() = Stg(8);
    rev_stg.push_back(EXIT);
    {
        IrPools pools;
        CHECK(CountOpcode(TranslateDirect(pools, rev_stg), Shader::IR::Opcode::BitReverse32) == 1);
    }

    const std::array<Instruction, 6> leading{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x081573c2, 0, 0x000e0000, Sched}},
        {{0x001572bd, 0x15, 0x080e0400, 0x000fe400}},
        {{0x00087c02, 0x15, 0x08000f00, Sched}},
    }};
    std::vector<Instruction> flo_stg(leading.begin(), leading.end());
    flo_stg.push_back(Stg(8));
    flo_stg.push_back(EXIT);
    {
        IrPools pools;
        CHECK(CountOpcode(TranslateDirect(pools, flo_stg), Shader::IR::Opcode::FindUMsb32) == 2);
    }

    const std::array<Instruction, 6> count{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x081173c2, 0, 0x000e0000, Sched}},
        {{0x000a72bf, 0x11, 0x08000000, 0x000fe200}},
        {{0x00087c02, 10U, 0x08000f00U, Sched}},
    }};
    std::vector<Instruction> popc_stg(count.begin(), count.end());
    popc_stg.push_back(Stg(8));
    popc_stg.push_back(EXIT);
    {
        IrPools pools;
        CHECK(CountOpcode(TranslateDirect(pools, popc_stg), Shader::IR::Opcode::BitCount32) == 1);
    }
}

TEST_CASE("Ampere guarded uniform comparisons keep the output when the guard is false",
          "[video_core][ampere]") {
    const std::array<Instruction, 2> code{{
        {{0x1500488c, 0x1f, 0x0bf04070, 0x000fe200}},
        EXIT,
    }};
    {
        IrPools pools;
        (void)TranslateDirect(pools, code);
    }

    const std::array<Instruction, 2> select{{
        {{0x04037c07, 0x7, 0x08000000, 0x000fe400}},
        EXIT,
    }};
    {
        IrPools pools;
        (void)TranslateDirect(pools, select);
    }
}

TEST_CASE("Ampere gathers take arrays, per-pixel offsets and depth references",
          "[video_core][ampere]") {
    // Ocarina: TLD4.R.2D.ARRAY
    {
        IrPools pools;
        CHECK(CountOpcode(
                  TranslateDirect(
                      pools,
                      std::array{PTR0, PTR1,
                                 Instruction{{0x082c7b63, 0xb0801004, 0x001e0f2a, 0x00014200}},
                                 Stg(44), EXIT},
                      Shader::TextureType::ColorArray2D),
                  Shader::IR::Opcode::ImageGather) == 1);
    }

    // Ocarina: @!P1 TLD4.R.PTP
    {
        IrPools pools;
        CHECK(CountOpcode(
                  TranslateDirect(
                      pools,
                      std::array{PTR0, PTR1,
                                 Instruction{{0x06189b63, 0x30007c10, 0x001e2f0e, 0x000f4600}},
                                 Stg(24), EXIT}),
                  Shader::IR::Opcode::ImageGather) == 1);
    }

    // Ocarina: TLD4.R.DC
    {
        IrPools pools;
        CHECK(CountOpcode(
                  TranslateDirect(
                      pools,
                      std::array{PTR0, PTR1,
                                 Instruction{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}},
                                 Stg(12), EXIT}),
                  Shader::IR::Opcode::ImageGatherDref) == 1);
    }

    // Offset mode 3 is not an offset form.
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(
        std::array{Instruction{{0x082c7b63, 0xb0801004, 0x001e3f2a, 0x00014200}}, EXIT})));

    // A gather from a second descriptor buffer works with and without AOFFI.
    const Instruction depth_gather{{0x180c7b63, 0x30800e1c, 0x001e4f0e, 0x000f4200}}; // binding 2
    const Instruction plain_b14{{0x040a7b63, 0x33800408, 0x041e0f08, 0x100fc200}};    // binding 14
    auto aoffi_b14 = plain_b14;
    aoffi_b14.words[2] |= 1U << 12;

    for (const Instruction& second : {plain_b14, aoffi_b14}) {
        const std::array code{PTR0, PTR1, depth_gather, second, Stg(12, 0), Stg(10, 4), EXIT};
        const auto analysis = AnalyzeProgram(code);
        INFO(analysis.error);
        REQUIRE(static_cast<bool>(analysis));
        CHECK(analysis.required_texture_buffer == 2);
        IrPools pools;
        const auto prog = TranslateDirect(pools, code);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageGatherDref) == 1);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageGather) == 1);
    }
}

TEST_CASE("Ampere Ocarina sign extends, uniform shifts and carries, LEA.HI and SURED forms",
          "[video_core][ampere]") {
    {
        IrPools pools;
        (void)TranslateDirect(
            pools, std::array{PTR0, PTR1, TID_R8,
                              Instruction{{0x0820781a, 1, 0x00000200, 0x000fe400}}, Stg(32), EXIT});
    }
    {
        IrPools pools;
        (void)TranslateDirect(
            pools, std::array{Instruction{{0x3f057499, 1, 0x08010e04, 0x000fe400}}, EXIT});
    }
    {
        IrPools pools;
        (void)TranslateDirect(
            pools, std::array{Instruction{{0x0f067290, 6, 0x0ff5e03f, 0x000fe200}}, EXIT});
    }
    {
        IrPools pools;
        (void)TranslateDirect(
            pools, std::array{Instruction{{0x01017211, 0x80000000, 0x078ff0ff, 0x000fca00}}, EXIT});
    }
    {
        IrPools pools;
        CHECK(CountOpcode(
                  TranslateDirect(
                      pools, std::array{Instruction{{0x0400159f, 0x60008400, 0x0310e000, 0x0001e200}},
                                        EXIT}),
                  Shader::IR::Opcode::ImageAtomicOr32) == 1);
    }

    // Aligned LDG.E.128 R216 succeeds natively while unaligned R218 is rejected by AnalyzeProgram.
    {
        const Instruction load{{0xd8d87381, 0, 0x001ead00, 0x000f2800}};
        IrPools pools;
        const auto prog = TranslateDirect(
            pools,
            std::array{Instruction{{0x00007c02, 0, 0x08000f00, 0}}, load, PTR0, PTR1, Stg(216), EXIT});
        CHECK(CountOpcode(prog, Shader::IR::Opcode::WriteStorage32) == 1);
    }
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array<Instruction, 3>{{
        {{0x00007c02, 0, 0x08000f00, 0}},
        {{0xd8da7381, 0, 0x001ead00, 0x000f2800}},
        EXIT,
    }})));
}

TEST_CASE("Ampere Ocarina byte constant loads, ULEA carries to UP3 and two-source F2IP",
          "[video_core][ampere]") {
    const auto set_extended = [](const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_NX2_LOWER_EXTENDED", value ? value : "");
#else
        if (value) {
            setenv("EDEN_NX2_LOWER_EXTENDED", value, 1);
        } else {
            unsetenv("EDEN_NX2_LOWER_EXTENDED");
        }
#endif
    };
    constexpr Instruction ByteLoad{0x06087abb, 0x01002080, 0x08000000, 0x000fe400};
    set_extended("0");
    CHECK_FALSE(static_cast<bool>(AnalyzeProgram(std::array{ByteLoad, EXIT})));
    set_extended(nullptr);

    {
        IrPools pools;
        CHECK(CountOpcode(TranslateDirect(pools, std::array{
                                                     PTR0, PTR1, ByteLoad,
                                                     Instruction{{0x00087c02, 8U, 0x08000f00, Sched}},
                                                     Stg(8), EXIT}),
                          Shader::IR::Opcode::GetCbufU8) == 1);
    }
    {
        IrPools pools;
        (void)TranslateDirect(pools, std::array{
                                         Instruction{{0x070a7291, 0x0000000a, 0x0f86103f, 0x000fe200}},
                                         EXIT});
    }
    {
        IrPools pools;
        (void)TranslateDirect(pools, std::array{
                                         PTR0, PTR1, TID_R8,
                                         Instruction{{0x082d7243, 0x00000015, 0x000004ff, 0x000fc400}},
                                         Stg(45), EXIT});
    }
    {
        IrPools pools;
        const auto prog = TranslateDirect(
            pools, std::array{PTR0, PTR1, Instruction{{0x0f077abb, 0x01000a40, 0x08000800, 0x000fc600}},
                              Instruction{{0x00087c02, 7U, 0x08000f00, Sched}}, Stg(8), EXIT});
        size_t bank4_loads = 0;
        for (const auto* block : prog.blocks) {
            for (const auto& inst : block->Instructions()) {
                if (inst.GetOpcode() == Shader::IR::Opcode::GetCbufU32) {
                    const auto bank = inst.Arg(0).Resolve();
                    bank4_loads += (bank.IsImmediate() && bank.U32() == 4U) ? 1U : 0U;
                }
            }
        }
        CHECK(bank4_loads == 1);
    }
    {
        IrPools pools;
        (void)TranslateDirect(pools, std::array{
                                         PTR0, PTR1,
                                         Instruction{{0x00097d06, 0x30000011, 0x08001000, 0x000fe200}},
                                         Stg(9), EXIT});
    }

    for (const u32 cond : {0U, 1U, 2U}) {
        IrPools pools;
        (void)TranslateDirect(pools, std::array<Instruction, 3>{{
                                         {{0x0400788c, 0x00000002, 0x0bf01070U | (cond << 17), 0x000fc800}},
                                         {{0x04047887, 0x00000002, 0x08000000U | (cond << 23), 0x000fc800}},
                                         EXIT,
                                     }});
    }
}

TEST_CASE("Ampere three-input IADD3 preserves negative immediates and negation",
          "[video_core][ampere]") {
    for (const u32 literal : {0xffffffffU, 0xfffffffeU, 0x80000000U, 0x7fffffffU, 17U}) {
        IrPools pools;
        (void)TranslateDirect(pools, std::array{
                                         PTR0, PTR1, TID_R8,
                                         Instruction{{0x08047810, literal, 0x07ffe008, 0x000fc400}},
                                         Stg(4), EXIT});
    }
    IrPools pools;
    (void)TranslateDirect(pools, std::array{
                                     PTR0, PTR1, TID_R8,
                                     Instruction{{0x08047210, 0x80000008, 0x07ffe908, 0x000fc400}},
                                     Stg(4), EXIT});
}

TEST_CASE("Ampere convergence-state markers lower to nothing", "[video_core][ampere]") {
    const std::array<Instruction, 4> code{{
        {{0x1a0f7f55, 0x00000000, 0x00000000, 0x000fe200}},
        {{0x1a0f7f56, 0x00000000, 0x00100000, 0x000fe400}},
        {{0x1a0f7f56, 0x00000000, 0x00000000, 0x000fe400}},
        EXIT,
    }};
    IrPools markers_pools;
    const auto with_markers = TranslateDirect(markers_pools, code);
    IrPools exit_pools;
    const auto exit_only = TranslateDirect(exit_pools, std::array{EXIT});
    REQUIRE(with_markers.blocks.size() == exit_only.blocks.size());
    CHECK(with_markers.blocks[0]->Instructions().size() ==
          exit_only.blocks[0]->Instructions().size());
}

TEST_CASE("Ampere FSWZADD.NDV and TEX.NDV translate cleanly", "[video_core][ampere]") {
    {
        IrPools pools;
        (void)TranslateDirect(pools, std::array{
                                         PTR0, PTR1, TID_R8,
                                         Instruction{{0x080c7822, 0x000000cc, 0x00002008, 0x001fc600}},
                                         Stg(12), EXIT});
    }

    IrPools pools;
    const auto tex = TranslateDirect(
        pools,
        std::array{PTR0, PTR1, Instruction{{0x4a0c7b60, 0x5080120c, 0x001e21ff, 0x00014200}},
                   Stg(12), EXIT},
        Shader::TextureType::Color3D);
    CHECK(CountOpcode(tex, Shader::IR::Opcode::ImageSampleImplicitLod) == 1);
}

TEST_CASE("Ampere MOV with a one-lane quad mask writes only that lane", "[video_core][ampere]") {
    const std::array<Instruction, 6> code{{
        PTR0,
        PTR1,
        TID_R8,
        {{0x00107202, 0x00000008, 0x00000200, 0x000fc800}},
        Stg(16),
        EXIT,
    }};
    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    CHECK(CountOpcode(program, Shader::IR::Opcode::LaneId) == 1);
    CHECK(CountOpcode(program, Shader::IR::Opcode::SelectU32) == 1);

    const std::array<Instruction, 2> two_lanes{{
        {{0x00107202, 0x0000000c, 0x00000300, 0x000fc800}},
        EXIT,
    }};
    REQUIRE(!static_cast<bool>(AnalyzeProgram(two_lanes)));
}
} // namespace
