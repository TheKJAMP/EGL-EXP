// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate/translate.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/pred.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/program_header.h"
#include "video_core/nvn/shader/nvn_program_map.h"

namespace {

using Shader::Ampere::AnalyzeProgram;
using Shader::Ampere::Instruction;
using Shader::Ampere::LoweringOptions;
using Shader::Ampere::TranslateProgram;

constexpr u32 Sched = 0x000fe200;
constexpr Instruction EXIT{{0x0000794d, 0, 0x03800000, Sched}};
constexpr Instruction NOP{{0x00007918, 0, 0, Sched}};

class TestEnv : public Shader::Environment {
public:
    explicit TestEnv(std::vector<u32> c1_words = {}, u32 tex_bound = 2,
                     Shader::Stage s = Shader::Stage::Compute)
        : c1{std::move(c1_words)}, texture_bound{tex_bound} {
        stage = s;
        is_proprietary_driver = true;
        start_address = (s == Shader::Stage::Compute) ? sizeof(Shader::ProgramHeader) : 0;
        std::memset(&sph, 0, sizeof(sph));
        if (s == Shader::Stage::Fragment) {
            sph.ps.omap.target = 0xf;
        }
    }

    u64 ReadInstruction(u32) override {
        return 0;
    }
    bool IsNativeProgram() const override {
        return true;
    }
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override {
        if (cbuf_index == 1) {
            const size_t w = cbuf_offset / 4;
            return w < c1.size() ? c1[w] : 0;
        }
        return 0x101 + cbuf_index * 16 + cbuf_offset;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    u32 TextureBoundBuffer() const override {
        return texture_bound;
    }
    u32 LocalMemorySize() const override {
        return 256;
    }
    u32 SharedMemorySize() const override {
        return 16 * 1024;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {32, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    std::vector<u32> c1;
    u32 texture_bound;
};

Shader::HostTranslateInfo MakeHostInfo() {
    Shader::HostTranslateInfo info{};
    info.support_float16 = true;
    info.support_int64 = true;
    info.support_float64 = true;
    info.min_ssbo_alignment = 16;
    info.support_viewport_index_layer = true;
    info.support_conditional_barrier = true;
    return info;
}

Shader::Profile MakeProfile() {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.unified_descriptor_binding = true;
    profile.support_int8 = true;
    profile.support_int16 = true;
    profile.support_int64 = true;
    profile.support_vote = true;
    profile.support_quad_shuffles = true;
    profile.support_shuffle_relative = true;
    profile.supported_subgroup_stages = 0x7f;
    profile.max_subgroup_size = 32;
    return profile;
}

struct IrPools {
    Shader::ObjectPool<Shader::IR::Inst> inst;
    Shader::ObjectPool<Shader::IR::Block> block;
};

Shader::IR::Program TranslateDirect(IrPools& pools, std::span<const Instruction> code,
                                    std::vector<u32> c1 = {},
                                    const LoweringOptions& options = {},
                                    Shader::Stage stage = Shader::Stage::Compute) {
    const auto analysis = AnalyzeProgram(code, options);
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));

    const u32 tex_bound =
        analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2U;
    TestEnv env{std::move(c1), tex_bound, stage};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    auto program = TranslateProgram(pools.inst, pools.block, flow_pool, env, code, options,
                                    analysis.uses_async_copy, &analysis.branch_targets,
                                    MakeHostInfo());
    Shader::Optimization::VerificationPass(program);
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(MakeProfile(), program);
    REQUIRE(!spirv.empty());
    return program;
}

size_t CountOpcode(const Shader::IR::Program& program, Shader::IR::Opcode opcode) {
    size_t count{};
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == opcode) {
                ++count;
            }
        }
    }
    return count;
}

size_t CountSyntaxNodes(const Shader::IR::Program& program, Shader::IR::AbstractSyntaxNode::Type type) {
    size_t count{};
    for (const auto& node : program.syntax_list) {
        if (node.type == type) {
            ++count;
        }
    }
    return count;
}

size_t CountOps(const Shader::IR::Program& program, std::initializer_list<Shader::IR::Opcode> ops) {
    size_t count{};
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            if (std::ranges::find(ops, inst.GetOpcode()) != ops.end()) {
                ++count;
            }
        }
    }
    return count;
}

bool UsesUniformPredBeforeSsa(std::span<const Instruction> code, Shader::IR::Pred target) {
    const auto analysis = AnalyzeProgram(code);
    REQUIRE(static_cast<bool>(analysis));
    TestEnv env{{}, 2U, Shader::Stage::Compute};
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    Shader::Ampere::Flow::CFG cfg{env, flow_pool, code, false};
    u32 req_tex{~0U};
    const auto syntax_list = Shader::Maxwell::BuildASL(
        inst_pool, block_pool, env, cfg.Functions(), MakeHostInfo(),
        [&](Shader::IR::Block* block, u32 begin, u32 end) {
            Shader::Ampere::TranslateBlock(env, block, code, begin, end, {},
                                           analysis.uses_async_copy, req_tex,
                                           &analysis.branch_targets);
        });
    for (const auto& node : syntax_list) {
        if (node.type == Shader::IR::AbstractSyntaxNode::Type::Block) {
            for (const Shader::IR::Inst& inst : node.data.block->Instructions()) {
                if ((inst.GetOpcode() == Shader::IR::Opcode::SetPred ||
                     inst.GetOpcode() == Shader::IR::Opcode::GetPred) &&
                    inst.Arg(0).Pred() == target) {
                    return true;
                }
            }
        }
    }
    return false;
}

std::array<u32, 2> Rel50(s64 byte_offset) {
    const u64 raw = static_cast<u64>(byte_offset) & ((1ULL << 50) - 1ULL);
    return {static_cast<u32>(raw), static_cast<u32>((raw >> 32) & 0x3ffffU)};
}

} // namespace

TEST_CASE("Direct Ampere IMAD eliminates XMAD bit-extraction bloat", "[ampere_direct][video_core]") {
    // Load R4 and R5 from c[0][0x20/0x24], compute IMAD R4, R4, R5, R6 (here via LDC + IMAD + STG),
    // or pure ALU sequence: IMAD R4, R4, R5, R6 -> STG [R2], R4.
    const std::array<Instruction, 5> program{{
        {{0xff007624U | (2U << 16), 0x20U << 6, 0x078e00ffU, Sched}}, // IMAD.MOV.U32 R2, RZ, RZ, c[0][0x20]
        {{0xff007624U | (3U << 16), 0x24U << 6, 0x078e00ffU, Sched}}, // R3 = c[0][0x24]
        {{0x04047224U | (5U << 24), 6U, 0x078e0007U, Sched}},          // IMAD R4, R5, R6, R7
        {{0x02007386U, 4U, 0x0010a900U, Sched}},                        // STG.E.32 [R2], R4
        EXIT,
    }};

    IrPools direct_pools;
    const auto direct = TranslateDirect(direct_pools, program);

    CHECK(CountOpcode(direct, Shader::IR::Opcode::BitFieldUExtract) == 0);
    CHECK(CountOps(direct, {Shader::IR::Opcode::IMul32, Shader::IR::Opcode::IAdd32}) <= 2);
}

TEST_CASE("Direct Ampere predicated instructions if-convert pure ALU and branch on side effects",
          "[ampere_direct][video_core]") {
    // Pure @P0 IMAD if-converts inline (0 Ifs); side-effecting @P0 STG keeps a single If diamond.
    const std::array<Instruction, 4> pure_program{{
        {{0xff007624U | (2U << 16), 0x20U << 6, 0x078e00ffU, Sched}},
        {{0x05040224U, 6U, 0x078e0007U, Sched}}, // @P0 IMAD R4, R5, R6, R7 (pure -> Select)
        {{0x02007386U, 4U, 0x0010a900U, Sched}}, // STG [R2], R4
        EXIT,
    }};
    const std::array<Instruction, 4> impure_program{{
        {{0xff007624U | (2U << 16), 0x20U << 6, 0x078e00ffU, Sched}},
        {{0x00047802U, 1U, 0xf00U, Sched}},
        {{0x02000386U, 4U, 0x0010a900U, Sched}}, // @P0 STG [R2], R4 (impure -> 1 If diamond)
        EXIT,
    }};

    IrPools pure_pools;
    IrPools impure_pools;
    const auto direct_pure = TranslateDirect(pure_pools, pure_program);
    const auto direct_impure = TranslateDirect(impure_pools, impure_program);

    CHECK(CountSyntaxNodes(direct_pure, Shader::IR::AbstractSyntaxNode::Type::If) == 0);
    CHECK(CountSyntaxNodes(direct_impure, Shader::IR::AbstractSyntaxNode::Type::If) == 1);
}

TEST_CASE("Direct Ampere uniform predicates eliminate R239 bit-packing",
          "[ampere_direct][video_core]") {
    // S2R R8, SR_TID_X -> R2UR UR4, R8 (non-constant uniform!) -> UISETP UP3 -> @UP3 UMOV UR5 ->
    // UPLOP3 UP4 -> @UP4 UIADD3 UR6 -> USEL UR7 -> MOV R4, UR7 -> STG [R2], R4
    std::array<Instruction, 11> fixed{{
        {{0xff007624U | (2U << 16), 0x20U << 6, 0x078e00ffU, Sched}}, // R2 = c[0][0x20]
        {{0xff007624U | (3U << 16), 0x24U << 6, 0x078e00ffU, Sched}}, // R3 = c[0][0x24]
        {{0x00087919U, 0U, 33U << 8, Sched}},                         // S2R R8, SR_TID_X
        {{0x080473c2U, 0U, 0x000e0000U, Sched}},                      // R2UR UR4, R8
        {{0x0400788cU, 8U, 0x0bf01070U | (3U << 17), Sched}},         // UISETP.LT.U32.AND UP3, UPT, UR4, 8
        {{0x00053882U, 0x42U, 0U, Sched}},                            // @UP3 UMOV UR5, 0x42
        {{0x0000789cU, 0U,
          (4U << 17) | (7U << 20) | (3U << 23) | (7U << 13) | (7U << 4) | ((0x0fU >> 3) << 8) |
              (0x0fU & 7U),
          Sched}},                                                    // UPLOP3.LUT UP4, !UP3
        {{0x05064890U, 1U, 0x0fffe03fU, Sched}},                      // @UP4 UIADD3 UR6, UR5, 1, URZ
        {{0x05077887U, 6U, 0x0a000000U, Sched}},                      // USEL UR7, UR5, UR6, UP4
        {{0x00047c02U, 7U, 0x08000f00U, Sched}},                      // MOV R4, UR7
        {{0x02007386U, 4U, 0x0010a900U, Sched}},                      // STG.E.32 [R2], R4
    }};
    std::vector<Instruction> prog(fixed.begin(), fixed.end());
    prog.push_back(EXIT);

    IrPools direct_pools;
    const auto direct = TranslateDirect(direct_pools, prog);

    CHECK(CountOps(direct,
                   {Shader::IR::Opcode::BitFieldInsert, Shader::IR::Opcode::BitFieldUExtract,
                    Shader::IR::Opcode::BitFieldSExtract}) == 0);
    CHECK(UsesUniformPredBeforeSsa(prog, Shader::IR::Pred::UP3));
    CHECK(UsesUniformPredBeforeSsa(prog, Shader::IR::Pred::UP4));
}

TEST_CASE("Direct Ampere handles high GPRs, high uniforms, and NVN stage pipeline",
          "[ampere_direct][video_core]") {
    // R220 + UR40 + FFMA -> STG [R2], R220
    const std::array<Instruction, 6> program{{
        {{0xff007624U | (2U << 16), 0x20U << 6, 0x078e00ffU, Sched}},       // R2 = c[0][0x20]
        {{0x00287882U, 0x3f800000U, 0U, Sched}},                            // UMOV UR40, 1.0f
        {{0x00dc7802U, 0x40000000U, 0xf00U, Sched}},                        // MOV R220, 2.0f
        {{0xdc007e23U | (220U << 16), 40U, 0x080000dcU, Sched}},            // FFMA R220, R220, R220, UR40
        {{0x02007386U, 220U, 0x0010a900U, Sched}},                          // STG [R2], R220
        EXIT,
    }};

    IrPools direct_pools;
    const auto direct = TranslateDirect(direct_pools, program);
    CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::If) == 0);

    // Also verify PrepareNvnProgram + NvnEnvironment + TranslateNvnStage end-to-end.
    constexpr u32 InstBytes = sizeof(program);
    constexpr u32 C1Offset = 0x100;
    std::vector<u8> control(0x100, 0);
    const auto put32 = [&](size_t off, u32 v) { std::memcpy(control.data() + off, &v, 4); };
    put32(0, VideoCommon::Nvn::ControlMagic);
    put32(4, VideoCommon::Nvn::ControlVersion);
    put32(0x38, InstBytes);
    put32(0x40, C1Offset);
    put32(0x44, C1Offset);
    control[0x4c] = 240;
    control[0x56] = 5;
    put32(0x58, 1);
    put32(0x5c, 1);
    put32(0x60, 1);
    control[0xb1] = 1;
    control[0xb2] = 1;
    control[0xba] = 0x30;
    std::vector<u8> code_bytes(C1Offset, 0);
    std::memcpy(code_bytes.data(), program.data(), InstBytes);

    const auto prepared = VideoCommon::Nvn::PrepareNvnProgram(
        VideoCommon::Nvn::NvnProgramBinary{VideoCommon::Nvn::ShaderStage::Compute, control, code_bytes});
    INFO(prepared->error);
    REQUIRE(static_cast<bool>(prepared));

    VideoCommon::Nvn::NvnEnvironment nvn_env{prepared};
    CHECK(nvn_env.NativeAmpereProgram().has_value());
    VideoCommon::Nvn::NvnShaderPools pools;
    auto nvn_ir = VideoCommon::Nvn::TranslateNvnStage(pools, nvn_env, MakeHostInfo());
    Shader::Optimization::VerificationPass(nvn_ir);
}

TEST_CASE("Direct Ampere translates control flow idioms (dual-pred, CALL/RET, BRX/BRXU)",
          "[ampere_direct][video_core]") {
    SECTION("Dual-predicated BRA and EXIT") {
        // @P0 BRA P1, target (forward) ; @!PT NOP ; @P0 EXIT P1 ; target: MOV R4, 1 ; EXIT
        const auto rel = Rel50(2 * 16);
        const std::array<Instruction, 5> code{{
            {{0x00000947U, rel[0], (1U << 23) | rel[1], Sched}},              // @P0 BRA P1, idx 3
            {{0x0000f918U, 0, 0, Sched}},                                       // @!PT NOP
            {{0x0000094dU, 0, (1U << 23), Sched}},                              // @P0 EXIT P1
            {{0x00047802U, 1, 0xf00U, Sched}},                                  // MOV R4, 1
            EXIT,
        }};
        IrPools direct_pools;
        const auto direct = TranslateDirect(direct_pools, code);
        CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::If) >= 2);
    }

    SECTION("CALL.REL.NOINC and RET.REL.NODEC") {
        const auto call_rel = Rel50(16);
        const auto ret_rel = Rel50(-4 * 16);
        const std::array<Instruction, 4> code{{
            {{0x00047802U, 0x20U, 0x00000f00U, Sched}},            // MOV R4, 0x20
            {{0x00007944U, call_rel[0], 0x03c00000U | call_rel[1], Sched}}, // CALL -> idx 3
            EXIT,
            {{0x04007950U, ret_rel[0], 0x03c00000U | ret_rel[1], Sched}},   // RET R4
        }};
        IrPools direct_pools;
        const auto direct = TranslateDirect(direct_pools, code);
        CHECK(static_cast<bool>(direct.syntax_list.size() > 0));
    }

    SECTION("BRXU uniform indirect jump table") {
        const auto brxu_rel = Rel50(-5 * 16);
        const std::array<Instruction, 7> code{{
            {{0x0400788cU, 1U, 0x0bf01070U, Sched}},                          // UISETP.LT.U32 UP0, UR4, 1
            {{0x04047887U, 1U, 0x08000000U, Sched}},                          // USEL UR4, UR4, 1, UP0
            {{0x04047899U, 2U, 0x0800063fU, Sched}},                          // USHF.L.U32 UR4, UR4, 2, URZ
            {{0x04047abbU, (1U << 22), 0x08000800U, Sched}},                  // ULDC UR4, c[1][UR4]
            {{0x04007958U, brxu_rel[0], 0x0b800000U | brxu_rel[1], Sched}},   // BRXU UR4, 0x0
            {{0x00057802U, 11U, 0xf00U, Sched}},                              // MOV R5, 11
            {{0x00057802U, 22U, 0xf00U, Sched}},                              // MOV R5, 22
        }};
        // Append two EXITs cleanly so jump table targets are valid terminating blocks, followed by BRA .
        const auto self_rel = Rel50(-16);
        const std::array<Instruction, 10> full_code{{
            code[0],
            code[1],
            code[2],
            code[3],
            code[4],
            code[5],
            EXIT,
            code[6],
            EXIT,
            {{0x00007947U, self_rel[0], 0x03800000U | self_rel[1], Sched}},
        }};
        // Note: idx 5 (80) moves 11, idx 7 (112) moves 22; idx 6 is EXIT! Let c1 point to 5*16=80 and 7*16=112.
        const std::vector<u32> table_c1{5U * 16U, 7U * 16U};
        IrPools direct_pools;
        const auto direct = TranslateDirect(direct_pools, full_code, table_c1);
        CHECK(static_cast<bool>(direct.syntax_list.size() > 0));
    }
}

TEST_CASE("Direct Ampere rejects unemulated HMMA when tensor option is off",
          "[ampere_direct][video_core]") {
    // HMMA.1688.F32.F16 R4, R0, R2, R4
    const std::array<Instruction, 2> code{{
        {{0x0004723cU, 2U, 0x00000004U, Sched}},
        EXIT,
    }};
    const auto rejected = AnalyzeProgram(code, {.tensor = false});
    CHECK_FALSE(static_cast<bool>(rejected));
    CHECK(rejected.error.find("native matrix multiply") != std::string_view::npos);
}

TEST_CASE("Direct Ampere Fragment EXIT emits color outputs (unconditional and dual-pred)",
          "[ampere_direct][video_core]") {
    const std::array<Instruction, 5> code{{
        {{0x00007802U, 0x3f800000U, 0xf00U, Sched}}, // MOV R0, 1.0f
        {{0x00017802U, 0x3f000000U, 0xf00U, Sched}}, // MOV R1, 0.5f
        {{0x00027802U, 0x00000000U, 0xf00U, Sched}}, // MOV R2, 0.0f
        {{0x00037802U, 0x3f800000U, 0xf00U, Sched}}, // MOV R3, 1.0f
        EXIT,
    }};
    IrPools pools;
    const auto direct = TranslateDirect(pools, code, {}, {}, Shader::Stage::Fragment);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::SetFragColor) == 4);

    const std::array<Instruction, 6> dual_code{{
        {{0x00007802U, 0x3f800000U, 0xf00U, Sched}}, // MOV R0, 1.0f
        {{0x00017802U, 0x3f000000U, 0xf00U, Sched}}, // MOV R1, 0.5f
        {{0x00027802U, 0x00000000U, 0xf00U, Sched}}, // MOV R2, 0.0f
        {{0x00037802U, 0x3f800000U, 0xf00U, Sched}}, // MOV R3, 1.0f
        {{0x0000094dU, 0U, (1U << 23), Sched}},      // @P0 EXIT P1
        EXIT,
    }};
    IrPools dual_pools;
    const auto dual_direct = TranslateDirect(dual_pools, dual_code, {}, {}, Shader::Stage::Fragment);
    CHECK(CountOpcode(dual_direct, Shader::IR::Opcode::SetFragColor) == 8);
}

TEST_CASE("Direct Ampere Tier 2 FP32 ALU, IPA, if-conversion, and IR opts minimize bloat",
          "[ampere_direct][video_core]") {
    // Exercise IPA, negated/abs FFMA, FMUL.M2.FTZ.SAT, FADD.FTZ, FMNMX.NAN, MUFU.RCP,
    // FSEL, FSET.BF, FSETP, CSE/GVN, and predicated pure if-conversion (zero If nodes!).
    const std::array<Instruction, 14> code{{
        // IPA.PASS R4, a[0x80] (Generic0.x)
        {{0x00047326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0X), Sched}},
        // IPA.PASS R5, a[0x84] (Generic0.y)
        {{0x00057326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0Y), Sched}},
        // FFMA R6, -R4, |R5|, R4
        {{0x04067223U, 0x40000005U, 0x00000104U, Sched}},
        // Duplicate pure FFMA R7, -R4, |R5|, R4 (GVN/CSE must merge with R6!)
        {{0x04077223U, 0x40000005U, 0x00000104U, Sched}},
        // FADD.FTZ R6, R6, R7
        {{0x06067221U, 0x00000007U, 0x00010000U, Sched}},
        // FMUL.M2.FTZ.SAT R6, R6, 0.5f
        {{0x06067820U, 0x3f000000U, (5U << 20) | (1U << 16) | (1U << 13), Sched}},
        // FSETP.LT.AND P0, PT, R6, R4, PT
        {{0x0600720bU, 0x00000004U, (7U << 23) | (7U << 17) | (1U << 12), Sched}},
        // @P0 FADD R0, R6, 1.0f  (predicated pure -> if-converted to SelectF32/U32, no If!)
        {{0x06000821U, 0x3f800000U, 0U, Sched}},
        // @!P0 MOV R0, 0.25f
        {{0x00008802U, 0x3e800000U, 0xf00U, Sched}},
        // FSEL R1, R0, R5, P0
        {{0x00017208U, 0x00000005U, 0U, Sched}},
        // MUFU.RCP R2, R1
        {{0x00027308U, 0x00000001U, (4U << 10), Sched}},
        // FMNMX.NAN R3, R2, R1, PT
        {{0x02037209U, 0x00000001U, (7U << 23) | (1U << 17), Sched}},
        // FSET.BF.GT.AND R3, R3, RZ, PT
        {{0x0303720aU, 0x000000ffU, (7U << 23) | (4U << 12), Sched}},
        EXIT,
    }};

    IrPools direct_pools;
    const auto direct = TranslateDirect(direct_pools, code, {}, {}, Shader::Stage::Fragment);
    CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::If) == 0);
    // Native FP + ValueNumbering CSE: duplicate FFMA collapsed (<=1 FPFma32), zero sign-XOR hacks!
    CHECK(CountOpcode(direct, Shader::IR::Opcode::FPFma32) <= 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::FPRecip32) == 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::SetFragColor) == 4);
}

TEST_CASE("Direct Ampere SsaRewritePass eliminates self-loop Phis across nested backedges",
          "[ampere_direct][video_core]") {
    // Loop initializing R4 before entering a inner loop that conditionally reads/writes R4 and loops.
    // Even if SSA constructs phi(&phi, R4_init), zero trivial self-loop Phi instructions remain.
    const auto back_rel = Rel50(-3 * 16);
    const std::array<Instruction, 8> code{{
        {{0x00047802U, 42U, 0xf00U, Sched}},                              // MOV R4, 42
        {{0x00057802U, 0U, 0xf00U, Sched}},                               // MOV R5, 0
        // loop: IADD3 R5, R5, 1, RZ
        {{0x05057810U, 1U, 0x07ffe0ffU, Sched}},
        // ISETP.LT.U32.AND P0, PT, R5, 4, PT
        {{0x0500780cU, 4U, (7U << 23) | (7U << 17) | (1U << 12), Sched}},
        // @P0 BRA loop
        {{0x00000947U, back_rel[0], (7U << 23) | back_rel[1], Sched}},
        {{0x00000000U | 0x7918U, 0U, 0U, Sched}},                         // NOP
        EXIT,
        EXIT,
    }};

    IrPools pools;
    const auto program = TranslateDirect(pools, code);
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != Shader::IR::Opcode::Phi) {
                continue;
            }
            const auto first = inst.Arg(0).Resolve();
            bool all_same = true;
            for (size_t i = 1; i < inst.NumArgs(); ++i) {
                const auto cur = inst.Arg(i).Resolve();
                if (cur != first && (cur.IsImmediate() || cur.Inst() != &inst)) {
                    all_same = false;
                    break;
                }
            }
            CHECK_FALSE(all_same);
        }
    }
}

TEST_CASE("ValueNumberingPass collapses bitcast sign-XOR, known bits, and GVN",
          "[ampere_direct][video_core]") {
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    Shader::IR::Program program;
    Shader::IR::Block* const block = block_pool.Create(inst_pool);
    program.blocks.push_back(block);
    program.post_order_blocks.push_back(block);
    program.syntax_list.push_back(Shader::IR::AbstractSyntaxNode{
        .data = {.block = block},
        .type = Shader::IR::AbstractSyntaxNode::Type::Block,
    });
    block->SsaSeal();

    Shader::IR::IREmitter ir{*block};
    const Shader::IR::F32 attr = ir.GetAttribute(Shader::IR::Attribute::PositionX);
    const Shader::IR::U32 neg_bits =
        ir.BitwiseXor(ir.BitCast<Shader::IR::U32>(attr), ir.Imm32(0x80000000U));
    const Shader::IR::F32 neg_fp1 = ir.BitCast<Shader::IR::F32>(neg_bits);
    const Shader::IR::F32 dbl_neg = ir.FPNeg(neg_fp1);
    const Shader::IR::U32 c1 = ir.GetCbuf(ir.Imm32(0U), ir.Imm32(16U));
    const Shader::IR::U32 c2 = ir.GetCbuf(ir.Imm32(0U), ir.Imm32(16U));
    const Shader::IR::U32 masked = ir.BitwiseAnd(ir.ShiftLeftLogical(c1, ir.Imm32(4U)), ir.Imm32(0xfU));
    const Shader::IR::U32 sum = ir.IAdd(ir.BitCast<Shader::IR::U32>(dbl_neg), ir.IAdd(c2, masked));
    ir.SetAttribute(Shader::IR::Attribute::PositionX, ir.BitCast<Shader::IR::F32>(sum), ir.Imm32(0U));

    Shader::Optimization::ValueNumberingPass(program);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::VerificationPass(program);

    CHECK(CountOpcode(program, Shader::IR::Opcode::BitwiseXor32) == 0);
    CHECK(CountOpcode(program, Shader::IR::Opcode::BitwiseAnd32) == 0);
    CHECK(CountOpcode(program, Shader::IR::Opcode::ShiftLeftLogical32) == 0);
    CHECK(CountOpcode(program, Shader::IR::Opcode::GetCbufU32) == 1);
}

TEST_CASE("Direct Ampere BRX/BRXU handles interleaved non-clobbering ops and R0/UR0",
          "[ampere_direct][video_core]") {
    SECTION("BRXU on UR0 with interleaved GPR arithmetic matching destination bits") {
        // UISETP.LT.U32 UP0, UR4, 1 ; IADD3 R0, R1, R2, RZ ; USEL UR0, UR4, 1, UP0 ;
        // IADD3 R0, R3, R4, RZ ; USHF.L.U32 UR0, UR0, 2, URZ ; MOV R0, 7 ;
        // ULDC UR0, c[1][UR0] ; MOV R0, 9 ; BRXU UR0, 0x0 ; target1 ; EXIT ; target2 ; EXIT ; BRA .
        const auto brxu_rel = Rel50(-9 * 16);
        const auto self_rel = Rel50(-16);
        const std::array<Instruction, 14> code{{
            {{0x0400788cU, 1U, 0x0bf01070U, Sched}},                            // 0: UISETP.LT.U32 UP0, UR4, 1
            {{0x01007210U, 2U, 0x07ffe0ffU, Sched}},                            // 1: IADD3 R0, R1, R2, RZ
            {{0x04007887U, 1U, 0x08000000U, Sched}},                            // 2: USEL UR0, UR4, 1, UP0
            {{0x03007210U, 4U, 0x07ffe0ffU, Sched}},                            // 3: IADD3 R0, R3, R4, RZ
            {{0x00007899U, 2U, 0x0800063fU | (1U << 31), Sched}},               // 4: USHF.L.U32 UR0, UR0, 2, URZ (extra top bit)
            {{0x00007802U, 7U, 0xf00U, Sched}},                                 // 5: MOV R0, 7
            {{0x00007abbU, (1U << 22), 0x08000800U, Sched}},                    // 6: ULDC UR0, c[1][UR0]
            {{0x00007802U, 9U, 0xf00U, Sched}},                                 // 7: MOV R0, 9
            {{0x00007958U, brxu_rel[0], 0x0b800000U | brxu_rel[1], Sched}},     // 8: BRXU UR0, 0x0
            {{0x00057802U, 11U, 0xf00U, Sched}},                                // 9: MOV R5, 11
            EXIT,                                                               // 10
            {{0x00057802U, 22U, 0xf00U, Sched}},                                // 11: MOV R5, 22
            EXIT,                                                               // 12
            {{0x00007947U, self_rel[0], 0x03800000U | self_rel[1], Sched}},     // 13
        }};
        const std::vector<u32> table_c1{9U * 16U, 11U * 16U};
        IrPools direct_pools;
        const auto direct = TranslateDirect(direct_pools, code, table_c1);
        CHECK(direct.syntax_list.size() > 0);
    }

    SECTION("BRX on R0 with interleaved ISETP/NOP before LDC") {
        // IMNMX R4, R4, 1, PT ; UMOV UR0, 3 ; SHF.L.U32 R4, R4, 2, RZ ;
        // ISETP.NE.U32 P0, PT, R5, RZ, PT ; LDC R0, c[1][R4] ;
        // @!PT NOP ; BRX R0, 0x0 ; target1 ; EXIT ; target2 ; EXIT ; BRA .
        const auto brx_rel = Rel50(-7 * 16);
        const auto self_rel = Rel50(-16);
        const std::array<Instruction, 12> code{{
            {{0x04047817U, 1U, 0x03800000U, Sched}},                            // 0: IMNMX R4, R4, 1, PT
            {{0x00007882U, 3U, 0U, Sched}},                                     // 1: UMOV UR0, 3
            {{0x04047819U, 2U, 0x000006ffU, Sched}},                            // 2: SHF.L.U32 R4, R4, 2, RZ
            {{0x0500720cU, 255U, (5U << 12) | (7U << 17) | (7U << 20) | (7U << 23), Sched}}, // 3: ISETP
            {{0x04007b82U, (1U << 22), 0x00000800U, Sched}},                    // 4: LDC R0, c[1][R4]
            {{0x0000f918U, 0U, 0U, Sched}},                                     // 5: @!PT NOP
            {{0x00007949U, brx_rel[0], 0x03800000U | brx_rel[1], Sched}},     // 6: BRX R0, 0x0
            {{0x00057802U, 11U, 0xf00U, Sched}},                                // 7
            EXIT,                                                               // 8
            {{0x00057802U, 22U, 0xf00U, Sched}},                                // 9
            EXIT,                                                               // 10
            {{0x00007947U, self_rel[0], 0x03800000U | self_rel[1], Sched}},     // 11
        }};
        const std::vector<u32> table_c1{7U * 16U, 9U * 16U};
        IrPools direct_pools;
        const auto direct = TranslateDirect(direct_pools, code, table_c1);
        CHECK(direct.syntax_list.size() > 0);
    }
}

TEST_CASE("Direct Ampere if-converts predicated Tier 1 integer and logic ALU",
          "[ampere_direct][video_core]") {
    const std::array<Instruction, 6> code{{
        // @P0 IADD3 R4, R4, 1, RZ
        {{0x04040810U, 1U, 0x07ffe0ffU, Sched}},
        // @!P0 LOP3.LUT R4, R4, 0xff, RZ, 0x80 (AND)
        {{0x04048812U, 0xffU, 0x078e00ffU | (0x80U << 8), Sched}},
        // @P0 IMAD.WIDE.U32 R6, R4, 3, RZ
        {{0x04060825U, 3U, 0x078e00ffU, Sched}},
        // @UP0 UIADD3 UR4, UR4, 2, URZ
        {{0x04040890U, 2U, 0x07ffe03fU, Sched}},
        // @P0 ULDC UR6, c[0][0x10] (static uniform cbuf)
        {{0x3f060abbU, 0x10U << 6, 0x08000800U, Sched}},
        EXIT,
    }};
    IrPools direct_pools;
    const auto direct = TranslateDirect(direct_pools, code);
    CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::If) == 0);
    CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::Loop) == 0);
}

TEST_CASE("Direct Ampere translates Batch B1 FP16, conversions, IPA.SC, and FSWZADD",
          "[ampere_direct][video_core]") {
    const std::array<Instruction, 18> code{{
        // F2I.U32.TRUNC R4, R4
        {{0x00047305U, 4U, 0x0020f000U, Sched}},
        // I2F.F32.S16 R5, R4.H1
        {{0x00057306U, (1U << 28) | 4U, 0x00101400U, Sched}},
        // I2FP.F32.S32.RN R6, R4
        {{0x00067245U, 4U, 0x00201400U, Sched}},
        // F2I.FTZ.U64.TRUNC R14, R5
        {{0x000e7311U, 5U, 0x0021d800U, Sched}},
        // I2F.F32.U64 R7, R14
        {{0x00077312U, 14U, 0x00301000U, Sched}},
        // F2IP.U8.F32.NTZ R8, R4, R5, R6
        {{0x04087243U, 5U, 0x0000c406U, Sched}},
        // ISETP.NE.AND P0, PT, R4, RZ, PT
        {{0x0400720cU, 0x000000ffU, (7U << 23) | (7U << 20) | (0U << 17) | (5U << 12), Sched}},
        // @P0 HADD2.SAT R16, R4, R5 (if-converted)
        {{0x04100230U, 5U, (1U << 13), Sched}},
        // @P0 HFMA2.RELU R17, R4, R5, R6, PT (if-converted)
        {{0x04110231U, 5U, (1U << 15) | (7U << 23) | 6U, Sched}},
        // @P0 HMUL2.FTZ R18, R4, 0x3c003c00 (if-converted)
        {{0x04120832U, 0x3c003c00U, (1U << 16), Sched}},
        // @P0 F2F.F16.F32 R19, |R4| (if-converted)
        {{0x00130304U, 0x40000004U, 0x00200800U, Sched}},
        // @P0 F2FP.MERGE_C R20, RZ, R4, R5 (if-converted)
        {{0xff14023eU, 4U, (1U << 14) | 5U, Sched}},
        // @P0 HSET2.LT.BF.AND R21, R4, R5, PT (if-converted)
        {{0x04150233U, 5U, (7U << 23) | (1U << 12) | (1U << 7), Sched}},
        // @P0 HSETP2.GT.AND P1, P2, R4, R5, PT (if-converted)
        {{0x04000234U, 5U, (7U << 23) | (2U << 20) | (1U << 17) | (4U << 12), Sched}},
        // @P0 HMNMX2 R22, R4, R5, PT (if-converted)
        {{0x04160240U, 5U, (7U << 23), Sched}},
        // @P0 FRND.F32.RP R23, R4 (if-converted)
        {{0x00170307U, 4U, (2U << 20) | (2U << 14) | (2U << 11), Sched}},
        // @P0 MUFU.RCP.F16 R24, R4.H1 (if-converted)
        {{0x00180308U, (1U << 28) | 4U, (4U << 10) | (1U << 9), Sched}},
        EXIT,
    }};
    IrPools pure_pools;
    const auto pure = TranslateDirect(pure_pools, code);
    CHECK(CountSyntaxNodes(pure, Shader::IR::AbstractSyntaxNode::Type::If) == 0);
    CHECK(CountOpcode(pure, Shader::IR::Opcode::FSwizzleAdd) == 0);

    // Quad-coupled FSWZADD must NOT be if-converted when predicated.
    const std::array<Instruction, 5> swz_code{{
        // IPA R4, a[0x80]; IPA R5, a[0x84]
        {{0x00047326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0X),
          Sched}},
        {{0x00057326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0Y),
          Sched}},
        // FSETP.LT.AND P0, PT, R4, R5, PT
        {{0x0400720bU, 5U, (7U << 23) | (7U << 20) | (0U << 17) | (1U << 12), Sched}},
        // @P0 FSWZADD R0, R4, R5, 0x55
        {{0x04000822U, 0x00000055U, 5U, Sched}},
        EXIT,
    }};
    IrPools swz_pools;
    const auto swz = TranslateDirect(swz_pools, swz_code, {}, {}, Shader::Stage::Fragment);
    CHECK(CountSyntaxNodes(swz, Shader::IR::AbstractSyntaxNode::Type::If) == 1);
    CHECK(CountOpcode(swz, Shader::IR::Opcode::FSwizzleAdd) == 1);

    // IPA.PASS on legacy ColorFrontDiffuseR multiplies by PositionW; IPA.SC does not.
    const std::array<Instruction, 2> ipa_pass{{
        {{0x00007326U, 0x000000ffU,
          (7U << 17) | static_cast<u32>(Shader::IR::Attribute::ColorFrontDiffuseR), Sched}},
        EXIT,
    }};
    const std::array<Instruction, 2> ipa_sc{{
        {{0x00007326U, 0x000000ffU,
          (7U << 17) | (1U << 14) | static_cast<u32>(Shader::IR::Attribute::ColorFrontDiffuseR),
          Sched}},
        EXIT,
    }};
    const auto translate_frag_ir = [](IrPools& pools, std::span<const Instruction> prog_code) {
        const auto analysis = AnalyzeProgram(prog_code, {});
        REQUIRE(static_cast<bool>(analysis));
        TestEnv env{{}, 2U, Shader::Stage::Fragment};
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
        return TranslateProgram(pools.inst, pools.block, flow_pool, env, prog_code, {},
                                analysis.uses_async_copy, &analysis.branch_targets,
                                MakeHostInfo());
    };
    IrPools pass_pools, sc_pools;
    const auto pass_prog = translate_frag_ir(pass_pools, ipa_pass);
    const auto sc_prog = translate_frag_ir(sc_pools, ipa_sc);
    CHECK(CountOpcode(pass_prog, Shader::IR::Opcode::FPMul32) == 1);
    CHECK(CountOpcode(sc_prog, Shader::IR::Opcode::FPMul32) == 0);
}

TEST_CASE("Direct Ampere translates Batch B2 memory, atomics, barriers, LDGSTS, and local spill",
          "[ampere_direct][video_core]") {
    const std::array<Instruction, 21> code{{
        // Initialize pointer & data via S2R + UMOV
        {{0x00067919U, 0U, 0x25U << 8, Sched}},                 // S2R R6, SR_CTAID_X
        {{0x00077802U, 0x10U, 0x00000f00U, Sched}},             // MOV R7, 0x10
        {{0x00047882U, 0x1000U, 0U, Sched}},                    // UMOV UR4, 0x1000
        {{0x00057882U, 0U, 0U, Sched}},                         // UMOV UR5, 0
        // LDG.E.U16 R4, [R6+0x4]
        {{0x06047381U, 0x400U, 0x001e8500U, Sched}},
        // LDG.E.128 R8, [UR4+R6+0x10]
        {{0x06087981U, (0x10U << 8) | 4U, 0x081e8d00U, Sched}},
        // STG.E.64 [R6+0x8], R8
        {{0x06007386U, (8U << 8) | 8U, 0x0010ab00U, Sched}},
        // STG.E.64 [UR4+0x20], RZ
        {{0xff007986U, (0x20U << 8) | 255U, 0x0810ab04U, Sched}},
        // ATOMG.E.ADD.STRONG.GPU R12, [R6+0x4], R4
        {{0x060c73a8U, (4U << 8) | 4U, 0x001ee100U, Sched}},
        // RED.E.ADD.F32.STRONG.GPU [UR4+0x8], R4
        {{0xff00798eU, (8U << 8) | 4U, 0x0c10e100U | (3U << 9) | 4U, Sched}},
        // LDS.U16 R13, [R6.X4+UR2+0x8]
        {{0x060d7984U, (8U << 8) | 2U, 0x08000000U | (2U << 9) | (1U << 14), Sched}},
        // STS.64 [R6+0x10], RZ
        {{0x06007388U, (0x10U << 8) | 255U, (5U << 9), Sched}},
        // ATOMS.ADD R14, [R6+0x4], R4
        {{0x060e738cU, (4U << 8) | 4U, 0U, Sched}},
        // ATOMS.POPC.INC.32 [R6.X4+0x8]
        {{0x06ff7f8cU, (8U << 8), 0x0d00003fU | (1U << 14), Sched}},
        // ISETP.NE.AND P0, PT, R4, RZ, PT
        {{0x0400720cU, 0x000000ffU, (7U << 23) | (7U << 20) | (5U << 12), Sched}},
        // Dynamic-enable LDGSTS P0 (must create 1 If diamond with LoadGlobal128 in true branch)
        {{0x06047faeU, (1U << 16) | 0x20U, 0x0c101c00U, Sched}},
        // STL.32 [RZ+0x10], R13 + LDL.S8 R15, [RZ+0x10]
        {{0xff007387U, (0x10U << 8) | 13U, 0x00100800U, Sched}},
        {{0xff0f7983U, (0x10U << 8), 0x00100200U, Sched}},
        // DEPBAR + LDGDEPBAR + ERRBAR + BAR.SYNC + MEMBAR
        {{0x0000791aU, 0x8000U, 0U, Sched}},
        {{0x000079afU, 0U, 0U, Sched}},
        EXIT,
    }};
    IrPools direct_pools;
    const auto analysis = AnalyzeProgram(code, {});
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    TestEnv env{{}, 2U, Shader::Stage::Compute};
    auto host = MakeHostInfo();
    host.nvn_raw_global_memory = true;
    host.nvn_raw_global_table_ssbo = true;
    auto profile = MakeProfile();
    profile.support_raw_global_memory = true;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    auto direct = TranslateProgram(direct_pools.inst, direct_pools.block, flow_pool, env, code,
                                   {}, analysis.uses_async_copy, &analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(direct);
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, direct);
    REQUIRE(!spirv.empty());
    CHECK(CountSyntaxNodes(direct, Shader::IR::AbstractSyntaxNode::Type::If) >= 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::LoadGlobalU16) == 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::LoadGlobal128) == 2); // LDG.128 + LDGSTS
    CHECK(CountOpcode(direct, Shader::IR::Opcode::WriteSharedU64) == 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::WriteSharedU128) == 2); // true & false LDGSTS
    CHECK(CountOpcode(direct, Shader::IR::Opcode::GlobalAtomicIAdd32) >= 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::GlobalAtomicAddF32) == 1);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::SharedAtomicIAdd32) == 2);

    // Isolated dynamic-enable LDGSTS lowers into one If diamond in the frontend CFG.
    const std::array<Instruction, 3> ldgsts_only{{
        {{0x0400720cU, 0x000000ffU, (7U << 23) | (7U << 20) | (5U << 12), Sched}}, // ISETP P0
        {{0x06047faeU, (1U << 16) | 0x20U, 0x0c101c00U, Sched}},                   // LDGSTS P0
        EXIT,
    }};
    const auto ldgsts_analysis = AnalyzeProgram(ldgsts_only, {});
    REQUIRE(static_cast<bool>(ldgsts_analysis));
    {
        IrPools cfg_pools;
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> cfg_flow_pool;
        Shader::Ampere::Flow::CFG cfg{env, cfg_flow_pool, ldgsts_only, false};
        CHECK(cfg.Functions()[0].blocks.size() == 5U);
        u32 req_tex = env.TextureBoundBuffer();
        const auto pre_opt_syntax = Shader::Maxwell::BuildASL(
            cfg_pools.inst, cfg_pools.block, env, cfg.Functions(), host,
            [&](Shader::IR::Block* block, u32 begin, u32 end) {
                Shader::Ampere::TranslateBlock(env, block, ldgsts_only, begin, end, {},
                                               ldgsts_analysis.uses_async_copy, req_tex,
                                               &ldgsts_analysis.branch_targets);
            });
        CHECK(!pre_opt_syntax.empty());
    }
    IrPools ldgsts_pools;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> ldgsts_flow;
    const auto ldgsts_prog = TranslateProgram(
        ldgsts_pools.inst, ldgsts_pools.block, ldgsts_flow, env, ldgsts_only, {},
        ldgsts_analysis.uses_async_copy, &ldgsts_analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(ldgsts_prog);
    CHECK(CountOpcode(ldgsts_prog, Shader::IR::Opcode::LoadGlobal128) == 1);
    CHECK(CountOpcode(ldgsts_prog, Shader::IR::Opcode::WriteSharedU128) == 2);
}

TEST_CASE("Direct Ampere translates Batch B2 ALD/AST, OUT, and ISBERD/ISBEWR",
          "[ampere_direct][video_core]") {
    LoweringOptions geom_opts{};
    geom_opts.geometry = true;
    const std::array<Instruction, 5> geom_code{{
        {{0xff047321U, (0x80U << 8) | 5U, 0U, Sched}},                        // ALD R4, a[0x80], R5
        {{0xff007322U, (0x80U << 8) | 4U, (1U << 10) | 5U, Sched}},           // AST.64 a[0x80], R4, R5
        {{0xff007322U, (0x5cU << 8) | 4U, 255U, Sched}},                      // AST.32 a[0x5c] (dropped)
        {{0x04047924U, 1U, (3U << 14), Sched}},                               // OUT.EMIT_CUT R4, R4, 1
        EXIT,
    }};
    IrPools geom_pools;
    const auto geom_analysis = AnalyzeProgram(geom_code, geom_opts);
    INFO(geom_analysis.error);
    REQUIRE(static_cast<bool>(geom_analysis));
    TestEnv geom_env{{}, 2U, Shader::Stage::Geometry};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> geom_flow;
    const auto geom = TranslateProgram(geom_pools.inst, geom_pools.block, geom_flow, geom_env,
                                       geom_code, geom_opts, geom_analysis.uses_async_copy,
                                       &geom_analysis.branch_targets, MakeHostInfo());
    CHECK(CountOpcode(geom, Shader::IR::Opcode::EmitVertex) == 1);
    CHECK(CountOpcode(geom, Shader::IR::Opcode::EndPrimitive) == 1);

    const std::array<Instruction, 3> mesh_code{{
        {{0x05047923U, (8U << 8), (1U << 15) | (3U << 12), Sched}},           // ISBERD.ATTR.O
        {{0x05007927U, (8U << 8) | 4U, (1U << 15) | (3U << 12) | (1U << 11), Sched}}, // ISBEWR.ATTR.32.O
        EXIT,
    }};
    IrPools mesh_pools;
    const auto mesh_analysis = AnalyzeProgram(mesh_code, {});
    INFO(mesh_analysis.error);
    REQUIRE(static_cast<bool>(mesh_analysis));
    TestEnv mesh_env{{}, 2U, Shader::Stage::Task};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> mesh_flow;
    const auto mesh = TranslateProgram(mesh_pools.inst, mesh_pools.block, mesh_flow, mesh_env,
                                       mesh_code, {}, mesh_analysis.uses_async_copy,
                                       &mesh_analysis.branch_targets, MakeHostInfo());
    CHECK(CountOpcode(mesh, Shader::IR::Opcode::IsbeReadOutput) == 1);
    CHECK(CountOpcode(mesh, Shader::IR::Opcode::IsbeWriteAttribute) == 1);
}

TEST_CASE("Direct Ampere translates Batch B3 textures, gathers, fetches, queries, and surfaces",
          "[ampere_direct][video_core]") {
    const auto mov_cbuf = [](u32 rd, u32 bank, u32 byte_off) -> Instruction {
        return {{0xff007a02U | (rd << 16), (bank << 22) | (byte_off << 6), 0x00000f00U, Sched}};
    };
    const auto uldc32 = [](u32 urd, u32 bank, u32 byte_off) -> Instruction {
        return {{0xff007ab9U | (urd << 16), (bank << 22) | (byte_off << 6), (4U << 9), Sched}};
    };
    const auto stg32 = [](u32 base_reg, u32 data_reg, u32 offset) -> Instruction {
        return {{0x00007386U | (base_reg << 24), data_reg | (offset << 8), 0x0010a900U, Sched}};
    };

    // 1. Bound & second-buffer TEX (FP32 + FP16), uniform 0xf60, TEX.B 0x361, TXQ, SULD, SUST, SUATOM, SURED
    // Note: keep coord/handle regs R2..R8 untouched; write outputs to R10..R29 and sink via STG.
    const std::array<Instruction, 23> mixed_code{{
        {{0x00027919U, 0U, 0x25U << 8, Sched}}, // S2R R2, SR_CTAID_X (coords + global ptr)
        {{0x00037802U, 0x3f800000U, 0x00000f00U, Sched}},
        {{0x00047802U, 0x3f000000U, 0x00000f00U, Sched}},
        {{0x00057802U, 0x40000000U, 0x00000f00U, Sched}},
        mov_cbuf(6U, 2U, 0x10U), // R6 = c[2][0x10] (bindless handle)
        mov_cbuf(7U, 2U, 0x14U), // R7 = c[2][0x14] (bindless TXQ handle)
        mov_cbuf(8U, 2U, 0x18U), // R8 = lod
        uldc32(4U, 2U, 0x1cU),   // UR4 = c[2][0x1c]
        // TEX 2D R10..R13 (second_dest=R12), binding 2
        {{0x020a7b60U, (1U << 29) | (2U << 22) | (0x10U << 8) | 3U, (7U << 17) | (0xfU << 8) | 12U, Sched}},
        // TEX.F16 2D R14 from second buffer binding 14 -> BindlessImageSampleImplicitLod + GetCbufU32
        {{0x020e7b60U, (1U << 29) | (14U << 22) | (0x20U << 8) | 3U, (7U << 17) | (1U << 15) | (3U << 8) | 255U, Sched}},
        // TEX 2D uniform handle UR4 (0xf60, coords=R4..5)
        {{0x04107f60U, (1U << 29) | (4U << 8) | 5U, (1U << 27) | (7U << 17) | (3U << 8) | 255U, Sched}},
        // TEX.B.LZ 2D (0x361: ra=4,5, handle=R6)
        {{0x04127361U, 0x28000006U, 0x008e01ffU, Sched}},
        // TXQ.B (0x370: ra=7,8) + TXQ bound binding 2 (0xb6f)
        {{0x07147370U, (1U << 27), (0x3U << 8) | 255U, Sched}},
        {{0x02167b6fU, (2U << 22) | (0x18U << 8), (0xfU << 8) | 24U, Sched}},
        // SUST.P.2D.STRONG.SM partial mask (channels=1) on second buffer binding 14 -> barriers + GetCbuf
        {{0x02007b9bU, (3U << 29) | (14U << 22) | (0x30U << 8) | 4U, 0x0010a100U, Sched}},
        // SURED.D.2D.ADD.U32 on binding 2
        {{0x0200759fU, (3U << 29) | (2U << 22) | (0x40U << 8) | 4U, 0x0010e000U, Sched}},
        stg32(2U, 10U, 0U),
        stg32(2U, 14U, 4U),
        stg32(2U, 16U, 8U),
        stg32(2U, 18U, 12U),
        stg32(2U, 20U, 16U),
        stg32(2U, 22U, 20U),
        EXIT,
    }};
    const auto mixed_analysis = AnalyzeProgram(mixed_code, {});
    INFO(mixed_analysis.error);
    REQUIRE(static_cast<bool>(mixed_analysis));
    CHECK(mixed_analysis.required_texture_buffer == 2U);
    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, mixed_code);
        CHECK(CountOps(prog, {Shader::IR::Opcode::ImageSampleImplicitLod,
                              Shader::IR::Opcode::BindlessImageSampleImplicitLod}) >= 2U);
        CHECK(CountOps(prog, {Shader::IR::Opcode::ImageSampleExplicitLod,
                              Shader::IR::Opcode::BindlessImageSampleExplicitLod}) >= 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::PackHalf2x16) >= 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageQueryDimensions) >= 2U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::DeviceMemoryBarrier) == 2U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageRead) >= 1U); // SUST partial read-modify-write
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageWrite) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageAtomicIAdd32) == 1U);
    }

    // 2. TLD4 (including PTP constant offsets & second-buffer fallback), TLD (0xb66, 0xf66, 0x367),
    //    TMML, TXD (0xf6c ARRAY_2D layer=coords+2), SULD, and SUATOM
    const std::array<Instruction, 16> b3_tex_code{{
        {{0x00027919U, 0U, 0x25U << 8, Sched}}, // S2R R2, SR_CTAID_X
        {{0x00037802U, 0U, 0x00000f00U, Sched}},
        mov_cbuf(4U, 2U, 0x10U), // R4 = bindless TLD.B handle
        uldc32(4U, 2U, 0x14U),   // UR4 = uniform handle for 0xf66 / 0xf6c
        // MOV R12, 0x01020304; MOV R13, 0x05060708 (constant PTP offsets)
        {{0x000c7802U, 0x01020304U, 0x00000f00U, Sched}},
        {{0x000d7802U, 0x05060708U, 0x00000f00U, Sched}},
        // TLD4.PTP 2D R16, R2, R12 (binding 2)
        {{0x02107b63U, (1U << 29) | (2U << 22) | (0x10U << 8) | 12U,
          (7U << 17) | (2U << 12) | (0xfU << 8) | 18U, Sched}},
        // TLD4 plain from second buffer binding 14 -> BindlessImageGather (ra=2, rb=6, dest=10)
        {{0x020a7b63U, 0x33800406U, 0x041e0f06U, Sched}},
        // TLD.LL.AOFFI 2D on binding 2 (ra=2..3, rb=12..13, dest=20..21)
        {{0x02147b66U, (1U << 29) | (1U << 28) | (2U << 22) | (0x20U << 8) | 12U,
          (3U << 23) | (7U << 17) | (1U << 12) | (0x3U << 8) | 255U, Sched}},
        // TLD.F16 2D on binding 2 -> dest=22
        {{0x02167b66U, (1U << 29) | (1U << 28) | (2U << 22) | (0x24U << 8) | 3U,
          (1U << 23) | (7U << 17) | (1U << 15) | (0xfU << 8) | 23U, Sched}},
        // TLD uniform 0xf66 (`ampere_lowering.cpp:63`, dest=24, second_dest=26, coords R2/R3, handle UR4)
        {{0x02187f66U, 0x30000403U, 0x089e071aU, Sched}},
        // TLD.B GPR 0x367 (ra=2,3, rb=4 handle, 2D, dest=28)
        {{0x021c7367U, (1U << 29) | (3U << 27) | 4U,
          (1U << 23) | (7U << 17) | (0x3U << 8) | 255U, Sched}},
        // TMML 2D (0xb69, mask=3, dest=30)
        {{0x021e7b69U, (1U << 29) | (2U << 22) | (0x28U << 8) | 3U, (3U << 8) | 255U, Sched}},
        // TXD.ARRAY_2D with uniform UR4 (`ampere_gradient.cpp:79`, coords=R2..4, deriv=R32..35, dest=40)
        {{0x02287f6cU, 0xa0000420U, 0x081e0f2aU, Sched}},
        // SULD 2D on binding 2 (dest=44)
        {{0x022c7b97U, (3U << 29) | (2U << 22) | (0x34U << 8), 0x001e0300U, Sched}},
        // SUATOM.D.2D.ADD on binding 2 (dest=46)
        {{0x022e7593U, (3U << 29) | (2U << 22) | (0x38U << 8) | 4U, 0x001ee000U, Sched}},
    }};
    std::vector<Instruction> b3_tex_sunk(b3_tex_code.begin(), b3_tex_code.end());
    b3_tex_sunk.push_back(stg32(2U, 16U, 0U));
    b3_tex_sunk.push_back(stg32(2U, 20U, 4U));
    b3_tex_sunk.push_back(stg32(2U, 30U, 8U));
    b3_tex_sunk.push_back(stg32(2U, 40U, 12U));
    b3_tex_sunk.push_back(stg32(2U, 44U, 16U));
    b3_tex_sunk.push_back(EXIT);

    const auto b3_tex_analysis = AnalyzeProgram(b3_tex_sunk, {});
    INFO(b3_tex_analysis.error);
    REQUIRE(static_cast<bool>(b3_tex_analysis));
    CHECK(b3_tex_analysis.required_texture_buffer == 2U);
    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, b3_tex_sunk);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageGather) >= 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageFetch) >= 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageQueryLod) >= 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageGradient) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageRead) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ImageAtomicIAdd32) == 1U);
    }

    // 3. Invalid per-tap gather offset (>31 && <224) must be rejected by AnalyzeProgram
    const std::array<Instruction, 4> bad_ptp{{
        {{0x000c7802U, 0x00400000U, 0x00000f00U, Sched}}, // out-of-range offset 0x40
        {{0x000d7802U, 0U, 0x00000f00U, Sched}},
        {{0x02107b63U, (1U << 29) | (2U << 22) | (0x10U << 8) | 12U,
          (7U << 17) | (2U << 12) | (0xfU << 8) | 18U, Sched}},
        EXIT,
    }};
    const auto bad_analysis = AnalyzeProgram(bad_ptp, {});
    CHECK_FALSE(static_cast<bool>(bad_analysis));
    CHECK(bad_analysis.error == "native per-tap gather offset range");
}

TEST_CASE("Direct Ampere translates Batch B3 SHFL, MATCH.ALL, MOVM, LDSM, and HMMA",
          "[ampere_direct][video_core]") {
    const auto stg32 = [](u32 base_reg, u32 data_reg, u32 offset) -> Instruction {
        return {{0x00007386U | (base_reg << 24), data_reg | (offset << 8), 0x0010a900U, Sched}};
    };
    const auto p2r = [](u32 dst, u32 mask) -> Instruction {
        return {{0xff007803U | (dst << 16), mask, 0U, Sched}};
    };
    // 1. Warp & Tensor core opcodes (live-sunk via STG so DCE keeps every IR opcode intact)
    const std::array<Instruction, 16> warp_tensor_code{{
        {{0x00027919U, 0U, 0x25U << 8, Sched}}, // S2R R2, SR_CTAID_X (base ptr)
        {{0x00037802U, 0U, 0x00000f00U, Sched}},
        {{0x00047882U, 0x20U, 0U, Sched}},      // UMOV UR4, 0x20
        // SHFL.DOWN PT, R5, R4, R2, 0x1f (`0x589`)
        {{0x04057589U, (2U << 26) | (0x1fU << 8) | 2U, (7U << 17), Sched}},
        // SHFL.IDX P0, R6, R4, 0, 0x1f (`0xf89`)
        {{0x04067f89U, (0x1fU << 8), 0x000e0000U, Sched}},
        // SHFL.BFLY P1, R7, R4, 1, R5 (`0x989`)
        {{0x04077989U, (3U << 26) | (1U << 21), (1U << 17) | 5U, Sched}},
        // MATCH.ALL.U32 P2, R6 + MATCH.ALL.U64 P3, R6
        {{0x06ff73a1U, 0U, (2U << 17), Sched}},
        {{0x06ff73a1U, 0U, (3U << 17) | (1U << 9), Sched}},
        // MOVM.16.MT88 R8, R4
        {{0x0408723aU, 0U, 0U, Sched}},
        // LDSM.16.M88.2 R10, [R4 + 0x10] + LDSM.16.MT88.1 R12, [R4 + UR4 + 0x20]
        {{0x040a783bU, (0x10U << 8), (1U << 8), Sched}},
        {{0x040c783bU, (0x20U << 8) | 4U, 0x08004000U, Sched}},
        p2r(14U, 0x0fU),
        stg32(2U, 7U, 0U),
        stg32(2U, 8U, 4U),
        stg32(2U, 10U, 8U),
        EXIT,
    }};
    // Also sink R11, R12 and R14 (overwrite EXIT cleanly)
    std::vector<Instruction> warp_tensor_sunk(warp_tensor_code.begin(), warp_tensor_code.end() - 1);
    warp_tensor_sunk.push_back(stg32(2U, 11U, 12U));
    warp_tensor_sunk.push_back(stg32(2U, 12U, 16U));
    warp_tensor_sunk.push_back(stg32(2U, 14U, 20U));
    warp_tensor_sunk.push_back(EXIT);

    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, warp_tensor_sunk);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ShuffleDown) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ShuffleButterfly) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::ShuffleIndex) >= 6U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::SubgroupBallot) >= 2U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::VoteAll) >= 2U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::LoadSharedU32) == 2U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::LoadSharedU16) == 2U);
    }

    // 2. Fragment QUAD_MASK fast path -> outputs to R0, R1 (FragColor0.x, .y)
    const u32 quad_mask = (28U << 8) | 3U;
    const std::array<Instruction, 3> frag_quad{{
        {{0x04007f89U, (0U << 26) | (2U << 21) | (quad_mask << 8), 0x000e0000U, Sched}},
        {{0x04017f89U, (3U << 26) | (1U << 21) | (quad_mask << 8), 0x000e0000U, Sched}},
        EXIT,
    }};
    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, frag_quad, {}, {}, Shader::Stage::Fragment);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::QuadBroadcast) == 1U);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::QuadSwap) == 1U);
    }

    // 3. HMMA.16816.F16: compact vs guarded compact vs expanded (store D0=R44, D1=R45 to STG)
    const std::array<Instruction, 6> hmma_code{{
        {{0x00027919U, 0U, 0x25U << 8, Sched}}, // S2R R2, SR_CTAID_X
        {{0x00037802U, 0U, 0x00000f00U, Sched}},
        {{0x102c723cU, 0x30U, 0x8ffU, Sched}},  // HMMA R44, R16, R48, RZ
        stg32(2U, 44U, 0U),
        stg32(2U, 45U, 4U),
        EXIT,
    }};
    auto guarded_hmma = hmma_code;
    guarded_hmma[2].words[0] = (guarded_hmma[2].words[0] & ~0xf000U) | (1U << 12);

    {
        IrPools pools;
        const auto compact =
            TranslateDirect(pools, hmma_code, {}, {.tensor = true, .compact_tensor = true});
        CHECK(CountOpcode(compact, Shader::IR::Opcode::TensorMulAdd) == 1U);
        CHECK(CountOpcode(compact, Shader::IR::Opcode::FPFma32) == 0U);
    }
    {
        IrPools pools;
        const auto guarded =
            TranslateDirect(pools, guarded_hmma, {}, {.tensor = true, .compact_tensor = true});
        CHECK(CountOpcode(guarded, Shader::IR::Opcode::TensorMulAdd) == 0U);
        CHECK(CountOpcode(guarded, Shader::IR::Opcode::FPFma32) == 64U);
    }
    {
        IrPools pools;
        const auto expanded =
            TranslateDirect(pools, hmma_code, {}, {.tensor = true, .compact_tensor = false});
        CHECK(CountOpcode(expanded, Shader::IR::Opcode::TensorMulAdd) == 0U);
        CHECK(CountOpcode(expanded, Shader::IR::Opcode::FPFma32) == 64U);
        CHECK(CountOpcode(expanded, Shader::IR::Opcode::FPRoundEven32) == 8U);
    }
}

TEST_CASE("Direct Ampere validates and translates KIL (0x95b) and TRAP (0x95c)",
          "[ampere_direct][video_core]") {
    const std::array<Instruction, 7> code{{
        // IPA R4, a[0x80]; IPA R5, a[0x84]
        {{0x00047326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0X),
          Sched}},
        {{0x00057326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0Y),
          Sched}},
        // FSETP.LT.AND P0, PT, R4, R5, PT ; FSETP.GT.AND P1, PT, R4, R5, PT
        {{0x0400720bU, 5U, (7U << 23) | (7U << 20) | (0U << 17) | (1U << 12), Sched}},
        {{0x0400720bU, 5U, (7U << 23) | (7U << 20) | (1U << 17) | (4U << 12), Sched}},
        // @P0 KIL ; @P1 TRAP ; unconditional KIL
        {{0x0000095bU, 0U, 0x03800000U, Sched}},
        {{0x0000195cU, 4U, 0x00300000U, Sched}},
        EXIT,
    }};
    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, code, {}, {}, Shader::Stage::Fragment);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::DemoteToHelperInvocation) >= 1U);
    }
    const std::array<Instruction, 2> uncond_kill{{
        {{0x0000795bU, 0U, 0x03800000U, Sched}},
        EXIT,
    }};
    {
        IrPools pools;
        const auto prog = TranslateDirect(pools, uncond_kill, {}, {}, Shader::Stage::Fragment);
        CHECK(CountOpcode(prog, Shader::IR::Opcode::DemoteToHelperInvocation) >= 1U);
    }

    auto bad_kill = code;
    bad_kill[4].words[2] = 0x03800001U;
    const auto bad_kill_analysis = AnalyzeProgram(bad_kill, {});
    CHECK_FALSE(static_cast<bool>(bad_kill_analysis));

    auto bad_trap = code;
    bad_trap[5].words[1] = 0U;
    const auto bad_trap_analysis = AnalyzeProgram(bad_trap, {});
    CHECK_FALSE(static_cast<bool>(bad_trap_analysis));
}

TEST_CASE("Direct Ampere tracks ULDC.64 + LDG/STG [UR + R] and IMAD.WIDE pointers to SSBO without raw_global",
          "[ampere_direct][video_core]") {
    class PointerTrackingEnv final : public TestEnv {
    public:
        PointerTrackingEnv() : TestEnv({}, 2U, Shader::Stage::Compute) {}

        bool IsNvnHleProgram() const override {
            return true;
        }
        bool IsCbufStorageBufferPointer(u32 index, u32 offset) override {
            return index == 0U && offset == 0x100U;
        }
    };

    const std::array<Instruction, 9> code{{
        {{0x00067919U, 0U, 0x25U << 8, Sched}},                                     // S2R R6, SR_CTAID_X
        {{0x00047802U, 0x10U, 0x00000f00U, Sched}},                                 // MOV R4, 0x10
        {{0x00047ab9U, 0x100U << 6, 5U << 9, Sched}},                               // ULDC.64 UR4, c[0x0][0x100]
        {{0x06007981U, (0x10U << 8) | 4U, 0x081e8900U, Sched}},                     // LDG.E.32 R0, [UR4+R6+0x10]
        {{0x06007986U, (0x14U << 8) | 0U, 0x0810a904U, Sched}},                     // STG.E.32 [UR4+R6+0x14], R0
        {{0x06087625U, 0x100U << 6, 0x07800000U | (7U << 17) | 4U, Sched}},         // IMAD.WIDE.U32 R8, R6, R4, c[0x0][0x100]
        {{0x08017381U, 0U, 0x001e8100U | (4U << 9), Sched}},                        // LDG.E.32 R1, [R8+0x0]
        {{0xff007986U, (0x18U << 8) | 1U, 0x0810a904U, Sched}},                     // STG.E.32 [UR4+0x18], R1
        EXIT,
    }};

    const auto analysis = AnalyzeProgram(code, {});
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));

    PointerTrackingEnv env;
    auto host = MakeHostInfo();
    host.nvn_raw_global_memory = true;
    host.nvn_raw_global_table_ssbo = true;
    auto profile = MakeProfile();
    profile.support_raw_global_memory = true;

    IrPools pools;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    auto direct = TranslateProgram(pools.inst, pools.block, flow_pool, env, code, {},
                                   analysis.uses_async_copy, &analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(direct);
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, direct);
    REQUIRE(!spirv.empty());

    CHECK_FALSE(direct.info.uses_raw_global_memory);
    CHECK_FALSE(direct.info.stores_raw_global_memory);
    REQUIRE(direct.info.storage_buffers_descriptors.size() == 1U);
    CHECK(direct.info.storage_buffers_descriptors[0].cbuf_index == 0U);
    CHECK(direct.info.storage_buffers_descriptors[0].cbuf_offset == 0x100U);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::LoadStorage32) == 2U);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::WriteStorage32) == 2U);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::LoadGlobal32) == 0U);
    CHECK(CountOpcode(direct, Shader::IR::Opcode::WriteGlobal32) == 0U);
}

TEST_CASE("AnalyzeProgram records generic_attributes_read from native ALD instructions",
          "[ampere_direct][video_core]") {
    const auto ald = [](u32 dest_reg, u32 address, u32 components) -> Instruction {
        return {{0xff007321U | (dest_reg << 16), (address << 8) | 0xffU,
                 ((components - 1U) << 10), Sched}};
    };
    const auto ast = [](u32 address, u32 src_reg, u32 components) -> Instruction {
        return {{0xff007322U, (address << 8) | src_reg,
                 ((components - 1U) << 10) | 0xffU, Sched}};
    };

    const std::array<Instruction, 6> code{{
        ald(2U, 0x70U, 4U),  // PositionX..PositionW (attrs 28..31) -> not generic
        ald(6U, 0x80U, 1U),  // Generic0X (attr 32) -> loc 0
        ald(8U, 0xb8U, 3U),  // Generic3Z, Generic3W, Generic4X (attrs 46..48) -> locs 3, 4
        ald(255U, 0xf0U, 2U), // R255 sink at Generic7X -> ignored
        ast(0x70U, 6U, 1U),
        EXIT,
    }};
    const auto analysis = AnalyzeProgram(code, {});
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));
    CHECK(analysis.generic_attributes_read == ((1u << 0) | (1u << 3) | (1u << 4)));
}

TEST_CASE("SPIR-V backend deduplicates bindless texture lookups per basic block",
          "[ampere_direct][video_core]") {
    const auto mov_cbuf = [](u32 rd, u32 bank, u32 byte_off) -> Instruction {
        return {{0xff007a02U | (rd << 16), (bank << 22) | (byte_off << 6), 0x00000f00U, Sched}};
    };
    const auto count_spv_op = [](const std::vector<u32>& code, spv::Op op) -> u32 {
        u32 count = 0;
        for (size_t offset = 5; offset < code.size();) {
            const u32 word_count = code[offset] >> 16;
            REQUIRE(word_count > 0);
            if (static_cast<spv::Op>(code[offset] & 0xffffU) == op) {
                ++count;
            }
            offset += word_count;
        }
        return count;
    };

    // Fragment shader: two TEX.B.LZ in entry block with identical varying handle R6 (from IPA,
    // different coords), then conditional branch around a third TEX.B.LZ in block 2 with the same
    // handle R6. Within block 1, BindlessSampled caches OpSampledImage; across block boundary it
    // re-emits to preserve SSA dominance.
    const auto skip_rel = Rel50(2 * 16);
    const std::array<Instruction, 18> code{{
        // IPA.PASS R6, Generic0X (dynamic per-pixel bindless handle)
        {{0x00067326U, 0x000000ffU, (7U << 17) | static_cast<u32>(Shader::IR::Attribute::Generic0X), Sched}},
        mov_cbuf(2U, 2U, 0x14U), // R2 = c[2][0x14] (branch predicate source)
        mov_cbuf(4U, 2U, 0x18U),
        mov_cbuf(5U, 2U, 0x1cU),
        mov_cbuf(8U, 2U, 0x20U),
        mov_cbuf(9U, 2U, 0x24U),
        mov_cbuf(16U, 2U, 0x28U),
        mov_cbuf(17U, 2U, 0x2cU),
        {{0x04127361U, 0x28000006U, 0x008e01ffU, Sched}}, // TEX.B.LZ R18, R4..5, R6 (block 1)
        {{0x08147361U, 0x28000006U, 0x008e01ffU, Sched}}, // TEX.B.LZ R20, R8..9, R6 (block 1, dup)
        {{0x12007221U, 20U, 0U, Sched}},                  // FADD R0, R18, R20
        {{0x0200720cU, 0x000000ffU, (5U << 12) | (7U << 17) | (7U << 20) | (7U << 23), Sched}}, // ISETP.NE P0
        {{0x00000947U, skip_rel[0], (7U << 23) | skip_rel[1], Sched}}, // @P0 BRA skip
        {{0x10167361U, 0x28000006U, 0x008e01ffU, Sched}}, // TEX.B.LZ R22, R16..17, R6 (block 2)
        {{0x00007221U, 22U, 0U, Sched}},                  // FADD R0, R0, R22
        // skip:
        {{0xff017802U, 0U, 0xf00U, Sched}},               // MOV R1..R2 = 0 (live exit)
        {{0xff027802U, 0U, 0xf00U, Sched}},
        EXIT,
    }};

    const auto analysis = AnalyzeProgram(code, {});
    INFO(analysis.error);
    REQUIRE(static_cast<bool>(analysis));

    TestEnv env{{}, analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2U,
               Shader::Stage::Fragment};
    auto host = MakeHostInfo();
    host.nvn_bindless_table = true;
    auto profile = MakeProfile();
    profile.support_sampled_image_array_nonuniform_indexing = true;

    IrPools pools;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    auto program = TranslateProgram(pools.inst, pools.block, flow_pool, env, code, {},
                                    analysis.uses_async_copy, &analysis.branch_targets, host);
    Shader::Optimization::VerificationPass(program);
    REQUIRE(program.info.texture_descriptors.size() == 1U);
    REQUIRE(program.info.texture_descriptors[0].bindless);

    Shader::RuntimeInfo runtime_info{};
    runtime_info.previous_stage_stores.Set(Shader::IR::Attribute::Generic0X);
    runtime_info.generic_input_types[0] = Shader::AttributeType::Float;
    Shader::Backend::Bindings bindings{};
    const auto spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, bindings);
    REQUIRE(!spirv.empty());
    // Note: entry block dedups 2 -> 1, block 2 emits 1 => 2 OpSampledImage total.
    CHECK(count_spv_op(spirv, spv::Op::OpImageSampleExplicitLod) == 3U);
    CHECK(count_spv_op(spirv, spv::Op::OpSampledImage) == 2U);
}
