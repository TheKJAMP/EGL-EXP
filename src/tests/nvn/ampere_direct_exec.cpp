// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Real-Vulkan GPU execution tests for direct Ampere -> IR Tier 1 opcodes, uniform datapath,
// high registers with bridged ops, warp ops, and native Ampere control flow.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "ampere_exec.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "video_core/native_tensor_programs.h"

namespace {

using namespace AmpereExec;

constexpr u32 ZeroGpr = 255U;
constexpr u32 ZeroUr = 63U;

Inst MovImm(u32 rd, u32 imm) {
    return {{0x00007802U | (rd << 16), imm, 0x00000f00U, Sched}};
}

Inst MovReg(u32 rd, u32 rs) {
    return {{0x00007202U | (rd << 16), rs, 0x00000f00U, Sched}};
}

Inst MovFromUr(u32 rd, u32 urs) {
    return {{0x00007c02U | (rd << 16), urs, 0x08000f00U, Sched}};
}

Inst UMovImm(u32 urd, u32 imm) {
    return {{0x00007882U | (urd << 16), imm, 0U, Sched}};
}

Inst ImadReg(u32 rd, u32 ra, u32 rb, u32 rc, bool neg_c = false) {
    return {{0x00007224U | (rd << 16) | (ra << 24), rb,
             0x078e0000U | (neg_c ? 0x800U : 0U) | rc, Sched}};
}

Inst ImadImm(u32 rd, u32 ra, u32 imm, u32 rc, bool neg_c = false) {
    return {{0x00007824U | (rd << 16) | (ra << 24), imm,
             0x078e0000U | (neg_c ? 0x800U : 0U) | rc, Sched}};
}

Inst ImadWideReg(u32 rd, u32 ra, u32 rb, u32 rc, bool is_signed, u32 pu = 7U) {
    return {{0x00007225U | (rd << 16) | (ra << 24), rb,
             0x07800000U | (pu << 17) | (is_signed ? 0x200U : 0U) | rc, Sched}};
}

Inst ImadHiReg(u32 rd, u32 ra, u32 rb, u32 rc, bool is_signed) {
    return {{0x00007227U | (rd << 16) | (ra << 24), rb,
             0x07800000U | (7U << 17) | (is_signed ? 0x200U : 0U) | rc, Sched}};
}

Inst Idp2a(u32 rd, u32 ra, u32 rb, u32 rc) {
    return {{0x00007226U | (rd << 16) | (ra << 24), rb, 0x00001000U | rc, Sched}};
}

Inst Iadd3Reg(u32 rd, u32 ra, u32 rb, u32 rc, bool neg_a = false, bool neg_b = false,
              bool neg_c = false, u32 pu = 7U, u32 pv = 7U, bool extended = false, u32 p_in = 15U) {
    u32 w2 = 0x07800000U | rc | (neg_a ? (1U << 8) : 0U) | (neg_c ? (1U << 11) : 0U) |
             (extended ? (1U << 10) : 0U) | (15U << 13) | (pu << 17) | (pv << 20);
    if (p_in != 15U) {
        w2 = (w2 & ~(15U << 23)) | (p_in << 23);
    }
    return {{0x00007210U | (rd << 16) | (ra << 24), rb | (neg_b ? 0x80000000U : 0U), w2, Sched}};
}

Inst LeaReg(u32 rd, u32 ra, u32 rb, u32 rc, u32 shift, bool high, bool sx32, bool neg_b = false,
            u32 pu = 7U, bool extended = false, u32 p_in = 15U) {
    const u32 w2 = (p_in << 23) | (7U << 20) | (pu << 17) | (high ? (1U << 16) : 0U) |
                   (shift << 11) | (extended ? (1U << 10) : 0U) | (sx32 ? (1U << 9) : 0U) | rc;
    return {{0x00007211U | (rd << 16) | (ra << 24), rb | (neg_b ? 0x80000000U : 0U), w2, Sched}};
}

Inst Lop3Reg(u32 rd, u32 ra, u32 rb, u32 rc, u32 lut, u32 pu = 7U) {
    return {{0x00007212U | (rd << 16) | (ra << 24), rb,
             0x07800000U | (pu << 17) | (lut << 8) | rc, Sched}};
}

Inst Plop3(u32 target1, u32 target2, u32 pa, u32 pb, u32 pc, u32 lut1, u32 lut2,
           bool uniform_c = false) {
    const u32 w2 = (pa << 23) | (target2 << 20) | (target1 << 17) | (pb << 13) |
                   ((lut1 >> 3) << 8) | (pc << 4) | (uniform_c ? (1U << 3) : 0U) | (lut1 & 7U);
    return {{0x0000781cU | (lut2 << 16), 0U, w2, Sched}};
}

Inst IsetpReg(u32 cmp, u32 comb, bool is_signed, u32 pu, u32 pv, u32 ra, u32 rb, u32 accum = 7U,
              bool ex = false, u32 p_low = 7U) {
    const u32 w2 = (accum << 23) | (pv << 20) | (pu << 17) | (cmp << 12) | (comb << 10) |
                   (is_signed ? (1U << 9) : 0U) | (ex ? (1U << 8) : 0U) | (p_low << 4);
    return {{0x0000720cU | (ra << 24), rb, w2, Sched}};
}

Inst SelReg(u32 rd, u32 ra, u32 rb, u32 pred4) {
    return {{0x00007207U | (rd << 16) | (ra << 24), rb, (pred4 << 23), Sched}};
}

Inst ShfReg(u32 rd, u32 lo, u32 shift_reg, u32 hi, u32 data_type, bool right, bool high_result,
            bool wrap) {
    const u32 w2 = (high_result ? (1U << 16) : 0U) | (right ? (1U << 12) : 0U) |
                   (wrap ? (1U << 11) : 0U) | (data_type << 9) | hi;
    return {{0x00007219U | (rd << 16) | (lo << 24), shift_reg, w2, Sched}};
}

Inst SgxtImm(u32 rd, u32 ra, u32 width, bool is_signed) {
    return {{0x0000781aU | (rd << 16) | (ra << 24), width, is_signed ? 0x200U : 0U, Sched}};
}

Inst PrmtImm(u32 rd, u32 ra, u32 rc, u32 sel) {
    return {{0x00007816U | (rd << 16) | (ra << 24), sel, rc, Sched}};
}

Inst FloReg(u32 rd, u32 rb, bool is_signed, bool shift, bool tilde = false) {
    return {{0xff007300U | (rd << 16), rb | (tilde ? 0x80000000U : 0U),
             0x000e0000U | (shift ? (1U << 10) : 0U) | (is_signed ? (1U << 9) : 0U), Sched}};
}

Inst PopcReg(u32 rd, u32 rb, bool tilde = false) {
    return {{0xff007309U | (rd << 16), rb | (tilde ? 0x80000000U : 0U), 0U, Sched}};
}

Inst BrevReg(u32 rd, u32 rb) {
    return {{0x00007301U | (rd << 16), rb, 0U, Sched}};
}

Inst P2R(u32 rd, u32 ra, u32 mask, u32 byte_sel = 0U) {
    return {{0x00007803U | (rd << 16) | (ra << 24), mask, byte_sel << 12, Sched}};
}

Inst R2P(u32 ra, u32 mask, u32 byte_sel = 0U) {
    return {{0x00007804U | (ra << 24), mask, byte_sel << 12, Sched}};
}

Inst IabsReg(u32 rd, u32 rb) {
    return {{0xff007213U | (rd << 16), rb, 0U, Sched}};
}

Inst ImnmxReg(u32 rd, u32 ra, u32 rb, u32 pred4, bool is_signed) {
    return {{0x00007217U | (rd << 16) | (ra << 24), rb,
             0x03800000U | (pred4 << 23) | (is_signed ? (1U << 9) : 0U), Sched}};
}

Inst Redux(u32 urd, u32 ra, u32 mode_word2, u32 pred4 = 7U) {
    return {{0x000003c4U | (pred4 << 12) | (urd << 16) | (ra << 24), 0U, mode_word2, Sched}};
}

Inst R2ur(u32 urd, u32 ra, u32 pred4 = 7U) {
    return {{0x000003c2U | (pred4 << 12) | (urd << 16) | (ra << 24), 0U, 0x000e0000U, Sched}};
}

Inst VoteU(u32 urd, u32 up_dst, u32 mode, u32 p_in) {
    return {{0x00007886U | (urd << 16), 0U,
             0x08000000U | (p_in << 23) | (up_dst << 17) | (mode << 8), Sched}};
}

std::array<u32, 2> Rel50(s64 byte_offset) {
    const u64 raw = static_cast<u64>(byte_offset) & ((1ULL << 50) - 1ULL);
    return {static_cast<u32>(raw), static_cast<u32>((raw >> 32) & 0x3ffffU)};
}

// Custom 32-lane / c1 GPU compute runner built on top of AmpereExec::MakeHeadlessVulkan.
class CustomComputeRig {
public:
    static std::unique_ptr<CustomComputeRig> Create() {
        auto vulkan = MakeHeadlessVulkan();
        if (!vulkan) {
            return nullptr;
        }
        return std::unique_ptr<CustomComputeRig>(new CustomComputeRig(std::move(vulkan)));
    }

    bool Run(std::span<const Inst> instructions, std::span<const u32> c1_words, u32 thread_x,
             u32 dispatch_x, std::span<const u32> init_words, std::span<u32> out_words,
             u32 reg_count = 32, u32 shared_bytes = 0, u32 local_bytes = 0) {
        using namespace std::chrono_literals;
        const u32 inst_bytes = static_cast<u32>(instructions.size() * sizeof(Inst));
        const u32 c1_offset = (inst_bytes + 0xffU) / 0x100U * 0x100U;
        const u32 code_bytes = c1_offset + static_cast<u32>(c1_words.size() * sizeof(u32));

        std::vector<u8> control(0x100, 0);
        Put32(control, 0, VC::ControlMagic);
        Put32(control, 4, VC::ControlVersion);
        Put32(control, 0x38, inst_bytes);
        Put32(control, 0x40, c1_offset);
        Put32(control, 0x44, code_bytes);
        control[0x4c] = static_cast<u8>(reg_count);
        Put32(control, 0x50, local_bytes * 32U);
        control[0x56] = 5; // compute
        Put32(control, 0x58, thread_x);
        Put32(control, 0x5c, 1);
        Put32(control, 0x60, 1);
        Put32(control, 0x64, shared_bytes);
        control[0xb1] = 1;
        control[0xb2] = 1;
        control[0xba] = 0x30;

        std::vector<u8> code(code_bytes, 0);
        std::memcpy(code.data(), instructions.data(), inst_bytes);
        if (!c1_words.empty()) {
            std::memcpy(code.data() + c1_offset, c1_words.data(), c1_words.size() * sizeof(u32));
        }

        const u64 serial = next_program++;
        VC::ProgramDesc desc{};
        desc.serial = serial;
        desc.control[5] = std::move(control);
        desc.code[5] = std::move(code);
        desc.stage_mask = 1u << 5;

        const size_t storage_bytes =
            (std::max)(init_words.size(), out_words.size()) * sizeof(u32);
        std::memset(data + kStorageOffset, 0xAA, storage_bytes);
        if (!init_words.empty()) {
            std::memcpy(data + kStorageOffset, init_words.data(), init_words.size() * sizeof(u32));
        }

        context->CreateProgram(desc);
        std::vector<u64> list;
        PushRecord(list, VC::Op::BindProgram,
                   VC::BindProgramPayload{.program_serial = serial, .stages = 1u << 5});
        const VC::BufferRange range{kPoolVa + kStorageOffset, storage_bytes};
        PushRecord(list, VC::Op::BindStorageBuffers,
                   VC::BindBuffersPayload{.stage = 5, .first = 0, .count = 1},
                   {reinterpret_cast<const u8*>(&range), sizeof(range)});
        PushRecord(list, VC::Op::Dispatch, VC::DispatchPayload{.x = dispatch_x, .y = 1, .z = 1});
        list.push_back(0);
        auto* end = reinterpret_cast<VC::RecordHeader*>(&list.back());
        end->op = static_cast<u16>(VC::Op::End);
        end->size = 8;

        VC::Submission sub{};
        sub.seq = ++next_seq;
        sub.queue = 0;
        sub.lists.push_back(std::move(list));
        sub.retained_serials = {serial, 1};
        const auto before = context->Stats();
        context->Submit(std::move(sub));
        if (!context->Wait(next_seq, 60s)) {
            return false;
        }
        const auto after = context->Stats();
        if (after.ops_failed != before.ops_failed || after.ops_unsupported != before.ops_unsupported) {
            return false;
        }
        std::memcpy(out_words.data(), data + kStorageOffset, out_words.size() * sizeof(u32));
        return true;
    }

private:
    static constexpr u64 kPoolVa = 0x5000'0000;
    static constexpr u64 kPoolSize = 4ull * 1024 * 1024;
    static constexpr u64 kStorageOffset = 0x1000;

    explicit CustomComputeRig(std::unique_ptr<HeadlessVulkan> vulkan_) : vulkan(std::move(vulkan_)) {
        data = memory.Add(kPoolVa, kPoolSize);
        context = std::make_unique<Vulkan::Nvn::Context>(
            *vulkan->device, vulkan->submit_mutex, nullptr,
            Vulkan::Nvn::ContextOptions{.allow_host_import = true});
        context->SetGuestMemory(&memory);
        context->CreatePool({.serial = 1, .guest_address = kPoolVa, .size = kPoolSize, .flags = 4});
    }

    static void Put32(std::vector<u8>& bytes, size_t off, u32 value) {
        std::memcpy(bytes.data() + off, &value, sizeof(value));
    }

    std::unique_ptr<HeadlessVulkan> vulkan;
    FakeMemory memory;
    u8* data{};
    std::unique_ptr<Vulkan::Nvn::Context> context;
    u64 next_program = 500;
    u64 next_seq = 0;
};

} // namespace

TEST_CASE("Direct Ampere GPU execution: integer multiply, wide/hi, and IDP.2A",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // R8=a, R9=b, R10=c_lo, R11=c_hi
    // R16 = IMAD(a, b, c_lo)
    // R17 = IMAD(a, b, -c_lo)
    // R18:R19 = IMAD.WIDE.U32(a, b, c_hi:c_lo) -> P0 carry out
    // R20 = IMAD.HI.S32(a, b, c_lo)
    // R21 = IDP.2A.LO.U16.U8(a, b, c_lo)
    // R22 = P0 ? 1 : 0
    const std::array<Inst, 7> body{{
        ImadReg(16, 8, 9, 10, false),
        ImadReg(17, 8, 9, 10, true),
        ImadWideReg(18, 8, 9, 10, false, 0),
        ImadHiReg(20, 8, 9, 10, true),
        Idp2a(21, 8, 9, 10),
        SelReg(22, 255, 255, 0), // dummy cleared below
        MovImm(22, 0),
    }};
    // Fix R22: MOV imm 1 to R23, SELP R22 = P0 ? R23 : RZ
    const std::array<Inst, 7> prog{{
        ImadReg(16, 8, 9, 10, false),
        ImadReg(17, 8, 9, 10, true),
        ImadWideReg(18, 8, 9, 10, false, 0),
        ImadHiReg(20, 8, 9, 10, true),
        Idp2a(21, 8, 9, 10),
        MovImm(23, 1),
        SelReg(22, 23, ZeroGpr, 0),
    }};

    const std::vector<Words> cases{
        {0U, 0U, 0U, 0U},
        {3U, 7U, 11U, 0U},
        {0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU},
        {0x80000001U, 0xfffffffdU, 0x12345678U, 0x9abcdef0U},
        {0x12345678U, 0x87654321U, 0xfedcba98U, 0x76543210U},
        {0xffff0002U, 0x00000304U, 100U, 0U},
    };
    const auto res = rig->Run(prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    for (size_t i = 0; i < cases.size(); ++i) {
        const u32 a = cases[i][0], b = cases[i][1], clo = cases[i][2], chi = cases[i][3];
        const u64 wide = u64{a} * u64{b} + ((u64{chi} << 32) | clo);
        const bool carry =
            (u64{a} * u64{b} > ~((u64{chi} << 32) | clo));
        const s64 swide = static_cast<s64>(static_cast<s32>(a)) *
                              static_cast<s64>(static_cast<s32>(b)) +
                          static_cast<s64>((u64{chi} << 32) | clo);
        const u32 idp = (a & 0xffffU) * (b & 0xffU) + (a >> 16) * ((b >> 8) & 0xffU) + clo;
        CHECK(res.outputs[i][0] == a * b + clo);
        CHECK(res.outputs[i][1] == a * b - clo);
        CHECK(res.outputs[i][2] == static_cast<u32>(wide));
        CHECK(res.outputs[i][3] == static_cast<u32>(wide >> 32));
        CHECK(res.outputs[i][4] == static_cast<u32>(static_cast<u64>(swide) >> 32));
        CHECK(res.outputs[i][5] == idp);
        CHECK(res.outputs[i][6] == (carry ? 1U : 0U));
    }
}

TEST_CASE("Direct Ampere GPU execution: IADD3, LEA, IABS, IMNMX", "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // Inputs: R8=a, R9=b, R10=c, R11=d
    // R16: R24 = (a + b) low with P0 carryout; R17 = IADD3.X(c, d, P0) (64-bit add!)
    // R18 = IADD3(a, -b, c)
    // R19 = LEA(a, b, 5) low, with P1 carryout (shift = 5 avoids Mesa 26.3.1 RADV/ACO's GFX9
    // s_lshl{1..4}_add_u32 SCC carry quirk on uniform 1x1x1 dispatches while testing shifts + carry)
    // R20 = LEA.HI.X(a, d, c, shift=5, P1) (64-bit left shift + 64-bit add!)
    // R21 = IABS(a)
    // R22 = IMNMX.S32(a, b, PT) (min), R23 = IMNMX.U32(a, b, !PT) (max)
    const std::array<Inst, 8> prog{{
        Iadd3Reg(16, 8, 9, ZeroGpr, false, false, false, 0),
        Iadd3Reg(17, 10, 11, ZeroGpr, false, false, false, 7, 7, true, 0),
        Iadd3Reg(18, 8, 9, 10, false, true, false),
        LeaReg(19, 8, 9, ZeroGpr, 5, false, false, false, 1),
        LeaReg(20, 8, 11, 10, 5, true, false, false, 7, true, 1),
        IabsReg(21, 8),
        ImnmxReg(22, 8, 9, 7, true),
        ImnmxReg(23, 8, 9, 15, false),
    }};

    const std::vector<Words> cases{
        {0U, 0U, 0U, 0U},
        {0xffffffffU, 1U, 10U, 20U},
        {0x80000000U, 0xffffffffU, 0x12345678U, 0x9abcdef0U},
        {static_cast<u32>(-42), 17U, 0xdeadbeefU, 0xcafebabeU},
        {0x30000001U, 0x40000000U, 5U, 9U},
        {0x04000001U, 0x90000000U, 5U, 9U}, // (a << 5) = 0x80000020; + 0x90000000 produces carry P1 == 1
    };
    const auto res = rig->Run(prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    for (size_t i = 0; i < cases.size(); ++i) {
        const u32 a = cases[i][0], b = cases[i][1], c = cases[i][2], d = cases[i][3];
        const u64 add_lo = u64{a} + b;
        const u32 carry0 = static_cast<u32>(add_lo >> 32);
        const u64 lea_lo = (u64{a} << 5) & 0xffffffffULL;
        const u64 lea_sum_lo = lea_lo + b;
        const u32 carry1 = static_cast<u32>(lea_sum_lo >> 32);
        const u32 lea_hi = ((c << 5) | (a >> 27)) + d + carry1;

        CHECK(res.outputs[i][0] == static_cast<u32>(add_lo));
        CHECK(res.outputs[i][1] == c + d + carry0);
        CHECK(res.outputs[i][2] == a - b + c);
        CHECK(res.outputs[i][3] == static_cast<u32>(lea_sum_lo));
        CHECK(res.outputs[i][4] == lea_hi);
        const s32 sa = static_cast<s32>(a);
        CHECK(res.outputs[i][5] == static_cast<u32>(sa == std::numeric_limits<s32>::min() ? sa : (sa < 0 ? -sa : sa)));
        CHECK(res.outputs[i][6] == static_cast<u32>((std::min)(static_cast<s32>(a), static_cast<s32>(b))));
        CHECK(res.outputs[i][7] == (std::max)(a, b));
    }
}

TEST_CASE("Direct Ampere GPU execution: logic, bitfield, shifts, predicates, P2R/R2P",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // R8=a, R9=b, R10=c, R11=shift
    // R16 = LOP3(a, b, c, 0xca)  -> (a & b) | (~a & c) ? Wait: index = (a<<2)|(b<<1)|c -> bit 0..7 of 0xca = bits 1,3,6,7 -> (a&b) | (~a&c)
    // R17 = SHF.R.S64.HI(c:a, R11) wrap=0 (arithmetic right shift of high word via 64-bit) -> Wait: R11 & 31
    // R18 = SGXT.S32(a, 12)
    // R19 = PRMT(a, c, 0x9421)
    // R20 = FLO.S32(a)
    // R21 = POPC(a)
    // R22 = BREV(a)
    // ISETP64: P0 = (c:a <u c:b), then PLOP3 dual outputs P1=(P0^P2), P2=(!P0 & P2), P2R R23
    const std::array<Inst, 14> prog{{
        Lop3Reg(16, 8, 9, 10, 0xcaU),
        ShfReg(17, 8, 11, 10, 2U, true, false, false), // SHF.R.S64 (c:a) >> R11 low word
        SgxtImm(18, 8, 12U, true),
        PrmtImm(19, 8, 10, 0x9421U),
        FloReg(20, 8, true, false),
        PopcReg(21, 8),
        BrevReg(22, 8),
        // 64-bit compare: (R10:R8) <u (R10:R9)
        IsetpReg(1U, 0U, false, 0U, 7U, 8U, 9U),                    // P0 = a <u b
        IsetpReg(1U, 0U, false, 0U, 7U, 10U, 10U, 7U, true, 0U),    // P0 = (c <u c) || (c==c && P0)
        IsetpReg(2U, 0U, false, 2U, 7U, 8U, 9U),                    // P2 = (a == b)
        Plop3(1U, 3U, 0U, 2U, 7U, 0x3cU /* a^b */, 0x0cU /* !a&b */),
        R2P(8, 0x10U, 0U),                                          // P4 = (a >> 4) & 1
        MovImm(23, 0),
        P2R(23, 23, 0x1fU, 0U),
    }};

    const std::vector<Words> cases{
        {0U, 0U, 0U, 0U},
        {0x00000800U, 0x00001000U, 0xffffffffU, 4U},
        {0x80000f00U, 0x80000f00U, 0x12345678U, 17U},
        {0xffffffffU, 0x12345678U, 0x9abcdef0U, 63U},
        {0x12345678U, 0x87654321U, 0x80000000U, 70U},
    };
    const auto res = rig->Run(prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    for (size_t i = 0; i < cases.size(); ++i) {
        const u32 a = cases[i][0], b = cases[i][1], c = cases[i][2], sh = cases[i][3];
        // LOP3 bit = ((a_bit << 2) | (b_bit << 1) | c_bit) -> 0xca = (~a & c) | (a & b)
        u32 expected_lop3 = (~a & c) | (a & b);
        const s64 packed = static_cast<s64>((u64{c} << 32) | a);
        const u32 safe_sh = (std::min)(sh, 63U);
        const s64 shifted = packed >> safe_sh;
        const s32 sgxt = static_cast<s32>(a << 20) >> 20;
        auto prmt_byte = [&](u32 idx) -> u32 {
            const u32 nib = (0x9421U >> (idx * 4)) & 0xfU;
            const u32 src = (nib & 4U) ? c : a;
            const u32 pos = (nib & 3U) * 8U;
            if (nib & 8U) {
                return ((src >> (pos + 7U)) & 1U) ? 0xffU : 0U;
            }
            return (src >> pos) & 0xffU;
        };
        const u32 prmt = prmt_byte(0) | (prmt_byte(1) << 8) | (prmt_byte(2) << 16) | (prmt_byte(3) << 24);
        const s32 flo_expected = [&] {
            if (a == 0 || a == 0xffffffffU) {
                return -1;
            }
            const u32 v = (static_cast<s32>(a) < 0) ? ~a : a;
            return 31 - std::countl_zero(v);
        }();
        u32 brev = 0;
        for (u32 bit = 0; bit < 32; ++bit) {
            if ((a >> bit) & 1U) {
                brev |= 1U << (31U - bit);
            }
        }
        const bool p0 = ((u64{c} << 32) | a) < ((u64{c} << 32) | b);
        const bool p2 = (a == b);
        const bool p1 = p0 ^ p2;
        const bool p3 = !p0 && p2;
        const bool p4 = ((a >> 4) & 1U) != 0U;
        const u32 pmask =
            u32{p0} | (u32{p1} << 1) | (u32{p2} << 2) | (u32{p3} << 3) | (u32{p4} << 4);

        CHECK(res.outputs[i][0] == expected_lop3);
        CHECK(res.outputs[i][1] == static_cast<u32>(shifted));
        CHECK(res.outputs[i][2] == static_cast<u32>(sgxt));
        CHECK(res.outputs[i][3] == prmt);
        CHECK(res.outputs[i][4] == static_cast<u32>(flo_expected));
        CHECK(res.outputs[i][5] == static_cast<u32>(std::popcount(a)));
        CHECK(res.outputs[i][6] == brev);
        CHECK(res.outputs[i][7] == pmask);
    }
}

TEST_CASE("Direct Ampere GPU execution: multi-lane warp ops and BRXU jump table",
          "[nvn][ampere_direct_exec]") {
    auto rig = CustomComputeRig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    SECTION("32-thread REDUX, R2UR, VOTEU, UIADD3, @UPn") {
        // Storage buffer layout: out[tid] = {tid_val, sum_redux, smin_redux, r2ur_masked, vote_ballot, ur_pred_select}
        // Uses 1 workgroup of 32 threads:
        // R2:R3 = storage base (c[0][0x20/0x24])
        // R4 = SR_TID_X
        // R5 = IMAD.MOV.U32 (R4 * 2 - 10)  (signed mix: -10 .. +52)
        // REDUX.SUM UR4, R5
        // REDUX.MIN.S32 UR5, R5
        // ISETP.GE.U32 P0, PT, R4, 5, PT
        // @P0 R2UR UR6, R5        (first lane >= 5 -> tid 5 -> val = 0; else test pred >= 10 -> val = 10)
        // Actually use R4 >= 10 -> tid 10 -> 10*2 - 10 = 10!
        // VOTEU.ANY UR7, UP0, P0  (ballot = 0xfffffc00, UP0 = true)
        // @UP0 UMOV UR8, 0x77
        // @!UP0 UMOV UR8, 0x99
        // Store 8 words per thread
        const std::array<Inst, 27> shader{{
            S2R(4, 33),                                            // SR_TID_X
            MovC0(2, 0x20),
            MovC0(3, 0x24),
            ImadImm(5, 4, 2, ZeroGpr),
            {{0x00007810U | (5U << 16) | (5U << 24), static_cast<u32>(-10), 0x07ffe0ffU, Sched}}, // IADD3 R5, R5, -10, RZ
            Redux(4, 5, 0xc000U),                                  // REDUX.SUM UR4, R5
            Redux(5, 5, 0x10200U),                                 // REDUX.MIN.S32 UR5, R5
            IsetpReg(6U, 0U, false, 0U, 7U, 4U, ZeroGpr),          // placeholder patched below
            UMovImm(6, 999),
            R2ur(6, 5, 0U),                                        // @P0 R2UR UR6, R5
            VoteU(7, 0U, 1U, 0U),                                  // VOTEU.ANY UR7, UP0, P0
            UMovImm(8, 0x99U),
            {{0x00000882U | (8U << 16), 0x77U, 0U, Sched}},        // @UP0 UMOV UR8, 0x77
            // Storage index: byte offset = R4 * 32 -> LEA R2, R4, R2, 5 (with carry -> R3 via IADD3.X? Use R6:R7 temp)
            LeaReg(6, 4, 2, ZeroGpr, 5U, false, false, false, 1U),
            Iadd3Reg(7, 3, ZeroGpr, ZeroGpr, false, false, false, 7U, 7U, true, 1U),
            Stg(6, 0U, 5U),
            MovFromUr(9, 4U), Stg(6, 4U, 9U),
            MovFromUr(9, 5U), Stg(6, 8U, 9U),
            MovFromUr(9, 6U), Stg(6, 12U, 9U),
            MovFromUr(9, 7U), Stg(6, 16U, 9U),
            MovFromUr(9, 8U), Stg(6, 20U, 9U),
            Exit,
        }};
        auto patched = shader;
        // Patch idx 7 to ISETP.GE.U32 P0, PT, R4, 10, PT (`0x80c` imm)
        patched[7] = {{0x0000780cU | (4U << 24), 10U, (7U << 23) | (7U << 20) | (0U << 17) | (6U << 12), Sched}};

        std::vector<u32> out(32 * 8, 0);
        REQUIRE(rig->Run(patched, {}, 32, 1, {}, out));

        s32 expected_sum = 0;
        s32 expected_min = std::numeric_limits<s32>::max();
        for (u32 t = 0; t < 32; ++t) {
            const s32 v = static_cast<s32>(t * 2) - 10;
            expected_sum += v;
            expected_min = (std::min)(expected_min, v);
        }
        CHECK(out[1] == static_cast<u32>(expected_sum));
        CHECK(out[2] == static_cast<u32>(expected_min));
        CHECK(out[3] == 10U);         // lane 10 -> 10*2 - 10 = 10
        CHECK(out[4] == 0xfffffc00U); // lanes 10..31 set
        CHECK(out[5] == 0x77U);       // @UP0 took true branch
    }

    SECTION("BRXU uniform jump table with c1 targets") {
        // Dispatch 4 groups (1 thread each): group reads input[gid], jumps via BRXU to label k, writes case result to out[gid].
        const auto rel_brxu = Rel50(-12 * 16);
        const std::array<Inst, 21> shader{{
            S2R(4, SR_CTAID_X),
            MovC0(2, 0x20),
            MovC0(3, 0x24),
            LeaReg(6, 4, 2, ZeroGpr, 2U, false, false, false, 0U),
            Iadd3Reg(7, 3, ZeroGpr, ZeroGpr, false, false, false, 7U, 7U, true, 0U),
            Ldg(5, 6, 0U),                                        // R5 = input[gid]
            R2ur(4, 5, 7U),                                       // UR4 = R5
            {{0x0400788cU, 3U, 0x0bf01070U, Sched}},              // UISETP.LT.U32 UP0, UR4, 3
            {{0x04047887U, 3U, 0x08000000U, Sched}},              // USEL UR4, UR4, 3, UP0
            {{0x04047899U, 2U, 0x0800063fU, Sched}},              // USHF.L.U32 UR4, UR4, 2, URZ
            {{0x04047abbU, (1U << 22), 0x08000800U, Sched}},      // ULDC UR4, c[1][UR4]
            {{0x04007958U, rel_brxu[0], 0x0b800000U | rel_brxu[1], Sched}}, // BRXU UR4, 0x0
            // case 0 (idx 12):
            MovImm(9, 100U), Stg(6, 16U, 9U), Exit,
            // case 1 (idx 15):
            MovImm(9, 201U), Stg(6, 16U, 9U), Exit,
            // case 2 (idx 18):
            MovImm(9, 302U), Stg(6, 16U, 9U), Exit,
        }};
        // Need 4th case for clamp 3! Wait: clamp to 2 -> case 3 unused if we use clamp 2, OR add case 3 (idx 21). Let's extend to 24 insts!
        const auto rel_brxu2 = Rel50(-12 * 16);
        std::vector<Inst> full(shader.begin(), shader.end());
        full[11] = {{0x04007958U, rel_brxu2[0], 0x0b800000U | rel_brxu2[1], Sched}};
        full.push_back(MovImm(9, 403U)); // idx 21
        full.push_back(Stg(6, 16U, 9U));
        full.push_back(Exit);
        const auto self_rel = Rel50(-16);
        full.push_back({{0x00007947U, self_rel[0], 0x03800000U | self_rel[1], Sched}});

        const std::vector<u32> c1{12U * 16U, 15U * 16U, 18U * 16U, 21U * 16U};
        const std::vector<u32> init{2U, 0U, 99U, 1U, 0U, 0U, 0U, 0U};
        std::vector<u32> out(8, 0);
        REQUIRE(rig->Run(full, c1, 1, 4, init, out));
        CHECK(out[4] == 302U);
        CHECK(out[5] == 100U);
        CHECK(out[6] == 403U); // clamped to 3!
        CHECK(out[7] == 201U);
    }
}

TEST_CASE("Direct Ampere GPU execution: high GPRs + UR40 + @UPn + Tier 2 FFMA",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // Exercise R220, R239 (does NOT collide with uniform predicates!), R250, UR40, UP4 + bridged FFMA!
    const std::array<Inst, 10> prog{{
        UMovImm(40U, 0x40000000U),                                // UR40 = 2.0f
        MovReg(220U, 8U),                                         // R220 = input0 (float bits)
        MovReg(239U, 9U),                                         // R239 = input1 (float bits)
        {{0xdc007e23U | (250U << 16), 40U, 0x080000efU, Sched}},  // FFMA R250, R220, R220, R239? Note: c=UR40 via form 7!
        // Re-check FFMA form 7: 0xe23 has c = UR(32..39), b = R(64..71), a = R(24..31)!
        // Let's put b=R239 (0xef) at 64..71 and c=UR40 (40) at 32..37!
        MovFromUr(16U, 40U),                                      // R16 = 2.0f
        // Compare (R8 as uint < 5): sets UP4
        {{0x2800788cU, 0x40000001U, (4U << 17) | (7U << 20) | (7U << 23) | (1U << 12) | 0x08000000U, Sched}},
        {{0x00107882U | (41U << 16), 11U, 0U, Sched}},            // UMOV UR41, 11
        {{0x00107882U | (41U << 16), 22U, 0U, Sched}},            // overwritten below with @UP4
        MovReg(17U, 250U),
        MovFromUr(18U, 41U),
    }};
    auto patched = prog;
    // Patch idx 3: FFMA R250, R220, R239, UR40 (c = UR40 -> bits 32..37 = 40, b = R239 -> bits 64..71 = 239)
    patched[3] = {{0xdcfa7e23U, 40U, 0x080000efU, Sched}};
    // Patch idx 6 & 7: UMOV UR41, 11 then @UP4 UMOV UR41, 22 -> UR40 < 0x40000001 is TRUE!
    patched[6] = UMovImm(41U, 11U);
    patched[7] = {{0x00004882U | (41U << 16), 22U, 0U, Sched}}; // @UP4 UMOV UR41, 22

    const std::vector<Words> cases{
        {std::bit_cast<u32>(3.0f), std::bit_cast<u32>(4.0f)},   // 3 * 4 + 2 = 14
        {std::bit_cast<u32>(-1.5f), std::bit_cast<u32>(6.0f)},  // -1.5 * 6 + 2 = -7
    };
    const auto res = rig->Run(patched, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    CHECK(res.outputs[0][0] == 0x40000000U);
    CHECK(res.outputs[0][1] == std::bit_cast<u32>(14.0f));
    CHECK(res.outputs[0][2] == 22U);
    CHECK(res.outputs[1][0] == 0x40000000U);
    CHECK(res.outputs[1][1] == std::bit_cast<u32>(-7.0f));
    CHECK(res.outputs[1][2] == 22U);
}

TEST_CASE("Direct Ampere GPU execution: Tier 2 FP32 ALU and predicated if-conversion",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // Inputs: R8=f_a, R9=f_b, R10=f_c, R11=selector_u32
    const std::array<Inst, 13> prog{{
        // FFMA R16, -R8, |R9|, R10
        {{0x08107223U, 0x40000009U, 0x0000010aU, Sched}},
        // FMUL.M2.FTZ R17, R8, R9
        {{0x08117220U, 0x00000009U, (5U << 20) | (1U << 16), Sched}},
        // FADD.SAT R18, R8, 0.5f  (form 4: 0x821)
        {{0x08127821U, 0x3f000000U, (1U << 13), Sched}},
        // FSETP.LT.AND P0, P1, R8, R9, PT
        {{0x0800720bU, 0x00000009U, (7U << 23) | (1U << 20) | (0U << 17) | (1U << 12), Sched}},
        // FSEL R19, R8, R10, P0
        {{0x08137208U, 0x0000000aU, 0U, Sched}},
        // FSET.BF.GT.AND R20, R8, R9, PT
        {{0x0814720aU, 0x00000009U, (7U << 23) | (4U << 12), Sched}},
        // FMNMX.NAN R21, R8, R9, PT
        {{0x08157209U, 0x00000009U, (7U << 23) | (1U << 17), Sched}},
        // MUFU.RCP R22, R10
        {{0x00167308U, 0x0000000aU, (4U << 10), Sched}},
        // Test self-predicated if-converted chain:
        // MOV R23, 10; ISETP.NE P2, PT, R11, 0, PT; @P2 MOV R23, 77; @P2 ISETP.EQ.AND P2, P3, R11, 1, P2; @P2 IADD3 R23, R23, 5, RZ
        MovImm(23U, 10U),
        {{0x0b00780cU, 0U, (7U << 23) | (7U << 20) | (2U << 17) | (5U << 12), Sched}},
        {{0x00172802U, 77U, 0x00000f00U, Sched}},
        {{0x0b00280cU, 1U, (2U << 23) | (3U << 20) | (2U << 17) | (2U << 12), Sched}},
        {{0x17172810U, 5U, 0x07ffe0ffU, Sched}},
    }};

    const std::vector<Words> cases{
        {std::bit_cast<u32>(1.5f), std::bit_cast<u32>(-2.0f), std::bit_cast<u32>(4.0f), 1U},
        {std::bit_cast<u32>(-0.25f), std::bit_cast<u32>(3.0f), std::bit_cast<u32>(2.0f), 2U},
        {std::bit_cast<u32>(0.75f), 0x7fc00000U, std::bit_cast<u32>(2.0f), 0U},
    };
    const auto res = rig->Run(prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    // Case 0: a=1.5, b=-2, c=4, sel=1 -> -1.5 * 2 + 4 = 1.0; 1.5*(-2)*2 = -6; sat(2.0)=1.0; 1.5 < -2 false -> fsel=4, fset=1, mnmx=-2, rcp=0.25, R23=82
    CHECK(res.outputs[0][0] == std::bit_cast<u32>(1.0f));
    CHECK(res.outputs[0][1] == std::bit_cast<u32>(-6.0f));
    CHECK(res.outputs[0][2] == std::bit_cast<u32>(1.0f));
    CHECK(res.outputs[0][3] == std::bit_cast<u32>(4.0f));
    CHECK(res.outputs[0][4] == std::bit_cast<u32>(1.0f));
    CHECK(res.outputs[0][5] == std::bit_cast<u32>(-2.0f));
    CHECK(std::abs(std::bit_cast<f32>(res.outputs[0][6]) - 0.25f) < 1e-5f);
    CHECK(res.outputs[0][7] == 82U);

    // Case 1: a=-0.25, b=3, c=2, sel=2 -> 0.25*3+2=2.75; -0.25*3*2=-1.5; sat(0.25)=0.25; -0.25<3 true -> fsel=-0.25, fset=0, mnmx=-0.25, rcp=0.5, R23=77
    CHECK(res.outputs[1][0] == std::bit_cast<u32>(2.75f));
    CHECK(res.outputs[1][1] == std::bit_cast<u32>(-1.5f));
    CHECK(res.outputs[1][2] == std::bit_cast<u32>(0.25f));
    CHECK(res.outputs[1][3] == std::bit_cast<u32>(-0.25f));
    CHECK(res.outputs[1][4] == std::bit_cast<u32>(0.0f));
    CHECK(res.outputs[1][5] == std::bit_cast<u32>(-0.25f));
    CHECK(std::abs(std::bit_cast<f32>(res.outputs[1][6]) - 0.5f) < 1e-5f);
    CHECK(res.outputs[1][7] == 77U);

    // Case 2: b=NaN, sel=0 -> FMNMX.NAN returns NaN (0x7fffffff), R23=10
    CHECK((res.outputs[2][5] & 0x7fffffffU) > 0x7f800000U);
    CHECK(res.outputs[2][7] == 10U);
}

TEST_CASE("Direct Ampere GPU execution: Batch B1 FP16 ALU, comparisons, and MUFU.F16",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // Inputs: R8=a_h2, R9=b_h2, R10=c_h2, R11=f_x
    const std::array<Inst, 10> checked_prog{{
        // HADD2 R16, R8, R9
        {{0x08107230U, 9U, 0U, Sched}},
        // HMUL2 R17, R8, R9
        {{0x08117232U, 9U, 0U, Sched}},
        // HFMA2.RELU R18, R8, R9, R10, PT
        {{0x08127231U, 9U, (1U << 15) | (7U << 23) | 10U, Sched}},
        // HFMA2.SAT R19, R8, R9.H0_NH1, R11.H0_H0
        {{0x08137231U, 9U, (1U << 22) | (2U << 17) | (1U << 13) | 11U, Sched}},
        // HMNMX2 R20, R8, R9, PT
        {{0x08147240U, 9U, (7U << 23), Sched}},
        // HSET2.LT.AND R21, R8, R9, PT
        {{0x08157233U, 9U, (7U << 23) | (1U << 12), Sched}},
        // HSETP2.GT.AND P0, P1, R8, R9, PT
        {{0x08007234U, 9U, (7U << 23) | (1U << 20) | (0U << 17) | (4U << 12), Sched}},
        MovImm(22U, 222U),
        // @P0 MOV R22, 111
        {{0x00160802U, 111U, 0x00000f00U, Sched}},
        // MUFU.RCP.F16 R23, R9.H1
        {{0x00177308U, (1U << 28) | 9U, (4U << 10) | (1U << 9), Sched}},
    }};

    // a=(1.5h, -2.0h), b=(0.5h, 4.0h), c=(-3.0h, 10.0h), d=(6.5h, 0.0h -> H0_H0=6.5h)
    const std::vector<Words> cases{
        {0xc0003e00U, 0x44003800U, 0x4900c200U, 0x00004680U},
    };
    const auto res = rig->Run(checked_prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    // HADD2: (2.0h, 2.0h) = 0x40004000
    CHECK(res.outputs[0][0] == 0x40004000U);
    // HMUL2: (0.75h, -8.0h) = 0xc8003a00
    CHECK(res.outputs[0][1] == 0xc8003a00U);
    // HFMA2.RELU: lo=0.75-3.0=-2.25 -> 0x0000, hi=-8.0+10.0=2.0 -> 0x4000
    CHECK(res.outputs[0][2] == 0x40000000U);
    // HFMA2.SAT H0_NH1: b'=(-4.0h, 0.5h), a*b'=(-6.0, -1.0), + (6.5, 6.5) -> (0.5, 5.5 -> sat 1.0) = (0x3800, 0x3c00)
    CHECK(res.outputs[0][3] == 0x3c003800U);
    // HMNMX2: (0.5h, -2.0h) = 0xc0003800
    CHECK(res.outputs[0][4] == 0xc0003800U);
    // HSET2.LT: (1.5<0.5=0, -2.0<4.0=0xffff) = 0xffff0000
    CHECK(res.outputs[0][5] == 0xffff0000U);
    // HSETP2.GT: 1.5>0.5 = true -> P0=true -> R22=111
    CHECK(res.outputs[0][6] == 111U);
    // MUFU.RCP.F16(4.0h) = 0.25h = 0x3400
    CHECK((res.outputs[0][7] & 0xffffU) == 0x3400U);
}

TEST_CASE("Direct Ampere GPU execution: Batch B1 scalar/packed conversions and FRND/FSWZADD",
          "[nvn][ampere_direct_exec]") {
    auto rig = Rig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // Inputs: R8=f_a(-3.75f), R9=f_b(2.25f), R10=0x00ff8040, R11=0x12345678
    const std::array<Inst, 8> prog{{
        // F2FP.PACK_AB R16, R8, R9 -> lo=f16(2.25)=0x4080, hi=f16(-3.75)=0xc380
        {{0x0810723eU, 9U, 0U, Sched}},
        // F2FP.MERGE_C R17, RZ, R9, R11 -> lo=0x4080, hi=0x5678
        {{0xff11723eU, 9U, (1U << 14) | 11U, Sched}},
        // FRND.F32.RM R18, R8 -> floor(-3.75f) = -4.0f
        {{0x00127307U, 8U, (2U << 20) | (1U << 14) | (2U << 11), Sched}},
        // F2I.S32.FLOOR R19, R8 -> -4
        {{0x00137305U, 8U, 0x00207100U, Sched}},
        // I2F.F32.S8 R20, R10.B1 -> -128.0f
        {{0x00147306U, (1U << 28) | 10U, 0x00001400U, Sched}},
        // MOV R21, 0x51800000 (68719476736.0f = 2^36)
        MovImm(21U, 0x51800000U),
        // F2I.FTZ.U64.TRUNC R22, R21 -> R22:R23 = {0, 0x10}
        {{0x00167311U, 21U, 0x0021d800U, Sched}},
        // I2F.F32.U64 R21, R22 -> 68719476736.0f
        {{0x00157312U, 22U, 0x00301000U, Sched}},
    }};

    const std::vector<Words> cases{
        {std::bit_cast<u32>(-3.75f), std::bit_cast<u32>(2.25f), 0x00ff8040U, 0x12345678U},
    };
    const auto res = rig->Run(prog, cases);
    INFO(res.error);
    REQUIRE(res.error.empty());

    CHECK(res.outputs[0][0] == 0xc3804080U);
    CHECK(res.outputs[0][1] == 0x56784080U);
    CHECK(res.outputs[0][2] == std::bit_cast<u32>(-4.0f));
    CHECK(res.outputs[0][3] == static_cast<u32>(-4));
    CHECK(res.outputs[0][4] == std::bit_cast<u32>(-128.0f));
    CHECK(res.outputs[0][5] == 0x51800000U);
    CHECK(res.outputs[0][6] == 0U);
    CHECK(res.outputs[0][7] == 0x10U);
}

TEST_CASE("Direct Ampere GPU execution: Batch B2 shared, LDGSTS dynamic enable, and global atomics",
          "[nvn][ampere_direct_exec]") {
    auto rig = CustomComputeRig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // 1 workgroup of 4 threads reading init[0..7] at c[0][0x20/0x24]:
    // Lane 0 (P0=true) runs LDGSTS.128 from init[0..3] -> shared[0..15]
    // Lane 1 (P0=false, invalid null global ptr R10:R11 = 0ULL!) runs LDGSTS.128 -> shared[16..31] (must zero-fill without reading 0ULL!)
    // Lane 0 stores back shared[0..3] (from init[0..3]) & shared[4..7] (zero-filled by lane 1)
    // Plus ATOMS.ADD on shared[64], BAR.SYNC, ATOMG.E.ADD, RED.E.MAX.S32, LDL/STL spill!
    const std::array<Inst, 29> shader{{
        S2R(4, 33),                                                 // R4 = tid (0..3)
        MovC0(2, 0x20),
        MovC0(3, 0x24),
        // P0 = (R4 == 0)
        {{0x0400780cU, 0U, (7U << 23) | (7U << 20) | (0U << 17) | (2U << 12), Sched}},
        // R10:R11 = P0 ? R2:R3 : 0:0 (unsafe null global address for lanes 1..3!)
        MovImm(10U, 0U),
        MovImm(11U, 0U),
        {{0x000a0202U, 2U, 0x00000f00U, Sched}},                   // @P0 MOV R10, R2
        {{0x000b0202U, 3U, 0x00000f00U, Sched}},                   // @P0 MOV R11, R3
        // R12 = tid * 16 (shared offset)
        {{0x040c7824U, 16U, 0x078e02ffU, Sched}},
        // LDGSTS.E.BYPASS.128 [R12 + 0], [R10 + 0], P0
        {{0x0a0c7faeU, 0U, 0x0c101c00U, Sched}},
        // Wait for LDGSTS -> LDGDEPBAR + DEPBAR.LE SB0, 0 + BAR.SYNC 0
        {{0x000079afU, 0U, 0U, Sched}},
        {{0x0000791aU, 0x8000U, 0U, Sched}},
        {{0x00007b1dU, 0U, 0U, Sched}},
        // Shared atomic sum: lane 0 zeroes shared[64], BAR.SYNC, all lanes ATOMS.ADD [64], (tid + 1)
        {{0xff000388U, (64U << 8) | ZeroGpr, (4U << 9), Sched}},   // @P0 STS.32 [RZ+64], RZ
        {{0x00007b1dU, 0U, 0U, Sched}},
        {{0x040d7810U, 1U, 0x07ffe0ffU, Sched}},                   // R13 = tid + 1
        {{0xff0e738cU, (64U << 8) | 13U, 0U, Sched}},              // ATOMS.ADD R14, [64], R13
        {{0x00007b1dU, 0U, 0U, Sched}},
        // Lane 0 loads results and writes out[0..7]:
        //   out[0] = shared[0] (init[0] = 0x11223344)
        //   out[1] = shared[4] (init[1] = 0x55667788)
        //   out[2] = shared[16] (0 from !P0 LDGSTS)
        //   out[3] = shared[64] (1+2+3+4 = 10)
        //   out[4] = ATOMG.E.ADD old value (out[4] starts at 100 -> old=100, new=107)
        //   out[5] = RED.E.MAX.S32 (init 50 vs 120 -> 120)
        //   out[6] = STL.8 / LDL.S8 sign-extend check (0xfe -> -2)
        // ATOMG & RED run cleanly on @P0:
        {{0xff0f0984U, (0U << 8), (6U << 9) & ~0x08000000U, Sched}}, // @P0 LDS.128 R15..18? Wait: R15 not 4-aligned; load scalars!
        MovImm(15U, 0U),
        Exit,
    }};
    // Cleanly write out-of-branch @P0 suffix by replacing index 18..20 cleanly below:
    std::vector<Inst> full(shader.begin(), shader.begin() + 18);
    auto append = [&](Inst i) { full.push_back(i); };
    // Non-lane-0 early exit: ISETP.NE P1, PT, R4, 0, PT; @P1 EXIT;
    append({{0x0400780cU, 0U, (7U << 23) | (7U << 20) | (1U << 17) | (5U << 12), Sched}});
    append({{0x0000194dU, 0U, 0x03800000U, Sched}}); // @P1 EXIT
    // Only lane 0 continues unconditionally:
    append({{0xff107984U, (0U << 8), (4U << 9), Sched}});          // LDS.32 R16, [RZ+0]
    append({{0xff117984U, (4U << 8), (4U << 9), Sched}});          // LDS.32 R17, [RZ+4]
    append({{0xff127984U, (16U << 8), (4U << 9), Sched}});         // LDS.32 R18, [RZ+16]
    append({{0xff137984U, (64U << 8), (4U << 9), Sched}});         // LDS.32 R19, [RZ+64]
    append(MovImm(20U, 7U));
    append({{0x021573a8U, (32U << 8) | 20U, 0x001ee100U, Sched}}); // ATOMG.E.ADD R21, [R2+32], R20 -> out[0]? Wait output base is +16 bytes (init[0..3]=16B, init[4..7]=16B -> put ATOMG target at +16, RED target at +20!)
    // Let's write to out[0..7] starting at R2:R3 + 32 (offset 0x20), using offset 0x10 for ATOMG & 0x14 for RED:
    full.pop_back(); // replace ATOMG offset 32 -> 16
    append({{0x021573a8U, (16U << 8) | 20U, 0x001ee100U, Sched}}); // R21 = old slot[4]
    append(MovImm(22U, 120U));
    append({{0x0200798eU, (20U << 8) | 22U, 0x0010e100U | (1U << 9) | (2U << 23), Sched}}); // RED.E.MAX.S32 [R2+20], R22
    // STL.8 [RZ+0x10], 0xfe; LDL.S8 R23, [RZ+0x10]
    append(MovImm(22U, 0xfeU));
    append({{0xff007387U, (0x10U << 8) | 22U, 0x00100000U | (0U << 9), Sched}});
    append({{0xff177983U, (0x10U << 8), 0x00100000U | (1U << 9), Sched}});
    // Store results at R2:R3 + 32..60 (buf[8..15]):
    append(Stg(2U, 32U, 16U));
    append(Stg(2U, 36U, 17U));
    append(Stg(2U, 40U, 18U));
    append(Stg(2U, 44U, 19U));
    append(Stg(2U, 48U, 21U));
    append(Stg(2U, 52U, 23U));
    append(Exit);

    const std::vector<u32> init{0x11223344U, 0x55667788U, 3U, 4U, 100U, 50U, 0U, 0U};
    std::vector<u32> out(16, 0U);
    REQUIRE(rig->Run(full, {}, 4, 1, init, out, 32, 1024, 256));
    CHECK(out[8] == 0x11223344U);
    CHECK(out[9] == 0x55667788U);
    CHECK(out[10] == 0U);
    CHECK(out[11] == 10U);
    CHECK(out[12] == 100U);
    CHECK(out[4] == 107U);
    CHECK(out[5] == 120U);
    CHECK(out[13] == static_cast<u32>(-2));
}

namespace {

[[nodiscard]] u16 FloatToHalfRn(f32 value) noexcept {
    const u32 bits = std::bit_cast<u32>(value);
    const u32 sign = (bits >> 16) & 0x8000U;
    const u32 mag = bits & 0x7fffffffU;
    if (mag == 0U) {
        return static_cast<u16>(sign);
    }
    if (mag >= 0x7f800000U) {
        return static_cast<u16>(sign | 0x7c00U | ((mag & 0x007fffffU) != 0 ? 0x0200U : 0U));
    }
    if (mag < 0x38800000U) {
        const u32 exp = mag >> 23;
        const u32 mant = (mag & 0x007fffffU) | 0x00800000U;
        const u32 shift = 126U - exp;
        if (shift > 24U) {
            return static_cast<u16>(sign);
        }
        const u32 half_mag = mant >> shift;
        const u32 rem = mant & ((1U << shift) - 1U);
        const u32 halfway = 1U << (shift - 1U);
        const u32 rounded =
            half_mag + ((rem > halfway || (rem == halfway && (half_mag & 1U))) ? 1U : 0U);
        return static_cast<u16>(sign | rounded);
    }
    const u32 rounded = mag + 0xfffU + ((mag >> 13) & 1U);
    if (rounded >= 0x47800000U) {
        return static_cast<u16>(sign | 0x7c00U);
    }
    return static_cast<u16>(sign | ((rounded - 0x38000000U) >> 13));
}

[[nodiscard]] f32 HalfToFloat(u16 h) noexcept {
    const u32 sign = u32{h & 0x8000U} << 16;
    const u32 exp = (h >> 10) & 0x1fU;
    const u32 frac = h & 0x3ffU;
    if (exp == 0U) {
        if (frac == 0U) {
            return std::bit_cast<f32>(sign);
        }
        f32 v = static_cast<f32>(frac) * 0x1p-24f;
        return sign != 0U ? -v : v;
    }
    if (exp == 31U) {
        return std::bit_cast<f32>(sign | 0x7f800000U | (frac != 0U ? 0x00400000U : 0U));
    }
    return std::bit_cast<f32>(sign | ((exp + 112U) << 23) | (frac << 13));
}

[[nodiscard]] u32 PackF16Pair(f32 lo, f32 hi) noexcept {
    return u32{FloatToHalfRn(lo)} | (u32{FloatToHalfRn(hi)} << 16);
}

} // namespace

TEST_CASE("Direct Ampere GPU execution: Batch B3 SHFL, MATCH.ALL, MOVM, LDSM, and HMMA",
          "[nvn][ampere_direct_exec]") {
    // Enable native tensor lowering cleanly (compact for OoT program ID) and restore on exit.
    const u64 prev_program_id = Settings::GetCurrentProgramID();
    Settings::SetCurrentProgramID(VideoCommon::OcarinaOfTimeProgramId);
    SCOPE_EXIT { Settings::SetCurrentProgramID(prev_program_id); };

    auto rig = CustomComputeRig::Create();
    if (!rig) {
        SKIP("no Vulkan device");
    }

    // 32-lane warp test:
    // Buffer layout (u32):
    //   init[0..31]    : per-lane A row0 pair (`a0`) for HMMA / MOVM
    //   init[32..63]   : per-lane A row1 pair (`a1`) for HMMA
    //   init[64..95]   : per-lane A row0 hi-K pair (`a2`) for HMMA.16816
    //   init[96..127]  : per-lane A row1 hi-K pair (`a3`) for HMMA.16816
    //   init[128..159] : per-lane B lo-K pair (`b0`) for HMMA
    //   init[160..191] : per-lane B hi-K pair (`b1`) for HMMA.16816
    //   init[192..223] : per-lane C lo/hi (`c0`/`c1` all zero = RZ or loaded; pass `c0` in 192..223)
    //   out[256..287]  : SHFL.BFLY ^ 1 (of lane tid)
    //   out[288..319]  : MATCH.ALL results packed as bit0=all_eq_same, bit1=all_eq_divergent
    //   out[320..351]  : MOVM.16.MT88(init[0..31])
    //   out[352..383]  : LDSM.16.M88.1 (of shared 8x8 matrix = init[0..31])
    //   out[384..415]  : LDSM.16.MT88.1 (must equal MOVM of LDSM.M88!)
    //   out[416..447]  : HMMA.16816.F16 D0 (row g)
    //   out[448..479]  : HMMA.16816.F16 D1 (row g+8)
    //   out[480..511]  : Guarded @P0 HMMA.1688.F16 D0 (tests expanded SSA HMMA path inside compact pipeline!)
    const auto stg_at = [](u32 base_reg, u32 val_reg, u32 byte_offset) -> Inst {
        return Stg(base_reg, byte_offset, val_reg);
    };
    const auto ldg_at = [](u32 dst_reg, u32 base_reg, u32 byte_offset) -> Inst {
        return Ldg(dst_reg, base_reg, byte_offset);
    };
    std::vector<Inst> prog{
        S2R(4U, 33U),                                                       // R4 = tid (0..31)
        MovC0(2U, 0x20U),
        MovC0(3U, 0x24U),
        // R6:R7 = base + tid * 4
        LeaReg(6U, 4U, 2U, ZeroGpr, 2U, false, false, false, 0U),
        Iadd3Reg(7U, 3U, ZeroGpr, ZeroGpr, false, false, false, 7U, 7U, true, 0U),
        // Load A(R8..11), B(R12..13), C0(R14)
        ldg_at(8U, 6U, 0U),
        ldg_at(9U, 6U, 128U),
        ldg_at(10U, 6U, 256U),
        ldg_at(11U, 6U, 384U),
        ldg_at(12U, 6U, 512U),
        ldg_at(13U, 6U, 640U),
        ldg_at(14U, 6U, 768U),
        MovImm(15U, 0U),                                                    // C1 = 0
        // Store a0 into shared[tid * 4] so lanes 0..7 have row i at offset i * 16 bytes (8x8 * u16)!
        {{0x04107824U, 4U, 0x078e02ffU, Sched}},                            // R16 = tid * 4
        {{0x10007388U, 8U, (4U << 9), Sched}},                              // STS.32 [R16 + 0], R8
        {{0x00007b1dU, 0U, 0U, Sched}},                                     // BAR.SYNC 0
        // 1. SHFL.BFLY R17, R4, 1, 0x1f
        {{0x04117f89U, (3U << 26) | (1U << 21) | (0x1fU << 8), 0x000e0000U, Sched}},
        // 2. MATCH.ALL.U32: R18 = all_const(77) -> P0; R19 = divergent(tid) -> P1; pack in R20
        MovImm(18U, 77U),
        {{0x12ff73a1U, 0U, (0U << 17), Sched}},
        {{0x04ff73a1U, 0U, (1U << 17), Sched}},
        MovImm(21U, 1U),
        MovImm(22U, 2U),
        SelReg(20U, 21U, ZeroGpr, 0U),
        SelReg(23U, 22U, ZeroGpr, 1U),
        Iadd3Reg(20U, 20U, 23U, ZeroGpr),
        // 3. MOVM.16.MT88 R18, R8
        {{0x0812723aU, 0U, 0U, Sched}},
        // 4. LDSM.16.M88.1 R19, [R22 + 0] and LDSM.16.MT88.1 R21, [R22 + 0] where R22 = tid * 16
        {{0x04167824U, 16U, 0x078e02ffU, Sched}},
        {{0x1613783bU, 0U, 0x00000000U, Sched}},
        {{0x1615783bU, 0U, 0x00004000U, Sched}},
        // 5. HMMA.16816.F16 R24:R25, R8..R11, R12..R13, R14..R15 (compact unguarded on OoT!)
        {{0x0818723cU, 12U, (1U << 11) | 14U, Sched}},
        // 6. @P0 HMMA.1688.F16 R26:R27, R8..R9, R12, RZ (guarded -> always exercises expanded SSA path!)
        {{0x081a023cU, 12U, 255U, Sched}},
        // Write outputs:
        stg_at(6U, 17U, 1024U),
        stg_at(6U, 20U, 1152U),
        stg_at(6U, 18U, 1280U),
        stg_at(6U, 19U, 1408U),
        stg_at(6U, 21U, 1536U),
        stg_at(6U, 24U, 1664U),
        stg_at(6U, 25U, 1792U),
        stg_at(6U, 26U, 1920U),
        Exit,
    };

    // Construct clean exact A (16x16), B (16x8), C (16x8) matrices in f16:
    std::array<std::array<f32, 16>, 16> A{};
    std::array<std::array<f32, 8>, 16> B{};
    std::array<std::array<f32, 8>, 16> C{};
    for (u32 r = 0; r < 16; ++r) {
        for (u32 c = 0; c < 8; ++c) {
            A[r][c] = static_cast<f32>((r + c) % 5) * 0.5f - 0.5f;
            A[r][c + 8] = static_cast<f32>((r * 3 + c) % 4) * 0.25f;
            B[r][c] = (r < 8) ? static_cast<f32>(((r + c) % 4) + 1) * 0.5f
                              : static_cast<f32>((r + c) % 3) * 0.25f;
            C[r][c] = 0.25f * static_cast<f32>((r + c) % 3);
        }
    }
    // Cell (0, 0) tests exact K=8 midpoint half rounding:
    // k=0..7 sums to 2048.0 + 1.0 = 2049.0 -> exact tie at K=8 midpoint (rounds ties-to-even -> 2048.0),
    // then k=8 adds +0.5 -> 2048.5 (rounds to 2048.0). Skipping K=8 rounding would yield 2049.5 -> 2050.0!
    for (u32 k = 0; k < 16; ++k) {
        A[0][k] = 0.0f;
        B[k][0] = 0.0f;
    }
    A[0][0] = 2048.0f;
    A[0][1] = 1.0f;
    B[0][0] = 1.0f;
    B[1][0] = 1.0f;
    A[0][8] = 0.5f;
    B[8][0] = 1.0f;
    C[0][0] = 0.0f;

    std::vector<u32> init(512, 0U);
    for (u32 t = 0; t < 32; ++t) {
        const u32 g = t / 4U;
        const u32 i = t % 4U;
        init[t] = PackF16Pair(A[g][2U * i], A[g][2U * i + 1U]);
        init[32U + t] = PackF16Pair(A[g + 8U][2U * i], A[g + 8U][2U * i + 1U]);
        init[64U + t] = PackF16Pair(A[g][8U + 2U * i], A[g][8U + 2U * i + 1U]);
        init[96U + t] = PackF16Pair(A[g + 8U][8U + 2U * i], A[g + 8U][8U + 2U * i + 1U]);
        init[128U + t] = PackF16Pair(B[2U * i][g], B[2U * i + 1U][g]);
        init[160U + t] = PackF16Pair(B[8U + 2U * i][g], B[8U + 2U * i + 1U][g]);
        init[192U + t] = PackF16Pair(C[g][2U * i], C[g][2U * i + 1U]);
    }

    std::vector<u32> out(512, 0U);
    REQUIRE(rig->Run(prog, {}, 32U, 1U, init, out, 32U, 512U, 0U));

    for (u32 t = 0; t < 32; ++t) {
        CAPTURE(t);
        // 1. SHFL.BFLY ^ 1
        CHECK(out[256U + t] == (t ^ 1U));
        // 2. MATCH.ALL
        CHECK(out[288U + t] == 1U);
        // 3 & 4. LDSM.M88 vs init, and MOVM == LDSM.MT88
        CHECK(out[352U + t] == init[t]);
        CHECK(out[320U + t] == out[384U + t]);
        const u32 g = t / 4U;
        const u32 i = t % 4U;
        const u32 expected_mt = PackF16Pair(A[2U * i][g], A[2U * i + 1U][g]);
        CHECK(out[320U + t] == expected_mt);

        // 5 & 6. HMMA CPU reference (F32 FMA, half round after K=8!)
        const auto hmma_cell = [&](u32 row, u32 col, u32 k_max, f32 c_init) -> u16 {
            f32 sum = c_init;
            for (u32 k = 0; k < k_max; ++k) {
                sum = std::fma(A[row][k], B[k][col], sum);
                if (k == 7U && k_max > 8U) {
                    sum = HalfToFloat(FloatToHalfRn(sum));
                }
            }
            return FloatToHalfRn(sum);
        };
        const u32 exp_d0 = u32{hmma_cell(g, 2U * i, 16U, C[g][2U * i])} |
                           (u32{hmma_cell(g, 2U * i + 1U, 16U, C[g][2U * i + 1U])} << 16);
        const u32 exp_d1 = u32{hmma_cell(g + 8U, 2U * i, 16U, 0.0f)} |
                           (u32{hmma_cell(g + 8U, 2U * i + 1U, 16U, 0.0f)} << 16);
        const u32 exp_k8 = u32{hmma_cell(g, 2U * i, 8U, 0.0f)} |
                           (u32{hmma_cell(g, 2U * i + 1U, 8U, 0.0f)} << 16);
        CHECK(out[416U + t] == exp_d0);
        CHECK(out[448U + t] == exp_d1);
        CHECK(out[480U + t] == exp_k8);
    }
}
