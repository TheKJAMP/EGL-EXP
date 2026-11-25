// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <optional>
#include <vector>

#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/reg.h"

namespace Shader::Ampere {
namespace {

struct Compare {
    bool operator()(const Flow::Block& lhs, Maxwell::Location rhs) const noexcept {
        return lhs.begin < rhs;
    }
    bool operator()(Maxwell::Location lhs, const Flow::Block& rhs) const noexcept {
        return lhs < rhs.begin;
    }
    bool operator()(const Flow::Block& lhs, const Flow::Block& rhs) const noexcept {
        return lhs.begin < rhs.begin;
    }
};

[[nodiscard]] constexpr IR::Reg GprReg(u32 reg) noexcept {
    return reg == 255 ? IR::Reg::RZ : static_cast<IR::Reg>(reg);
}

[[nodiscard]] constexpr IR::Reg UniformReg(u32 reg) noexcept {
    return reg == 63 ? IR::Reg::RZ : static_cast<IR::Reg>(256U + reg);
}

[[nodiscard]] constexpr IR::Pred ThreadPred(u32 index) noexcept {
    return static_cast<IR::Pred>(index & 7U);
}

[[nodiscard]] constexpr IR::Pred UniformPred(u32 index) noexcept {
    const u32 slot = index & 7U;
    return slot == 7U ? IR::Pred::PT : static_cast<IR::Pred>(8U + slot);
}

[[nodiscard]] constexpr s64 SignExtend50(u64 raw) noexcept {
    return static_cast<s64>(raw << 14) >> 14;
}

[[nodiscard]] constexpr u32 CbufBank(const Instruction& insn) noexcept {
    const u32 value = insn.Bits(54, 5);
    return value == 24 ? 17U : value;
}

[[nodiscard]] constexpr bool ValidCbuf(const Instruction& insn) noexcept {
    return !insn.Bits(91, 1) && CbufBank(insn) < 18 && (insn.Bits(38, 16) % 4) == 0;
}

void Split(Flow::Block* old_block, Flow::Block* new_block, Maxwell::Location pc) {
    if (pc <= old_block->begin || pc >= old_block->end) {
        throw InvalidArgument("Invalid Ampere CFG address to split={:#x}", pc.Offset());
    }
    *new_block = Flow::Block{};
    new_block->begin = pc;
    new_block->end = old_block->end;
    new_block->end_class = old_block->end_class;
    new_block->cond = old_block->cond;
    new_block->stack = old_block->stack;
    new_block->branch_true = old_block->branch_true;
    new_block->branch_false = old_block->branch_false;
    new_block->function_call = old_block->function_call;
    new_block->return_block = old_block->return_block;
    new_block->branch_reg = old_block->branch_reg;
    new_block->branch_offset = old_block->branch_offset;
    new_block->indirect_branches = std::move(old_block->indirect_branches);

    const Maxwell::Location old_begin{old_block->begin};
    Flow::Stack old_stack{std::move(old_block->stack)};
    *old_block = Flow::Block{};
    old_block->begin = old_begin;
    old_block->end = pc;
    old_block->end_class = Flow::EndClass::Branch;
    old_block->cond = IR::Condition{true};
    old_block->stack = std::move(old_stack);
    old_block->branch_true = new_block;
    old_block->branch_false = nullptr;
}

[[nodiscard]] bool IsSpecialRegSupported(u32 special) noexcept {
    return special == 0 || (special >= 17 && special <= 19) || (special >= 29 && special <= 35) ||
           special == 37 || special == 38 || special == 39 || (special >= 56 && special <= 60);
}

[[nodiscard]] bool ValidateTier1Instruction(const Instruction& insn, const LoweringOptions& options,
                                            std::string_view& error) noexcept {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 pred = insn.Bits(12, 4);
    const u32 form = insn.Bits(9, 3);
    const auto fail = [&](std::string_view msg) {
        error = msg;
        return false;
    };
    const auto valid_alu_form = [&](bool check_cbuf = true) {
        if (form < 1 || form > 7) {
            return fail("native ALU source form");
        }
        const bool wide1 = form == 4 || form == 5 || form == 6;
        const bool wide2 = form == 2 || form == 3 || form == 7;
        if (check_cbuf && ((wide1 && form == 5) || (wide2 && form == 3)) && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    };
    switch (opcode) {
    case 0x202: case 0x802: case 0xa02: case 0xc02: {
        const u32 lane_mask = insn.Bits(72, 4);
        if (opcode == 0x202 && std::popcount(lane_mask) == 1 && !insn.Bits(76, 16) &&
            !insn.Bits(40, 24)) {
            return true;
        }
        if (lane_mask != 15 || (insn.Bits(76, 16) & ~(opcode == 0xc02 ? 0x8000U : 0U))) {
            return fail("native MOV modifiers");
        }
        if (opcode == 0xc02) {
            if (insn.words[1] >= 64 || insn.words[2] != 0x08000f00U) {
                return fail("native uniform move source");
            }
        } else if (opcode == 0x202) {
            if (insn.Bits(40, 24)) {
                return fail("native MOV register modifiers");
            }
        } else if (opcode == 0xa02) {
            if (!ValidCbuf(insn) || insn.Bits(62, 2)) {
                return fail("native MOV constant buffer");
            }
        }
        return true;
    }
    case 0x882:
        return (insn.words[2] == 0 && dest <= 63) ? true : fail("native uniform immediate move mode");
    case 0xc82:
        return (insn.words[1] < 64 && dest <= 63 && insn.words[2] == 0x08000000U)
                   ? true
                   : fail("native uniform move mode");
    case 0xb82: {
        const u32 width = insn.Bits(73, 3);
        const u32 mode = insn.Bits(78, 2);
        if (width > 5 || (mode != 0 && mode != 2) || insn.Bits(91, 1) || CbufBank(insn) >= 18 ||
            (width == 5 && ((dest & 1) || dest >= 254))) {
            return fail("native indexed constant load mode");
        }
        return true;
    }
    case 0xab9: case 0xabb: {
        const u32 width = insn.Bits(73, 3);
        const u32 alignment = !LowerExtended() ? 4U
                              : opcode == 0xabb ? 1U
                              : width >= 4     ? 4U
                                               : (width >= 2 ? 2U : 1U);
        if ((!LowerExtended() && width < 4) || dest >= 63 || width > 5 ||
            (width == 5 && ((dest & 1) || dest + 1 >= 63)) || CbufBank(insn) >= 18 ||
            (insn.Bits(38, 16) % alignment) != 0 || insn.Bits(62, 2) ||
            (insn.words[2] & ~0xe00U) != (opcode == 0xab9 ? 0U : 0x08000000U) ||
            (opcode == 0xabb && insn.Bits(24, 8) > 63)) {
            return fail("native uniform constant load mode");
        }
        return true;
    }
    case 0x919: case 0x9c3: {
        const u32 special = insn.Bits(72, 8);
        if (opcode == 0x9c3 && dest > 63) {
            return fail("native uniform special destination");
        }
        if (special == 132 && !insn.words[1] && insn.words[2] == (special << 8)) {
            return true;
        }
        if (insn.words[1] || insn.words[2] != (special << 8) || !IsSpecialRegSupported(special)) {
            return fail("native special register");
        }
        return true;
    }
    case 0x805:
        return (insn.Bits(72, 8) == 255 && insn.Bits(80, 1) && dest < 254)
                   ? true
                   : fail("native special-register read");
    case 0x34e:
        if (!options.geometry) return fail("unsupported native opcode");
        return (!insn.Bits(24, 8) && !insn.words[1] && !insn.words[2] && dest != 254 && pred == 7)
                   ? true
                   : fail("native program counter load mode");
    case 0x918:
        return (!insn.words[1] && !insn.words[2]) ? true : fail("native NOP mode");
    case 0x357:
        if (!options.geometry) return fail("unsupported native opcode");
        return (!insn.Bits(16, 8) && !insn.words[1] && !insn.words[2] && pred == 7)
                   ? true
                   : fail("native geometry exit setup mode");
    case 0x957:
        return options.geometry ? true : fail("unsupported native opcode");
    case 0x945: case 0x941: case 0x942: case 0x948: case 0x348: case 0xf55: case 0xf56:
        return true;
    case 0x210: case 0x810: case 0xa10: case 0xc10:
    case 0x290: case 0x890: {
        const bool is_uniform = opcode == 0x290 || opcode == 0x890;
        const bool extended = insn.Bits(74, 1) != 0;
        if (is_uniform) {
            if (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63 ||
                (opcode == 0x290 && insn.Bits(32, 8) > 63)) {
                return fail("native uniform arithmetic register");
            }
        }
        if (insn.Bits(73, 1) || insn.Bits(76, 1)) {
            return fail("native integer carry/third source");
        }
        const u32 first_out = insn.Bits(81, 3);
        const u32 second_out = insn.Bits(84, 3);
        if (first_out != 7 && first_out == second_out) {
            return fail("native three-input carry mode");
        }
        if (!extended && (insn.Bits(77, 4) != 15 || insn.Bits(87, 4) != 15)) {
            return fail("native three-input carry mode");
        }
        if (form == 5 || form == 3) {
            if (!ValidCbuf(insn) || insn.Bits(62, 1)) {
                return fail("native constant buffer address");
            }
        }
        return true;
    }
    case 0x211: case 0x811: case 0xa11: case 0xc11:
    case 0x291: case 0x891: {
        const bool is_uniform = opcode == 0x291 || opcode == 0x891;
        if (is_uniform) {
            if (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63 ||
                (opcode == 0x291 && insn.Bits(32, 8) > 63)) {
                return fail("native uniform arithmetic register");
            }
        }
        if (insn.Bits(72, 1)) return fail("native LEA mode");
        if (form == 5 || form == 3) {
            if (!ValidCbuf(insn) || insn.Bits(62, 2)) {
                return fail("native constant LEA mode");
            }
        }
        if (insn.Bits(80, 1)) {
            if (insn.Bits(81, 3) != 7 || (insn.Bits(64, 8) != (is_uniform ? 63U : 255U) && insn.Bits(73, 1))) {
                return fail("native high effective address mode");
            }
            if ((opcode == 0x211 || opcode == 0x291) && (insn.Bits(62, 1) || (insn.Bits(63, 1) && insn.Bits(74, 1)))) {
                return fail("native high effective address mode");
            }
        } else {
            const u32 carry_out = insn.Bits(81, 3);
            if (insn.Bits(74, 1) || (opcode == 0x211 && insn.Bits(62, 1)) ||
                (carry_out != 7 && opcode != 0x811 && opcode != 0x891 && insn.Bits(63, 1))) {
                return fail("native LEA carry/high mode");
            }
        }
        return true;
    }
    case 0x213: case 0xa13: case 0xc13:
        if (insn.words[2] & ~(opcode == 0xc13 ? 0x08000000U : 0U)) {
            return fail("native integer absolute mode");
        }
        return valid_alu_form();
    case 0x217: case 0x817: case 0xa17: case 0xc17:
        if ((insn.words[2] & ~(0x04000200U | (opcode == 0xc17 ? 0x08000000U : 0U))) != 0x03800000 ||
            (opcode != 0xa17 && opcode != 0x817 && insn.words[1] > 255U) ||
            (opcode == 0xa17 && (!ValidCbuf(insn) || insn.Bits(62, 2)))) {
            return fail("native integer min/max mode");
        }
        return true;
    case 0x224: case 0x424: case 0x624: case 0x824: case 0xa24: case 0xc24: case 0xe24:
    case 0x2a4: case 0x4a4: case 0x8a4: {
        const bool is_uniform = opcode == 0x2a4 || opcode == 0x4a4 || opcode == 0x8a4;
        if (is_uniform && (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63 ||
                           (opcode == 0x2a4 && insn.Bits(32, 8) > 63))) {
            return fail("native uniform arithmetic register");
        }
        return valid_alu_form();
    }
    case 0x225: case 0x227: case 0x625: case 0x825: case 0x827: case 0xa27:
    case 0x2a5: case 0x8a5: {
        const bool is_uniform = opcode == 0x2a5 || opcode == 0x8a5;
        const bool wide = opcode == 0x225 || opcode == 0x625 || opcode == 0x825 || is_uniform;
        const bool signed_product = !is_uniform && insn.Bits(73, 1);
        const u32 c = opcode == 0x625 ? (is_uniform ? 63U : 255U) : insn.Bits(64, 8);
        const u32 carry_out = insn.Bits(81, 3);
        const u32 zero_reg = is_uniform ? 63U : 255U;
        if (is_uniform) {
            if (dest >= 62 || (dest & 1) || insn.Bits(24, 8) > 63 ||
                (opcode == 0x2a5 && insn.Bits(32, 8) > 63) || (c != 63 && (c >= 62 || (c & 1)))) {
                return fail("native uniform wide multiply mode");
            }
            return true;
        }
        if ((insn.words[2] & ~0x000e02ffU) != 0x07800000 ||
            ((opcode == 0x225 || opcode == 0x227) && (insn.words[1] & ~0xffU)) ||
            ((opcode == 0xa27 || opcode == 0x625) && (!ValidCbuf(insn) || insn.Bits(62, 2))) ||
            (opcode == 0x625 && insn.Bits(38, 16) > 0xfff8) || (c != zero_reg && (c & 1)) ||
            (wide && (dest >= 254 || (dest & 1))) || (signed_product && carry_out != 7)) {
            return fail("native high multiply mode");
        }
        return true;
    }
    case 0x226:
        return ((insn.words[2] & ~0xffU) == 0x1000U && (insn.words[1] & ~0xffU) == 0U)
                   ? true
                   : fail("native packed dot-product mode");
    case 0x212: case 0x812: case 0xa12: case 0xc12:
    case 0x292: case 0x892: {
        const bool is_uniform = opcode == 0x292 || opcode == 0x892;
        if (is_uniform && (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63 ||
                           (opcode == 0x292 && insn.Bits(32, 8) > 63))) {
            return fail("native uniform arithmetic register");
        }
        if (insn.Bits(80, 1) || insn.Bits(87, 4) != 15 ||
            ((opcode == 0x212 || opcode == 0x292) && insn.Bits(40, 24)) ||
            (opcode == 0xa12 && (!ValidCbuf(insn) || insn.Bits(62, 2)))) {
            return fail("native LUT/predicate mode");
        }
        return true;
    }
    case 0x219: case 0x419: case 0x819: case 0xa19: case 0xc19:
    case 0x299: case 0x499: case 0x899: {
        const bool is_uniform = opcode == 0x299 || opcode == 0x499 || opcode == 0x899;
        if (is_uniform && (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63)) {
            return fail("native uniform shift mode");
        }
        if (opcode == 0xa19 && (!ValidCbuf(insn) || insn.Bits(62, 2))) {
            return fail("native constant buffer address");
        }
        if (insn.Bits(73, 2) == 0) {
            return fail("native funnel-shift mode");
        }
        if (opcode == 0x419 || opcode == 0x499) {
            const u32 zero_reg = is_uniform ? 63U : 255U;
            const u32 expected_w2 = is_uniform ? 0x08010600U : 0x00010600U;
            if (insn.Bits(24, 8) != zero_reg || (insn.words[2] & ~0x000008ffU) != expected_w2) {
                return fail("native funnel-shift mode");
            }
        }
        return true;
    }
    case 0x81a:
        return (!(insn.words[2] & ~0x200U) && insn.words[1] <= 32 && dest < 255)
                   ? true
                   : fail("native sign extend mode");
    case 0x816: case 0x896: {
        const bool is_uniform = opcode == 0x896;
        if (is_uniform && (dest > 63 || insn.Bits(24, 8) > 63 || insn.Bits(64, 8) > 63)) {
            return fail("native uniform arithmetic register");
        }
        return (insn.words[1] <= 0xffffU && (insn.words[2] & ~0xffU) == 0U)
                   ? true
                   : fail("native permutation mode");
    }
    case 0x300: case 0xd00:
        return ((insn.words[2] & ~(0x600U | (opcode == 0xd00 ? 0x08000000U : 0U))) == 0x000e0000U &&
                valid_alu_form())
                   ? true
                   : fail("native leading bit mode");
    case 0x2bd:
        return (dest < 63 && insn.Bits(32, 8) <= 63 && !(insn.words[1] & ~0xffU) &&
                (insn.words[2] & ~0x400U) == 0x080e0000U)
                   ? true
                   : fail("native uniform leading bit mode");
    case 0x309: case 0xd09:
        return (insn.words[2] == (opcode == 0xd09 ? 0x08000000U : 0U) && valid_alu_form())
                   ? true
                   : fail("native population count mode");
    case 0x2bf:
        return (dest < 63 && insn.Bits(32, 8) <= 63 && !(insn.words[1] & ~0x800000ffU) &&
                insn.words[2] == 0x08000000U)
                   ? true
                   : fail("native uniform population count mode");
    case 0x301:
        return (!insn.words[2] && insn.words[1] <= 255U) ? true : fail("native bit reverse mode");
    case 0x2be:
        return (dest < 63 && insn.Bits(32, 8) <= 63 && !(insn.words[1] & ~0xffU) &&
                insn.words[2] == 0x08000000U)
                   ? true
                   : fail("native uniform bit reverse mode");
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: {
        if (insn.Bits(72, 1)) {
            const u32 cmp = insn.Bits(76, 3);
            if (cmp == 0 || cmp == 7 || insn.Bits(74, 2) == 3) {
                return fail("native extended integer comparison mode");
            }
        } else if (insn.Bits(74, 2) > 2) {
            return fail("native predicate combination");
        }
        return valid_alu_form();
    }
    case 0x28c: case 0x88c:
        return (insn.Bits(81, 3) <= 6 && insn.Bits(84, 3) == 7 && !insn.Bits(72, 1) &&
                insn.Bits(74, 2) != 3 && insn.Bits(24, 8) <= 63 &&
                (opcode == 0x88c || insn.Bits(32, 8) <= 63))
                   ? true
                   : fail("native uniform comparison mode");
    case 0x207: case 0x807: case 0xa07: case 0xc07:
        return valid_alu_form();
    case 0x287: case 0x887:
        return (insn.Bits(87, 3) <= 6 && dest <= 63 && insn.Bits(24, 8) <= 63 &&
                (opcode == 0x887 || insn.Bits(32, 8) <= 63) &&
                (insn.words[2] & ~0x07800000U) == 0x08000000U)
                   ? true
                   : fail("native uniform select mode");
    case 0x81c: {
        const u32 t1 = insn.Bits(81, 3), t2 = insn.Bits(84, 3);
        return (!insn.words[1] && !(t1 != 7 && t1 == t2)) ? true
                                                           : fail("native predicate LUT mode");
    }
    case 0x89c:
        return (insn.Bits(84, 3) == 7 && !insn.Bits(67, 1) && !insn.Bits(16, 8) && !insn.words[1])
                   ? true
                   : fail("native uniform predicate LUT mode");
    case 0x803:
        return (insn.words[1] <= 0x7f && !(insn.words[2] & ~0x3000U))
                   ? true
                   : fail("native predicate-to-register mode");
    case 0x804:
        return (!dest && insn.words[1] <= 0x7f && !(insn.words[2] & ~0x3000U))
                   ? true
                   : fail("native register-to-predicate mode");
    case 0x883:
        return (dest <= 63 && insn.Bits(24, 8) == 63 && pred == 7 &&
                insn.words[2] == 0x08000000U && insn.words[1] <= 0x7f)
                   ? true
                   : fail("native uniform predicate move mode");
    case 0x80d:
        return (insn.Bits(87, 4) == 7 && !insn.Bits(16, 16)) ? true : fail("native CSM test mode");
    case 0x806:
        return (!insn.words[1] && insn.Bits(72, 2) <= 2 && !(insn.words[2] & ~0x078e0300U))
                   ? true
                   : fail("native vote mode");
    case 0x886:
        return (dest <= 63 && !insn.words[1] && insn.Bits(72, 2) <= 2 && pred == 7)
                   ? true
                   : fail("native uniform vote mode");
    case 0x3c2:
        return (dest < 63 && !insn.words[1] && insn.words[2] == 0x000e0000U)
                   ? true
                   : fail("native register-to-uniform mode");
    case 0x3c4: {
        const u32 mode = insn.words[2] & ~0x200U;
        const bool minmax = mode == 0x10000U || mode == 0x14000U;
        const bool logical = insn.words[2] == 0x4000U || insn.words[2] == 0x8000U;
        const bool signed_order = insn.Bits(73, 1) != 0;
        if (dest >= 63 || insn.words[1] ||
            (insn.words[2] != 0xc000U && !minmax && !logical) || (signed_order && !minmax) ||
            pred != 7) {
            return fail("native warp sum mode");
        }
        return true;
    }
    default:
        return false;
    }
}

[[nodiscard]] std::optional<u32> FindCallLinkRegister(std::span<const Instruction> code,
                                                      u32 index) noexcept {
    std::array<bool, 256> written{};
    for (u32 prior = index; prior-- > 0 && index - prior <= 64;) {
        const Instruction& candidate = code[prior];
        const u32 prior_opcode = candidate.Bits(0, 12);
        if ((candidate.words[0] & ~0x00ff0000U) == 0x00007802 &&
            candidate.words[1] == (index + 1) * 16 && candidate.words[2] == 0xf00) {
            const u32 link = candidate.Bits(16, 8);
            return (link < 250 && !written[link]) ? std::optional<u32>{link} : std::nullopt;
        }
        if (prior_opcode == 0x947 || prior_opcode == 0x949 || prior_opcode == 0x958 ||
            prior_opcode == 0x944 || prior_opcode == 0x950 || prior_opcode == 0x94d ||
            prior_opcode == 0x945 || prior_opcode == 0x941 || prior_opcode == 0x942) {
            break;
        }
        const u32 written_reg = candidate.Bits(16, 8);
        written[written_reg] = true;
        if (prior_opcode == 0x225 || prior_opcode == 0x625 || prior_opcode == 0x825) {
            written[(written_reg + 1) & 0xff] = true;
        }
    }
    return std::nullopt;
}

[[nodiscard]] bool VerifyReturnLink(std::span<const Instruction> code, u32 index, u32 active_entry,
                                    std::optional<u32> active_link, u32 return_reg) noexcept {
    if (!active_link) {
        return false;
    }
    if (return_reg == *active_link) {
        return true;
    }
    const auto copies_link = [&](const Instruction& prior) {
        if (prior.Bits(16, 8) != return_reg || prior.Bits(12, 4) != 7) {
            return false;
        }
        if ((prior.words[0] & 0xff00ffffU) == 0x00007202U) {
            return prior.words[1] == *active_link && prior.words[2] == 0xf00;
        }
        if ((prior.words[0] & 0xff00ffffU) == 0xff007224U) {
            return prior.words[1] == 0xff && prior.words[2] == (0x078e0000U | *active_link);
        }
        return false;
    };
    for (u32 prior = index; prior-- > active_entry;) {
        if (copies_link(code[prior])) {
            return true;
        }
        if (code[prior].Bits(16, 8) == return_reg) {
            break;
        }
    }
    return false;
}

struct AmpereBranchTableInfo {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 num_entries{};
    IR::Reg branch_reg{};
};

[[nodiscard]] std::optional<u32> ExtractPow2ShiftFromIMAD(const Instruction& insn) noexcept {
    if (insn.Bits(64, 8) != 255 || insn.Bits(75, 1)) {
        return std::nullopt;
    }
    const u32 mul = insn.words[1];
    if (mul == 0 || (mul & (mul - 1)) != 0 || (insn.words[2] & ~0x2ffU) != 0x078e0000) {
        return std::nullopt;
    }
    return static_cast<u32>(std::countr_zero(mul));
}

[[nodiscard]] constexpr bool IsControlFlowOpcode(u32 op) noexcept {
    return op == 0x947 || op == 0x944 || op == 0x950 || op == 0x949 || op == 0x958 ||
           op == 0x94d || op == 0x95b || op == 0x95c || op == 0x945 || op == 0x941 ||
           op == 0x942 || op == 0x948 || op == 0x348 || op == 0x918 || op == 0xf55 ||
           op == 0xf56 || op == 0x357 || op == 0x957;
}

[[nodiscard]] bool StopsJumpTableScan(const Instruction& insn) noexcept {
    if (insn.Bits(12, 4) == 15U) {
        return false; // @!PT predicate-disabled slot never executes
    }
    const u32 op = insn.Bits(0, 12);
    if (op == 0x947) {
        return insn.Bits(32, 2) != 2U; // BRA.DIV is a no-op
    }
    return op == 0x944 || op == 0x950 || op == 0x949 || op == 0x958 || op == 0x94d ||
           op == 0x95b || op == 0x95c || op == 0x941 || op == 0x942;
}

[[nodiscard]] constexpr bool IsUniformRegWriterOpcode(u32 op) noexcept {
    switch (op) {
    case 0x882: case 0xc82: // UMOV
    case 0xab9: case 0xabb: // ULDC
    case 0x9c3:             // S2UR
    case 0x290: case 0x890: // UIADD3
    case 0x291: case 0x891: // ULEA
    case 0x2a4: case 0x4a4: case 0x8a4: // UIMAD
    case 0x2a5: case 0x8a5: // UIMAD.WIDE
    case 0x292: case 0x892: // ULOP3
    case 0x299: case 0x499: case 0x899: // USHF
    case 0x896:             // UPRMT
    case 0x2bd: case 0x2be: case 0x2bf: // UFLO, UBREV, UPOPC
    case 0x287: case 0x887: // USEL
    case 0x883:             // UP2UR
    case 0x886:             // VOTEU
    case 0x3c2:             // R2UR
    case 0x3c4:             // REDUX
        return true;
    default:
        return false;
    }
}

[[nodiscard]] bool InstructionWritesUniform(const Instruction& insn, u32 ur) noexcept {
    if (ur >= 63U || insn.Bits(12, 4) == 15U) {
        return false;
    }
    const u32 op = insn.Bits(0, 12);
    if (!IsUniformRegWriterOpcode(op)) {
        return false;
    }
    const u32 rd = insn.Bits(16, 8);
    if (rd == ur) {
        return true;
    }
    const bool writes_pair =
        op == 0x2a5 || op == 0x8a5 || ((op == 0xab9 || op == 0xabb) && insn.Bits(73, 3) == 5U);
    return writes_pair && rd < 62U && (rd + 1U) == ur;
}

[[nodiscard]] bool InstructionWritesGpr(const Instruction& insn, u32 reg) noexcept {
    if (reg >= 255U || insn.Bits(12, 4) == 15U) {
        return false;
    }
    const u32 op = insn.Bits(0, 12);
    if (IsControlFlowOpcode(op) || IsUniformRegWriterOpcode(op)) {
        return false;
    }
    switch (op) {
    // Barriers, fences, predicate-only compares, stores, geometry output, no-result atomics
    case 0x91a: case 0x992: case 0x9ab: case 0x9af: case 0xb1d:
    case 0x28c: case 0x88c: case 0x89c:
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c:
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b:
    case 0x234: case 0x434: case 0x634:
    case 0x81c: case 0x804: case 0x80d:
    case 0x386: case 0x986: case 0x387: case 0x983:
    case 0x388: case 0x984: case 0x988:
    case 0x322: case 0x324: case 0x924: case 0x927:
    case 0xb9b: case 0xf8c: case 0xfae:
        return false;
    default:
        break;
    }
    const u32 rd = insn.Bits(16, 8);
    if (rd == reg) {
        return true;
    }
    if (rd >= 254U) {
        return false;
    }
    // Wide / multi-register writers
    if (op == 0x225 || op == 0x625 || op == 0x825 || op == 0x311 || op == 0x805 || op == 0x34e ||
        (op == 0xb82 && insn.Bits(73, 3) >= 5U)) {
        return reg == rd + 1U;
    }
    if (op == 0x381 || op == 0x981 || op == 0xb97 || op == 0xb60 || op == 0xf60 || op == 0x361 ||
        op == 0xb63 || op == 0xf63 || op == 0xb66 || op == 0xf66 || op == 0x367 || op == 0xb69 ||
        op == 0xf69 || op == 0x36a || op == 0xb6c || op == 0xf6c || op == 0xb6f || op == 0x370) {
        return reg >= rd && reg < rd + 4U;
    }
    return false;
}

[[nodiscard]] bool InstructionWritesPred(const Instruction& insn, bool uniform, u32 pred_idx) noexcept {
    if (pred_idx >= 7U || insn.Bits(12, 4) == 15U) {
        return false;
    }
    const u32 op = insn.Bits(0, 12);
    if (uniform) {
        if (op == 0x28c || op == 0x88c || op == 0x89c) {
            return insn.Bits(81, 3) == pred_idx || (op != 0x89c && insn.Bits(84, 3) == pred_idx);
        }
        if (op == 0x292 || op == 0x892 || op == 0x886 || op == 0x290 || op == 0x890 ||
            op == 0x291 || op == 0x891) {
            return insn.Bits(81, 3) == pred_idx;
        }
        return false;
    }
    switch (op) {
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c:
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b:
    case 0x234: case 0x434: case 0x634:
    case 0x81c: case 0x80d:
        return insn.Bits(81, 3) == pred_idx || insn.Bits(84, 3) == pred_idx;
    case 0x212: case 0x812: case 0xa12: case 0xc12:
    case 0x210: case 0x810: case 0xa10: case 0xc10:
    case 0x211: case 0x811: case 0xa11: case 0xc11:
    case 0x225: case 0x625: case 0x825: case 0x227: case 0x827: case 0xa27:
    case 0x806:
        return insn.Bits(81, 3) == pred_idx || insn.Bits(84, 3) == pred_idx;
    case 0x804: // R2P
        return (insn.words[1] & (1U << pred_idx)) != 0U;
    default:
        return false;
    }
}

[[nodiscard]] std::optional<AmpereBranchTableInfo> TrackGprJumpTable(
    std::span<const Instruction> code, u32 start_idx, u32 ldc_reg, u32 cbuf_index,
    u32 cbuf_offset, IR::Reg branch_reg) noexcept {
    std::optional<u32> shl_idx;
    u32 shl_reg = ldc_reg;
    for (u32 i = start_idx; i-- > 0;) {
        const Instruction& insn = code[i];
        if (StopsJumpTableScan(insn)) {
            return std::nullopt;
        }
        if (!InstructionWritesGpr(insn, shl_reg)) {
            continue;
        }
        const u32 op = insn.Bits(0, 12);
        if (op == 0x202 && insn.Bits(12, 4) == 7 && !insn.Bits(72, 20) && !insn.Bits(40, 24)) {
            shl_reg = insn.Bits(32, 8);
            continue;
        }
        if (op == 0x819 && !insn.Bits(80, 1) && !insn.Bits(76, 1) && insn.Bits(64, 8) == 255 &&
            insn.Bits(73, 2) == 3 && insn.words[1] == 2) {
            shl_idx = i;
            shl_reg = insn.Bits(24, 8);
            break;
        }
        if (op == 0x824 && ExtractPow2ShiftFromIMAD(insn) == 2U) {
            shl_idx = i;
            shl_reg = insn.Bits(24, 8);
            break;
        }
        if ((op == 0x211 || op == 0x811) && !insn.Bits(80, 1) && insn.Bits(75, 5) == 2 &&
            ((op == 0x211 && insn.Bits(32, 8) == 255) ||
             (op == 0x811 && insn.words[1] == 0))) {
            shl_idx = i;
            shl_reg = insn.Bits(24, 8);
            break;
        }
        return std::nullopt;
    }
    if (!shl_idx || shl_reg >= 255) {
        return std::nullopt;
    }
    for (u32 i = *shl_idx; i-- > 0;) {
        const Instruction& insn = code[i];
        if (StopsJumpTableScan(insn)) {
            return std::nullopt;
        }
        if (!InstructionWritesGpr(insn, shl_reg)) {
            continue;
        }
        const u32 op = insn.Bits(0, 12);
        if (op == 0x202 && insn.Bits(12, 4) == 7 && !insn.Bits(72, 20) && !insn.Bits(40, 24)) {
            shl_reg = insn.Bits(32, 8);
            continue;
        }
        if (op == 0x817 && insn.Bits(87, 4) == 7 && (insn.words[1] >> 31) == 0) {
            return AmpereBranchTableInfo{
                .cbuf_index = cbuf_index,
                .cbuf_offset = cbuf_offset,
                .num_entries = insn.words[1] + 1,
                .branch_reg = branch_reg,
            };
        }
        if (op == 0x807 && insn.Bits(12, 4) == 7 && (insn.words[1] >> 31) == 0) {
            const u32 condition = insn.Bits(87, 3);
            const bool negated = insn.Bits(90, 1) != 0;
            for (u32 j = i; j-- > 0;) {
                const Instruction& cmp = code[j];
                if (StopsJumpTableScan(cmp)) {
                    break;
                }
                if (!InstructionWritesPred(cmp, false, condition)) {
                    continue;
                }
                if (cmp.Bits(0, 12) == 0x80c && cmp.Bits(12, 4) == 7 &&
                    cmp.Bits(24, 8) == insn.Bits(24, 8) && !cmp.Bits(73, 1) &&
                    cmp.Bits(87, 4) == 7U && !cmp.Bits(72, 1)) {
                    const u32 cmp_op = cmp.Bits(76, 3);
                    const bool normal_match =
                        (cmp.Bits(81, 3) == condition && !negated) ||
                        (cmp.Bits(84, 3) == condition && negated);
                    const bool inverted_match =
                        (cmp.Bits(81, 3) == condition && negated) ||
                        (cmp.Bits(84, 3) == condition && !negated);
                    const bool valid_lt = (cmp_op == 1U || cmp_op == 3U) && normal_match;
                    const bool valid_ge = (cmp_op == 4U || cmp_op == 6U) && inverted_match;
                    if (valid_lt || valid_ge) {
                        const u32 max_val = cmp.words[1];
                        const u32 sel_val = insn.words[1];
                        const u32 n =
                            (cmp_op == 1U && sel_val + 1U <= max_val) ? max_val : (sel_val + 1U);
                        if (n > 0U && n < 0x10000U) {
                            return AmpereBranchTableInfo{
                                .cbuf_index = cbuf_index,
                                .cbuf_offset = cbuf_offset,
                                .num_entries = n,
                                .branch_reg = branch_reg,
                            };
                        }
                    }
                }
                break;
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<AmpereBranchTableInfo> TrackAmpereIndirectBranchTable(
    std::span<const Instruction> code, u32 brx_index) noexcept {
    const Instruction& brx = code[brx_index];
    const u32 brx_op = brx.Bits(0, 12);
    const bool is_uniform = brx_op == 0x958;
    u32 src = brx.Bits(24, 8);

    if (!is_uniform) {
        std::optional<u32> ldc_idx;
        u32 ldc_reg = 255;
        for (u32 i = brx_index; i-- > 0;) {
            const Instruction& insn = code[i];
            if (StopsJumpTableScan(insn)) {
                break;
            }
            if (!InstructionWritesGpr(insn, src)) {
                continue;
            }
            const u32 op = insn.Bits(0, 12);
            if (op == 0x202 && insn.Bits(12, 4) == 7 && !insn.Bits(72, 20) &&
                !insn.Bits(40, 24)) {
                src = insn.Bits(32, 8);
                continue;
            }
            if (op == 0xb82 && insn.Bits(73, 3) == 4 && insn.Bits(78, 2) == 0 &&
                !insn.Bits(91, 1) && CbufBank(insn) < 18) {
                ldc_idx = i;
                ldc_reg = insn.Bits(24, 8);
                break;
            }
            break;
        }
        if (!ldc_idx || ldc_reg >= 255) {
            return std::nullopt;
        }
        const Instruction& ldc = code[*ldc_idx];
        return TrackGprJumpTable(code, *ldc_idx, ldc_reg, CbufBank(ldc), ldc.Bits(38, 16),
                                 GprReg(brx.Bits(24, 8)));
    }

    // Uniform jump table path (BRXU URx, 0x0)
    std::optional<u32> uldc_idx;
    u32 ldc_ur = 63;
    for (u32 i = brx_index; i-- > 0;) {
        const Instruction& insn = code[i];
        if (StopsJumpTableScan(insn)) {
            break;
        }
        if (!InstructionWritesUniform(insn, src)) {
            continue;
        }
        const u32 op = insn.Bits(0, 12);
        if (op == 0xc82 && insn.Bits(12, 4) == 7 && insn.words[2] == 0x08000000U &&
            insn.words[1] < 63U) {
            src = insn.words[1];
            continue;
        }
        if (op == 0xabb && insn.Bits(73, 3) == 4 && CbufBank(insn) < 18) {
            uldc_idx = i;
            ldc_ur = insn.Bits(24, 8);
            break;
        }
        break;
    }
    if (!uldc_idx || ldc_ur >= 63) {
        return std::nullopt;
    }
    const Instruction& uldc = code[*uldc_idx];
    const u32 cbuf_index = CbufBank(uldc);
    const u32 cbuf_offset = uldc.Bits(38, 16);

    std::optional<u32> shl_idx;
    u32 shl_ur = ldc_ur;
    for (u32 i = *uldc_idx; i-- > 0;) {
        const Instruction& insn = code[i];
        if (StopsJumpTableScan(insn)) {
            return std::nullopt;
        }
        if (!InstructionWritesUniform(insn, shl_ur)) {
            continue;
        }
        const u32 op = insn.Bits(0, 12);
        if (op == 0xc82 && insn.Bits(12, 4) == 7 && insn.words[2] == 0x08000000U &&
            insn.words[1] < 63U) {
            shl_ur = insn.words[1];
            continue;
        }
        if (op == 0x3c2) { // R2UR URd, Rs: trace GPR source backwards
            return TrackGprJumpTable(code, i, insn.Bits(24, 8), cbuf_index, cbuf_offset,
                                     UniformReg(brx.Bits(24, 8)));
        }
        if (op == 0x899 && !insn.Bits(80, 1) && !insn.Bits(76, 1) && insn.Bits(64, 8) == 63 &&
            insn.Bits(73, 2) == 3 && insn.words[1] == 2) {
            shl_idx = i;
            shl_ur = insn.Bits(24, 8);
            break;
        }
        if (op == 0x8a4 && insn.Bits(64, 8) == 63 && !insn.Bits(74, 1) && !insn.Bits(75, 1) &&
            insn.words[1] == 4) {
            shl_idx = i;
            shl_ur = insn.Bits(24, 8);
            break;
        }
        if ((op == 0x291 || op == 0x891) && !insn.Bits(80, 1) && insn.Bits(75, 5) == 2 &&
            ((op == 0x291 && insn.Bits(32, 8) == 63) || (op == 0x891 && insn.words[1] == 0))) {
            shl_idx = i;
            shl_ur = insn.Bits(24, 8);
            break;
        }
        return std::nullopt;
    }
    if (!shl_idx || shl_ur >= 63) {
        return std::nullopt;
    }
    for (u32 i = *shl_idx; i-- > 0;) {
        const Instruction& insn = code[i];
        if (StopsJumpTableScan(insn)) {
            return std::nullopt;
        }
        if (!InstructionWritesUniform(insn, shl_ur)) {
            continue;
        }
        const u32 op = insn.Bits(0, 12);
        if (op == 0xc82 && insn.Bits(12, 4) == 7 && insn.words[2] == 0x08000000U &&
            insn.words[1] < 63U) {
            shl_ur = insn.words[1];
            continue;
        }
        if (op == 0x3c2) { // Clamped in GPR before R2UR -> continue tracing clamp in GPR
            const u32 gpr_src = insn.Bits(24, 8);
            for (u32 k = i; k-- > 0;) {
                const Instruction& g = code[k];
                if (StopsJumpTableScan(g)) {
                    return std::nullopt;
                }
                if (!InstructionWritesGpr(g, gpr_src)) {
                    continue;
                }
                if (g.Bits(0, 12) == 0x817 && g.Bits(87, 4) == 7 && (g.words[1] >> 31) == 0) {
                    return AmpereBranchTableInfo{
                        .cbuf_index = cbuf_index,
                        .cbuf_offset = cbuf_offset,
                        .num_entries = g.words[1] + 1,
                        .branch_reg = UniformReg(brx.Bits(24, 8)),
                    };
                }
                return std::nullopt;
            }
            return std::nullopt;
        }
        if (op == 0x887 && insn.Bits(12, 4) == 7 && insn.words[1] <= 0x7ffffU) {
            const u32 condition = insn.Bits(87, 3);
            const bool negated = insn.Bits(90, 1) != 0;
            for (u32 j = i; j-- > 0;) {
                const Instruction& cmp = code[j];
                if (StopsJumpTableScan(cmp)) {
                    break;
                }
                if (!InstructionWritesPred(cmp, true, condition)) {
                    continue;
                }
                if (cmp.Bits(0, 12) == 0x88c && cmp.Bits(12, 4) == 7 &&
                    cmp.Bits(24, 8) == insn.Bits(24, 8) && !cmp.Bits(73, 1) &&
                    cmp.Bits(87, 4) == 7U && !cmp.Bits(72, 1)) {
                    const u32 cmp_op = cmp.Bits(76, 3);
                    const bool normal_match =
                        (cmp.Bits(81, 3) == condition && !negated) ||
                        (cmp.Bits(84, 3) == condition && negated);
                    const bool inverted_match =
                        (cmp.Bits(81, 3) == condition && negated) ||
                        (cmp.Bits(84, 3) == condition && !negated);
                    const bool valid_lt = (cmp_op == 1U || cmp_op == 3U) && normal_match;
                    const bool valid_ge = (cmp_op == 4U || cmp_op == 6U) && inverted_match;
                    if (valid_lt || valid_ge) {
                        const u32 max_val = cmp.words[1];
                        const u32 sel_val = insn.words[1];
                        const u32 n =
                            (cmp_op == 1U && sel_val + 1U <= max_val) ? max_val : (sel_val + 1U);
                        if (n > 0U && n < 0x10000U) {
                            return AmpereBranchTableInfo{
                                .cbuf_index = cbuf_index,
                                .cbuf_offset = cbuf_offset,
                                .num_entries = n,
                                .branch_reg = UniformReg(brx.Bits(24, 8)),
                            };
                        }
                    }
                }
                break;
            }
        }
        return std::nullopt;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> ResolveTensorMode(
    const LoweringOptions& options) noexcept {
    if (const char* const forced_mode = std::getenv("EDEN_NX2_TENSOR"); forced_mode != nullptr) {
        return std::string_view{forced_mode};
    }
    if (!options.tensor) {
        return std::nullopt;
    }
    return options.compact_tensor ? std::string_view{"compact"} : std::string_view{"1"};
}

[[nodiscard]] bool ValidateTier2Instruction(std::span<const Instruction> code, u32 index,
                                            const Instruction& insn,
                                            const LoweringOptions& options,
                                            u32& required_texture_buffer,
                                            const std::vector<u8>& branch_targets,
                                            std::string_view& error) noexcept {
    const u32 opcode = insn.Bits(0, 12);
    const u32 form = insn.Bits(9, 3);
    const auto fail = [&](std::string_view msg) {
        error = msg;
        return false;
    };
    const auto bind_texture_buffer = [&](u32 binding) {
        if (required_texture_buffer != ~0U && required_texture_buffer != binding) {
            return false;
        }
        required_texture_buffer = binding;
        return true;
    };
    const auto valid_alu_form = [&] {
        if (form < 1U || form > 7U) {
            return fail("native ALU source form");
        }
        if ((form == 3U || form == 5U) && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        if ((form == 6U || form == 7U) && insn.Bits(32, 8) > 63U) {
            return fail("native uniform source register");
        }
        return true;
    };

    switch (opcode) {
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23: // FFMA
        if (insn.Bits(76, 1) ||
            (insn.Bits(81, 11) & ~(opcode == 0xe23 || opcode == 0xc23 ? 0x400U : 0U))) {
            return fail("native FFMA denormal/source mode");
        }
        return valid_alu_form();
    case 0x220: case 0x820: case 0xa20: case 0xc20: { // FMUL
        const u32 scale = insn.Bits(84, 3);
        if (insn.Bits(76, 1) || scale == 0U || insn.Bits(81, 3) || insn.Bits(87, 4) ||
            insn.Bits(91, 1) != static_cast<u32>(opcode == 0xc20) ||
            (scale != 4U && (!insn.Bits(80, 1) || insn.Bits(78, 2)))) {
            return fail("native FMUL scale/mode");
        }
        return valid_alu_form();
    }
    case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21: // FADD
        if (insn.Bits(81, 10) || insn.Bits(91, 1) != static_cast<u32>(opcode == 0xe21)) {
            return fail("native FADD mode");
        }
        return valid_alu_form();
    case 0x209: case 0x809: case 0xa09: case 0xc09: // FMNMX
        if (insn.Bits(74, 6) || insn.Bits(82, 5) || (opcode != 0xc09 && insn.Bits(91, 1))) {
            return fail("native floating min/max mode");
        }
        return valid_alu_form();
    case 0x308: case 0xb08: case 0xd08: { // MUFU
        const u32 op = insn.Bits(74, 6);
        const bool is_fp16 = insn.Bits(73, 1) != 0U;
        if (op > 9U || op == 6U || op == 7U || (op == 9U && is_fp16) ||
            insn.Bits(80, 12) != (opcode == 0xd08 ? 0x800U : 0U)) {
            return fail("native MUFU mode");
        }
        if (is_fp16 && (opcode != 0x308 || insn.Bits(60, 2) > 1U)) {
            return fail("native half special-function source");
        }
        if (opcode == 0xb08 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        if (opcode == 0xd08 && insn.Bits(32, 8) > 63U) {
            return fail("native MUFU uniform source");
        }
        return true;
    }
    case 0x526: // Indexed IPA
        if (insn.Bits(24, 8) > 63U || insn.Bits(32, 8) != 255U ||
            (insn.Bits(76, 4) != 0U && insn.Bits(76, 4) != 4U) || insn.Bits(81, 3) != 7U) {
            return fail("native indexed IPA mode");
        }
        return true;
    case 0x326: // IPA
        if (insn.Bits(32, 8) != 255U || insn.Bits(76, 2) > 1U || insn.Bits(79, 1) ||
            insn.Bits(81, 3) != 7U) {
            return fail("native IPA interpolation mode");
        }
        return true;
    case 0x208: case 0x808: case 0xa08: // FSEL
        return valid_alu_form();
    case 0x20a: case 0x80a: case 0xa0a: case 0xc0a: // FSET
        if (insn.Bits(64, 8) || insn.Bits(74, 2) == 3U || insn.Bits(81, 6) ||
            (opcode != 0xc0a && insn.Bits(91, 1))) {
            return fail("native floating comparison output mode");
        }
        return valid_alu_form();
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b: // FSETP
        if (insn.Bits(74, 2) > 2U) {
            return fail("native predicate combination");
        }
        return valid_alu_form();
    case 0x304: // F2F.F16.F32
        if (insn.words[2] != 0x00200800U || insn.Bits(24, 8) != 0U ||
            (insn.words[1] & ~0xc00000ffU) != 0U) {
            return fail("native scalar float conversion mode");
        }
        return true;
    case 0x23e: case 0xa3e: // F2FP
        if (insn.Bits(62, 2) || insn.Bits(72, 1) || insn.Bits(73, 5) || insn.Bits(79, 2) ||
            insn.Bits(81, 10) || insn.Bits(91, 1)) {
            return fail("native float packing modifiers");
        }
        if (opcode == 0xa3e && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    case 0x232: case 0x832: case 0xa32: // HMUL2
        if (insn.Bits(78, 2) || insn.Bits(81, 10) || insn.Bits(91, 1)) {
            return fail("native half multiply mode");
        }
        if (opcode == 0xa32 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    case 0x230: case 0x430: case 0x630: { // HADD2
        if (insn.Bits(76, 1) || insn.Bits(79, 1) || insn.Bits(81, 10) || insn.Bits(91, 1)) {
            return fail("native half add mode");
        }
        if (opcode == 0x630 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        if (insn.Bits(78, 1)) {
            const u32 a_reg = insn.Bits(24, 8);
            const u32 a_swizzle = insn.Bits(74, 2);
            const bool a_abs = insn.Bits(73, 1) != 0U;
            const bool a_neg = insn.Bits(72, 1) != 0U;
            const u32 b_swizzle = opcode == 0x230 ? insn.Bits(60, 2)
                                                  : (opcode == 0x430 ? 0U : insn.Bits(60, 2));
            if (a_reg != 255U || !a_neg || a_abs || a_swizzle != 0U || b_swizzle < 2U ||
                insn.Bits(77, 1) || insn.Bits(80, 1)) {
                return fail("native float-output half add mode");
            }
        }
        return true;
    }
    case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31: { // HFMA2
        const bool relu = insn.Bits(79, 1) != 0U;
        if (insn.Bits(78, 1) || insn.Bits(85, 1) || insn.Bits(91, 1) ||
            (relu ? insn.Bits(77, 1) != 0U : insn.Bits(87, 4) != 0U)) {
            return fail("native half fused multiply-add mode");
        }
        if ((opcode == 0x631 || opcode == 0xa31) && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        const bool b_wide = form == 4U || form == 5U;
        const bool b_imm = b_wide && form == 4U;
        const bool b_upper = form == 2U || form == 3U;
        const u32 b_swizzle =
            b_wide ? (b_imm ? 0U : insn.Bits(60, 2)) : insn.Bits(b_upper ? 81U : 60U, 2);
        const bool b_abs =
            b_wide ? (!b_imm && insn.Bits(62, 1) != 0U) : (insn.Bits(b_upper ? 83U : 62U, 1) != 0U);
        const bool c_wide = form == 2U || form == 3U;
        const bool c_imm = c_wide && form == 2U;
        const bool c_abs =
            c_wide ? (!c_imm && insn.Bits(62, 1) != 0U) : (insn.Bits(83, 1) != 0U);
        if (b_abs || c_abs) {
            return fail("native half fused multiply-add absolute source");
        }
        if (insn.Bits(86, 1) && (b_swizzle != 0U || (!b_wide && insn.Bits(b_upper ? 64U : 32U, 8) >= 250U))) {
            return fail("native half mixed-sign source mode");
        }
        return true;
    }
    case 0x233: case 0x433: case 0x633: // HSET2
        if (insn.Bits(64, 5) || insn.Bits(69, 2) > 2U || insn.Bits(81, 6) || insn.Bits(91, 1) ||
            (opcode == 0x233 && (insn.words[1] & 0x0fffff00U))) {
            return fail("native packed half comparison mode");
        }
        if (opcode == 0x633 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    case 0x234: case 0x434: case 0x634: // HSETP2
        if (insn.Bits(69, 2) > 2U) {
            return fail("native predicate combination");
        }
        if ((opcode == 0x234 && ((insn.words[1] & 0x0fffff00U) || insn.Bits(64, 5))) ||
            (opcode == 0x634 && insn.Bits(64, 5))) {
            return fail("native half comparison mode");
        }
        if (opcode == 0x634 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    case 0x240: case 0x840: // HMNMX2
        if (insn.Bits(76, 11) || insn.Bits(91, 1)) {
            return fail("native half min/max mode");
        }
        if (insn.Bits(74, 2) == 1U || (opcode == 0x240 && insn.Bits(60, 2) == 1U)) {
            return fail("native half min/max source mode");
        }
        return true;
    case 0x822: // FSWZADD
        if (insn.words[1] > 0xffU || (insn.words[2] & ~0x2000U) > 255U) {
            return fail("native swizzled add mode");
        }
        return true;
    case 0x311: { // F2I.FTZ.U64.TRUNC
        const u32 dest = insn.Bits(16, 8);
        if (insn.words[2] != 0x0021d800U || (insn.words[1] & ~0xffU) || insn.Bits(24, 8) ||
            (dest & 1U) || dest >= 254U) {
            return fail("native unsigned 64-bit float conversion mode");
        }
        return true;
    }
    case 0x312: { // I2F.F32.(U|S)64
        const u32 src = insn.Bits(32, 8);
        if ((insn.words[2] & ~0xc400U) != 0x00301000U || (insn.words[1] & ~0xffU) ||
            insn.Bits(24, 8) || (src & 1U) || src >= 254U) {
            return fail("native 64-bit integer-to-float mode");
        }
        return true;
    }
    case 0x243: // F2IP
        if ((insn.words[1] & ~0xffU) || (insn.words[2] & ~0xd8ffU) != 0x400U ||
            (!LowerExtended() && insn.Bits(24, 8) != 255U)) {
            return fail("native packed float-to-byte mode");
        }
        return true;
    case 0x305: case 0xb05: case 0xd05: { // F2I
        const u32 mode = insn.words[2] & ~(opcode == 0xd05 ? 0x08000000U : 0U);
        const bool half = opcode == 0x305 && (mode & ~0x100U) == 0x0010f000U;
        const bool small = mode == 0x00212800U;
        const bool floor = (mode & ~0x10100U) == 0x00207000U;
        const bool nearest = (mode & ~0x10100U) == 0x00203000U;
        const bool round_up = (mode & ~0x10100U) == 0x0020b000U;
        const bool signed_byte = opcode == 0x305 && mode == 0x0021e100U;
        const bool short_trunc = opcode == 0x305 && (mode & ~0x100U) == 0x0021e800U;
        const bool unsigned_byte = opcode == 0x305 && (mode & ~0x10000U) == 0x0020e000U;
        const bool abs = opcode != 0xd05 && !half && insn.Bits(62, 1) != 0U;
        const bool neg = opcode != 0xd05 && !half && insn.Bits(63, 1) != 0U;
        if ((!half && !small && !floor && !nearest && !round_up && !signed_byte && !short_trunc &&
             !unsigned_byte && (mode & ~0x10100U) != 0x0020f000U) ||
            ((unsigned_byte || opcode == 0xb05) &&
             (insn.Bits(24, 8) || (insn.words[3] & 0xc00001ffU))) ||
            (opcode == 0xb05 && insn.Bits(59, 3)) ||
            (opcode != 0xb05 &&
             (insn.words[1] &
              ~(half ? 0x100000ffU
                     : (0xffU | (abs ? 0x40000000U : 0U) | (neg ? 0x80000000U : 0U))))) ||
            (opcode == 0xd05 && insn.Bits(32, 8) > 63U)) {
            return fail("native float-to-integer conversion mode");
        }
        if (opcode == 0xb05 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    }
    case 0x307: case 0xb07: // FRND
        if (insn.Bits(75, 2) != 2U || insn.Bits(84, 2) != 2U || insn.Bits(81, 3) ||
            insn.Bits(86, 6)) {
            return fail("native floating round format");
        }
        if (opcode == 0xb07 && !ValidCbuf(insn)) {
            return fail("native constant buffer address");
        }
        return true;
    case 0x306: case 0xd06: { // I2F 8/16/32-bit
        const bool uniform = opcode == 0xd06;
        if (uniform && insn.Bits(32, 8) > 63U) {
            return fail("native byte-to-float conversion mode");
        }
        if ((insn.words[2] & ~0xc400U) == (uniform ? 0x08201000U : 0x00201000U) &&
            !(insn.words[1] & ~0xffU)) {
            return true;
        }
        if ((insn.words[2] & ~0x400U) == (uniform ? 0x08101000U : 0x00101000U) &&
            !(insn.words[1] & 0xefffff00U)) {
            return true;
        }
        if ((insn.words[2] & ~0x400U) != (uniform ? 0x08001000U : 0x1000U) ||
            (insn.words[1] & (!uniform || LowerExtended() ? 0xcfffff00U : 0xffffff00U))) {
            return fail("native byte-to-float conversion mode");
        }
        return true;
    }
    case 0x906: // I2F imm
        if ((insn.words[2] & ~0xc400U) != 0x00201000U) {
            return fail("native immediate integer-to-float mode");
        }
        return true;
    case 0x245: case 0x845: case 0xa45: case 0xc45: // I2FP
        if ((insn.words[2] & ~(0x400U | 0xc000U | (opcode == 0xc45 ? 0x08000000U : 0U))) !=
            0x00201000U) {
            return fail("native packed integer-to-float format");
        }
        return valid_alu_form();
    case 0x381: case 0x981: { // LDG
        const u32 size = insn.Bits(73, 3);
        const u32 dest = insn.Bits(16, 8);
        const u32 address = insn.Bits(24, 8);
        if (size > 6U || (opcode == 0x381 && insn.Bits(32, 8)) || (size == 5U && (dest & 1U)) ||
            (size == 6U && (dest & 3U))) {
            return fail("native global load mode");
        }
        if (opcode == 0x381) {
            if ((insn.words[2] & ~0x2e00U) != 0x001e8100U || (address & 1U) || address >= 254U) {
                return fail("native global load mode");
            }
            return true;
        }
        const u32 u_raw = insn.Bits(32, 6);
        const bool descriptor = (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c1e1100U;
        if ((!descriptor &&
             (insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U | 0x6000U)) != 0x081e8100U) ||
            insn.Bits(38, 2) || u_raw >= 63U || (u_raw & 1U)) {
            return fail("native global load uniform addressing mode");
        }
        if ((descriptor || (address != 255U && insn.Bits(90, 1))) &&
            ((address & 1U) || address >= 254U)) {
            return fail("native global load register address pair");
        }
        return true;
    }
    case 0x386: case 0x986: { // STG
        const u32 size = insn.Bits(73, 3);
        const u32 data = insn.Bits(32, 8);
        const u32 address = insn.Bits(24, 8);
        const u32 words = size < 4U ? 1U : (1U << (size - 4U));
        if (size > 6U || (data != 255U && ((size == 5U && (data & 1U)) ||
                                            (size == 6U && (data & 3U)) || data + words > 255U))) {
            return fail("native global store size");
        }
        if (opcode == 0x986) {
            const u32 u_raw = insn.Bits(64, 6);
            const bool descriptor = (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c101100U;
            if ((!descriptor &&
                 (insn.words[2] & ~(0x04000000U | 0x3fU | 0xe00U | 0x4000U)) != 0x0810a100U) ||
                u_raw >= 63U || (u_raw & 1U) || !insn.Bits(72, 1)) {
                return fail("native global store uniform addressing mode");
            }
            if ((descriptor || (address != 255U && insn.Bits(90, 1))) &&
                ((address & 1U) || address >= 254U)) {
                return fail("native global store register address pair");
            }
            return true;
        }
        if ((insn.words[2] & ~0xe00U) != 0x0010a100U) {
            return fail("native global store register addressing mode");
        }
        if ((address & 1U) || address >= 254U) {
            return fail("native global store register address pair");
        }
        return true;
    }
    case 0x3a8: { // ATOMG GPR
        const u32 address = insn.Bits(24, 8);
        const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
        if (insn.words[2] != 0x001ee100U || (address & 1U) || address >= 254U ||
            offset < -0x80000 || offset > 0x7ffff) {
            return fail("native global atomic register mode");
        }
        return true;
    }
    case 0x9a8: { // ATOMG uniform
        const u32 u_raw = insn.Bits(64, 6);
        const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
        if (insn.Bits(24, 8) != 255U || offset < -0x80000 || offset > 0x7ffff ||
            (insn.words[2] & ~0x3fU) != 0x081ee140U || u_raw >= 63U || (u_raw & 1U)) {
            return fail("native global atomic mode");
        }
        return true;
    }
    case 0x98e: { // RED
        const u32 type = insn.Bits(73, 3);
        const u32 operation = insn.Bits(87, 3);
        const bool uniform_addr = insn.Bits(24, 8) == 255U;
        const u32 variable = 0x03800e00U | (uniform_addr ? 0x3fU : 0U);
        const u32 mode = uniform_addr ? 0x0c10e100U : 0x0010e100U;
        const s32 offset = static_cast<s32>(insn.Bits(40, 24) << 8) >> 8;
        if ((insn.words[2] & ~variable) != mode || (type != 0U && type != 1U && type != 3U) ||
            (type == 3U && operation != 0U)) {
            return fail("native global reduction mode");
        }
        const u32 addr = uniform_addr ? insn.Bits(64, 6) : insn.Bits(24, 8);
        if ((uniform_addr ? (addr >= 63U) : (addr >= 254U)) || (addr & 1U) || offset < -0x80000 ||
            offset > 0x7ffff) {
            return fail("native global reduction address");
        }
        return true;
    }
    case 0x388: case 0x984: case 0x988: { // LDS / STS
        const u32 width = insn.Bits(73, 3);
        const u32 stride = insn.Bits(78, 2);
        const bool uniform_base = (opcode == 0x984 || opcode == 0x988) && insn.Bits(91, 1);
        const u32 u_raw = uniform_base ? insn.Bits(opcode == 0x988 ? 64 : 32, 8) : 63U;
        const u32 ignored = uniform_base ? (opcode == 0x988 ? 0x080000ffU : 0x08000000U) : 0U;
        if (width > 6U || (insn.words[2] & ~ignored) != ((width << 9) | (stride << 14)) ||
            (opcode == 0x984 && !uniform_base && insn.Bits(32, 8)) ||
            (opcode == 0x988 && !uniform_base) || u_raw > 63U) {
            return fail("native shared memory addressing mode");
        }
        return true;
    }
    case 0x38c: // ATOMS
        if ((insn.words[2] & ~0x07800200U) || insn.Bits(87, 4) > 8U ||
            (insn.Bits(40, 24) & 3U)) {
            return fail("native shared atomic mode");
        }
        return true;
    case 0xf8c: // ATOMS.POPC.INC
        if (insn.Bits(16, 8) != 255U || (insn.words[1] & 0xffU) || (insn.Bits(40, 24) & 3U) ||
            (insn.words[2] & ~0xc000U) != 0x0d00003fU) {
            return fail("native shared increment mode");
        }
        return true;
    case 0xfae: { // LDGSTS
        const u32 global_addr = insn.Bits(24, 8);
        const u32 desc = insn.Bits(64, 6);
        if (insn.Bits(12, 4) != 7U || (global_addr & 1U) || global_addr >= 254U || desc >= 63U ||
            (desc & 1U) || (insn.words[1] & 0xf000U) ||
            (insn.words[2] & ~(0x03800000U | 0x7fU)) != 0x0c101c00U) {
            return fail("native global-to-shared copy mode");
        }
        return true;
    }
    case 0x387: case 0x983: { // LDL / STL
        const bool store = opcode == 0x387;
        const u32 size = insn.Bits(73, 3);
        const u32 data = insn.Bits(32, 8);
        const u32 offset = insn.Bits(40, 24);
        const u32 address = insn.Bits(24, 8);
        const u32 words = size == 6U ? 4U : (size == 5U ? 2U : 1U);
        const u32 value = store ? data : insn.Bits(16, 8);
        const u32 alignment = size < 2U ? 1U : (size < 4U ? 2U : 4U);
        if ((address != 255U && address >= 254U) || (store && insn.Bits(16, 8)) ||
            (!store && data) ||
            (value != 255U && (value + words > 255U || (value & (words - 1U)))) ||
            (!store && value == 255U) || (offset & (alignment - 1U)) ||
            offset + 4U * words > 0x800000U || size > 6U ||
            (insn.words[2] & ~(store ? 0xe00U : 0x200e00U)) != 0x00100000U) {
            return fail("native local scalar spill mode");
        }
        return true;
    }
    case 0x91a: // DEPBAR
        if (insn.words[0] != 0x791aU || (insn.words[1] & ~0x7fffU) != 0x8000U || insn.words[2]) {
            return fail("native dependency barrier mode");
        }
        return true;
    case 0x992: { // MEMBAR
        const u32 scope = insn.Bits(76, 3);
        if (insn.words[1] || insn.words[2] != (scope << 12) ||
            (scope != 0U && scope != 2U && scope != 3U && scope != 5U)) {
            return fail("native memory barrier scope");
        }
        return true;
    }
    case 0x9ab: // ERRBAR
        if (insn.words[1] || insn.words[2]) {
            return fail("native error barrier mode");
        }
        return true;
    case 0x9af: // LDGDEPBAR
        if (insn.words[0] != 0x79afU || insn.words[1] || insn.words[2]) {
            return fail("native copy-group commit mode");
        }
        return true;
    case 0xb1d: // BAR.SYNC
        if (insn.words[1] || (insn.words[2] & ~0x10000U) || insn.Bits(12, 4) != 7U) {
            return fail("native workgroup barrier mode");
        }
        return true;
    case 0x321: case 0x322: { // ALD / AST
        if (insn.Bits(76, 2) || insn.Bits(79, 1)) {
            return fail("native indexed/patch attribute access");
        }
        const u32 address = insn.Bits(40, 10);
        const u32 components = insn.Bits(74, 2) + 1U;
        if ((address % 4U) != 0U || address + components * 4U > 0x400U ||
            insn.Bits(24, 8) != 255U) {
            return fail("native attribute address or indexing");
        }
        const u32 value_reg = opcode == 0x321 ? insn.Bits(16, 8) : insn.Bits(32, 8);
        if (!(opcode == 0x322 && address == 0x5cU && components == 1U) && value_reg != 255U &&
            value_reg + components > 255U) {
            return fail("native attribute register range");
        }
        return true;
    }
    case 0x324: case 0x924: // OUT
        if (!options.geometry) {
            return fail("unsupported native opcode");
        }
        if (insn.Bits(80, 12) || insn.Bits(64, 14)) {
            return fail("native OUT mode");
        }
        if (opcode == 0x924 && insn.Bits(78, 2) != 0U && insn.words[1] >= (1U << 19)) {
            return fail("native OUT stream immediate");
        }
        return true;
    case 0x923: case 0x927: { // ISBERD / ISBEWR
        if (options.geometry) {
            if (opcode != 0x923 || insn.Bits(75, 5) || insn.Bits(40, 24)) {
                return fail("native geometry ISBERD mode");
            }
            return true;
        }
        const bool write = opcode == 0x927;
        const u32 region = insn.Bits(76, 2);
        const bool output = insn.Bits(79, 1) != 0U;
        if ((insn.words[2] & ~0xf800U) != 0U || (region != 0U && region != 3U) ||
            (write && !output) || (!write && region != 3U) || (!write && insn.Bits(16, 8) == 255U)) {
            return fail("native stage buffer access mode");
        }
        return true;
    }
    case 0xb60: case 0xf60: case 0x361: { // TEX
        const bool bindless = opcode != 0xb60;
        const u32 handle = insn.Bits(40, 6);
        if (bindless && ((opcode == 0xf60 && (handle >= 63U || insn.Bits(46, 13))) ||
                         (opcode == 0x361 && (insn.Bits(40, 19) || insn.Bits(87, 3) > 3U)))) {
            return fail("native bindless texture mode");
        }
        const u32 mask = insn.Bits(72, 4);
        const u32 dest = insn.Bits(16, 8);
        const u32 second_dest = insn.Bits(64, 8);
        const bool split = second_dest != 255U;
        const bool depth_compare = insn.Bits(78, 1) != 0U;
        const u32 dimension = insn.Bits(61, 3);
        const u32 lod = insn.Bits(87, 3);
        const u32 source_a = insn.Bits(24, 8);
        const u32 source_b = insn.Bits(32, 8);
        const bool offsets = insn.Bits(76, 1) != 0U;
        const u32 binding = insn.Bits(54, 5);
        constexpr std::array<u32, 8> coordinate_count{1, 2, 3, 3, 2, 3, 0, 4};
        if (dimension == 6U || insn.Bits(81, 3) != 7U ||
            insn.Bits(59, 1) != (opcode == 0x361 ? 1U : 0U) ||
            (!bindless && (insn.Bits(40, 14) > 0x1fffU || binding >= 18U)) ||
            (lod > 3U && lod != 6U && lod != 7U) || (depth_compare && mask != 1U) ||
            mask == 0U) {
            return fail("native texture mode or descriptor binding");
        }
        const u32 components = static_cast<u32>(std::popcount(mask));
        if (insn.Bits(79, 1) && components > 2U && !split && dest != 255U && dest >= 254U) {
            return fail("native half-precision texture result registers");
        }
        const u32 coordinates = coordinate_count[dimension];
        const bool has_lod = lod == 2U || lod == 3U || lod == 6U || lod == 7U;
        const u32 handle_slot = opcode == 0x361 ? 1U : 0U;
        const u32 count =
            coordinates + handle_slot + (has_lod ? 1U : 0U) + (offsets ? 1U : 0U) +
            (depth_compare ? 1U : 0U);
        const u32 in_a = opcode == 0x361 && !insn.Bits(60, 1) ? coordinates
                         : count <= 4U                        ? (count + 1U) / 2U
                         : opcode == 0x361                    ? coordinates
                                                              : 4U;
        if ((source_a != 255U && source_a + in_a > 255U) ||
            (count > in_a && source_b != 255U && source_b + (count - in_a) > 255U)) {
            return fail("native texture mode or descriptor binding");
        }
        if (opcode == 0x361) {
            const u32 hreg = coordinates < in_a
                                 ? (source_a == 255U ? 255U : source_a + coordinates)
                                 : (source_b == 255U ? 255U : source_b + (coordinates - in_a));
            if (hreg >= 255U) {
                return fail("native bindless texture mode");
            }
        }
        if (!bindless) {
            (void)bind_texture_buffer(binding);
        }
        return true;
    }
    case 0xb63: case 0xf63: { // TLD4
        const bool bindless = opcode == 0xf63;
        const bool array = insn.Bits(61, 3) == 5U;
        const bool f16 = !bindless && insn.Bits(79, 2) == 1U;
        const u32 mask = insn.Bits(72, 4);
        const u32 components = static_cast<u32>(std::popcount(mask));
        const u32 dest = insn.Bits(16, 8);
        const u32 second_dest = insn.Bits(64, 8);
        const u32 source_a = insn.Bits(24, 8);
        const u32 source_b = insn.Bits(32, 8);
        const u32 binding = insn.Bits(54, 5);
        const u32 offset_mode = insn.Bits(76, 2);
        const bool depth = insn.Bits(78, 1) != 0U;
        if (f16 && (dest == 255U || array || (second_dest == 255U && components > 2U))) {
            return fail("native half-precision texture result registers");
        }
        if ((insn.Bits(61, 3) != 1U && !array) || offset_mode == 3U || (!f16 && insn.Bits(79, 1)) ||
            insn.Bits(81, 3) != 7U || insn.Bits(59, 1) ||
            (bindless && (offset_mode || depth)) || (offset_mode && source_b == 255U) ||
            insn.Bits(89, 1) || mask == 0U ||
            (bindless ? (insn.Bits(40, 6) >= 63U || insn.Bits(46, 13) || !insn.Bits(91, 1))
                      : (insn.Bits(40, 14) > 0x1fffU || binding >= 18U))) {
            return fail("native texture gather mode");
        }
        const u32 coordinates = array ? 3U : 2U;
        const u32 meta = (offset_mode == 1U ? 1U : offset_mode == 2U ? 2U : 0U) +
                         (depth ? 1U : 0U);
        const u32 in_a = meta ? coordinates : (coordinates + 1U) / 2U;
        const u32 in_b = meta ? meta : coordinates - in_a;
        if ((source_a != 255U && source_a + in_a > 255U) ||
            (source_b != 255U && source_b + in_b > 255U)) {
            return fail("native texture gather mode");
        }
        if (offset_mode == 2U) {
            for (u32 i = 0; i < 2U; ++i) {
                std::string_view ptp_error;
                (void)FindGatherPtpOffsetImmediate(code, index,
                                                   source_b == 255U ? 255U : source_b + i,
                                                   insn.Bits(12, 4), &branch_targets, ptp_error);
                if (!ptp_error.empty() &&
                    (f16 || ptp_error != "native per-tap gather nonconstant offsets")) {
                    return fail(ptp_error);
                }
            }
        }
        if (!bindless) {
            (void)bind_texture_buffer(binding);
        }
        return true;
    }
    case 0xb66: { // TLD bound
        const u32 mask = insn.Bits(72, 4);
        const bool f16 = insn.Bits(79, 2) == 1U;
        const u32 components = static_cast<u32>(std::popcount(mask));
        const u32 dest = insn.Bits(16, 8);
        const u32 second_dest = insn.Bits(64, 8);
        const u32 dimension = insn.Bits(61, 3);
        if (f16 && (dest == 255U || dimension == 5U || (second_dest == 255U && components > 2U))) {
            return fail("native half-precision texture result registers");
        }
        const bool level = insn.Bits(87, 3) == 3U;
        const bool offsets = insn.Bits(76, 1) != 0U;
        const bool multisample = insn.Bits(78, 1) != 0U;
        const bool array = dimension == 5U;
        const bool volume = dimension == 2U;
        const u32 source_a = insn.Bits(24, 8);
        const u32 source_b = insn.Bits(32, 8);
        const u32 binding = insn.Bits(54, 5);
        const bool split = second_dest != 255U;
        const u32 flags76 = insn.Bits(76, 5) & ~(f16 ? 8U : 0U);
        if (!insn.Bits(60, 1) || (dimension != 0U && dimension != 1U && !array && !volume) ||
            mask == 0U || (flags76 & ~5U) ||
            (multisample && (dimension != 1U || level || offsets || (source_a & 1U) ||
                             source_a >= 254U || source_b == 255U)) ||
            (offsets && (dimension != 1U || (source_a & 1U) || source_a >= 254U ||
                         source_b == 255U || (level && ((source_b & 1U) || source_b >= 254U)))) ||
            insn.Bits(81, 3) != 7U || (insn.Bits(87, 3) != 1U && !level) || (level && volume) ||
            (level && dimension == 1U && ((source_a & 1U) || source_a >= 254U)) ||
            (level && array && ((source_b & 1U) || source_b >= 254U)) || insn.Bits(59, 1) ||
            insn.Bits(40, 14) > 0x1fffU || binding >= 18U || dest == 255U ||
            (!f16 && (mask & (mask - 1U)) && ((dest & 1U) || dest >= 254U)) ||
            (!f16 && split && mask == 15U && ((second_dest & 1U) || second_dest >= 254U)) ||
            ((array || volume) && ((source_a & 1U) || source_a >= 254U))) {
            return fail("native texel fetch mode or descriptor binding");
        }
        if (!bind_texture_buffer(binding) && (level || offsets || multisample)) {
            return fail("native texture descriptor buffer conflict");
        }
        return true;
    }
    case 0x367: { // TLD.B GPR handle
        constexpr std::array<u32, 8> coordinate_count{1, 2, 3, 0, 0, 3, 0, 0};
        const u32 mask = insn.Bits(72, 4);
        const u32 source_a = insn.Bits(24, 8);
        const u32 source_b = insn.Bits(32, 8);
        const u32 dimension = insn.Bits(61, 3);
        const u32 coordinates = coordinate_count[dimension];
        const u32 count = coordinates + 1U;
        const u32 in_a = (count + 1U) / 2U;
        if (coordinates == 0U || !insn.Bits(59, 1) || !insn.Bits(60, 1) || insn.Bits(40, 19) ||
            insn.Bits(76, 5) || insn.Bits(81, 3) != 7U || insn.Bits(87, 3) != 1U ||
            insn.Bits(91, 1) || mask == 0U || source_a == 255U || source_a + in_a > 255U ||
            source_b == 255U || source_b + (count - in_a) > 255U) {
            return fail("native bindless texel fetch mode");
        }
        return true;
    }
    case 0xf66: { // TLD.LZ uniform handle
        const u32 handle = insn.Bits(40, 6);
        const u32 mask = insn.Bits(72, 4);
        const u32 components = static_cast<u32>(std::popcount(mask));
        const u32 dest = insn.Bits(16, 8);
        const u32 second_dest = insn.Bits(64, 8);
        const u32 source_a = insn.Bits(24, 8);
        const u32 dimension = insn.Bits(61, 3);
        const bool array = dimension == 5U;
        const bool volume = dimension == 2U;
        const bool split = second_dest != 255U;
        if (handle >= 63U || insn.Bits(46, 13) || !insn.Bits(91, 1) || !insn.Bits(60, 1) ||
            insn.Bits(59, 1) || insn.Bits(76, 5) || insn.Bits(81, 3) != 7U ||
            insn.Bits(87, 3) != 1U || (dimension != 0U && dimension != 1U && !array && !volume) ||
            mask == 0U || ((array || volume) && source_a >= 254U) ||
            (dest != 255U && dest + (split ? (std::min)(components, 2U) : components) > 255U) ||
            (split && components > 2U && second_dest + components - 2U > 255U)) {
            return fail("native bindless texel fetch mode");
        }
        return true;
    }
    case 0xb69: case 0xf69: case 0x36a: { // TMML
        const bool bindless = opcode != 0xb69;
        const u32 mask = insn.Bits(72, 4);
        const u32 source_a = insn.Bits(24, 8);
        const u32 source_b = insn.Bits(32, 8);
        const u32 binding = insn.Bits(54, 5);
        const u32 dimension = insn.Bits(61, 3);
        constexpr std::array<u32, 4> coordinate_count{1, 2, 3, 3};
        const u32 coordinates = dimension < 4U ? coordinate_count[dimension] : 0U;
        const u32 in_a =
            (source_b == 255U || opcode == 0x36a) ? coordinates : (coordinates + 1U) / 2U;
        if (coordinates == 0U || insn.Bits(76, 4) || insn.Bits(80, 10) || mask == 0U ||
            insn.Bits(59, 1) != (opcode == 0x36a ? 1U : 0U) ||
            (source_a != 255U && source_a + in_a > 255U) ||
            (in_a < coordinates && source_b != 255U && source_b + (coordinates - in_a) > 255U) ||
            (opcode == 0xf69 && (insn.Bits(40, 6) >= 63U || insn.Bits(46, 13) ||
                                 !insn.Bits(91, 1))) ||
            (opcode == 0x36a && (source_b >= 255U || insn.Bits(40, 19) || insn.Bits(91, 1))) ||
            (!bindless && (insn.Bits(40, 14) > 0x1fffU || binding >= 18U))) {
            return fail("native texture LOD query mode");
        }
        if (!bindless && !bind_texture_buffer(binding)) {
            return fail("native texture LOD query mode");
        }
        return true;
    }
    case 0xb6c: case 0xf6c: { // TXD
        const bool bindless = opcode == 0xf6c;
        const u32 binding = insn.Bits(54, 5);
        const u32 offset = insn.Bits(40, 14);
        const u32 coordinates = insn.Bits(24, 8);
        const u32 derivatives = insn.Bits(32, 8);
        const u32 dimension = insn.Bits(61, 3);
        const u32 dest = insn.Bits(16, 8);
        const u32 high = insn.Bits(64, 8);
        const u32 mask = insn.Bits(72, 4);
        const u32 channels = static_cast<u32>(std::popcount(mask));
        if ((insn.words[2] & ~(0xfffU | (1U << 26))) != (bindless ? 0x081e0000U : 0x001e0000U) ||
            mask == 0U || insn.Bits(59, 2) || (dimension != 1U && dimension != 5U) ||
            (bindless ? (insn.Bits(40, 6) >= 63U || insn.Bits(46, 13))
                      : (binding >= 18U || offset > 0x1fffU)) ||
            dest >= 255U || (channels <= 2U && high != 255U) ||
            coordinates + (dimension == 5U ? 3U : 2U) > 255U || derivatives + 4U > 255U ||
            (high != 255U && high + (channels > 2U ? channels - 2U : 0U) > 255U)) {
            return fail("native gradient texture mode");
        }
        if (!bindless && !bind_texture_buffer(binding)) {
            return fail("native texture descriptor buffer conflict");
        }
        return true;
    }
    case 0xb6f: case 0x370: { // TXQ
        const bool bindless = opcode == 0x370;
        const u32 binding = insn.Bits(54, 5);
        const u32 mask = insn.Bits(72, 4);
        const u32 components = static_cast<u32>(std::popcount(mask));
        const u32 dest = insn.Bits(16, 8);
        const u32 second_dest = insn.Bits(64, 8);
        const u32 ra = insn.Bits(24, 8);
        const bool direct = second_dest == 255U;
        if (insn.Bits(59, 5) != (bindless ? 1U : 0U) || mask == 0U || (direct && components > 2U) ||
            insn.Bits(76, 14) || insn.Bits(91, 1) || (!bindless && binding >= 18U) ||
            (bindless && (insn.Bits(32, 27) || ra >= 254U)) ||
            (!direct && (second_dest >= 255U || dest >= 255U)) ||
            insn.Bits(40, 14) > 0x1fffU || (direct && components == 2U && (dest & 1U)) ||
            dest >= 254U) {
            return fail("native texture dimension query mode");
        }
        if (!bindless && !bind_texture_buffer(binding)) {
            return fail("native texture descriptor buffer conflict");
        }
        return true;
    }
    case 0xb97: { // SULD
        const u32 binding = insn.Bits(54, 5);
        const u32 offset = insn.Bits(40, 14);
        const u32 coordinate = insn.Bits(24, 8);
        const u32 channels = insn.Bits(72, 4);
        const u32 dimension = insn.Bits(61, 3);
        const u32 dest = insn.Bits(16, 8);
        if ((insn.words[2] & ~0xf00U) != 0x001e0000U || channels == 0U ||
            (dimension != 0U && dimension < 3U) || dimension > 5U || binding >= 18U ||
            offset > 0x1fffU || insn.Bits(59, 2) || dest == 255U ||
            coordinate > (dimension >= 4U ? 252U : dimension == 0U ? 254U : 253U)) {
            return fail("native formatted surface load mode");
        }
        if (!bind_texture_buffer(binding)) {
            return fail("native surface descriptor buffer conflict");
        }
        return true;
    }
    case 0xb9b: { // SUST
        const u32 binding = insn.Bits(54, 5);
        const u32 offset = insn.Bits(40, 14);
        const u32 data = insn.Bits(32, 8);
        const u32 coordinate = insn.Bits(24, 8);
        const u32 dimension = insn.Bits(61, 3);
        const u32 channels = insn.Bits(72, 4);
        const u32 mode = insn.words[2] & ~0xf00U;
        const bool strong = mode == 0x0010a000U;
        if ((mode != 0x00100000U && !strong) || channels == 0U || dimension < 3U ||
            dimension > 5U || binding >= 18U || offset > 0x1fffU || insn.Bits(59, 2) ||
            (data != 255U && data + static_cast<u32>(std::popcount(channels)) > 255U) ||
            coordinate > (dimension >= 4U ? 252U : 253U)) {
            return fail("native formatted surface store mode");
        }
        (void)bind_texture_buffer(binding);
        return true;
    }
    case 0x593: { // SUATOM
        const u32 binding = insn.Bits(54, 5);
        const u32 offset = insn.Bits(40, 14);
        const u32 coordinate = insn.Bits(24, 8);
        const u32 dimension = insn.Bits(61, 3);
        const u32 operation = insn.Bits(87, 3);
        if ((insn.words[2] & ~0x03800000U) != 0x001ee000U ||
            (operation != 0U && operation != 6U) || (dimension != 3U && dimension != 5U) ||
            binding >= 18U || offset > 0x1fffU || insn.Bits(59, 2) ||
            coordinate > (dimension == 5U ? 252U : 253U)) {
            return fail("native surface reduction mode");
        }
        if (!bind_texture_buffer(binding)) {
            return fail("native surface descriptor buffer conflict");
        }
        return true;
    }
    case 0x59f: { // SURED
        const u32 coordinate = insn.Bits(24, 8);
        const u32 offset = insn.Bits(40, 14);
        const u32 binding = insn.Bits(54, 5);
        const u32 dimension = insn.Bits(61, 3);
        const u32 type = insn.Bits(73, 3);
        const u32 operation = insn.Bits(87, 4);
        const u32 coordinates = dimension <= 1U ? 1U : dimension <= 3U ? 2U : 3U;
        if ((insn.words[2] & ~((7U << 9) | (15U << 23))) != 0x0010e000U || insn.Bits(59, 2) ||
            dimension > 5U || type > 1U || operation > 7U || operation == 3U ||
            operation == 4U || binding >= 18U || offset > 0x1fffU ||
            (coordinate == 255U ? coordinates > 1U : coordinate + coordinates > 255U)) {
            return fail("native surface reduction mode");
        }
        if (!bind_texture_buffer(binding)) {
            return fail("native surface descriptor buffer conflict");
        }
        return true;
    }
    case 0x589: // SHFL
        if ((insn.words[2] & ~0x000e0000U) || insn.Bits(53, 5) || insn.Bits(60, 4)) {
            return fail("native shuffle mode");
        }
        return true;
    case 0x989:
        if ((insn.words[1] & 0xf01fffffU) || (insn.words[2] & ~0x000e00ffU)) {
            return fail("native register-mask shuffle mode");
        }
        return true;
    case 0xf89:
        if (insn.words[2] != 0x000e0000U || insn.Bits(32, 8) || insn.Bits(60, 4)) {
            return fail("native immediate shuffle mode");
        }
        return true;
    case 0x3a1: { // MATCH.ALL
        const u32 source = insn.Bits(24, 8);
        const u32 target = insn.Bits(81, 3);
        const u32 words = insn.Bits(73, 1) ? 2U : 1U;
        if (insn.Bits(16, 8) != 255U || source >= 254U || (words == 2U && (source & 1U)) ||
            target == 7U || insn.Bits(12, 4) != 7U || insn.words[1] ||
            (insn.words[2] & ~0xe0200U) != 0U) {
            return fail("native warp match mode");
        }
        return true;
    }
    case 0x23a: // MOVM.16.MT88
        if (insn.words[1] || insn.words[2] || insn.Bits(24, 8) >= 255U ||
            insn.Bits(16, 8) >= 255U) {
            return fail("native matrix move mode");
        }
        return true;
    case 0x23c: { // HMMA.1688/.16816.F16
        if (!ResolveTensorMode(options)) {
            return fail("native matrix multiply (set EDEN_NX2_TENSOR to emulate)");
        }
        const u32 dest = insn.Bits(16, 8);
        const u32 a = insn.Bits(24, 8);
        const u32 b = insn.Bits(32, 8);
        const u32 c = insn.Bits(64, 8);
        const u32 shape = insn.words[2] & ~0xffU;
        const u32 depth = shape == 0U ? 8U : 16U;
        const u32 a_count = depth / 4U;
        const u32 b_count = depth / 8U;
        if ((shape != 0U && shape != 0x800U) || (insn.words[1] & ~0xffU) || (dest & 1U) ||
            dest + 2U > 255U || (a & (a_count - 1U)) || a + a_count > 255U ||
            (b & (b_count - 1U)) || b + b_count > 255U ||
            (c != 255U && ((c & 1U) || c + 2U > 255U))) {
            return fail("native matrix multiply mode");
        }
        return true;
    }
    case 0x83b: { // LDSM.16.M88.{1,2,4}
        const u32 count_code = insn.Bits(72, 2);
        const u32 count = 1U << count_code;
        const bool uniform_offset = insn.Bits(91, 1) != 0U;
        const u32 dest = insn.Bits(16, 8);
        if (count_code == 3U || (insn.words[2] & ~0x08004300U) ||
            (!uniform_offset && insn.Bits(32, 8)) || (uniform_offset && insn.Bits(32, 8) > 63U) ||
            dest + count > 255U) {
            return fail(uniform_offset && insn.Bits(32, 8) > 63U
                            ? "native matrix load uniform offset"
                            : "native matrix load mode");
        }
        return true;
    }
    default:
        return false;
    }
}

} // Anonymous namespace

std::optional<u32> FindGatherPtpOffsetImmediate(std::span<const Instruction> code, u32 index,
                                                u32 reg, u32 guard_pred,
                                                const std::vector<u8>* branch_targets,
                                                std::string_view& error) noexcept {
    error = {};
    if (reg >= 255U || index > code.size()) {
        return std::nullopt;
    }
    const auto is_branch_target = [&](u32 cursor) {
        return branch_targets && cursor < branch_targets->size() && (*branch_targets)[cursor] != 0;
    };
    for (u32 cursor = index; cursor > 0U;) {
        if (is_branch_target(cursor)) {
            break;
        }
        const Instruction& def = code[--cursor];
        if (def.Bits(12, 4) == 15U) {
            continue;
        }
        const u32 op = def.Bits(0, 12);
        const u32 target = def.Bits(16, 8);
        if (op == 0x94d || op == 0x950 || op == 0x944 || op == 0x943) {
            break;
        }
        if (op == 0x947 || op == 0x945 || op == 0x918 || op == 0x91a || (op & 0x1ffU) == 0x00cU ||
            (op & 0x1ffU) == 0x00bU) {
            continue;
        }
        if (op == 0xb63) {
            const u32 count = static_cast<u32>(std::popcount(def.Bits(72, 4)));
            const u32 high = def.Bits(64, 8);
            const bool packed = def.Bits(79, 1) != 0U;
            const u32 first_count = packed ? (high == 255U ? (count + 1U) / 2U : 1U)
                                           : (high == 255U ? count : (std::min)(count, 2U));
            const u32 second_count =
                packed ? (count > 2U ? 1U : 0U) : (count > 2U ? count - 2U : 0U);
            if ((target != 255U && reg >= target && reg < target + first_count) ||
                (high != 255U && reg >= high && reg < high + second_count)) {
                error = "native per-tap gather offset clobber";
                return std::nullopt;
            }
            continue;
        }
        if (op == 0xb9b || op == 0x59f || op == 0x386 || op == 0x986 || op == 0x388 ||
            op == 0x988 || op == 0x992 || op == 0xb1d || op == 0x948 || op == 0x348 ||
            op == 0x941 || op == 0x942 || op == 0x95b || op == 0xf55 || op == 0xf56) {
            continue;
        }
        const u32 count = (op == 0x225 || op == 0x625 || op == 0x825 || op == 0x805) ? 2U
                          : (op == 0x305 && def.Bits(75, 2) == 3U)                  ? 2U
                                                                                    : 1U;
        if (target == 255U || reg < target || reg >= target + count) {
            continue;
        }
        const bool move = op == 0x802 && !def.Bits(24, 8) && def.words[2] == 0xf00U;
        const bool imad =
            op == 0x424 && def.Bits(24, 8) == 255U && def.words[2] == 0x078e00ffU;
        if ((!move && !imad) || (def.Bits(12, 4) != 7U && def.Bits(12, 4) != guard_pred) ||
            (def.words[3] & 0xc00001ffU)) {
            error = "native per-tap gather nonconstant offsets";
            return std::nullopt;
        }
        for (u32 shift = 0U; shift < 32U; shift += 8U) {
            const u32 value = (def.words[1] >> shift) & 255U;
            if (value > 31U && value < 224U) {
                error = "native per-tap gather offset range";
                return std::nullopt;
            }
        }
        return def.words[1];
    }
    return std::nullopt;
}

bool UsesUniformGuardPredicate(u32 opcode) noexcept {
    switch (opcode) {
    case 0x287: case 0x887: // USEL
    case 0x28c: case 0x88c: // UISETP
    case 0x290: case 0x890: // UIADD3
    case 0x291: case 0x891: // ULEA
    case 0x292: case 0x892: // ULOP3
    case 0x299: case 0x499: case 0x899: // USHF
    case 0x2a4: case 0x4a4: case 0x8a4: // UIMAD
    case 0x2a5: case 0x8a5: // UIMAD.WIDE
    case 0x2bd: case 0x2be: case 0x2bf: // UFLO, UBREV, UPOPC
    case 0x882: case 0xc82: // UMOV
    case 0x883: // UP2UR
    case 0x886: // VOTEU
    case 0x896: // UPRMT
    case 0x89c: // UPLOP3
    case 0xab9: case 0xabb: // ULDC
    case 0x9c3: // S2UR
    case 0x3c4: // REDUX
    case 0x958: // BRXU
        return true;
    default:
        return false;
    }
}

IR::Condition GuardCondition(u32 opcode, u32 pred_bits) noexcept {
    if (pred_bits == 7U) {
        return IR::Condition{true};
    }
    if (pred_bits == 15U) {
        return IR::Condition{false};
    }
    const bool negated = (pred_bits >> 3) != 0;
    const IR::Pred pred =
        UsesUniformGuardPredicate(opcode) ? UniformPred(pred_bits) : ThreadPred(pred_bits);
    return IR::Condition{pred, negated};
}

bool IsDirectTier1Opcode(u32 opcode) noexcept {
    switch (opcode) {
    // Moves, constant loads, special registers, no-ops
    case 0x202: case 0x802: case 0xa02: case 0xc02:
    case 0x882: case 0xc82:
    case 0xb82: case 0xab9: case 0xabb:
    case 0x919: case 0x9c3: case 0x805: case 0x34e:
    case 0x918:
    case 0x945: case 0x941: case 0x942: case 0x948: case 0x348: case 0xf55: case 0xf56:
    case 0x357: case 0x957:
    // Integer add / LEA / abs / min-max
    case 0x210: case 0x810: case 0xa10: case 0xc10: case 0x290: case 0x890:
    case 0x211: case 0x811: case 0xa11: case 0xc11: case 0x291: case 0x891:
    case 0x213: case 0xa13: case 0xc13:
    case 0x217: case 0x817: case 0xa17: case 0xc17:
    // Integer multiply
    case 0x224: case 0x424: case 0x624: case 0x824: case 0xa24: case 0xc24: case 0xe24:
    case 0x2a4: case 0x4a4: case 0x8a4:
    case 0x225: case 0x227: case 0x625: case 0x825: case 0x827: case 0xa27:
    case 0x2a5: case 0x8a5:
    case 0x226:
    // Logic & bit operations
    case 0x212: case 0x812: case 0xa12: case 0xc12: case 0x292: case 0x892:
    case 0x219: case 0x419: case 0x819: case 0xa19: case 0xc19:
    case 0x299: case 0x499: case 0x899:
    case 0x81a:
    case 0x816: case 0x896:
    case 0x300: case 0xd00: case 0x2bd:
    case 0x309: case 0xd09: case 0x2bf:
    case 0x301: case 0x2be:
    // Predicate & select
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: case 0x28c: case 0x88c:
    case 0x207: case 0x807: case 0xa07: case 0xc07: case 0x287: case 0x887:
    case 0x81c: case 0x89c:
    case 0x803: case 0x804: case 0x883:
    case 0x80d:
    // Warp & uniform
    case 0x806: case 0x886: case 0x3c2: case 0x3c4:
        return true;
    default:
        return false;
    }
}

bool IsDirectTier2Opcode(u32 opcode) noexcept {
    switch (opcode) {
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23: // FFMA
    case 0x220: case 0x820: case 0xa20: case 0xc20:                                     // FMUL
    case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21:             // FADD
    case 0x209: case 0x809: case 0xa09: case 0xc09:                                     // FMNMX
    case 0x308: case 0xb08: case 0xd08:                                                 // MUFU
    case 0x326: case 0x526:                                                             // IPA
    case 0x208: case 0x808: case 0xa08:                                                 // FSEL
    case 0x20a: case 0x80a: case 0xa0a: case 0xc0a:                                     // FSET
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b:                                     // FSETP
    // FP16 ALU, scalar/packed conversions, and swizzled FP add
    case 0x304:                                                                         // F2F
    case 0x23e: case 0xa3e:                                                             // F2FP
    case 0x232: case 0x832: case 0xa32:                                                 // HMUL2
    case 0x230: case 0x430: case 0x630:                                                 // HADD2
    case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31:                         // HFMA2
    case 0x233: case 0x433: case 0x633:                                                 // HSET2
    case 0x234: case 0x434: case 0x634:                                                 // HSETP2
    case 0x240: case 0x840:                                                             // HMNMX2
    case 0x822:                                                                         // FSWZADD
    case 0x311:                                                                         // F2I.64
    case 0x312:                                                                         // I2F.64
    case 0x243:                                                                         // F2IP
    case 0x305: case 0xb05: case 0xd05:                                                 // F2I
    case 0x307: case 0xb07:                                                             // FRND
    case 0x306: case 0xd06: case 0x906:                                                 // I2F
    case 0x245: case 0x845: case 0xa45: case 0xc45:                                     // I2FP
    // Memory, atomics, barriers, attributes, geometry OUT, stage buffers
    case 0x381: case 0x981:                                                             // LDG
    case 0x386: case 0x986:                                                             // STG
    case 0x3a8: case 0x9a8:                                                             // ATOMG
    case 0x98e:                                                                         // RED
    case 0x388: case 0x984: case 0x988:                                                 // LDS / STS
    case 0x38c: case 0xf8c:                                                             // ATOMS
    case 0xfae:                                                                         // LDGSTS
    case 0x387: case 0x983:                                                             // LDL / STL
    case 0x91a: case 0x992: case 0x9ab: case 0x9af: case 0xb1d:                         // Barriers
    case 0x321: case 0x322:                                                             // ALD / AST
    case 0x324: case 0x924:                                                             // OUT
    case 0x923: case 0x927:                                                             // ISBE
    // Texture operations (Tier 2)
    case 0xb60: case 0xf60: case 0x361:                                                 // TEX / TEX.B
    case 0xb63: case 0xf63:                                                             // TLD4
    case 0xb66: case 0xf66: case 0x367:                                                 // TLD / TLD.B
    case 0xb69: case 0xf69: case 0x36a:                                                 // TMML / TMML.B
    case 0xb6c: case 0xf6c:                                                             // TXD
    case 0xb6f: case 0x370:                                                             // TXQ / TXQ.B
    // Surface, warp, and tensor operations (Tier 2)
    case 0xb97: case 0xb9b: case 0x593: case 0x59f:                                     // SULD / SUST / SUATOM / SURED
    case 0x589: case 0x989: case 0xf89:                                                 // SHFL
    case 0x3a1:                                                                         // MATCH.ALL
    case 0x23a: case 0x23c: case 0x83b:                                                 // MOVM / HMMA / LDSM
        return true;
    default:
        return false;
    }
}

bool CanIfConvertPredicated(const Instruction& insn) noexcept {
    const u32 opcode = insn.Bits(0, 12);
    switch (opcode) {
    // Moves, static constant loads, special registers
    case 0x202: case 0x802: case 0xa02: case 0xc02:
    case 0x882: case 0xc82:
    case 0xab9:
    case 0x919: case 0x9c3: case 0x805:
        return true;
    case 0xb82:
        return insn.Bits(24, 8) == 255U && insn.Bits(78, 2) == 0U;
    case 0xabb:
        return insn.Bits(24, 8) == 63U;
    // Integer add / LEA / abs / min-max / multiply / dot
    case 0x210: case 0x810: case 0xa10: case 0xc10: case 0x290: case 0x890:
    case 0x211: case 0x811: case 0xa11: case 0xc11: case 0x291: case 0x891:
    case 0x213: case 0xa13: case 0xc13:
    case 0x217: case 0x817: case 0xa17: case 0xc17:
    case 0x224: case 0x424: case 0x624: case 0x824: case 0xa24: case 0xc24: case 0xe24:
    case 0x2a4: case 0x4a4: case 0x8a4:
    case 0x225: case 0x227: case 0x625: case 0x825: case 0x827: case 0xa27:
    case 0x2a5: case 0x8a5:
    case 0x226:
    // Logic & bit operations
    case 0x212: case 0x812: case 0xa12: case 0xc12: case 0x292: case 0x892:
    case 0x219: case 0x419: case 0x819: case 0xa19: case 0xc19:
    case 0x299: case 0x499: case 0x899:
    case 0x81a:
    case 0x816: case 0x896:
    case 0x300: case 0xd00: case 0x2bd:
    case 0x309: case 0xd09: case 0x2bf:
    case 0x301: case 0x2be:
    // Predicate & select
    case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: case 0x28c: case 0x88c:
    case 0x207: case 0x807: case 0xa07: case 0xc07: case 0x287: case 0x887:
    case 0x81c: case 0x89c:
    case 0x803: case 0x804: case 0x883:
    case 0x80d:
    // Tier 2 FP32 ALU, FP16 ALU, and conversions (excludes quad-coupled 0x822 FSWZADD)
    case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23:
    case 0x220: case 0x820: case 0xa20: case 0xc20:
    case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21:
    case 0x209: case 0x809: case 0xa09: case 0xc09:
    case 0x308: case 0xb08: case 0xd08:
    case 0x208: case 0x808: case 0xa08:
    case 0x20a: case 0x80a: case 0xa0a: case 0xc0a:
    case 0x20b: case 0x80b: case 0xa0b: case 0xc0b:
    case 0x304:
    case 0x23e: case 0xa3e:
    case 0x232: case 0x832: case 0xa32:
    case 0x230: case 0x430: case 0x630:
    case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31:
    case 0x233: case 0x433: case 0x633:
    case 0x234: case 0x434: case 0x634:
    case 0x240: case 0x840:
    case 0x311:
    case 0x312:
    case 0x243:
    case 0x305: case 0xb05: case 0xd05:
    case 0x307: case 0xb07:
    case 0x306: case 0xd06: case 0x906:
    case 0x245: case 0x845: case 0xa45: case 0xc45:
        return true;
    default:
        return false;
    }
}

ProgramAnalysis AnalyzeProgram(std::span<const Instruction> code, const LoweringOptions& options) {
    ProgramAnalysis analysis;
    if (code.empty() || code.size() > 65536) {
        analysis.error_offset = 0;
        analysis.error = "invalid native program size";
        return analysis;
    }
    analysis.uses_async_copy = std::any_of(code.begin(), code.end(), [](const Instruction& insn) {
        return insn.Bits(0, 12) == 0xfae;
    });
    analysis.branch_targets.assign(code.size(), 0);

    struct PendingReloc {
        s64 target{};
        bool call{};
    };
    std::vector<PendingReloc> relocations;
    std::vector<u32> call_targets;
    std::vector<u32> call_links;
    bool main_exited = false;
    bool computed_branch = false;

    const auto fail = [&](u32 index, std::string_view reason) {
        analysis.error_offset = index * static_cast<u32>(sizeof(Instruction));
        analysis.error = reason;
        analysis.source_size = 0;
        return analysis;
    };
    const auto mark_target = [&](s64 target) {
        if (target >= 0 && static_cast<size_t>(target) < analysis.branch_targets.size()) {
            analysis.branch_targets[static_cast<size_t>(target)] = 1;
        }
    };
    const auto finish = [&](u32 index) {
        for (const auto& relocation : relocations) {
            if (relocation.target > index) {
                return fail(index, "native branch or call leaves captured program");
            }
        }
        analysis.source_size = (index + 1) * static_cast<u32>(sizeof(Instruction));
        analysis.instruction_count = index + 1;
        return analysis;
    };

    for (u32 index = 0; index < code.size(); ++index) {
        const Instruction& insn = code[index];
        const u32 opcode = insn.Bits(0, 12);
        const u32 native_predicate = insn.Bits(12, 4);
        if (native_predicate == 15) {
            continue; // @!PT: compiler padding
        }
        std::optional<u32> active_link;
        if (main_exited) {
            u32 active_entry = 0;
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] <= index && call_targets[call] >= active_entry) {
                    active_entry = call_targets[call];
                    active_link = call_links[call];
                }
            }
            const bool no_destination = opcode == 0x950 || opcode == 0x947 || opcode == 0x918 ||
                                        opcode == 0x95b || opcode == 0x95c ||
                                        (opcode & 0x1ff) == 0x00c || (opcode & 0x1ff) == 0x00b;
            const bool pair = opcode == 0x225 || opcode == 0x625 || opcode == 0x825;
            if (active_link && !no_destination &&
                (insn.Bits(16, 8) == *active_link ||
                 (pair && insn.Bits(16, 8) + 1 == *active_link))) {
                return fail(index, "native callee overwrites link register");
            }
        }

        switch (opcode) {
        case 0x949: case 0x958: { // BRX / BRXU
            if (native_predicate != 7 ||
                (insn.words[2] & ~0x3ffffU) != (opcode == 0x958 ? 0x0b800000U : 0x03800000U)) {
                return fail(index, "native computed branch mode");
            }
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = SignExtend50(raw);
            if (s64{index} * 16 + 16 + relative != 0) {
                return fail(index, "native computed branch relative base");
            }
            if (opcode == 0x958 && insn.Bits(24, 8) >= 63) {
                return fail(index, "native uniform computed branch source");
            }
            computed_branch = true;
            break;
        }
        case 0x944: { // CALL.REL.NOINC
            if (native_predicate != 7 && insn.words[1] == 16 && insn.words[2] == 0x03c00000 &&
                index + 2 < code.size() && code[index + 1].Bits(0, 12) == 0x947) {
                relocations.push_back({static_cast<s64>(index + 2), false});
                mark_target(static_cast<s64>(index + 2));
                break;
            }
            const std::optional<u32> link = FindCallLinkRegister(code, index);
            if (main_exited || insn.words[0] != 0x7944 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000 || !link) {
                return fail(index, "native call linkage mode");
            }
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = SignExtend50(raw);
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) {
                return fail(index, "native call destination alignment");
            }
            const u32 target_idx = static_cast<u32>(target / 16);
            for (size_t call = 0; call < call_targets.size(); ++call) {
                if (call_targets[call] == target_idx && call_links[call] != *link) {
                    return fail(index, "native inconsistent call linkage");
                }
            }
            call_targets.push_back(target_idx);
            call_links.push_back(*link);
            relocations.push_back({static_cast<s64>(target_idx), true});
            mark_target(target_idx);
            break;
        }
        case 0x950: { // RET.REL.NODEC
            const u32 return_reg = insn.Bits(24, 8);
            u32 active_entry = 0;
            if (main_exited) {
                for (size_t call = 0; call < call_targets.size(); ++call) {
                    if (call_targets[call] <= index && call_targets[call] >= active_entry) {
                        active_entry = call_targets[call];
                    }
                }
            }
            const bool linked =
                main_exited && VerifyReturnLink(code, index, active_entry, active_link, return_reg);
            if (!linked || (insn.words[0] & 0x00ffffffU) != 0x00007950 || return_reg >= 249 ||
                (insn.words[2] & ~0x3ffffU) != 0x03c00000) {
                return fail(index, "native return linkage mode");
            }
            const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
            if (s64{index} * 16 + 16 + SignExtend50(raw) != 0) {
                return fail(index, "native return relative base");
            }
            bool complete = !computed_branch;
            for (u32 target : call_targets) {
                complete &= target <= index;
            }
            if (complete) {
                return finish(index);
            }
            break;
        }
        case 0x947: { // BRA
            const u32 mode = insn.Bits(32, 2);
            if (mode == 0 && insn.Bits(91, 1)) {
                return fail(index, "native branch convergence mode");
            }
            if (mode == 2) {
                break; // .DIV is a no-op
            }
            if (insn.Bits(87, 4) != 7 && native_predicate != 7) {
                mark_target(static_cast<s64>(index + 1));
            }
            const u64 raw = (insn.words[1] & ~3U) | (u64{insn.words[2] & 0x3ffff} << 32);
            const s64 relative = SignExtend50(raw);
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16) {
                return fail(index, "native branch destination alignment");
            }
            const s64 target_idx = target / 16;
            relocations.push_back({target_idx, false});
            mark_target(target_idx);
            const auto unconditional_jump = [&](const Instruction& before) {
                const u32 op = before.Bits(0, 12);
                return before.Bits(12, 4) == 7 &&
                       ((op == 0x947 && before.Bits(87, 4) == 7) || op == 0x94d);
            };
            const bool after_jump =
                index > 0 && call_targets.empty() && unconditional_jump(code[index - 1]);
            const u32 effective_pred =
                insn.Bits(87, 4) != 7 ? insn.Bits(87, 4) : native_predicate;
            if ((computed_branch || after_jump) && target == s64{index} * 16 &&
                effective_pred == 7) {
                return finish(index);
            }
            break;
        }
        case 0x94d: { // EXIT
            if (insn.Bits(84, 2)) {
                return fail(index, "native conditional/special EXIT");
            }
            if (insn.Bits(87, 4) != 7) {
                if (native_predicate != 7) {
                    relocations.push_back({static_cast<s64>(index + 1), false});
                    mark_target(static_cast<s64>(index + 1));
                }
                break;
            }
            if (native_predicate != 7) {
                break;
            }
            if (call_targets.empty()) {
                const bool pending_branch = std::any_of(relocations.begin(), relocations.end(),
                    [&](const PendingReloc& r) { return r.target > index; });
                if (!pending_branch && !computed_branch) {
                    return finish(index);
                }
                break;
            }
            const bool pending_main_branch = std::any_of(relocations.begin(), relocations.end(),
                [&](const PendingReloc& r) { return !r.call && r.target > index; });
            if (!main_exited && pending_main_branch) {
                break;
            }
            if (main_exited) {
                bool complete = std::none_of(relocations.begin(), relocations.end(),
                    [&](const PendingReloc& r) { return r.target > index; });
                for (u32 target : call_targets) {
                    complete &= target <= index;
                }
                if (complete && !computed_branch) {
                    return finish(index);
                }
                break;
            }
            for (u32 target : call_targets) {
                if (target <= index) {
                    return fail(index, "native call target overlaps main program");
                }
            }
            const bool pending_branch = std::any_of(relocations.begin(), relocations.end(),
                [&](const PendingReloc& r) {
                    return r.target > index &&
                           std::find(call_targets.begin(), call_targets.end(),
                                     static_cast<u32>(r.target)) == call_targets.end();
                });
            if (!pending_branch) {
                main_exited = true;
            }
            break;
        }
        case 0x95b: // KIL: fragment discard; execution predicate is preserved.
            if (insn.words[1] || insn.words[2] != 0x03800000U) {
                return fail(index, "native fragment discard mode");
            }
            break;
        case 0x95c: // TRAP: aborts the invocation as EXIT does.
            if ((insn.words[0] & ~0xf000U) != 0x095cU || insn.words[1] != 4U ||
                insn.words[2] != 0x00300000U) {
                return fail(index, "native trap mode");
            }
            break;
        default: {
            if (IsDirectTier1Opcode(opcode)) {
                std::string_view tier1_error;
                if (!ValidateTier1Instruction(insn, options, tier1_error)) {
                    return fail(index, tier1_error);
                }
                break;
            }
            if (IsDirectTier2Opcode(opcode)) {
                std::string_view tier2_error;
                if (!ValidateTier2Instruction(code, index, insn, options,
                                              analysis.required_texture_buffer,
                                              analysis.branch_targets, tier2_error)) {
                    return fail(index, tier2_error);
                }
                if (opcode == 0x321U && insn.Bits(16, 8) != 255U) {
                    const u32 address = insn.Bits(40, 10);
                    const u32 components = insn.Bits(74, 2) + 1U;
                    const u32 base_attr = address / 4U;
                    for (u32 c = 0; c < components; ++c) {
                        const u32 attr_idx = base_attr + c;
                        if (attr_idx >= static_cast<u32>(IR::Attribute::Generic0X) &&
                            attr_idx <= static_cast<u32>(IR::Attribute::Generic31W)) {
                            analysis.generic_attributes_read |=
                                1u << ((attr_idx - static_cast<u32>(IR::Attribute::Generic0X)) / 4U);
                        }
                    }
                }
                break;
            }
            return fail(index, "unsupported native opcode");
        }
        }
    }
    return fail(static_cast<u32>(code.size()),
                main_exited ? "native program lacks captured library return"
                            : "native program has no unconditional EXIT");
}

namespace Flow {

CFG::CFG(Environment& env_, ObjectPool<Block>& block_pool_, std::span<const Instruction> code_,
         bool exits_to_dispatcher_)
    : env{env_}, block_pool{block_pool_}, code{code_},
      exits_to_dispatcher{exits_to_dispatcher_} {
    if (exits_to_dispatcher) {
        dispatch_block = block_pool.Create(Block{});
        dispatch_block->begin = {};
        dispatch_block->end = {};
        dispatch_block->end_class = EndClass::Exit;
        dispatch_block->cond = IR::Condition{true};
        dispatch_block->stack = {};
        dispatch_block->branch_true = nullptr;
        dispatch_block->branch_false = nullptr;
    }
    functions.emplace_back(block_pool, LocationOf(0));
    for (FunctionId function_id = 0; function_id < functions.size(); ++function_id) {
        while (!functions[function_id].labels.empty()) {
            Function& function{functions[function_id]};
            Label label{function.labels.back()};
            function.labels.pop_back();
            AnalyzeLabel(function_id, label);
        }
    }
    if (exits_to_dispatcher) {
        const auto last_block{functions[0].blocks.rbegin()};
        dispatch_block->begin = NextLocation(last_block->end);
        dispatch_block->end = dispatch_block->begin;
        functions[0].blocks.insert(*dispatch_block);
    }
}

void CFG::AnalyzeLabel(FunctionId function_id, Label& label) {
    if (InspectVisitedBlocks(function_id, label)) {
        return;
    }
    Function* const function{&functions[function_id]};
    Maxwell::Location pc{label.address};
    const auto next_it{function->blocks.upper_bound(pc, Compare{})};
    const bool is_last{next_it == function->blocks.end()};
    Block* const next{is_last ? nullptr : &*next_it};
    Block* const block{label.block};
    bool is_branch{false};
    while (!next || pc < next->begin) {
        is_branch = AnalyzeInst(block, function_id, pc) == AnalysisState::Branch;
        if (is_branch) {
            break;
        }
        pc = NextLocation(pc);
    }
    if (!is_branch) {
        block->end = pc;
        block->cond = IR::Condition{true};
        block->branch_true = next;
        block->branch_false = nullptr;
    }
    functions[function_id].blocks.insert(*block);
}

bool CFG::InspectVisitedBlocks(FunctionId function_id, const Label& label) {
    const Maxwell::Location pc{label.address};
    Function& function{functions[function_id]};
    auto it{function.blocks.upper_bound(pc, Compare{})};
    if (it == function.blocks.begin() || !std::prev(it)->Contains(pc)) {
        return false;
    }
    --it;
    Block* const visited_block{&*it};
    if (visited_block->begin == pc) {
        throw LogicError("Dangling Ampere block at {:#x}", pc.Offset());
    }
    Block* const new_block{label.block};
    Split(visited_block, new_block, pc);
    function.blocks.insert(it, *new_block);
    return true;
}

void CFG::AnalyzeCondInst(Block* block, FunctionId function_id, Maxwell::Location pc,
                          EndClass insn_end_class, IR::Condition cond) {
    if (block->begin != pc) {
        block->end = pc;
        block->cond = IR::Condition{true};
        block->branch_true = AddLabel(block, block->stack, pc, function_id);
        block->branch_false = nullptr;
        return;
    }
    Block* const conditional_block{block_pool.Create()};
    Block virtual_block{};
    virtual_block.begin = block->begin.Virtual();
    virtual_block.end = block->begin.Virtual();
    virtual_block.end_class = EndClass::Branch;
    virtual_block.stack = block->stack;
    virtual_block.cond = cond;
    virtual_block.branch_true = conditional_block;
    virtual_block.branch_false = nullptr;
    *conditional_block = std::move(*block);
    *block = std::move(virtual_block);
    conditional_block->end = NextLocation(pc);
    conditional_block->end_class = insn_end_class;
    Block* const endif_block{
        AddLabel(conditional_block, block->stack, NextLocation(pc), function_id)};
    block->branch_false = endif_block;
    if (insn_end_class == EndClass::Branch || insn_end_class == EndClass::Kill) {
        conditional_block->cond = IR::Condition{true};
        conditional_block->branch_true = endif_block;
        conditional_block->branch_false = nullptr;
    }
    functions[function_id].blocks.insert(*conditional_block);
}

CFG::AnalysisState CFG::AnalyzeInst(Block* block, FunctionId function_id, Maxwell::Location pc) {
    const u32 index = IndexOf(pc);
    if (index >= code.size()) {
        throw RuntimeError("Ampere control flow walked out of bounds at instruction {}", index);
    }
    const Instruction& insn = code[index];
    const u32 opcode = insn.Bits(0, 12);
    const u32 pred_bits = insn.Bits(12, 4);
    if (pred_bits == 15) {
        return AnalysisState::Continue; // @!PT: never executes
    }

    switch (opcode) {
    case 0x947:
        return AnalyzeBRA(block, function_id, pc, index, insn);
    case 0x94d:
        return AnalyzeEXIT(block, function_id, pc, index, insn);
    case 0x95b:
        return AnalyzeKILL(block, function_id, pc, insn);
    case 0x95c:
        return AnalyzeTRAP(block, function_id, pc, insn);
    case 0x944:
        return AnalyzeCALL(block, function_id, pc, index, insn);
    case 0x950:
        return AnalyzeRET(block, pc, index, insn);
    case 0x949:
    case 0x958:
        return AnalyzeBRX(block, function_id, pc, index, insn);
    case 0x3c2:
        // R2UR must run warp-wide so inactive lanes receive the broadcast uniform value.
        return AnalysisState::Continue;
    case 0xfae: {
        const u32 enable = insn.Bits(87, 3) | (insn.Bits(70, 1) << 3);
        if (enable == 7U || enable == 15U) {
            return AnalysisState::Continue;
        }
        if (block->begin != pc) {
            block->end = pc;
            block->cond = IR::Condition{true};
            block->branch_true = AddLabel(block, block->stack, pc, function_id);
            block->branch_false = nullptr;
            return AnalysisState::Branch;
        }
        Block* const true_block{block_pool.Create()};
        Block* const false_block{block_pool.Create()};
        Block* const endif_block{AddLabel(block, block->stack, NextLocation(pc), function_id)};
        const Maxwell::Location mid_pc = Maxwell::Location::Raw(pc.Offset() + 8U);

        *true_block = *block;
        true_block->begin = pc;
        true_block->end = mid_pc;
        true_block->end_class = EndClass::Branch;
        true_block->cond = IR::Condition{true};
        true_block->branch_true = endif_block;
        true_block->branch_false = nullptr;

        *false_block = *block;
        false_block->begin = mid_pc;
        false_block->end = NextLocation(pc);
        false_block->end_class = EndClass::Branch;
        false_block->cond = IR::Condition{true};
        false_block->branch_true = endif_block;
        false_block->branch_false = nullptr;

        block->begin = pc.Virtual();
        block->end = pc.Virtual();
        block->end_class = EndClass::Branch;
        block->cond = GuardCondition(0xfae, enable);
        block->branch_true = true_block;
        block->branch_false = false_block;

        functions[function_id].blocks.insert(*true_block);
        functions[function_id].blocks.insert(*false_block);
        return AnalysisState::Branch;
    }
    default:
        break;
    }

    if (pred_bits == 7 || CanIfConvertPredicated(insn)) {
        return AnalysisState::Continue;
    }
    AnalyzeCondInst(block, function_id, pc, EndClass::Branch, GuardCondition(opcode, pred_bits));
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeBRA(Block* block, FunctionId function_id, Maxwell::Location pc,
                                   u32 index, const Instruction& insn) {
    const u32 mode = insn.Bits(32, 2);
    if (mode == 0 && insn.Bits(91, 1)) {
        throw RuntimeError("Unsupported native branch convergence mode");
    }
    if (mode == 2) {
        return AnalysisState::Continue; // .DIV never branches in converged execution
    }
    const u32 pg = insn.Bits(12, 4);
    const u32 pcond = insn.Bits(87, 4);
    if (pg == 15 || pcond == 15) {
        return AnalysisState::Continue;
    }
    const u64 raw = (insn.words[1] & ~3U) | (u64{insn.words[2] & 0x3ffff} << 32);
    const s64 byte_target = s64{index} * 16 + 16 + SignExtend50(raw);
    if (byte_target < 0 || (byte_target % 16) != 0 ||
        static_cast<size_t>(byte_target / 16) >= code.size()) {
        throw RuntimeError("Invalid native BRA destination {}", byte_target);
    }
    const Maxwell::Location target_pc = LocationOf(static_cast<u32>(byte_target / 16));

    if (pg != 7 && pcond != 7) {
        // Dual predicate (@Pg BRA Pc): split to pc first, then guard inner branch check.
        if (block->begin != pc) {
            block->end = pc;
            block->cond = IR::Condition{true};
            block->branch_true = AddLabel(block, block->stack, pc, function_id);
            block->branch_false = nullptr;
            return AnalysisState::Branch;
        }
        Block* const inner = block_pool.Create();
        inner->begin = pc;
        inner->end = pc;
        inner->end_class = EndClass::Branch;
        inner->stack = block->stack;
        inner->cond = GuardCondition(0x947, pcond);
        functions[function_id].blocks.insert(*inner);

        Block virtual_block{};
        virtual_block.begin = pc.Virtual();
        virtual_block.end = pc.Virtual();
        virtual_block.end_class = EndClass::Branch;
        virtual_block.stack = block->stack;
        virtual_block.cond = GuardCondition(0x947, pg);
        virtual_block.branch_true = inner;

        Block* const endif_block =
            AddLabel(inner, virtual_block.stack, NextLocation(pc), function_id);
        virtual_block.branch_false = endif_block;
        inner->branch_false = endif_block;
        inner->branch_true = (target_pc == pc)
                                 ? block
                                 : AddLabel(inner, virtual_block.stack, target_pc, function_id);
        *block = std::move(virtual_block);
        return AnalysisState::Branch;
    }

    const u32 effective_pred = pcond != 7 ? pcond : pg;
    const IR::Condition cond = GuardCondition(0x947, effective_pred);
    block->end = NextLocation(pc);
    block->end_class = EndClass::Branch;
    block->cond = cond;
    block->branch_true = AddLabel(block, block->stack, target_pc, function_id);
    block->branch_false =
        cond != IR::Condition{true} ? AddLabel(block, block->stack, NextLocation(pc), function_id)
                                    : nullptr;
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeEXIT(Block* block, FunctionId function_id, Maxwell::Location pc,
                                    u32, const Instruction& insn) {
    if (insn.Bits(84, 2)) {
        throw RuntimeError("Unsupported native conditional/special EXIT");
    }
    const u32 pg = insn.Bits(12, 4);
    const u32 pcond = insn.Bits(87, 4);
    if (pg == 15 || pcond == 15) {
        return AnalysisState::Continue;
    }
    if (exits_to_dispatcher && function_id != 0) {
        throw NotImplementedException("Dispatch EXIT on external function");
    }
    if (pg != 7 && pcond != 7) {
        if (block->begin != pc) {
            block->end = pc;
            block->cond = IR::Condition{true};
            block->branch_true = AddLabel(block, block->stack, pc, function_id);
            block->branch_false = nullptr;
            return AnalysisState::Branch;
        }
        Block* const inner = block_pool.Create();
        inner->begin = pc;
        inner->end = pc;
        inner->end_class = EndClass::Branch;
        inner->stack = block->stack;
        inner->cond = GuardCondition(0x94d, pcond);
        functions[function_id].blocks.insert(*inner);

        Block* exit_dest = dispatch_block;
        if (!exits_to_dispatcher) {
            exit_dest = block_pool.Create();
            exit_dest->begin = Maxwell::Location::Raw(pc.Offset() + 4);
            exit_dest->end = NextLocation(pc);
            exit_dest->end_class = EndClass::Exit;
            exit_dest->cond = IR::Condition{true};
            exit_dest->stack = block->stack;
            functions[function_id].blocks.insert(*exit_dest);
        }

        Block virtual_block{};
        virtual_block.begin = pc.Virtual();
        virtual_block.end = pc.Virtual();
        virtual_block.end_class = EndClass::Branch;
        virtual_block.stack = block->stack;
        virtual_block.cond = GuardCondition(0x94d, pg);
        virtual_block.branch_true = inner;

        Block* const endif_block =
            AddLabel(inner, virtual_block.stack, NextLocation(pc), function_id);
        virtual_block.branch_false = endif_block;
        inner->branch_true = exit_dest;
        inner->branch_false = endif_block;
        *block = std::move(virtual_block);
        return AnalysisState::Branch;
    }

    const u32 effective_pred = pcond != 7 ? pcond : pg;
    const IR::Condition cond = GuardCondition(0x94d, effective_pred);
    if (cond != IR::Condition{true}) {
        if (exits_to_dispatcher) {
            block->end = NextLocation(pc);
            block->end_class = EndClass::Branch;
            block->cond = cond;
            block->branch_true = dispatch_block;
            block->branch_false = AddLabel(block, block->stack, NextLocation(pc), function_id);
            return AnalysisState::Branch;
        }
        AnalyzeCondInst(block, function_id, pc, EndClass::Exit, cond);
        return AnalysisState::Branch;
    }
    if (exits_to_dispatcher) {
        block->cond = IR::Condition{true};
        block->end = NextLocation(pc);
        block->end_class = EndClass::Branch;
        block->branch_true = dispatch_block;
        block->branch_false = nullptr;
        return AnalysisState::Branch;
    }
    block->end = NextLocation(pc);
    block->end_class = EndClass::Exit;
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeKILL(Block* block, FunctionId function_id, Maxwell::Location pc,
                                    const Instruction& insn) {
    if (insn.words[1] || insn.words[2] != 0x03800000U) {
        throw RuntimeError("Invalid native KILL encoding");
    }
    const IR::Condition cond = GuardCondition(0x95b, insn.Bits(12, 4));
    AnalyzeCondInst(block, function_id, pc, EndClass::Kill, cond);
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeTRAP(Block* block, FunctionId function_id, Maxwell::Location pc,
                                    const Instruction& insn) {
    if ((insn.words[0] & ~0xf000U) != 0x095c || insn.words[1] != 4 ||
        insn.words[2] != 0x00300000) {
        throw RuntimeError("Invalid native TRAP encoding");
    }
    const u32 pred_bits = insn.Bits(12, 4);
    if (pred_bits != 7) {
        AnalyzeCondInst(block, function_id, pc, EndClass::Exit, GuardCondition(0x95c, pred_bits));
        return AnalysisState::Branch;
    }
    if (exits_to_dispatcher && function_id == 0) {
        block->cond = IR::Condition{true};
        block->end = pc;
        block->end_class = EndClass::Branch;
        block->branch_true = dispatch_block;
        block->branch_false = nullptr;
        return AnalysisState::Branch;
    }
    block->end = NextLocation(pc);
    block->end_class = EndClass::Exit;
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeCALL(Block* block, FunctionId function_id, Maxwell::Location pc,
                                    u32 index, const Instruction& insn) {
    const u32 pg = insn.Bits(12, 4);
    if (pg != 7 && insn.words[1] == 16 && insn.words[2] == 0x03c00000 &&
        index + 2 < code.size() && code[index + 1].Bits(0, 12) == 0x947) {
        const IR::Condition cond = GuardCondition(0x944, pg);
        block->end = NextLocation(pc);
        block->end_class = EndClass::Branch;
        block->cond = cond;
        block->branch_true = AddLabel(block, block->stack, LocationOf(index + 2), function_id);
        block->branch_false = AddLabel(block, block->stack, NextLocation(pc), function_id);
        return AnalysisState::Branch;
    }
    const std::optional<u32> link = FindCallLinkRegister(code, index);
    if (pg != 7 || insn.words[0] != 0x7944 || (insn.words[2] & ~0x3ffffU) != 0x03c00000 ||
        !link) {
        throw RuntimeError("Invalid native CALL linkage at instruction {}", index);
    }
    const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
    const s64 byte_target = s64{index} * 16 + 16 + SignExtend50(raw);
    if (byte_target < 0 || (byte_target % 16) != 0 ||
        static_cast<size_t>(byte_target / 16) >= code.size()) {
        throw RuntimeError("Invalid native CALL target {}", byte_target);
    }
    const u32 target_idx = static_cast<u32>(byte_target / 16);
    const Maxwell::Location cal_pc = LocationOf(target_idx);
    const auto it = std::ranges::find(functions, cal_pc, &Function::entrypoint);
    const bool exists = it != functions.end();
    const FunctionId call_id =
        exists ? static_cast<FunctionId>(std::distance(functions.begin(), it)) : functions.size();
    if (!exists) {
        functions.emplace_back(block_pool, cal_pc);
        call_sites.push_back({target_idx, *link});
    }
    block->end_class = EndClass::Call;
    block->function_call = call_id;
    block->return_block = AddLabel(block, block->stack, NextLocation(pc), function_id);
    block->end = NextLocation(pc);
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeRET(Block* block, Maxwell::Location pc, u32 index,
                                   const Instruction& insn) {
    if (insn.Bits(12, 4) != 7 || (insn.words[0] & 0x00ffffffU) != 0x00007950 ||
        insn.Bits(24, 8) >= 249 || (insn.words[2] & ~0x3ffffU) != 0x03c00000) {
        throw RuntimeError("Invalid native RET mode at instruction {}", index);
    }
    const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
    if (s64{index} * 16 + 16 + SignExtend50(raw) != 0) {
        throw RuntimeError("Invalid native RET relative base at instruction {}", index);
    }
    block->cond = IR::Condition{true};
    block->end_class = EndClass::Return;
    block->end = NextLocation(pc);
    return AnalysisState::Branch;
}

CFG::AnalysisState CFG::AnalyzeBRX(Block* block, FunctionId function_id, Maxwell::Location pc,
                                   u32 index, const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    if (insn.Bits(12, 4) != 7 ||
        (insn.words[2] & ~0x3ffffU) != (opcode == 0x958 ? 0x0b800000U : 0x03800000U)) {
        throw RuntimeError("Conditional or invalid native BRX at instruction {}", index);
    }
    const u64 raw = insn.words[1] | (u64{insn.words[2] & 0x3ffff} << 32);
    if (s64{index} * 16 + 16 + SignExtend50(raw) != 0) {
        throw RuntimeError("Invalid native BRX relative base at instruction {}", index);
    }
    const auto table = TrackAmpereIndirectBranchTable(code, index);
    if (!table) {
        const u32 lo = index > 20 ? index - 20 : 0;
        for (u32 k = lo; k <= index; ++k) {
            const Instruction& p = code[k];
            LOG_WARNING(Shader,
                        "BRX track miss near idx={}: [{}] op={:#05x} w=({:#010x},{:#010x},{:#010x})",
                        index, k, p.Bits(0, 12), p.words[0], p.words[1], p.words[2]);
        }
        throw NotImplementedException("Failed to track native Ampere indirect branch table");
    }
    std::vector<u32> targets;
    targets.reserve(table->num_entries);
    for (u32 i = 0; i < table->num_entries; ++i) {
        const u32 target = env.ReadCbufValue(table->cbuf_index, table->cbuf_offset + i * 4);
        if ((target % 16) != 0 || static_cast<size_t>(target / 16) >= code.size()) {
            throw RuntimeError("Invalid native indirect branch target {:#x}", target);
        }
        targets.push_back(target);
    }
    std::ranges::sort(targets);
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());

    block->indirect_branches.reserve(targets.size());
    for (const u32 target : targets) {
        Block* const branch = AddLabel(block, block->stack, LocationOf(target / 16), function_id);
        block->indirect_branches.push_back({
            .block = branch,
            .address = target,
        });
    }
    block->cond = IR::Condition{true};
    block->end = NextLocation(pc);
    block->end_class = EndClass::IndirectBranch;
    block->branch_reg = table->branch_reg;
    block->branch_offset = 0;
    return AnalysisState::Branch;
}

Block* CFG::AddLabel(Block* block, Stack stack, Maxwell::Location pc, FunctionId function_id) {
    Function& function{functions[function_id]};
    if (block->begin == pc) {
        return block;
    }
    if (const auto it{function.blocks.find(pc, Compare{})}; it != function.blocks.end()) {
        if (function.blocks.begin() != it) {
            const auto prev{std::prev(it)};
            if (it->begin.Virtual() == prev->begin) {
                return &*prev;
            }
        }
        return &*it;
    }
    const auto label_it{std::ranges::find(function.labels, pc, &Label::address)};
    if (label_it != function.labels.end()) {
        return label_it->block;
    }
    Block* const new_block{block_pool.Create()};
    new_block->begin = pc;
    new_block->end = pc;
    new_block->end_class = EndClass::Branch;
    new_block->cond = IR::Condition{true};
    new_block->stack = stack;
    new_block->branch_true = nullptr;
    new_block->branch_false = nullptr;
    function.labels.push_back(Label{
        .address{pc},
        .block = new_block,
        .stack{std::move(stack)},
    });
    return new_block;
}

} // namespace Flow
} // namespace Shader::Ampere
