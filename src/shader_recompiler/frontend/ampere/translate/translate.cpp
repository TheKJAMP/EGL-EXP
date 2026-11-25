// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"
#include "shader_recompiler/frontend/ampere/translate/translate.h"

namespace Shader::Ampere {

void TranslateBlock(Environment& env, IR::Block* block, std::span<const Instruction> code,
                    u32 begin_offset, u32 end_offset, const LoweringOptions& options,
                    bool uses_async_copy, u32& required_texture_buffer,
                    const std::vector<u8>* branch_targets, bool mkw_zero_depth_fallback) {
    if (begin_offset >= end_offset || begin_offset < AmpereLocationBase) {
        return;
    }
    TranslatorVisitor visitor{env,
                              *block,
                              code,
                              options,
                              uses_async_copy,
                              required_texture_buffer,
                              branch_targets};
    if (((begin_offset - AmpereLocationBase) & 15U) == 8U) {
        const u32 index = (begin_offset - AmpereLocationBase) / static_cast<u32>(sizeof(Instruction));
        if (index < code.size()) {
            visitor.LDGSTS(code[index], false);
        }
        return;
    }
    for (u32 offset = begin_offset; offset < end_offset; offset += sizeof(Instruction)) {
        const u32 index = (offset - AmpereLocationBase) / static_cast<u32>(sizeof(Instruction));
        if (index >= code.size()) {
            break;
        }
        if (mkw_zero_depth_fallback && index == (0x2d0U / sizeof(Instruction))) {
            // Workaround for a missing ground-height map in this exact MKW shader: set empty pair flag.
            const IR::U32 both_zero = visitor.ir.BitwiseOr(visitor.R(8U), visitor.R(9U));
            const IR::U1 is_empty = visitor.ir.IEqual(both_zero, visitor.ir.Imm32(0U));
            const IR::U32 fallback = IR::U32{
                visitor.ir.Select(is_empty, visitor.ir.Imm32(0x3f800000U), visitor.ir.Imm32(0U))};
            visitor.R(4U, visitor.ir.BitwiseOr(visitor.R(4U), fallback));
        }
        const Instruction& insn = code[index];
        const u32 pred_bits = insn.Bits(12, 4);
        if (pred_bits == 15U) {
            continue; // @!PT: never executes
        }
        if (pred_bits != 7U && CanIfConvertPredicated(insn)) {
            visitor.SetActiveGuard(visitor.EvalGuard(insn));
        }
        const u32 opcode = insn.Bits(0, 12);
        switch (opcode) {
        // Control flow handled cleanly by Ampere::Flow::CFG
        case 0x947: case 0x944: case 0x950: case 0x949: case 0x958: case 0x95b:
        // Reconvergence / NOP / geometry exit no-ops
        case 0x918: case 0x945: case 0x941: case 0x942: case 0x948: case 0x348:
        case 0xf55: case 0xf56: case 0x357: case 0x957:
            break;
        case 0x94d: case 0x95c:
            visitor.EXIT();
            break;

        // Moves, constant loads, special registers
        case 0x202: case 0x802: case 0xa02: case 0xc02:
            visitor.MOV(insn);
            break;
        case 0x882: case 0xc82:
            visitor.UMOV(insn);
            break;
        case 0xb82:
            visitor.LDC(insn);
            break;
        case 0xab9: case 0xabb:
            visitor.ULDC(insn);
            break;
        case 0x919:
            visitor.S2R(insn);
            break;
        case 0x9c3:
            visitor.S2UR(insn);
            break;
        case 0x805:
            visitor.CS2R(insn);
            break;
        case 0x34e:
            visitor.LEPC(insn, index);
            break;

        // Integer add, LEA, abs, min/max
        case 0x210: case 0x810: case 0xa10: case 0xc10: case 0x290: case 0x890:
            visitor.IADD3(insn);
            break;
        case 0x211: case 0x811: case 0xa11: case 0xc11: case 0x291: case 0x891:
            visitor.LEA(insn);
            break;
        case 0x213: case 0xa13: case 0xc13:
            visitor.IABS(insn);
            break;
        case 0x217: case 0x817: case 0xa17: case 0xc17:
            visitor.IMNMX(insn);
            break;

        // Integer multiply & dot product
        case 0x224: case 0x424: case 0x624: case 0x824: case 0xa24: case 0xc24: case 0xe24:
        case 0x2a4: case 0x4a4: case 0x8a4:
            visitor.IMAD(insn);
            break;
        case 0x225: case 0x227: case 0x625: case 0x825: case 0x827: case 0xa27:
        case 0x2a5: case 0x8a5:
            visitor.IMAD_WIDE(insn);
            break;
        case 0x226:
            visitor.IDP2A(insn);
            break;

        // Logic & bit manipulation
        case 0x212: case 0x812: case 0xa12: case 0xc12: case 0x292: case 0x892:
            visitor.LOP3(insn);
            break;
        case 0x219: case 0x419: case 0x819: case 0xa19: case 0xc19:
        case 0x299: case 0x499: case 0x899:
            visitor.SHF(insn);
            break;
        case 0x81a:
            visitor.SGXT(insn);
            break;
        case 0x816: case 0x896:
            visitor.PRMT(insn);
            break;
        case 0x300: case 0xd00: case 0x2bd:
            visitor.FLO(insn);
            break;
        case 0x309: case 0xd09: case 0x2bf:
            visitor.POPC(insn);
            break;
        case 0x301: case 0x2be:
            visitor.BREV(insn);
            break;

        // Predicate & select
        case 0x20c: case 0x80c: case 0xa0c: case 0xc0c: case 0x28c: case 0x88c:
            visitor.ISETP(insn);
            break;
        case 0x207: case 0x807: case 0xa07: case 0xc07: case 0x287: case 0x887:
            visitor.SEL(insn);
            break;
        case 0x81c: case 0x89c:
            visitor.PLOP3(insn);
            break;
        case 0x803:
            visitor.P2R(insn);
            break;
        case 0x804:
            visitor.R2P(insn);
            break;
        case 0x883:
            visitor.UP2UR(insn);
            break;
        case 0x80d:
            visitor.CSMTEST(insn);
            break;

        // Warp & uniform
        case 0x806: case 0x886:
            visitor.VOTE(insn);
            break;
        case 0x3c2:
            visitor.R2UR(insn);
            break;
        case 0x3c4:
            visitor.REDUX(insn);
            break;

        // Floating-point ALU & fragment interpolation (Tier 2)
        case 0x223: case 0x423: case 0x623: case 0x823: case 0xa23: case 0xc23: case 0xe23:
            visitor.FFMA(insn);
            break;
        case 0x220: case 0x820: case 0xa20: case 0xc20:
            visitor.FMUL(insn);
            break;
        case 0x221: case 0x421: case 0x621: case 0x821: case 0xa21: case 0xe21:
            visitor.FADD(insn);
            break;
        case 0x209: case 0x809: case 0xa09: case 0xc09:
            visitor.FMNMX(insn);
            break;
        case 0x308: case 0xb08: case 0xd08:
            visitor.MUFU(insn);
            break;
        case 0x326: case 0x526:
            visitor.IPA(insn);
            break;
        case 0x208: case 0x808: case 0xa08:
            visitor.FSEL(insn);
            break;
        case 0x20a: case 0x80a: case 0xa0a: case 0xc0a:
            visitor.FSET(insn);
            break;
        case 0x20b: case 0x80b: case 0xa0b: case 0xc0b:
            visitor.FSETP(insn);
            break;

        // FP16 ALU, scalar/packed conversions, and swizzled FP add
        case 0x304:
            visitor.F2F(insn);
            break;
        case 0x23e: case 0xa3e:
            visitor.F2FP(insn);
            break;
        case 0x232: case 0x832: case 0xa32:
            visitor.HMUL2(insn);
            break;
        case 0x230: case 0x430: case 0x630:
            visitor.HADD2(insn);
            break;
        case 0x231: case 0x431: case 0x631: case 0x831: case 0xa31:
            visitor.HFMA2(insn);
            break;
        case 0x233: case 0x433: case 0x633:
            visitor.HSET2(insn);
            break;
        case 0x234: case 0x434: case 0x634:
            visitor.HSETP2(insn);
            break;
        case 0x240: case 0x840:
            visitor.HMNMX2(insn);
            break;
        case 0x822:
            visitor.FSWZADD(insn);
            break;
        case 0x311:
            visitor.F2I_64(insn);
            break;
        case 0x312:
            visitor.I2F_64(insn);
            break;
        case 0x243:
            visitor.F2IP(insn);
            break;
        case 0x305: case 0xb05: case 0xd05:
            visitor.F2I(insn);
            break;
        case 0x307: case 0xb07:
            visitor.FRND(insn);
            break;
        case 0x306: case 0xd06: case 0x906:
            visitor.I2F(insn);
            break;
        case 0x245: case 0x845: case 0xa45: case 0xc45:
            visitor.I2FP(insn);
            break;

        // Memory, atomics, barriers, attributes, geometry OUT, stage buffers
        case 0x381: case 0x981:
            visitor.LDG(insn);
            break;
        case 0x386: case 0x986:
            visitor.STG(insn);
            break;
        case 0x3a8: case 0x9a8:
            visitor.ATOMG(insn);
            break;
        case 0x98e:
            visitor.RED(insn);
            break;
        case 0x388: case 0x984: case 0x988:
            visitor.LDS_STS(insn);
            break;
        case 0x38c:
            visitor.ATOMS(insn);
            break;
        case 0xf8c:
            visitor.ATOMS_POPC_INC(insn);
            break;
        case 0xfae: {
            const bool load_global =
                (end_offset == offset + 8U) ||
                ((insn.Bits(87, 3) | (insn.Bits(70, 1) << 3)) == 7U);
            visitor.LDGSTS(insn, load_global);
            break;
        }
        case 0x387: case 0x983:
            visitor.LDL_STL(insn);
            break;
        case 0x91a:
            visitor.DEPBAR(insn);
            break;
        case 0x992:
            visitor.MEMBAR(insn);
            break;
        case 0x9ab:
            visitor.ERRBAR();
            break;
        case 0x9af:
            visitor.LDGDEPBAR();
            break;
        case 0xb1d:
            visitor.BAR_SYNC();
            break;
        case 0x321: case 0x322:
            visitor.ALD_AST(insn);
            break;
        case 0x324: case 0x924:
            visitor.OUT(insn);
            break;
        case 0x923: case 0x927:
            visitor.ISBE(insn);
            break;

        // Texture operations (Tier 2)
        case 0xb60: case 0xf60: case 0x361:
            visitor.TEX(insn);
            break;
        case 0xb63: case 0xf63:
            visitor.TLD4(insn, index);
            break;
        case 0xb66: case 0xf66: case 0x367:
            visitor.TLD(insn);
            break;
        case 0xb69: case 0xf69: case 0x36a:
            visitor.TMML(insn);
            break;
        case 0xb6c: case 0xf6c:
            visitor.TXD(insn);
            break;
        case 0xb6f: case 0x370:
            visitor.TXQ(insn);
            break;

        // Surface and warp operations (Tier 2)
        case 0xb97:
            visitor.SULD(insn);
            break;
        case 0xb9b:
            visitor.SUST(insn);
            break;
        case 0x593:
            visitor.SUATOM(insn);
            break;
        case 0x59f:
            visitor.SURED(insn);
            break;
        case 0x589: case 0x989: case 0xf89:
            visitor.SHFL(insn);
            break;
        case 0x3a1:
            visitor.MATCH_ALL(insn);
            break;

        // Tensor / matrix operations (Tier 2)
        case 0x23a:
            visitor.MOVM(insn);
            break;
        case 0x23c:
            visitor.HMMA(insn);
            break;
        case 0x83b:
            visitor.LDSM(insn);
            break;

        default:
            throw NotImplementedException("Unsupported native Ampere opcode {:#05x} at {}", opcode,
                                          index);
        }
        visitor.SetActiveGuard(std::nullopt);
    }
}

} // namespace Shader::Ampere
