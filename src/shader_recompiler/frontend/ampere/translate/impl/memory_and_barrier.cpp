// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <optional>
#include <utility>

#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate/impl/impl.h"

namespace Shader::Ampere {
namespace {

[[nodiscard]] constexpr s32 SignExtend24(u32 raw) noexcept {
    return static_cast<s32>(raw << 8) >> 8;
}

[[nodiscard]] IR::U64 AddSignedOffset64(IR::IREmitter& ir, const IR::U64& base, u64 offset) {
    if (offset == 0) {
        return base;
    }
    return IR::U64{ir.IAdd(base, ir.Imm64(offset))};
}

[[nodiscard]] IR::U32 AddByteOffset32(IR::IREmitter& ir, const IR::U32& base, u32 raw_off) {
    const s32 offset = SignExtend24(raw_off);
    if (offset == 0) {
        return base;
    }
    return IR::U32{ir.IAdd(base, ir.Imm32(offset))};
}

[[nodiscard]] std::pair<int, bool> SharedLocalBitSize(u32 width) {
    switch (width) {
    case 0U:
        return {8, false};
    case 1U:
        return {8, true};
    case 2U:
        return {16, false};
    case 3U:
        return {16, true};
    case 4U:
        return {32, false};
    case 5U:
        return {64, false};
    case 6U:
        return {128, false};
    default:
        throw NotImplementedException("Invalid shared/local memory width {}", width);
    }
}

[[nodiscard]] IR::U32 LoadBoundedLocal(TranslatorVisitor& v, const IR::U32& word_offset,
                                       const IR::U32& byte_offset) {
    const IR::U32 local_size = v.ir.Imm32(v.env.LocalMemorySize());
    const IR::U1 in_bounds = v.ir.ILessThan(byte_offset, local_size, false);
    return IR::U32{v.ir.Select(in_bounds, v.ir.LoadLocal(word_offset), v.ir.Imm32(0U))};
}

} // Anonymous namespace

IR::U64 TranslatorVisitor::PackU64Gpr(u32 reg) {
    if (reg == 255U) {
        return ir.Imm64(u64{0});
    }
    return ir.PackUint2x32(ir.CompositeConstruct(R(reg), R(reg + 1U)));
}

IR::U64 TranslatorVisitor::PackU64Ur(u32 ureg) {
    if (ureg >= 63U) {
        return ir.Imm64(u64{0});
    }
    return ir.PackUint2x32(ir.CompositeConstruct(UR(ureg), UR(ureg + 1U)));
}

IR::U64 TranslatorVisitor::GlobalAddress(const Instruction& insn, bool is_uniform_opcode,
                                         u32 ureg_bit, bool is_descriptor) {
    const u32 ra = insn.Bits(24, 8);
    const u32 raw_off = insn.Bits(40, 24);
    if (!is_uniform_opcode || is_descriptor) {
        const IR::U64 base = PackU64Gpr(ra);
        const u64 offset = (ra == 255U) ? u64{raw_off}
                                        : static_cast<u64>(static_cast<s64>(SignExtend24(raw_off)));
        return AddSignedOffset64(ir, base, offset);
    }
    const u32 ureg = insn.Bits(ureg_bit, 6);
    IR::U32 base_lo = ureg >= 63U ? ir.Imm32(0U) : UR(ureg);
    IR::U32 base_hi = ureg >= 63U ? ir.Imm32(0U) : UR(ureg + 1U);
    if (ra != 255U) {
        const IR::U32 idx_lo = R(ra);
        if (ureg >= 63U) {
            base_lo = idx_lo;
            base_hi = insn.Bits(90, 1) != 0U ? R(ra + 1U) : ir.Imm32(0U);
        } else {
            const IR::U32 sum_lo = ir.IAdd(base_lo, idx_lo);
            const IR::U1 carry = ir.GetCarryFromOp(sum_lo);
            const IR::U32 carry_u32{ir.Select(carry, ir.Imm32(1U), ir.Imm32(0U))};
            IR::U32 sum_hi =
                insn.Bits(90, 1) != 0U ? IR::U32{ir.IAdd(base_hi, R(ra + 1U))} : base_hi;
            sum_hi = ir.IAdd(sum_hi, carry_u32);
            base_lo = sum_lo;
            base_hi = sum_hi;
        }
    }
    const IR::U64 base = ir.PackUint2x32(ir.CompositeConstruct(base_lo, base_hi));
    const u64 offset = static_cast<u64>(static_cast<s64>(SignExtend24(raw_off)));
    return AddSignedOffset64(ir, base, offset);
}

void TranslatorVisitor::LDG(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 size = insn.Bits(73, 3);
    const bool descriptor =
        opcode == 0x981U && (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c1e1100U;
    const bool strong = StrongGlobalFence() && opcode == 0x981U && insn.Bits(78, 1) != 0U;
    const IR::U64 address = GlobalAddress(insn, opcode == 0x981U, 32U, descriptor);

    if (strong) {
        ir.DeviceMemoryBarrier();
    }
    switch (size) {
    case 0U:
        R(dest, ir.LoadGlobalU8(address));
        break;
    case 1U:
        R(dest, ir.LoadGlobalS8(address));
        break;
    case 2U:
        R(dest, ir.LoadGlobalU16(address));
        break;
    case 3U:
        R(dest, ir.LoadGlobalS16(address));
        break;
    case 4U:
        R(dest, ir.LoadGlobal32(address));
        break;
    case 5U: {
        const IR::Value vector = ir.LoadGlobal64(address);
        if (dest != 255U) {
            for (u32 i = 0; i < 2U; ++i) {
                R(dest + i, IR::U32{ir.CompositeExtract(vector, i)});
            }
        }
        break;
    }
    case 6U: {
        const IR::Value vector = ir.LoadGlobal128(address);
        if (dest != 255U) {
            for (u32 i = 0; i < 4U; ++i) {
                R(dest + i, IR::U32{ir.CompositeExtract(vector, i)});
            }
        }
        break;
    }
    default:
        throw NotImplementedException("Invalid Ampere LDG size {}", size);
    }
    if (strong) {
        ir.DeviceMemoryBarrier();
    }
}

void TranslatorVisitor::STG(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 data = insn.Bits(32, 8);
    const u32 size = insn.Bits(73, 3);
    const bool descriptor =
        opcode == 0x986U && (insn.words[2] & ~(0x3fU | 0xe00U)) == 0x0c101100U;
    const bool strong = StrongGlobalFence() && opcode == 0x986U && insn.Bits(78, 1) != 0U;
    const IR::U64 address = GlobalAddress(insn, opcode == 0x986U, 64U, descriptor);

    if (strong) {
        ir.DeviceMemoryBarrier();
    }
    switch (size) {
    case 0U:
        ir.WriteGlobalU8(address, R(data));
        break;
    case 1U:
        ir.WriteGlobalS8(address, R(data));
        break;
    case 2U:
        ir.WriteGlobalU16(address, R(data));
        break;
    case 3U:
        ir.WriteGlobalS16(address, R(data));
        break;
    case 4U:
        ir.WriteGlobal32(address, R(data));
        break;
    case 5U: {
        const IR::U32 v0 = data == 255U ? ir.Imm32(0U) : R(data);
        const IR::U32 v1 = data == 255U ? ir.Imm32(0U) : R(data + 1U);
        ir.WriteGlobal64(address, ir.CompositeConstruct(v0, v1));
        break;
    }
    case 6U: {
        const IR::U32 v0 = data == 255U ? ir.Imm32(0U) : R(data);
        const IR::U32 v1 = data == 255U ? ir.Imm32(0U) : R(data + 1U);
        const IR::U32 v2 = data == 255U ? ir.Imm32(0U) : R(data + 2U);
        const IR::U32 v3 = data == 255U ? ir.Imm32(0U) : R(data + 3U);
        ir.WriteGlobal128(address, ir.CompositeConstruct(v0, v1, v2, v3));
        break;
    }
    default:
        throw NotImplementedException("Invalid Ampere STG size {}", size);
    }
    if (strong) {
        ir.DeviceMemoryBarrier();
    }
}

void TranslatorVisitor::ATOMG(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u64 offset = static_cast<u64>(static_cast<s64>(SignExtend24(insn.Bits(40, 24))));
    const IR::U64 base =
        opcode == 0x9a8U ? PackU64Ur(insn.Bits(64, 6)) : PackU64Gpr(insn.Bits(24, 8));
    const IR::U64 address = AddSignedOffset64(ir, base, offset);
    R(dest, IR::U32{ir.GlobalAtomicIAdd(address, R(insn.Bits(32, 8)))});
}

void TranslatorVisitor::RED(const Instruction& insn) {
    const u32 type = insn.Bits(73, 3);
    const u32 operation = insn.Bits(87, 3);
    const bool uniform_address = insn.Bits(24, 8) == 255U;
    const u64 offset = static_cast<u64>(static_cast<s64>(SignExtend24(insn.Bits(40, 24))));
    const IR::U64 base =
        uniform_address ? PackU64Ur(insn.Bits(64, 6)) : PackU64Gpr(insn.Bits(24, 8));
    const IR::U64 address = AddSignedOffset64(ir, base, offset);

    if (type == 3U) {
        static constexpr IR::FpControl f32_control{
            .no_contraction = false,
            .rounding = IR::FpRounding::RN,
            .fmz_mode = IR::FmzMode::FTZ,
        };
        (void)ir.GlobalAtomicF32Add(address, F(insn.Bits(32, 8)), f32_control);
        return;
    }
    const bool is_signed = type == 1U;
    const IR::U32 data = R(insn.Bits(32, 8));
    switch (operation) {
    case 0U:
        (void)ir.GlobalAtomicIAdd(address, data);
        break;
    case 1U:
        (void)ir.GlobalAtomicIMin(address, data, is_signed);
        break;
    case 2U:
        (void)ir.GlobalAtomicIMax(address, data, is_signed);
        break;
    case 3U:
    case 4U:
        // Maxwell AtomOpNotApplicable: INC/DEC for U32/S32 in RED is a no-op read.
        (void)ir.LoadGlobal32(address);
        break;
    case 5U:
        (void)ir.GlobalAtomicAnd(address, data);
        break;
    case 6U:
        (void)ir.GlobalAtomicOr(address, data);
        break;
    case 7U:
        (void)ir.GlobalAtomicXor(address, data);
        break;
    default:
        throw NotImplementedException("Invalid Ampere RED operation {}", operation);
    }
}

void TranslatorVisitor::LDS_STS(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool store = opcode != 0x984U;
    const u32 width = insn.Bits(73, 3);
    const u32 stride = insn.Bits(78, 2);
    const bool uniform_base = (opcode == 0x984U || opcode == 0x988U) && insn.Bits(91, 1) != 0U;
    const u32 ra = insn.Bits(24, 8);
    const u32 raw_off = insn.Bits(40, 24);

    std::optional<IR::U32> addr;
    if (ra != 255U) {
        addr = stride != 0U ? IR::U32{ir.ShiftLeftLogical(R(ra), ir.Imm32(stride + 1U))} : R(ra);
    }
    if (uniform_base) {
        const u32 ureg = insn.Bits(opcode == 0x988U ? 64U : 32U, 8);
        if (ureg != 63U) {
            addr = addr ? IR::U32{ir.IAdd(*addr, UR(ureg))} : UR(ureg);
        }
    }
    const IR::U32 offset = addr ? AddByteOffset32(ir, *addr, raw_off) : ir.Imm32(raw_off);

    const auto [bit_size, is_signed] = SharedLocalBitSize(width);
    if (!store) {
        const u32 dest = insn.Bits(16, 8);
        const IR::Value value = ir.LoadShared(bit_size, is_signed, offset);
        if (dest == 255U) {
            return;
        }
        if (bit_size <= 32) {
            R(dest, IR::U32{value});
        } else {
            const u32 elems = static_cast<u32>(bit_size) / 32U;
            for (u32 i = 0; i < elems; ++i) {
                R(dest + i, IR::U32{ir.CompositeExtract(value, i)});
            }
        }
        return;
    }

    const u32 data = insn.Bits(32, 8);
    switch (bit_size) {
    case 8:
    case 16:
    case 32:
        ir.WriteShared(bit_size, offset, R(data));
        break;
    case 64: {
        const IR::U32 v0 = data == 255U ? ir.Imm32(0U) : R(data);
        const IR::U32 v1 = data == 255U ? ir.Imm32(0U) : R(data + 1U);
        ir.WriteShared(64, offset, ir.CompositeConstruct(v0, v1));
        break;
    }
    case 128: {
        const IR::U32 v0 = data == 255U ? ir.Imm32(0U) : R(data);
        const IR::U32 v1 = data == 255U ? ir.Imm32(0U) : R(data + 1U);
        const IR::U32 v2 = data == 255U ? ir.Imm32(0U) : R(data + 2U);
        const IR::U32 v3 = data == 255U ? ir.Imm32(0U) : R(data + 3U);
        ir.WriteShared(128, offset, ir.CompositeConstruct(v0, v1, v2, v3));
        break;
    }
    default:
        throw NotImplementedException("Invalid Ampere STS bit size {}", bit_size);
    }
}

void TranslatorVisitor::ATOMS(const Instruction& insn) {
    const u32 dest = insn.Bits(16, 8);
    const u32 ra = insn.Bits(24, 8);
    const u32 raw_off = insn.Bits(40, 24);
    const IR::U32 offset = ra == 255U ? ir.Imm32(raw_off) : AddByteOffset32(ir, R(ra), raw_off);
    const bool is_signed = insn.Bits(73, 1) != 0U;
    const u32 op = insn.Bits(87, 4);
    const IR::U32 data = R(insn.Bits(32, 8));

    IR::U32 result;
    switch (op) {
    case 0U:
        result = ir.SharedAtomicIAdd(offset, data);
        break;
    case 1U:
        result = ir.SharedAtomicIMin(offset, data, is_signed);
        break;
    case 2U:
        result = ir.SharedAtomicIMax(offset, data, is_signed);
        break;
    case 3U:
        result = ir.SharedAtomicInc(offset, data);
        break;
    case 4U:
        result = ir.SharedAtomicDec(offset, data);
        break;
    case 5U:
        result = ir.SharedAtomicAnd(offset, data);
        break;
    case 6U:
        result = ir.SharedAtomicOr(offset, data);
        break;
    case 7U:
        result = ir.SharedAtomicXor(offset, data);
        break;
    case 8U:
        result = IR::U32{ir.SharedAtomicExchange(offset, data)};
        break;
    default:
        throw NotImplementedException("Invalid Ampere ATOMS operation {}", op);
    }
    R(dest, result);
}

void TranslatorVisitor::ATOMS_POPC_INC(const Instruction& insn) {
    const u32 ra = insn.Bits(24, 8);
    const u32 stride = insn.Bits(78, 2);
    const u32 raw_off = insn.Bits(40, 24);
    IR::U32 offset;
    if (ra == 255U) {
        offset = ir.Imm32(raw_off);
    } else {
        const IR::U32 base =
            stride != 0U ? IR::U32{ir.ShiftLeftLogical(R(ra), ir.Imm32(stride + 1U))} : R(ra);
        offset = AddByteOffset32(ir, base, raw_off);
    }
    (void)ir.SharedAtomicIAdd(offset, ir.Imm32(1U));
}

void TranslatorVisitor::LDGSTS(const Instruction& insn, bool load_global) {
    const u32 global_imm = insn.words[1] & 0xfffU;
    const s32 g_off = static_cast<s32>(global_imm) - ((global_imm & 0x800U) ? 0x1000 : 0);
    const s32 s_off = static_cast<s32>(static_cast<s16>(insn.words[1] >> 16)) * 16;
    const u32 shared_reg = insn.Bits(16, 8);
    const IR::U32 shared_addr =
        shared_reg == 255U ? ir.Imm32(static_cast<u32>(s_off))
                           : IR::U32{ir.IAdd(R(shared_reg), ir.Imm32(s_off))};

    if (load_global) {
        const IR::U64 g_base = PackU64Gpr(insn.Bits(24, 8));
        const IR::U64 g_addr =
            AddSignedOffset64(ir, g_base, static_cast<u64>(static_cast<s64>(g_off)));
        ir.WriteShared(128, shared_addr, ir.LoadGlobal128(g_addr));
    } else {
        const IR::U32 zero = ir.Imm32(0U);
        ir.WriteShared(128, shared_addr, ir.CompositeConstruct(zero, zero, zero, zero));
    }
}

void TranslatorVisitor::LDL_STL(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool store = opcode == 0x387U;
    const u32 size = insn.Bits(73, 3);
    const auto [bit_size, is_signed] = SharedLocalBitSize(size);
    const u32 ra = insn.Bits(24, 8);
    const u32 raw_off = insn.Bits(40, 24);

    const IR::U32 offset = ra == 255U ? ir.Imm32(raw_off) : AddByteOffset32(ir, R(ra), raw_off);
    const IR::U32 word_offset = offset.IsImmediate()
                                    ? ir.Imm32(offset.U32() / 4U)
                                    : IR::U32{ir.ShiftRightArithmetic(offset, ir.Imm32(2U))};
    const IR::U32 byte_shift =
        ir.BitwiseAnd(IR::U32{ir.ShiftLeftLogical(offset, ir.Imm32(3U))}, ir.Imm32(24U));
    const IR::U32 short_shift =
        ir.BitwiseAnd(IR::U32{ir.ShiftLeftLogical(offset, ir.Imm32(3U))}, ir.Imm32(16U));

    if (!store) {
        const u32 dest = insn.Bits(16, 8);
        const IR::U32 word = LoadBoundedLocal(*this, word_offset, offset);
        switch (bit_size) {
        case 8:
            R(dest, ir.BitFieldExtract(word, byte_shift, ir.Imm32(8U), is_signed));
            break;
        case 16:
            R(dest, ir.BitFieldExtract(word, short_shift, ir.Imm32(16U), is_signed));
            break;
        case 32:
        case 64:
        case 128: {
            R(dest, word);
            const u32 words = static_cast<u32>(bit_size) / 32U;
            for (u32 i = 1U; i < words; ++i) {
                R(dest + i,
                  LoadBoundedLocal(*this, IR::U32{ir.IAdd(word_offset, ir.Imm32(i))},
                                   IR::U32{ir.IAdd(offset, ir.Imm32(i * 4U))}));
            }
            break;
        }
        }
        return;
    }

    if (offset.IsImmediate() && offset.U32() >= env.LocalMemorySize()) {
        LOG_WARNING(Shader, "Storing local memory at {:#x} with a size of {:#x}, dropping",
                    offset.U32(), env.LocalMemorySize());
        return;
    }
    const u32 data = insn.Bits(32, 8);
    const IR::U32 first = R(data);
    switch (bit_size) {
    case 8:
        ir.WriteLocal(word_offset, ir.BitFieldInsert(ir.LoadLocal(word_offset), first, byte_shift,
                                                     ir.Imm32(8U)));
        break;
    case 16:
        ir.WriteLocal(word_offset, ir.BitFieldInsert(ir.LoadLocal(word_offset), first, short_shift,
                                                     ir.Imm32(16U)));
        break;
    case 32:
    case 64:
    case 128: {
        ir.WriteLocal(word_offset, first);
        const u32 words = static_cast<u32>(bit_size) / 32U;
        for (u32 i = 1U; i < words; ++i) {
            const IR::U32 val = data == 255U ? ir.Imm32(0U) : R(data + i);
            ir.WriteLocal(IR::U32{ir.IAdd(word_offset, ir.Imm32(i))}, val);
        }
        break;
    }
    }
}

void TranslatorVisitor::DEPBAR(const Instruction& insn) {
    if (uses_async_copy && (insn.words[1] & ~0xfffU) == 0x8000U) {
        ir.WorkgroupMemoryBarrier();
    }
}

void TranslatorVisitor::MEMBAR(const Instruction& insn) {
    if (insn.Bits(76, 3) == 0U) {
        ir.WorkgroupMemoryBarrier();
    } else {
        ir.DeviceMemoryBarrier();
    }
}

void TranslatorVisitor::ERRBAR() {
    ir.DeviceMemoryBarrier();
}

void TranslatorVisitor::LDGDEPBAR() {
    ir.WorkgroupMemoryBarrier();
}

void TranslatorVisitor::BAR_SYNC() {
    ir.Barrier();
}

void TranslatorVisitor::ALD_AST(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const bool store = opcode == 0x322U;
    const u32 address = insn.Bits(40, 10);
    const u32 components = insn.Bits(74, 2) + 1U;
    if (store && address == 0x5cU && components == 1U) {
        return;
    }
    const u32 value_reg = store ? insn.Bits(32, 8) : insn.Bits(16, 8);
    if (!store && value_reg == 255U) {
        return;
    }
    const IR::U32 vertex = R(store ? insn.Bits(64, 8) : insn.Bits(32, 8));
    const u32 base_attr = address / 4U;
    for (u32 i = 0; i < components; ++i) {
        const auto attr = static_cast<IR::Attribute>(base_attr + i);
        if (store) {
            ir.SetAttribute(attr, value_reg == 255U ? ir.Imm32(0.0f) : F(value_reg + i), vertex);
        } else {
            F(value_reg + i, ir.GetAttribute(attr, vertex));
        }
    }
}

void TranslatorVisitor::OUT(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    const u32 dest = insn.Bits(16, 8);
    const u32 type = insn.Bits(78, 2);
    if (type == 0U) {
        R(dest, R(insn.Bits(24, 8)));
        return;
    }
    const IR::U32 raw_stream = opcode == 0x324U ? R(insn.Bits(32, 8)) : ir.Imm32(insn.words[1]);
    const IR::U32 stream = ir.BitwiseAnd(raw_stream, ir.Imm32(3U));
    if ((type & 1U) != 0U) {
        ir.EmitVertex(stream);
    }
    if ((type & 2U) != 0U) {
        ir.EndPrimitive(stream);
    }
    R(dest, ir.Imm32(0U));
}

void TranslatorVisitor::ISBE(const Instruction& insn) {
    const u32 opcode = insn.Bits(0, 12);
    if (options.geometry) {
        R(insn.Bits(16, 8), R(insn.Bits(24, 8)));
        return;
    }
    const Stage stage = env.ShaderStage();
    if (stage != Stage::Task && stage != Stage::Mesh) {
        throw NotImplementedException("Stage buffer access outside task and mesh programs");
    }
    const IR::U32 addr = IR::U32{ir.IAdd(R(insn.Bits(24, 8)), ir.Imm32(insn.Bits(40, 24)))};
    if (opcode == 0x923U) {
        const u32 dest = insn.Bits(16, 8);
        R(dest, insn.Bits(79, 1) != 0U ? ir.IsbeReadOutput(addr) : ir.IsbeRead(addr));
        return;
    }
    const IR::U32 value = R(insn.Bits(32, 8));
    const bool is_32 = insn.Bits(75, 1) != 0U;
    const bool is_attr = insn.Bits(76, 2) == 3U;
    if (is_attr) {
        if (!is_32) {
            throw NotImplementedException("U8 stage buffer attribute write");
        }
        ir.IsbeWriteAttribute(addr, value);
    } else if (is_32) {
        ir.IsbeWriteMapU32(addr, value);
    } else {
        ir.IsbeWriteMapU8(addr, value);
    }
}

} // namespace Shader::Ampere
