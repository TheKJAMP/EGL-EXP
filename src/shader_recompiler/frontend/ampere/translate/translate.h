// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ir/basic_block.h"

namespace Shader::Ampere {

void TranslateBlock(Environment& env, IR::Block* block, std::span<const Instruction> code,
                    u32 begin_offset, u32 end_offset, const LoweringOptions& options,
                    bool uses_async_copy, u32& required_texture_buffer,
                    const std::vector<u8>* branch_targets, bool mkw_zero_depth_fallback = false);

} // namespace Shader::Ampere
