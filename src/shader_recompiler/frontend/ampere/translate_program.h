// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <span>
#include <vector>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"

namespace Shader::Ampere {

[[nodiscard]] IR::Program TranslateProgram(
    ObjectPool<IR::Inst>& inst_pool, ObjectPool<IR::Block>& block_pool,
    ObjectPool<Maxwell::Flow::Block>& flow_block_pool, Environment& env,
    std::span<const Instruction> code, const LoweringOptions& options, bool uses_async_copy,
    const std::vector<u8>* branch_targets, const HostTranslateInfo& host_info);

} // namespace Shader::Ampere
