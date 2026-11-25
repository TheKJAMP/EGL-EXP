// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/translate/translate.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"

namespace Shader::Ampere {

IR::Program TranslateProgram(ObjectPool<IR::Inst>& inst_pool, ObjectPool<IR::Block>& block_pool,
                             ObjectPool<Maxwell::Flow::Block>& flow_block_pool, Environment& env,
                             std::span<const Instruction> code, const LoweringOptions& options,
                             bool uses_async_copy, const std::vector<u8>* branch_targets,
                             const HostTranslateInfo& host_info) {
    HostTranslateInfo normalized_host_info{host_info};
    normalized_host_info.ApplyDescriptorLimitPolicy();

    Flow::CFG cfg{env, flow_block_pool, code, false};
    u32 required_texture_buffer{env.TextureBoundBuffer()};
    const bool mkw_zero_depth_fallback =
        code.size_bytes() == 0x3b0 && NativeProgramKey(code) == 0x5a4f3726590efb17ULL;
    const Maxwell::BlockTranslator block_translator = [&](IR::Block* block, u32 begin, u32 end) {
        TranslateBlock(env, block, code, begin, end, options, uses_async_copy,
                       required_texture_buffer, branch_targets, mkw_zero_depth_fallback);
    };
    IR::AbstractSyntaxList syntax_list = Maxwell::BuildASL(
        inst_pool, block_pool, env, cfg.Functions(), normalized_host_info, block_translator);
    return Maxwell::FinalizeProgram(std::move(syntax_list), env, normalized_host_info);
}

} // namespace Shader::Ampere
