// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

// Offline shader translation bench over raw native dumps (EDEN_NX2_SHADER_DUMP output). Hidden:
//   EGLANTINE_SHADER_BENCH=<dump dir> ./tests "[.shader_bench]"
// Each program is lowered, translated (CFG, frontend and IR passes) and emitted to SPIR-V, timed
// per step. Dumps carry no program header, so graphics programs are translated as fragment
// programs (vertex when that fails) with an empty header: the code paths and their cost are the
// real ones, the outputs are not the game's. EGLANTINE_SHADER_BENCH_OUT=<file> writes one line
// per program (key, stage, SPIR-V hash or error) to compare two builds.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "common/cityhash.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/control_flow.h"
#include "shader_recompiler/frontend/ampere/instruction.h"
#include "shader_recompiler/frontend/ampere/translate_program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/program_header.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr u32 HeaderSize = sizeof(Shader::ProgramHeader);

class BenchEnvironment final : public Shader::Environment {
public:
    BenchEnvironment(Shader::Stage stage_, u32 texture_bound_) : texture_bound{texture_bound_} {
        stage = stage_;
        is_proprietary_driver = true;
        start_address = stage_ == Shader::Stage::Compute ? HeaderSize : 0;
    }

    u64 ReadInstruction(u32 address) override {
        throw Shader::RuntimeError("BenchEnvironment does not support Maxwell reads at {:#x}",
                                   address);
    }
    bool IsNativeProgram() const override {
        return true;
    }
    u32 ReadCbufValue(u32, u32) override {
        return 0;
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
        return 0;
    }
    u32 SharedMemorySize() const override {
        return 48 * 1024;
    }
    std::array<u32, 3> WorkgroupSize() const override {
        return {64, 1, 1};
    }
    bool HasHLEMacroState() const override {
        return false;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    u32 texture_bound;
};

struct StepTimes {
    double lower{}, cfg{}, translate{}, emit{};
};

struct Sample {
    std::string key;
    std::string stage;
    std::string result;
    StepTimes ms;
};

const char* StageName(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::Compute:
        return "cs";
    case Shader::Stage::Fragment:
        return "fs";
    case Shader::Stage::VertexB:
        return "vs";
    default:
        return "?";
    }
}

double Ms(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

u64 TranslateDirect(std::span<const Shader::Ampere::Instruction> native, Shader::Stage stage,
                    u32 texture_bound, bool uses_async_copy,
                    const std::vector<u8>& branch_targets, StepTimes& times) {
    Shader::ObjectPool<Shader::IR::Inst> inst_pool;
    Shader::ObjectPool<Shader::IR::Block> block_pool;
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_pool;
    BenchEnvironment env{stage, texture_bound};
    Shader::HostTranslateInfo host{};
    host.support_float16 = true;
    host.support_int64 = true;
    host.support_float64 = true;
    host.min_ssbo_alignment = 16;
    host.support_viewport_index_layer = true;
    host.support_conditional_barrier = true;
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.support_int64 = true;
    profile.support_int16 = true;
    profile.support_int8 = true;
    profile.support_descriptor_aliasing = true;
    profile.support_explicit_workgroup_layout = true;
    profile.support_vote = true;
    profile.warp_size_potentially_larger_than_guest = false;

    const auto begin = Clock::now();
    Shader::IR::Program program{Shader::Ampere::TranslateProgram(
        inst_pool, block_pool, flow_pool, env, native, {.tensor = true}, uses_async_copy,
        &branch_targets, host)};
    const auto translated = Clock::now();
    const std::vector<u32> spirv{Shader::Backend::SPIRV::EmitSPIRV(profile, program)};
    const auto emitted = Clock::now();
    times.translate += Ms(translated - begin);
    times.emit += Ms(emitted - translated);
    return Common::CityHash64(reinterpret_cast<const char*>(spirv.data()),
                              spirv.size() * sizeof(u32));
}

Sample Run(const std::filesystem::path& path) {
    Sample sample;
    const std::string name = path.filename().string();
    sample.key = name.substr(0, name.find('-'));
    const bool compute = name.find("-cs-") != std::string::npos;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto length = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<Shader::Ampere::Instruction> native(length / sizeof(Shader::Ampere::Instruction));
    file.read(reinterpret_cast<char*>(native.data()), native.size() * sizeof(native[0]));

    const std::vector<Shader::Stage> stages =
        compute ? std::vector{Shader::Stage::Compute}
                : std::vector{Shader::Stage::Fragment, Shader::Stage::VertexB};
    const auto analyze_begin = Clock::now();
    const auto analysis = Shader::Ampere::AnalyzeProgram(native, {.tensor = true});
    sample.ms.lower = Ms(Clock::now() - analyze_begin);
    if (!analysis) {
        sample.stage = compute ? "cs" : "gfx";
        sample.result = fmt::format("lower-reject {}", analysis.error);
        return sample;
    }
    const u32 texture_bound =
        analysis.required_texture_buffer != ~0U ? analysis.required_texture_buffer : 2;
    std::string error;
    for (const Shader::Stage stage : stages) {
        sample.stage = StageName(stage);
        try {
            const u64 hash = TranslateDirect(native, stage, texture_bound, analysis.uses_async_copy,
                                             analysis.branch_targets, sample.ms);
            sample.result = fmt::format("{:016x}", hash);
            return sample;
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    }
    sample.result = "error " + error.substr(0, 120);
    return sample;
}

} // Anonymous namespace

TEST_CASE("Shader translation bench over native dumps", "[.shader_bench]") {
    const char* const directory = std::getenv("EGLANTINE_SHADER_BENCH");
    if (directory == nullptr) {
        SKIP("EGLANTINE_SHADER_BENCH is not set");
    }
    // One file per native program key (dumps repeat a program at each address it was met).
    std::map<std::string, std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        if (entry.path().extension() == ".bin" && name.find('-') != std::string::npos) {
            files.try_emplace(name.substr(0, name.find('-')), entry.path());
        }
    }
    REQUIRE(!files.empty());

    std::vector<Sample> samples;
    samples.reserve(files.size());
    for (const auto& [key, path] : files) {
        samples.push_back(Run(path));
    }

    std::map<std::string, std::pair<size_t, StepTimes>> by_stage;
    std::vector<double> totals;
    size_t errors{};
    for (const Sample& sample : samples) {
        auto& [count, sum] = by_stage[sample.stage];
        ++count;
        sum.lower += sample.ms.lower;
        sum.cfg += sample.ms.cfg;
        sum.translate += sample.ms.translate;
        sum.emit += sample.ms.emit;
        totals.push_back(sample.ms.lower + sample.ms.cfg + sample.ms.translate + sample.ms.emit);
        errors += sample.result.starts_with("error") || sample.result.starts_with("lower-reject");
    }
    std::sort(totals.begin(), totals.end());
    const auto percentile = [&](double q) {
        return totals[std::min(totals.size() - 1, static_cast<size_t>(q * totals.size()))];
    };
    std::printf("shader bench: %zu programs, %zu failed\n", samples.size(), errors);
    for (const auto& [stage, entry] : by_stage) {
        const auto& [count, sum] = entry;
        std::printf("  %-4s %5zu programs: lower %8.1f ms, cfg %8.1f ms, translate %8.1f ms, "
                    "emit %8.1f ms\n",
                    stage.c_str(), count, sum.lower, sum.cfg, sum.translate, sum.emit);
    }
    std::printf("  per program: p50 %.2f ms, p90 %.2f ms, p99 %.2f ms, max %.2f ms\n",
                percentile(0.5), percentile(0.9), percentile(0.99), totals.back());

    if (const char* const out = std::getenv("EGLANTINE_SHADER_BENCH_OUT")) {
        std::ofstream file(out);
        for (const Sample& sample : samples) {
            file << sample.key << ' ' << sample.stage << ' ' << sample.result << '\n';
        }
    }
}
