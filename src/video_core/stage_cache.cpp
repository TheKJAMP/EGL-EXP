// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "video_core/stage_cache.h"

namespace VideoCommon {
namespace {

enum ReadKind : u8 {
    CbufValue,
    StoragePointer,
    IndirectStoragePointer,
    TextureType,
    TexturePixelFormat,
    TextureInteger,
    ViewportTransform,
    Replacement,
};

// Past this, the least recently used translations are dropped down to EvictTo (translations hold
// program metadata and SPIR-V).
constexpr size_t MaxBytes = 256ULL << 20;
constexpr size_t EvictTo = MaxBytes / 4 * 3;

class Writer {
public:
    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void Put(const T& value) {
        const auto* const begin = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    std::vector<u8> bytes;
};

u64 EncodeReplacement(const std::optional<Shader::ReplaceConstant>& value) {
    return value ? (u64{1} << 32) | static_cast<u32>(*value) : 0;
}

} // Anonymous namespace

StageCache::Recorder::Recorder(Shader::Environment& inner_) : inner{inner_} {
    sph = inner.SPH();
    gp_passthrough_mask = inner.GpPassthroughMask();
    mesh_registers = inner.MeshRegisters();
    mesh_primitive_header = inner.MeshPrimitiveHeader();
    stage = inner.ShaderStage();
    start_address = inner.StartAddress();
    is_proprietary_driver = inner.IsProprietaryDriver();
}

u64 StageCache::Recorder::Log(u8 kind, u32 a, u32 b, u32 c, u64 value) {
    // Only first reads: within one translation a repeated read gives the same answer.
    const u64 id = Common::CityHash64(reinterpret_cast<const char*>(std::array<u32, 4>{kind, a, b, c}.data()),
                                      sizeof(u32) * 4);
    if (logged.insert(id).second) {
        reads.push_back({kind, a, b, c, value});
    }
    return value;
}

u64 StageCache::Recorder::ReadInstruction(u32 address) {
    return inner.ReadInstruction(address);
}

bool StageCache::Recorder::IsNativeProgram() const {
    return inner.IsNativeProgram();
}

std::optional<Shader::Ampere::NativeAmpereView> StageCache::Recorder::NativeAmpereProgram() const {
    return inner.NativeAmpereProgram();
}

bool StageCache::Recorder::IsCbufStorageBufferPointer(u32 index, u32 offset) {
    return Log(StoragePointer, index, offset, 0, inner.IsCbufStorageBufferPointer(index, offset)) != 0;
}

bool StageCache::Recorder::IsCbufIndirectStorageBufferPointer(u32 index, u32 offset, u32 indirect) {
    return Log(IndirectStoragePointer, index, offset, indirect,
               inner.IsCbufIndirectStorageBufferPointer(index, offset, indirect)) != 0;
}

u32 StageCache::Recorder::ReadCbufValue(u32 index, u32 offset) {
    return static_cast<u32>(Log(CbufValue, index, offset, 0, inner.ReadCbufValue(index, offset)));
}

Shader::TextureType StageCache::Recorder::ReadTextureType(u32 handle) {
    return static_cast<Shader::TextureType>(
        Log(TextureType, handle, 0, 0, static_cast<u64>(inner.ReadTextureType(handle))));
}

Shader::TexturePixelFormat StageCache::Recorder::ReadTexturePixelFormat(u32 handle) {
    return static_cast<Shader::TexturePixelFormat>(
        Log(TexturePixelFormat, handle, 0, 0, static_cast<u64>(inner.ReadTexturePixelFormat(handle))));
}

bool StageCache::Recorder::IsTexturePixelFormatInteger(u32 handle) {
    return Log(TextureInteger, handle, 0, 0, inner.IsTexturePixelFormatInteger(handle)) != 0;
}

u32 StageCache::Recorder::ReadViewportTransformState() {
    return static_cast<u32>(Log(ViewportTransform, 0, 0, 0, inner.ReadViewportTransformState()));
}

u32 StageCache::Recorder::TextureBoundBuffer() const {
    return inner.TextureBoundBuffer();
}

u32 StageCache::Recorder::LocalMemorySize() const {
    return inner.LocalMemorySize();
}

u32 StageCache::Recorder::SharedMemorySize() const {
    return inner.SharedMemorySize();
}

std::array<u32, 3> StageCache::Recorder::WorkgroupSize() const {
    return inner.WorkgroupSize();
}

bool StageCache::Recorder::HasHLEMacroState() const {
    return inner.HasHLEMacroState();
}

std::optional<Shader::ReplaceConstant> StageCache::Recorder::GetReplaceConstBuffer(u32 bank,
                                                                                   u32 offset) {
    const auto value = inner.GetReplaceConstBuffer(bank, offset);
    Log(Replacement, bank, offset, 0, EncodeReplacement(value));
    return value;
}

void StageCache::Recorder::Dump(u64 pipeline_hash, u64 shader_hash) {
    inner.Dump(pipeline_hash, shader_hash);
}

StageCache::StageCache() {
    // EDEN_NX2_STAGE_CACHE=0 translates every stage again; =verify also builds every reused stage
    // and compares the SPIR-V.
    const char* const value = std::getenv("EDEN_NX2_STAGE_CACHE");
    enabled = value == nullptr || value[0] != '0';
    verify = enabled && value != nullptr && std::string_view{value} == "verify";
    if (verify) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_STAGE_CACHE=verify: reused stages are rebuilt and compared");
    }
}

std::vector<u8> StageCache::Signature(Shader::Environment& env, u64 unique_hash, size_t slot) {
    Writer writer;
    writer.Put(unique_hash);
    writer.Put(static_cast<u32>(slot));
    writer.Put(env.ShaderStage());
    writer.Put(env.StartAddress());
    writer.Put(env.IsProprietaryDriver());
    writer.Put(env.IsNativeProgram());
    writer.Put(env.SPH());
    writer.Put(env.GpPassthroughMask());
    writer.Put(env.MeshRegisters());
    writer.Put(env.MeshPrimitiveHeader());
    writer.Put(env.TextureBoundBuffer());
    writer.Put(env.LocalMemorySize());
    writer.Put(env.SharedMemorySize());
    writer.Put(env.WorkgroupSize());
    writer.Put(env.HasHLEMacroState());
    // What the recompiler reads from the settings, which can change during a session.
    const auto& resolution = Settings::values.resolution_info;
    writer.Put(resolution.up_scale);
    writer.Put(resolution.down_shift);
    writer.Put(resolution.up_factor);
    writer.Put(resolution.down_factor);
    writer.Put(resolution.active);
    writer.Put(Settings::values.renderer_debug.GetValue());
    writer.Put(Settings::values.disable_shader_loop_safety_checks.GetValue());
    writer.Put(Settings::values.fix_bloom_effects.GetValue());
    return std::move(writer.bytes);
}

StageCache::InFlight::InFlight(StageCache* cache_, std::vector<u64> hashes_)
    : cache{cache_}, hashes{std::move(hashes_)} {}

StageCache::InFlight::~InFlight() {
    Release();
}

StageCache::InFlight::InFlight(InFlight&& other) noexcept
    : cache{std::exchange(other.cache, nullptr)}, hashes{std::move(other.hashes)} {}

StageCache::InFlight& StageCache::InFlight::operator=(InFlight&& other) noexcept {
    if (this != &other) {
        Release();
        cache = std::exchange(other.cache, nullptr);
        hashes = std::move(other.hashes);
    }
    return *this;
}

void StageCache::InFlight::Release() {
    if (cache == nullptr) {
        return;
    }
    if (!hashes.empty()) {
        {
            std::scoped_lock lock{cache->in_flight_mutex};
            for (const u64 hash : hashes) {
                cache->in_flight.erase(hash);
            }
        }
        cache->in_flight_cv.notify_all();
    }
    cache = nullptr;
    hashes.clear();
}

u64 StageCache::SignatureHash(const std::vector<u8>& signature) {
    return Common::CityHash64(reinterpret_cast<const char*>(signature.data()), signature.size());
}

bool StageCache::WaitForInFlight(std::span<const u64> hashes, std::chrono::milliseconds limit,
                                 const std::function<bool()>& help) {
    const auto busy = [&] {
        for (const u64 hash : hashes) {
            if (in_flight.contains(hash)) {
                return true;
            }
        }
        return false;
    };
    std::unique_lock lock{in_flight_mutex};
    if (!busy()) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (help && busy() && std::chrono::steady_clock::now() < deadline) {
        lock.unlock();
        const bool helped = help();
        lock.lock();
        if (!helped) {
            break;
        }
        in_flight_helped.fetch_add(1, std::memory_order_relaxed);
    }
    in_flight_cv.wait_until(lock, deadline, [&] { return !busy(); });
    return true;
}

StageCache::InFlight StageCache::ClaimInFlight(std::span<const u64> hashes) {
    std::vector<u64> claimed;
    {
        std::scoped_lock lock{in_flight_mutex};
        for (const u64 hash : hashes) {
            if (in_flight.insert(hash).second) {
                claimed.push_back(hash);
            }
        }
    }
    return InFlight{this, std::move(claimed)};
}

std::shared_ptr<const StageCache::Translation> StageCache::Find(const std::vector<u8>& signature,
                                                                Shader::Environment& env) {
    const u64 hash = SignatureHash(signature);
    std::vector<std::shared_ptr<const Translation>> candidates;
    {
        std::scoped_lock lock{mutex};
        const auto it = translations.find(hash);
        if (it == translations.end()) {
            return nullptr;
        }
        candidates = it->second;
    }
    for (const auto& candidate : candidates) {
        if (candidate->signature != signature) {
            continue;
        }
        bool same = true;
        try {
            // In the order translation read them: up to the first difference, every read is one
            // translating here would make too.
            for (const Translation::Read& read : candidate->reads) {
                u64 value{};
                switch (read.kind) {
                case CbufValue:
                    value = env.ReadCbufValue(read.a, read.b);
                    break;
                case StoragePointer:
                    value = env.IsCbufStorageBufferPointer(read.a, read.b);
                    break;
                case IndirectStoragePointer:
                    value = env.IsCbufIndirectStorageBufferPointer(read.a, read.b, read.c);
                    break;
                case TextureType:
                    value = static_cast<u64>(env.ReadTextureType(read.a));
                    break;
                case TexturePixelFormat:
                    value = static_cast<u64>(env.ReadTexturePixelFormat(read.a));
                    break;
                case TextureInteger:
                    value = env.IsTexturePixelFormatInteger(read.a);
                    break;
                case ViewportTransform:
                    value = env.ReadViewportTransformState();
                    break;
                case Replacement:
                    value = EncodeReplacement(env.GetReplaceConstBuffer(read.a, read.b));
                    break;
                }
                if (value != read.value) {
                    same = false;
                    break;
                }
            }
        } catch (const Shader::Exception&) {
            // A disk cache environment that never recorded this read.
            same = false;
        }
        if (same) {
            candidate->last_used.store(use_clock.fetch_add(1, std::memory_order_relaxed) + 1,
                                       std::memory_order_relaxed);
            return candidate;
        }
    }
    return nullptr;
}

std::shared_ptr<const StageCache::Emitted> StageCache::FindEmitted(
    const Translation& translation, const std::vector<u8>& runtime_key) {
    std::scoped_lock lock{translation.emitted_mutex};
    for (const auto& emitted : translation.emitted) {
        if (emitted->runtime_key == runtime_key) {
            return emitted;
        }
    }
    return nullptr;
}

void StageCache::Insert(std::shared_ptr<Translation> translation,
                        std::shared_ptr<const Emitted> emitted) {
    const u64 hash = SignatureHash(translation->signature);
    const auto same_reads = [&](const Translation& other) {
        return other.signature == translation->signature &&
               std::ranges::equal(other.reads, translation->reads,
                                  [](const Translation::Read& a, const Translation::Read& b) {
                                      return a.kind == b.kind && a.a == b.a && a.b == b.b &&
                                             a.c == b.c && a.value == b.value;
                                  });
    };
    std::shared_ptr<const Translation> existing;
    {
        std::scoped_lock lock{mutex};
        auto& bucket{translations[hash]};
        // Another thread translated the same stage meanwhile: keep one, with both emissions.
        for (const auto& candidate : bucket) {
            if (same_reads(*candidate)) {
                existing = candidate;
                break;
            }
        }
        if (!existing) {
            translation->emitted.push_back(emitted);
            translation->last_used.store(use_clock.fetch_add(1, std::memory_order_relaxed) + 1,
                                         std::memory_order_relaxed);
            bucket.push_back(translation);
        }
    }
    CountTranslated();
    if (existing) {
        deduplicated.fetch_add(1, std::memory_order_relaxed);
        if (!FindEmitted(*existing, emitted->runtime_key)) {
            AddEmitted(*existing, std::move(emitted));
        }
        return;
    }
    const size_t size = translation->reads.size() * sizeof(Translation::Read) +
                        translation->signature.size() + sizeof(Translation) +
                        emitted->spirv.size() * sizeof(u32) + sizeof(Emitted);
    Account(*translation, size);
}

void StageCache::AddEmitted(const Translation& translation, std::shared_ptr<const Emitted> emitted) {
    const size_t size = emitted->spirv.size() * sizeof(u32) + sizeof(Emitted);
    {
        std::scoped_lock lock{translation.emitted_mutex};
        translation.emitted.push_back(std::move(emitted));
    }
    Account(translation, size);
}

void StageCache::Account(const Translation& translation, size_t added) {
    translation.bytes.fetch_add(added, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    bytes += added;
    if (bytes <= MaxBytes) {
        return;
    }
    // Translations in use elsewhere stay alive through their shared pointers.
    struct Entry {
        u64 last_used;
        u64 hash;
        const Translation* translation;
    };
    std::vector<Entry> entries;
    for (const auto& [hash, bucket] : translations) {
        for (const auto& candidate : bucket) {
            entries.push_back({candidate->last_used.load(std::memory_order_relaxed), hash,
                               candidate.get()});
        }
    }
    std::ranges::sort(entries, {}, &Entry::last_used);
    size_t dropped{};
    for (const Entry& entry : entries) {
        if (bytes <= EvictTo) {
            break;
        }
        // Read the size first: erasing may destroy the translation.
        const size_t size = entry.translation->bytes.load(std::memory_order_relaxed);
        auto& bucket{translations[entry.hash]};
        std::erase_if(bucket, [&](const auto& candidate) {
            return candidate.get() == entry.translation;
        });
        if (bucket.empty()) {
            translations.erase(entry.hash);
        }
        bytes -= (std::min)(bytes, size);
        ++dropped;
    }
    evicted.fetch_add(dropped, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "Stage cache: over {} MiB, dropped the {} least recently used stages",
             MaxBytes >> 20, dropped);
}

std::vector<u8> StageCache::RuntimeKey(const Shader::RuntimeInfo& info,
                                       const Shader::Backend::Bindings& bindings) {
    // Every field of RuntimeInfo: one left out would let different SPIR-V share a key.
    Writer writer;
    writer.Put(bindings);
    writer.Put(info.generic_input_types);
    for (size_t index = 0; index < info.previous_stage_stores.mask.size(); index += 64) {
        u64 word{};
        for (size_t bit = 0; bit < 64; ++bit) {
            word |= u64{info.previous_stage_stores.mask[index + bit]} << bit;
        }
        writer.Put(word);
    }
    writer.Put(static_cast<u32>(info.per_primitive_generics.to_ulong()));
    writer.Put(info.mesh_as_compute);
    writer.Put(static_cast<u32>(info.screen_linear_generics.to_ulong()));
    writer.Put(static_cast<u32>(info.previous_stage_legacy_stores_mapping.size()));
    for (const auto& [from, to] : info.previous_stage_legacy_stores_mapping) {
        writer.Put(from);
        writer.Put(to);
    }
    writer.Put(info.convert_depth_mode);
    writer.Put(info.force_early_z);
    writer.Put(info.tess_primitive);
    writer.Put(info.tess_spacing);
    writer.Put(info.tess_clockwise);
    writer.Put(info.input_topology);
    writer.Put(info.fixed_state_point_size.has_value());
    writer.Put(info.fixed_state_point_size.value_or(0.0f));
    writer.Put(info.alpha_test_func.has_value());
    writer.Put(info.alpha_test_func.value_or(Shader::CompareFunction{}));
    writer.Put(info.alpha_test_reference);
    writer.Put(info.y_negate);
    writer.Put(info.glasm_use_storage_buffers);
    writer.Put(info.xfb_count);
    for (u32 index = 0; index < info.xfb_count && index < info.xfb_varyings.size(); ++index) {
        const auto& varying = info.xfb_varyings[index];
        writer.Put(varying.buffer);
        writer.Put(varying.stream);
        writer.Put(varying.stride);
        writer.Put(varying.offset);
        writer.Put(varying.components);
    }
    writer.Put(info.color_output_types);
    writer.Put(info.dual_source_blend);
    writer.Put(info.nan_safe_min_max);
    return std::move(writer.bytes);
}

Shader::IR::Program StageCache::Summary(const Shader::IR::Program& program) {
    Shader::IR::Program summary;
    summary.info = program.info;
    summary.stage = program.stage;
    summary.workgroup_size = program.workgroup_size;
    summary.output_topology = program.output_topology;
    summary.output_vertices = program.output_vertices;
    summary.invocations = program.invocations;
    summary.local_memory_size = program.local_memory_size;
    summary.shared_memory_size = program.shared_memory_size;
    summary.is_geometry_passthrough = program.is_geometry_passthrough;
    summary.mesh = program.mesh;
    return summary;
}

bool StageCache::SameSummary(const Shader::IR::Program& a, const Shader::IR::Program& b) {
    return a.info == b.info && a.stage == b.stage && a.workgroup_size == b.workgroup_size &&
           a.output_topology == b.output_topology && a.output_vertices == b.output_vertices &&
           a.invocations == b.invocations && a.local_memory_size == b.local_memory_size &&
           a.shared_memory_size == b.shared_memory_size &&
           a.is_geometry_passthrough == b.is_geometry_passthrough && a.mesh == b.mesh;
}

void StageCache::CountReuse(bool emitted) {
    (emitted ? reused_emitted : reused_translation).fetch_add(1, std::memory_order_relaxed);
    Log();
}

void StageCache::CountTranslated() {
    translated.fetch_add(1, std::memory_order_relaxed);
    Log();
}

void StageCache::CountVerifyMismatch(u64 unique_hash) {
    verify_mismatches.fetch_add(1, std::memory_order_relaxed);
    LOG_ERROR(Render_Vulkan, "Stage cache: reused stage {:016x} differs from a fresh build",
              unique_hash);
}

void StageCache::CountInFlightWait(size_t stages, size_t reused) {
    in_flight_waits.fetch_add(1, std::memory_order_relaxed);
    in_flight_waited_stages.fetch_add(stages, std::memory_order_relaxed);
    in_flight_reused.fetch_add(reused, std::memory_order_relaxed);
}

void StageCache::Log() {
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = last_log.load(std::memory_order_relaxed);
    if (now - last < 10 || !last_log.compare_exchange_strong(last, now)) {
        return;
    }
    if (last == 0) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Stage cache: {} stages reused whole, {} translations reused, {} translated; "
             "{} pipelines waited for {} stages another thread was translating, {} of them "
             "reused, {} translations run while waiting; {} duplicates merged, {} evicted{}",
             reused_emitted.load(), reused_translation.load(), translated.load(),
             in_flight_waits.load(), in_flight_waited_stages.load(), in_flight_reused.load(),
             in_flight_helped.load(),
             deduplicated.load(), evicted.load(),
             verify ? fmt::format(", {} verify mismatches", verify_mismatches.load())
                    : std::string{});
}

} // namespace VideoCommon
