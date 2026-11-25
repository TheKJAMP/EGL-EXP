// SPDX-FileCopyrightText: Copyright 2026 Vortex727 (FUKKO)
// SPDX-License-Identifier: GPL-3.0-or-later

// The bindless texture table of the replay (shader_recompiler/bindless_table.h). A shader that
// reads a texture handle at run time (a light or material record in a buffer the game fills)
// indexes arrays of every texture and sampler the game registered in its pools. The arrays are
// filled from those registrations, not guessed: entry `id` of an array is the image view of the
// texture registered at pool slot `id`, made with the view the game described (target, format,
// level and layer range), for the shader image type that target gives, and entry `id` of the
// sampler array is the sampler registered at that slot.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <utility>

#include "common/logging.h"
#include "video_core/renderer_vulkan/nvn/nvn_bindless.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_layout.h"
#include "video_core/surface.h"

namespace Vulkan::Nvn {
namespace {

constexpr u32 TEXTURE_BINDINGS = Shader::BINDLESS_BINDING_COUNT - 1;

[[nodiscard]] Shader::TextureType TypeOfBinding(u32 binding) {
    return static_cast<Shader::TextureType>((binding - 1) % Shader::BINDLESS_TYPE_COUNT);
}

[[nodiscard]] VkImageLayout ReadLayoutOf(const Storage& s) {
    const bool depth = s.host.depth || s.host.stencil;
    return depth && (s.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
               ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
               : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

[[nodiscard]] bool NeedsBindlessPrime(const Context::Impl& impl, const Storage& s) noexcept {
    if (s.image || s.image_failed || s.samples != VK_SAMPLE_COUNT_1_BIT ||
        s.scale_owner != nullptr) {
        return false;
    }
    constexpr VkImageUsageFlags attachment = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((s.usage & attachment) == 0) {
        return true;
    }
    // Every uncompressed format gets COLOR_ATTACHMENT_BIT in BuildTexture. Prime small
    // CPU-authored uncompressed bindless textures (LUTs, 3D light grids, irradiance cubemaps) when
    // their storage lacks explicit display/compressible/linear render-target flags and lives in a
    // CPU-readable non-virtual pool.
    constexpr s32 RT_HINT_FLAGS =
        0x01 /* Display */ | TexFlag::Compressible | TexFlag::LinearRenderTarget;
    if ((s.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ||
        (s.desc.flags & RT_HINT_FLAGS) != 0 || s.vram_bytes > (u64{16} << 20)) {
        return false;
    }
    const auto pool_it = impl.pools.find(s.pool_serial);
    if (pool_it == impl.pools.end() || !pool_it->second) {
        return false;
    }
    const Pool& pool = *pool_it->second;
    return !pool.IsVirtual() && (pool.flags & PoolFlag::CpuNoAccess) == 0;
}

template <typename Key, typename SlotListT>
void EraseSlotId(std::unordered_map<Key, SlotListT>& map, const Key& key, u32 id) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return;
    }
    it->second.Remove(id);
    if (it->second.Empty()) {
        map.erase(it);
    }
}

/// The shader image types a texture registered with `target` is sampled as (what a shader declares
/// for it): its own, and the one-layer array / array-less twin where the view allows it.
[[nodiscard]] std::array<std::optional<Shader::TextureType>, 2> TypesOfTarget(s32 target) {
    using T = Shader::TextureType;
    switch (static_cast<TexTarget>(target)) {
    case TexTarget::Dim1D:
        return {T::Color1D, std::nullopt};
    case TexTarget::Dim1DArray:
        return {T::ColorArray1D, std::nullopt};
    case TexTarget::Dim2D:
        return {T::Color2D, T::ColorArray2D};
    case TexTarget::Dim2DArray:
        return {T::ColorArray2D, T::Color2D};
    case TexTarget::Rectangle:
        return {T::Color2DRect, T::Color2D};
    case TexTarget::Dim3D:
        return {T::Color3D, std::nullopt};
    case TexTarget::Cubemap:
        return {T::ColorCube, std::nullopt};
    case TexTarget::CubemapArray:
        return {T::ColorArrayCube, std::nullopt};
    default: // multisampled and buffer textures have no table
        return {std::nullopt, std::nullopt};
    }
}

} // namespace

VkImageViewType BindlessViewType(Shader::TextureType type) {
    switch (type) {
    case Shader::TextureType::Color1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::ColorArray2D:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::Color3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
        return VK_IMAGE_VIEW_TYPE_2D;
    }
}

vk::DescriptorSetLayout MakeBindlessSetLayout(const Device& device) {
    constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS;
    std::array<VkDescriptorSetLayoutBinding, Shader::BINDLESS_BINDING_COUNT> bindings{};
    std::array<VkDescriptorBindingFlags, Shader::BINDLESS_BINDING_COUNT> flags{};
    constexpr VkDescriptorBindingFlags update_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    bindings[0] = {
        .binding = Shader::BINDLESS_SAMPLER_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = Shader::BINDLESS_SAMPLER_CAPACITY,
        .stageFlags = stages,
        .pImmutableSamplers = nullptr,
    };
    flags[0] = update_flags;
    for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
        bindings[binding] = {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = Shader::BINDLESS_TEXTURE_CAPACITY,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        };
        flags[binding] = update_flags;
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = static_cast<u32>(flags.size()),
        .pBindingFlags = flags.data(),
    };
    return device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
}

bool DrawContext::BindlessReady() {
    BindlessState& state = bindless;
    if (state.set != VK_NULL_HANDLE) {
        return true;
    }
    if (state.failed) {
        return false;
    }
    try {
        const Device& device = impl.device;
        state.layout = MakeBindlessSetLayout(device);
        const std::array<VkDescriptorPoolSize, 2> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, Shader::BINDLESS_SAMPLER_CAPACITY},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             Shader::BINDLESS_TEXTURE_CAPACITY * TEXTURE_BINDINGS},
        }};
        state.pool = device.GetLogical().CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = static_cast<u32>(sizes.size()),
            .pPoolSizes = sizes.data(),
        });
        const VkDescriptorSetLayout layout_handle = *state.layout;
        const VkDescriptorSetAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = *state.pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout_handle,
        };
        if (impl.dld.vkAllocateDescriptorSets(impl.logical, &allocate, &state.set) != VK_SUCCESS) {
            throw vk::Exception(VK_ERROR_OUT_OF_POOL_MEMORY);
        }
        // Every entry starts as the stand-in of its type, so a handle no registration names reads
        // black (partially bound arrays must not be read unwritten).
        std::vector<VkDescriptorImageInfo> infos(Shader::BINDLESS_TEXTURE_CAPACITY);
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(Shader::BINDLESS_BINDING_COUNT);
        for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
            const VkImageView dummy =
                resources.DummyView(BindlessViewType(TypeOfBinding(binding)));
            state.dummy[binding] = dummy;
            state.written[binding].assign(
                Shader::BINDLESS_TEXTURE_CAPACITY,
                BindlessState::WrittenEntry{dummy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        }
        // One binding at a time: the infos are a copy of the binding's stand-in.
        for (u32 binding = 1; binding < Shader::BINDLESS_BINDING_COUNT; ++binding) {
            std::fill(infos.begin(), infos.end(),
                      VkDescriptorImageInfo{VK_NULL_HANDLE, state.dummy[binding],
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = Shader::BINDLESS_TEXTURE_CAPACITY,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .pImageInfo = infos.data(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            };
            device.GetLogical().UpdateDescriptorSets(std::span{&write, 1}, {});
        }
        state.written_samplers.assign(Shader::BINDLESS_SAMPLER_CAPACITY,
                                      *resources.fallback_sampler);
        std::vector<VkDescriptorImageInfo> sampler_infos(
            Shader::BINDLESS_SAMPLER_CAPACITY,
            VkDescriptorImageInfo{*resources.fallback_sampler, VK_NULL_HANDLE,
                                  VK_IMAGE_LAYOUT_UNDEFINED});
        const VkWriteDescriptorSet sampler_write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = state.set,
            .dstBinding = Shader::BINDLESS_SAMPLER_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = Shader::BINDLESS_SAMPLER_CAPACITY,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = sampler_infos.data(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        device.GetLogical().UpdateDescriptorSets(std::span{&sampler_write, 1}, {});
        LOG_INFO(Render_Vulkan,
                 "[nvn] bindless table created: {} texture arrays of {} entries and {} samplers",
                 TEXTURE_BINDINGS, Shader::BINDLESS_TEXTURE_CAPACITY,
                 Shader::BINDLESS_SAMPLER_CAPACITY);
        return true;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] bindless table creation failed: {}", exception.what());
        state.failed = true;
        state.set = VK_NULL_HANDLE;
        return false;
    }
}

void DrawContext::BindlessDetachEntry(BindlessState& state, u32 id) {
    BindlessState::Entry& entry = state.entries[id];
    if (entry.serial != 0 && entry.texture != nullptr) {
        EraseSlotId(state.serial_slots, entry.serial, id);
    }
    if (entry.storage != nullptr) {
        EraseSlotId(state.storage_slots, entry.storage, id);
    }
    if (const std::shared_ptr<Storage> s = entry.storage_ref.lock()) {
        --s->bindless_tracked_count;
        if (entry.any) {
            --s->bindless_active_count;
        }
    }
    entry = {};
    state.synced_serials[id] = 0;
}

void DrawContext::BindlessAttachEntry(BindlessState& state, u32 id, u64 serial,
                                      const std::shared_ptr<Texture>* texture_node,
                                      const std::shared_ptr<Storage>& storage_sp, u32 image_gen,
                                      VkFormat format, VkImageLayout layout, bool ready, bool any) {
    BindlessState::Entry& entry = state.entries[id];
    entry.serial = serial;
    entry.texture = texture_node;
    entry.image_gen = image_gen;
    entry.format = format;
    entry.layout = layout;
    entry.ready = ready;
    entry.any = any;
    state.synced_serials[id] = serial;
    if (serial != 0 && texture_node != nullptr) {
        state.serial_slots[serial].Add(id);
    }
    if (storage_sp) {
        entry.storage = storage_sp.get();
        entry.storage_ref = storage_sp;
        state.storage_slots[storage_sp.get()].Add(id);
        ++storage_sp->bindless_tracked_count;
        if (any) {
            ++storage_sp->bindless_active_count;
        }
    }
}

void DrawContext::BindlessSync(bool force, DrawPlan* plan) {
    BindlessState& state = bindless;
    const u64 seq = host->HostFlight().seq;
    const std::vector<u64>& texture_ids = submission->texture_pool_snapshot.Values();
    const std::vector<u64>& sampler_ids = submission->sampler_pool_snapshot.Values();
    const bool textures_same = !state.entries.empty() && state.synced_textures == texture_ids.data() &&
                               state.synced_texture_size == texture_ids.size();
    const bool samplers_same = !state.entries.empty() && state.synced_samplers == sampler_ids.data() &&
                               state.synced_sampler_size == sampler_ids.size() &&
                               state.synced_samplers_epoch == resources.samplers_epoch;
    const u64 cur_tex_epoch = impl.texture_epoch;
    const u64 cur_storage_epoch = impl.bindless_storage_change_epoch;
    // The registrations only change between submissions; what the table holds follows them (and
    // the images/storages the replay updated meanwhile) without walking unchanged entries.
    if (!force && state.synced_seq == seq && textures_same && samplers_same &&
        state.entries_epoch == cur_tex_epoch && state.storage_change_epoch == cur_storage_epoch &&
        state.retry_slots.empty() && !state.types_mask_dirty) {
        return;
    }
    state.synced_seq = seq;

    const u32 count = static_cast<u32>(
        std::min<size_t>(std::max(texture_ids.size(), size_t{state.synced_count}),
                         Shader::BINDLESS_TEXTURE_CAPACITY));
    struct Run {
        u32 binding;
        u32 first_id;
        u32 first_info;
        u32 count;
    };
    thread_local std::vector<Run> runs;
    thread_local std::vector<VkDescriptorImageInfo> infos;
    runs.clear();
    infos.clear();
    std::array<VkImageView, TEXTURE_BINDINGS + 1> want{};
    std::array<VkImageLayout, TEXTURE_BINDINGS + 1> want_layout{};
    bool full_scan = state.entries.empty() || state.types_mask_dirty;
    state.types_mask_dirty = false;
    if (state.entries.empty()) {
        state.entries.resize(Shader::BINDLESS_TEXTURE_CAPACITY);
        state.synced_serials.assign(Shader::BINDLESS_TEXTURE_CAPACITY, 0);
        state.entries_epoch = cur_tex_epoch;
        state.storage_change_epoch = cur_storage_epoch;
    } else if ((cur_tex_epoch - state.entries_epoch) > impl.texture_erased.size() ||
               (cur_storage_epoch - state.storage_change_epoch) >
                   impl.bindless_storage_changed.size()) {
        full_scan = true;
    }
    if (full_scan) {
        state.unprimed_queued.assign(Shader::BINDLESS_TEXTURE_CAPACITY, 0);
        state.unprimed_slots.clear();
    }

    thread_local std::vector<u32> dirty_ids;
    dirty_ids.clear();
    if (full_scan) {
        dirty_ids.reserve(count);
        for (u32 id = 0; id < count; ++id) {
            dirty_ids.push_back(id);
        }
    } else {
        if (!state.retry_slots.empty()) {
            dirty_ids.insert(dirty_ids.end(), state.retry_slots.begin(), state.retry_slots.end());
            state.retry_slots.clear();
        }
        for (u64 n = state.entries_epoch - 1; n != cur_tex_epoch - 1; ++n) {
            const u64 erased_serial = impl.texture_erased[n % impl.texture_erased.size()];
            if (const auto it = state.serial_slots.find(erased_serial);
                it != state.serial_slots.end()) {
                it->second.AppendTo(dirty_ids);
            }
        }
        for (u64 n = state.storage_change_epoch - 1; n != cur_storage_epoch - 1; ++n) {
            const Storage* changed_storage =
                impl.bindless_storage_changed[n % impl.bindless_storage_changed.size()];
            if (const auto it = state.storage_slots.find(changed_storage);
                it != state.storage_slots.end()) {
                it->second.AppendTo(dirty_ids);
            }
        }
        if (!textures_same) {
            for (u32 id = 0; id < count; ++id) {
                const u64 serial = id != 0 && id < texture_ids.size() ? texture_ids[id] : 0;
                if (serial != state.synced_serials[id]) {
                    dirty_ids.push_back(id);
                }
            }
        }
        if (dirty_ids.size() > 1) {
            std::ranges::sort(dirty_ids);
            dirty_ids.erase(std::unique(dirty_ids.begin(), dirty_ids.end()), dirty_ids.end());
        }
    }
    state.entries_epoch = cur_tex_epoch;
    state.storage_change_epoch = cur_storage_epoch;
    state.synced_texture_snapshot = submission->texture_pool_snapshot;
    state.synced_textures = texture_ids.data();
    state.synced_texture_size = texture_ids.size();

    const auto has_data = [](const Storage& s) {
        // Only what has an image holding the game's data: the table never creates one (a texture
        // the game registers and never uses would cost its memory).
        return s.image && !s.image_failed && s.contents_valid && s.samples == VK_SAMPLE_COUNT_1_BIT;
    };
    for (const u32 id : dirty_ids) {
        if (id >= Shader::BINDLESS_TEXTURE_CAPACITY) {
            continue;
        }
        // Pool slot 0 is "no texture" (the rebased handle keeps id 0 at 0).
        const u64 serial = id != 0 && id < texture_ids.size() ? texture_ids[id] : 0;
        if (full_scan && serial == 0 && state.entries[id].serial == 0) {
            continue;
        }
        BindlessDetachEntry(state, id);
        for (u32 binding = 1; binding <= TEXTURE_BINDINGS; ++binding) {
            want[binding] = state.dummy[binding];
            want_layout[binding] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        const std::shared_ptr<Texture>* texture_node = nullptr;
        std::shared_ptr<Storage> storage_sp;
        bool retry = false;
        u32 image_gen = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool ready = false;
        bool any = false;
        bool eligible_for_prime = false;
        if (serial != 0) {
            const std::shared_ptr<Texture>& texture = TextureOfId(id);
            if (texture) {
                // (TextureOfId returns the node of Impl::textures)
                texture_node = &texture;
                storage_sp = texture->storage;
            }
            if (texture && texture->storage) {
                Storage& s = *texture->storage;
                image_gen = s.image_gen;
                format = s.host.format;
                ready = has_data(s);
                const s32 target = texture->desc.view.set_target ? texture->desc.view.target
                                                                 : texture->desc.target;
                const auto target_types = TypesOfTarget(target);
                if (ready) {
                    layout = ReadLayoutOf(s);
                    const bool is_integer =
                        VideoCore::Surface::IsPixelFormatInteger(texture->host.pixel_format);
                    for (const std::optional<Shader::TextureType>& type : target_types) {
                        if (!type ||
                            (state.requested_types_mask & (1u << static_cast<u32>(*type))) == 0) {
                            continue;
                        }
                        ViewKey key{};
                        if (!ViewKeyOf(*texture, s, *type, false, key)) {
                            continue;
                        }
                        eligible_for_prime = true;
                        if (s.pending_unrestricted_formats || !s.FormatListAllows(key.format)) {
                            if (!pass_open) {
                                host->HostUpgradeImageCaps(s, false, true);
                                image_gen = s.image_gen;
                                if (const auto it = state.storage_slots.find(&s);
                                    it != state.storage_slots.end() && !it->second.Empty()) {
                                    state.dirty = true;
                                }
                            } else if (plan != nullptr) {
                                plan->UseImage(texture->storage, layout, true, false, false, false,
                                               key.format);
                                state.dirty = true;
                                retry = true;
                                continue;
                            }
                        }
                        const VkImageView view = resources.GetView(*texture, key);
                        if (!view) {
                            continue;
                        }
                        for (const u32 klass : {0u, 1u, 2u}) {
                            // Float and depth-compare samplers share the views; an integer
                            // sampler takes only an integer format.
                            if ((klass == 1) != is_integer) {
                                continue;
                            }
                            const u32 binding = Shader::BindlessTextureBinding(*type, klass);
                            want[binding] = view;
                            want_layout[binding] = layout;
                        }
                        any = true;
                    }
                } else if (NeedsBindlessPrime(impl, s)) {
                    for (const std::optional<Shader::TextureType>& type : target_types) {
                        if (!type ||
                            (state.requested_types_mask & (1u << static_cast<u32>(*type))) == 0) {
                            continue;
                        }
                        ViewKey key{};
                        if (ViewKeyOf(*texture, s, *type, false, key)) {
                            eligible_for_prime = true;
                            break;
                        }
                    }
                }
            }
        }
        BindlessAttachEntry(state, id, serial, texture_node, storage_sp, image_gen, format, layout,
                            ready, any);
        if (eligible_for_prime && storage_sp && NeedsBindlessPrime(impl, *storage_sp) &&
            !state.unprimed_queued[id]) {
            state.unprimed_queued[id] = 1;
            state.unprimed_slots.push_back(id);
        }
        if (retry) {
            state.retry_slots.push_back(id);
        }
        if (any && storage_sp && storage_sp->current_layout != ReadLayoutOf(*storage_sp)) {
            impl.NoteBindlessLayoutDirty(*storage_sp);
        }
        for (u32 binding = 1; binding <= TEXTURE_BINDINGS; ++binding) {
            const BindlessState::WrittenEntry desired{want[binding], want_layout[binding]};
            if (state.written[binding][id] == desired) {
                continue;
            }
            state.written[binding][id] = desired;
            infos.push_back({VK_NULL_HANDLE, want[binding], want_layout[binding]});
            runs.push_back({binding, id, static_cast<u32>(infos.size() - 1), 1});
        }
    }
    thread_local std::vector<VkDescriptorImageInfo> sorted;
    thread_local std::vector<Run> merged;
    thread_local std::vector<VkWriteDescriptorSet> writes;
    writes.clear();
    if (!runs.empty()) {
        // Merge consecutive entries of one binding into one write each.
        std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) {
            return a.binding != b.binding ? a.binding < b.binding : a.first_id < b.first_id;
        });
        sorted.clear();
        sorted.reserve(infos.size());
        merged.clear();
        for (const Run& run : runs) {
            if (!merged.empty() && merged.back().binding == run.binding &&
                merged.back().first_id + merged.back().count == run.first_id) {
                ++merged.back().count;
            } else {
                merged.push_back(
                    {run.binding, run.first_id, static_cast<u32>(sorted.size()), 1});
            }
            sorted.push_back(infos[run.first_info]);
        }
        for (const Run& run : merged) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = run.binding,
                .dstArrayElement = run.first_id,
                .descriptorCount = run.count,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .pImageInfo = sorted.data() + run.first_info,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            });
        }
    }
    state.synced_count = static_cast<u32>(std::min<size_t>(texture_ids.size(),
                                                            Shader::BINDLESS_TEXTURE_CAPACITY));
    if (state.sync_logs++ % 600 == 0) {
        u32 registered = 0;
        for (u32 id = 1; id < texture_ids.size() && id < Shader::BINDLESS_TEXTURE_CAPACITY; ++id) {
            registered += texture_ids[id] != 0;
        }
        u32 cube = 0;
        u32 d2 = 0;
        u32 d3 = 0;
        u32 d2_array = 0;
        u32 cube_array = 0;
        u32 unprimed_cpu_no_access = 0;
        u32 unprimed_rt_flag = 0;
        u32 unprimed_other = 0;
        const u32 cube_binding = Shader::BindlessTextureBinding(Shader::TextureType::ColorCube, 0);
        const u32 d2_binding = Shader::BindlessTextureBinding(Shader::TextureType::Color2D, 0);
        const u32 d3_binding = Shader::BindlessTextureBinding(Shader::TextureType::Color3D, 0);
        const u32 d2a_binding =
            Shader::BindlessTextureBinding(Shader::TextureType::ColorArray2D, 0);
        const u32 ca_binding =
            Shader::BindlessTextureBinding(Shader::TextureType::ColorArrayCube, 0);
        constexpr s32 RT_HINT_FLAGS =
            0x01 /* Display */ | TexFlag::Compressible | TexFlag::LinearRenderTarget;
        for (u32 id = 0; id < state.synced_count; ++id) {
            cube += state.written[cube_binding][id].view != state.dummy[cube_binding];
            d2 += state.written[d2_binding][id].view != state.dummy[d2_binding];
            d3 += state.written[d3_binding][id].view != state.dummy[d3_binding];
            d2_array += state.written[d2a_binding][id].view != state.dummy[d2a_binding];
            cube_array += state.written[ca_binding][id].view != state.dummy[ca_binding];
            if (const Storage* const st = state.entries[id].storage; st && !st->image) {
                const auto pool_it = impl.pools.find(st->pool_serial);
                const Pool* const pool =
                    (pool_it != impl.pools.end() && pool_it->second) ? pool_it->second.get()
                                                                     : nullptr;
                if (pool && (pool->flags & PoolFlag::CpuNoAccess) != 0) {
                    ++unprimed_cpu_no_access;
                } else if ((st->desc.flags & RT_HINT_FLAGS) != 0 ||
                           (st->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
                    ++unprimed_rt_flag;
                } else {
                    ++unprimed_other;
                }
            }
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] bindless table: {} textures registered, {} 2D / {} 2DArray / {} 3D / {} "
                 "cube / {} cubeArray entries hold an image ({} created for it so far), "
                 "unprimed(cpu_no_access={}, rt_flag={}, other={}), {} samplers registered, "
                 "{} writes",
                 registered, d2, d2_array, d3, cube, cube_array, state.primed_total,
                 unprimed_cpu_no_access, unprimed_rt_flag, unprimed_other,
                 static_cast<u32>(std::count_if(sampler_ids.begin(), sampler_ids.end(),
                                                [](u64 serial) { return serial != 0; })),
                 writes.size());
    }

    // Samplers: entry `id` is the sampler the game registered at that slot.
    thread_local std::vector<VkDescriptorImageInfo> sampler_infos;
    thread_local std::vector<std::pair<u32, u32>> sampler_runs;
    thread_local std::vector<std::pair<u32, u32>> sampler_writes;
    if (full_scan || !samplers_same) {
        const u32 sampler_count = static_cast<u32>((std::min<size_t>)(
            (std::max)(sampler_ids.size(), size_t{state.synced_sampler_count}),
            Shader::BINDLESS_SAMPLER_CAPACITY));
        sampler_infos.clear();
        sampler_runs.clear();
        for (u32 id = 0; id < sampler_count; ++id) {
            VkSampler sampler = *resources.fallback_sampler;
            if (id != 0 && id < sampler_ids.size() && sampler_ids[id] != 0) {
                sampler = resources.GetSampler(sampler_ids[id], false);
            }
            if (state.written_samplers[id] == sampler) {
                continue;
            }
            state.written_samplers[id] = sampler;
            sampler_infos.push_back({sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
            sampler_runs.emplace_back(id, static_cast<u32>(sampler_infos.size() - 1));
        }
        sampler_writes.clear();
        for (size_t i = 0; i < sampler_runs.size(); ++i) {
            if (!sampler_writes.empty() &&
                sampler_writes.back().first + sampler_writes.back().second == sampler_runs[i].first) {
                ++sampler_writes.back().second;
            } else {
                sampler_writes.emplace_back(sampler_runs[i].first, 1u);
            }
        }
        size_t info_index = 0;
        for (const auto& [first, run_count] : sampler_writes) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = Shader::BINDLESS_SAMPLER_BINDING,
                .dstArrayElement = first,
                .descriptorCount = run_count,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .pImageInfo = sampler_infos.data() + info_index,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            });
            info_index += run_count;
        }
        state.synced_sampler_snapshot = submission->sampler_pool_snapshot;
        state.synced_samplers = sampler_ids.data();
        state.synced_sampler_size = sampler_ids.size();
        state.synced_samplers_epoch = resources.samplers_epoch;
        state.synced_sampler_count = static_cast<u32>(
            (std::min<size_t>)(sampler_ids.size(), Shader::BINDLESS_SAMPLER_CAPACITY));
    }
    if (!writes.empty()) {
        impl.device.GetLogical().UpdateDescriptorSets(writes, {});
    }
}

void DrawContext::BindlessPrime(DrawPlan& plan) {
    BindlessState& state = bindless;
    // What a submission may create: the budget is memory and a count (a create is a pass break
    // and an upload), so a pool of thousands of textures fills over a few frames, not at once.
    static const u64 budget_bytes = [] {
        const char* text = std::getenv("EDEN_NX2_NVN_BINDLESS_PRIME_MB");
        return (text ? std::strtoull(text, nullptr, 10) : 128ull) << 20;
    }();
    constexpr u32 BUDGET_COUNT = 256;
    const u64 seq = host->HostFlight().seq;
    if (state.prime_seq != seq) {
        state.prime_seq = seq;
        state.prime_bytes = 0;
        state.prime_count = 0;
    }
    if (budget_bytes == 0 || state.unprimed_slots.empty()) {
        return;
    }
    const u64 device_budget = impl.device.GetDeviceMemoryBudget();
    u64 device_used =
        device_budget != 0
            ? impl.device.GetDeviceMemoryUsage() + impl.device.GetDeviceMemorySlack()
            : 0;
    if (device_budget != 0 &&
        (device_used >= device_budget - device_budget / 8 ||
         impl.storage_cache_bytes >= impl.storage_cache_budget)) {
        impl.TrimStorageCache();
        device_used = impl.device.GetDeviceMemoryUsage() + impl.device.GetDeviceMemorySlack();
    }
    if (device_budget != 0 && device_used >= device_budget) {
        if (state.prime_paused_logs++ % 600 == 0) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] bindless prime paused at VRAM budget: device {}/{} MiB, "
                     "storage cache {}/{} MiB, {} slots queued",
                     device_used >> 20, device_budget >> 20, impl.storage_cache_bytes >> 20,
                     impl.storage_cache_budget >> 20, state.unprimed_slots.size());
        }
        return;
    }
    u64 sub_budget_bytes = budget_bytes;
    u32 sub_budget_count = BUDGET_COUNT;
    const bool device_pressure =
        device_budget != 0 && device_used > device_budget - (device_budget * 15) / 100;
    if (impl.storage_cache_bytes >= impl.storage_cache_budget || device_pressure) {
        sub_budget_bytes = (std::min)(sub_budget_bytes, u64{16} << 20);
        sub_budget_count = (std::min)(sub_budget_count, 32u);
    }
    if (state.prime_count >= sub_budget_count || state.prime_bytes >= sub_budget_bytes) {
        return;
    }
    const std::vector<u64>& texture_ids = submission->texture_pool_snapshot.Values();
    while (!state.unprimed_slots.empty()) {
        const u32 id = state.unprimed_slots.back();
        std::shared_ptr<Storage> storage;
        if (id != 0 && id < texture_ids.size() && texture_ids[id] != 0 &&
            state.entries[id].serial == texture_ids[id]) {
            storage = state.entries[id].storage_ref.lock();
        }
        if (!storage || !NeedsBindlessPrime(impl, *storage)) {
            if (id < state.unprimed_queued.size()) {
                state.unprimed_queued[id] = 0;
            }
            state.unprimed_slots.pop_back();
            continue;
        }
        const u64 vram_bytes = storage->vram_bytes;
        if (state.prime_bytes + vram_bytes > sub_budget_bytes && state.prime_count != 0) {
            return;
        }
        if (id < state.unprimed_queued.size()) {
            state.unprimed_queued[id] = 0;
        }
        state.unprimed_slots.pop_back();
        const VkImageLayout layout = ReadLayoutOf(*storage);
        plan.UseImage(std::move(storage), layout, true, false, false, false, VK_FORMAT_UNDEFINED,
                      false, true);
        state.prime_bytes += vram_bytes;
        ++state.primed_total;
        state.dirty = true;
        if (++state.prime_count >= sub_budget_count || state.prime_bytes >= sub_budget_bytes) {
            break;
        }
    }
}

bool DrawContext::BindlessPrepare(DrawPlan& plan, u32 types_mask) {
    if (!BindlessReady()) {
        return false;
    }
    const u32 effective_mask = types_mask != 0 ? types_mask : ~u32{0};
    if ((bindless.requested_types_mask | effective_mask) != bindless.requested_types_mask) {
        bindless.requested_types_mask |= effective_mask;
        bindless.types_mask_dirty = true;
        LOG_INFO(Render_Vulkan, "[nvn] bindless table: requested texture types mask -> {:#x}",
                 bindless.requested_types_mask);
    }
    BindlessSync(false, &plan);
    BindlessPrime(plan);
    // A shader may read any entry: the images of the table must be in the layout the descriptors
    // name. Those a render pass or a copy left in another layout are moved like any sampled image
    // (a render target of this very draw stays: it cannot be both).
    size_t out = 0;
    for (size_t i = 0; i < impl.bindless_layout_dirty.size(); ++i) {
        std::shared_ptr<Storage>& storage = impl.bindless_layout_dirty[i];
        Storage& s = *storage;
        if (s.bindless_active_count == 0 || !s.image) {
            s.bindless_layout_queued = false;
            continue;
        }
        const VkImageLayout wanted = ReadLayoutOf(s);
        if (s.current_layout == wanted) {
            s.bindless_layout_queued = false;
            continue;
        }
        if (!IsBoundTarget(&s)) {
            plan.UseImage(storage, wanted, true, false, false, false, VK_FORMAT_UNDEFINED, true);
        }
        if (out != i) {
            impl.bindless_layout_dirty[out] = std::move(storage);
        }
        ++out;
    }
    impl.bindless_layout_dirty.resize(out);
    return true;
}

void DrawContext::BindlessBind(PipelineBase& pipe, VkPipelineBindPoint bind_point) {
    if (!pipe.layout || bindless.set == VK_NULL_HANDLE) {
        return;
    }
    const bool track =
        bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS && pass_open && !Debug::Get().no_state_elide;
    if (bindless.dirty) {
        // The plan created images before this draw recorded: their entries are written before
        // the set is bound (update after bind), so this very draw reads them.
        bindless.dirty = false;
        BindlessSync(true);
    } else if (track && bound_state.TakeBindlessSet(pipe.layout, bindless.set)) {
        return;
    }
    if (track) {
        static_cast<void>(bound_state.TakeBindlessSet(pipe.layout, bindless.set));
    }
    const VkDescriptorSet set = bindless.set;
    host->HostCmd().BindDescriptorSets(bind_point, pipe.layout, Shader::BINDLESS_SET,
                                       std::span{&set, 1}, {});
}

} // namespace Vulkan::Nvn
