//! Family-free forward machinery for the Vulkan model executors: the
//! routing-aware expert-residency state machine, the GPU-slot paging
//! primitives, and the per-layer KV cache container.
//!
//! This is the Vulkan analogue of the native backend's `OpsToolbox`: the
//! substrate-facing machinery every executor composes, carrying no family
//! semantics. The decode graphs (the per-family layer stacking) live in the
//! registration files; this module is what they drive.

use std::collections::HashMap;

use super::weights::{Linear, upload_slot};

/// Ant bucketing for one weight tensor: page it between the GPU bucket
/// (persistent upload) and the host-streaming bucket. The host copy in
/// `WeightData` is always authoritative, so eviction only releases the
/// device allocation and a page-in re-uploads from it.
pub(crate) fn page_linear(linear: &mut Linear, resident: bool) {
    match (resident, &mut linear.slot) {
        (true, slot @ None) => {
            let row_bytes = (linear.input_size + 3) / 4 * 4;
            *slot = upload_slot(&linear.name, &linear.weight, row_bytes);
        }
        (false, slot @ Some(_)) => {
            if let Some(slot) = slot.take() {
                cheetah::native::vulkan::weight_release(slot.handle);
            }
        }
        _ => {}
    }
}

/// One sparse feed-forward expert: the three projections that page together
/// between the resident GPU bucket and the host-streaming bucket.
#[derive(Clone)]
pub(crate) struct Expert {
    pub(crate) gate: Linear,
    pub(crate) up: Linear,
    pub(crate) down: Linear,
}

impl Expert {
    /// Page all three projections of this expert together.
    pub(crate) fn page(&mut self, resident: bool) {
        page_linear(&mut self.gate, resident);
        page_linear(&mut self.up, resident);
        page_linear(&mut self.down, resident);
    }
}

/// Routing-aware per-expert residency and the substrate paging primitives.
/// The map is keyed `(layer_index, expert_index)` so it holds across
/// positions and generation steps; transitions act on real device
/// allocations (upload into the resident bucket / release it), never on the
/// authoritative host copies.
#[derive(Clone, Default)]
pub(crate) struct VulkanForwardContext {
    /// RC2-B: per-expert residency, kept in sync with the MoE router's
    /// expert selections while a generation walks the layers.
    expert_residency: HashMap<(usize, usize), bool>,
}

impl VulkanForwardContext {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// RC2-B: routing-aware per-expert residency for one layer's experts.
    /// Experts the current router selection no longer picks are evicted
    /// (release the resident GPU bucket); freshly picked experts are paged
    /// in (upload into the resident bucket). `resident` reports, per expert
    /// index, whether the router selection needs it resident.
    pub(crate) fn sync_expert_residency(
        &mut self,
        layer_index: usize,
        experts: &mut [Expert],
        resident: &[bool],
    ) {
        for (expert_index, expert) in experts.iter_mut().enumerate() {
            let resident = resident.get(expert_index).copied().unwrap_or(false);
            let key = (layer_index, expert_index);
            let was_resident = self.expert_residency.get(&key).copied().unwrap_or(false);
            if was_resident != resident {
                expert.page(resident);
                self.expert_residency.insert(key, resident);
            }
        }
    }
}

/// One layer's KV cache: the appended keys/values rows the attention shader
/// reads. The host vectors are authoritative storage; the container just
/// keeps the two sides in lockstep.
#[derive(Clone, Default)]
pub(crate) struct GpuKvCache {
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
}

impl GpuKvCache {
    /// Append one position's key/value rows (each `kv_dim` long).
    pub(crate) fn append(&mut self, keys: &[f32], values: &[f32]) {
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
    }

    /// Drop the cached rows between generations.
    pub(crate) fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
}
