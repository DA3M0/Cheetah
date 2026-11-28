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

/// The gate·up activation a feed-forward applies, as resolved from the
/// model config's `hidden_act`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activation {
    Silu,
    Gelu,
    NewGelu,
}

/// Applies the configured activation (silu / gelu / tanh-gelu) to the
/// gate·up pair as one shader call.
pub(crate) fn apply_activation(
    activation: Activation,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>, String> {
    match activation {
        Activation::Silu => cheetah::native::vulkan::silu_mul(gate, up),
        Activation::Gelu => cheetah::native::vulkan::gelu_mul(gate, up, false),
        Activation::NewGelu => cheetah::native::vulkan::gelu_mul(gate, up, true),
    }
}

/// Sequence-level input validation shared by every transformer executor:
/// non-empty ids, vocabulary bounds, position limits, and cache/offset
/// consistency (`cached_elements` is the per-layer KV element count). The
/// error strings are part of the surface — keep them identical when touching.
pub(crate) fn validate_sequence(
    tokens: &[u32],
    offset: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    kv_heads: usize,
    head_dim: usize,
    cached_elements: usize,
) -> Result<usize, String> {
    let sequence = tokens.len();
    if sequence == 0 {
        return Err("Vulkan transformer requires a non-empty token sequence.".into());
    }
    let last_position = offset
        .checked_add(sequence)
        .ok_or_else(|| "Vulkan transformer sequence position overflow.".to_string())?;
    if last_position > max_position_embeddings {
        return Err("Vulkan transformer input exceeds max_position_embeddings.".into());
    }
    let expected_cache = offset
        .checked_mul(kv_heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| "Vulkan transformer cache size overflow.".to_string())?;
    if cached_elements != expected_cache {
        return Err(
            "Vulkan transformer KV cache position does not match the input offset.".into(),
        );
    }
    for token_id in tokens {
        let token = *token_id as usize;
        if token >= vocab_size {
            return Err(format!("Token ID {token} exceeds the model vocabulary."));
        }
    }
    Ok(sequence)
}

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
