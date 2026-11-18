//! Native CPU inference backend. Weights stay mmap-backed from
//! the prepared Ant shards and every Ant tier acts on real pages via madvise,
//! so the resident set only holds what the plan keeps resident.

pub(super) mod decoder;
// kernels and store are reached by the precision operator registry
// (src/chat/precision/<format>.rs wires the registered gemv operators).
pub(crate) mod kernels;
pub(crate) mod store;

use super::{AntTierControl, ModelRuntime};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Registry-facing loader for the native decoder-family CPU backend. The
/// llama-shaped baseline (llama / mixtral / qwen2 / qwen3: GQA attention,
/// rotate-half rope, silu MLP) plus the dense delta families mistral (sliding
/// window), gemma (offset norms + embedding scale + GeGLU) and phi
/// (LayerNorm + parallel residual + fused projections); attention biases and
/// tied embeddings are per-family. The backend capability tables expose
/// this as the native backend loader, so selecting the native backend goes
/// through the same backend dispatch as the Vulkan loader.
pub(super) fn native_family_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &serde_json::Value,
    plan: &BTreeMap<String, String>,
    family: super::ModelFamily,
) -> Result<Box<dyn ModelRuntime>, String> {
    let store = Arc::new(store::MmapWeightStore::open(model_dir)?);
    let cfg = decoder::DecoderConfig::resolve(Some(raw_config), &store, Some(family))?;
    let mut decoder = decoder::DecoderNative::with_store(cfg, Arc::clone(&store));
    decoder.apply_tiers(plan);
    Ok(Box::new(NativeCpuRuntime { decoder }))
}

pub(super) struct NativeCpuRuntime {
    decoder: decoder::DecoderNative,
}

impl Clone for NativeCpuRuntime {
    fn clone(&self) -> Self {
        Self {
            decoder: self.decoder.clone(),
        }
    }
}

impl ModelRuntime for NativeCpuRuntime {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.decoder.forward(tokens, offset)
    }

    fn clear_cache(&mut self) {
        self.decoder.clear_kv();
    }

    fn device_name(&self) -> &'static str {
        "native-cpu"
    }

    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

impl AntTierControl for NativeCpuRuntime {
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.decoder.page_in_unit(unit_id)
    }

    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.decoder.evict_unit(unit_id)
    }
}
