pub(super) mod native;
mod registry;
mod templates;

// Model runtime implementations. Every file in `registrations/` declares one
// family's capability (architectures, chat template, stop markers, backends,
// family constants); build.rs collects them into the generated `REGISTRY`
// below, and the runtime lookups derive from it — adding a family is adding
// a registration file, no table edits. The Vulkan side is one generic
// loader: executor features derive from the model's own `model_type`.
mod vulkan_executors;

use serde_json::Value;
use std::path::PathBuf;

/// Backend device descriptor for engine bookkeeping and backend dispatch.
/// Host-resident backends (the native mmap backend, the Vulkan
/// executor) run against the CPU device; real accelerator handles live inside
/// backend executors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Device {
    Cpu,
}

#[derive(Clone, Copy)]
pub(super) struct ChatTemplate {
    pub(super) id: &'static str,
    pub(super) render: PromptRenderer,
    pub(super) header_renderer: Option<PromptRenderer>,
}

pub(super) type PromptRenderer = fn(&[super::ChatMessage], Option<&str>) -> String;

pub(super) trait ModelRuntime: ModelRuntimeClone + Send {
    /// Run the model over `tokens` starting at absolute position `offset` and
    /// return the logits of the final position. Backend-agnostic inference boundary.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String>;
    fn clear_cache(&mut self);
    fn device_name(&self) -> &'static str;
    /// Ant: real tier transitions on backends that own mmap-backed weights
    /// (the native CPU backend); host-resident backends report None.
    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        None
    }
}

/// Ant: tier mutations that act on real memory pages, not just bookkeeping.
pub(super) trait AntTierControl {
    fn page_in_unit(&mut self, unit_id: &str, tier: i32) -> bool;
    fn evict_unit(&mut self, unit_id: &str) -> bool;
}

pub(super) trait ModelRuntimeClone {
    fn clone_box(&self) -> Box<dyn ModelRuntime>;
}

impl<T> ModelRuntimeClone for T
where
    T: ModelRuntime + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ModelRuntime> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelRuntime> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// One inference backend implementation per architecture. Backend capability
/// is resolved by architecture name through the runtime backend tables in
/// `registry`; selection happens by backend name.
#[derive(Clone, Copy)]
pub(super) enum BackendLoader {
    /// Vulkan compute path: receives the aligned shard paths. Only built when
    /// the `vulkan` feature is compiled in (`vulkan_loader_for_architecture`).
    #[allow(dead_code)]
    Vulkan(
        fn(
            &[PathBuf],
            &[u8],
            Value,
            &std::collections::BTreeMap<String, String>,
        ) -> Result<Box<dyn ModelRuntime>, String>,
    ),
    /// Native mmap + Ant tiering: receives the prepared model directory, the
    /// resolved residency plan, and the family constants for parameter
    /// fallbacks.
    Native(
        fn(
            &std::path::Path,
            &[u8],
            &Value,
            &std::collections::BTreeMap<String, String>,
            ModelFamily,
        ) -> Result<Box<dyn ModelRuntime>, String>,
    ),
}

/// Family-level inference constants for a model family: used as fallbacks when
/// the model directory does not carry them (missing or incomplete
/// config.json), keeping the native backend independent of directory
/// completeness.
#[derive(Clone, Copy)]
pub(super) struct ModelFamily {
    pub(super) rope_theta: f64,
    pub(super) rms_norm_eps: f64,
    pub(super) tie_word_embeddings: bool,
    /// Known size profiles for this family, keyed by `(hidden_size, vocab_size)`:
    /// `(q_heads, kv_heads, head_dim)`. The q/kv/head-dim split cannot be
    /// derived from tensor shapes alone (and hidden size alone can be ambiguous
    /// within a family), so it is declared here.
    pub(super) profiles: &'static [(usize, usize, usize, usize, usize)],
    /// Family-specific config.json fields
    /// require (e.g. hidden_act / sliding_window), merged into a synthesized
    /// config when the model directory has none. Empty JSON object or "" = none.
    pub(super) extra_config_json: &'static str,
    /// Family-level default context length (`max_position_embeddings`). `None`
    /// means the family declares no default; config.json / the units manifest
    /// then govern (with a generic fallback in the engine).
    pub(super) max_position_embeddings: Option<usize>,
    /// Sparse-MoE expert naming layout for this family (Mixtral-compatible
    /// defaults). Drives router + expert weight slicing so the native forward
    /// never hardcodes one architecture's naming.
    pub(super) moe_layout: native::decoder::MoEExpertLayout,
    /// Dense-architecture deltas over the llama-shaped baseline (norm kind,
    /// activation, projection naming, residual topology, partial rope) for the
    /// native forward pass.
    pub(super) features: native::decoder::DenseFeatures,
}

impl ModelFamily {
    pub(super) fn profile_for(
        &self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Option<(usize, usize, usize)> {
        self.profiles
            .iter()
            .find(|(hidden, vocab, _, _, _)| *hidden == hidden_size && *vocab == vocab_size)
            .map(|(_, _, q_heads, kv_heads, head_dim)| (*q_heads, *kv_heads, *head_dim))
    }
}

/// One declarative model registration: what a family serves (architecture
/// names, chat template, stop markers), which backends back it, and the
/// family constants for native parameter fallbacks. Entries live in
/// `src/chat/models/registrations/<family>.rs` and carry no runtime code;
/// build.rs assembles them into `REGISTRY` (generated below).
#[derive(Clone, Copy)]
pub(super) struct ModelRegistration {
    /// Architecture names this registration serves (`Qwen2ForCausalLM` and
    /// `qwen2` style names alike; lookups normalize case + suffix).
    pub(super) architectures: &'static [&'static str],
    pub(super) chat_template: ChatTemplate,
    /// Generation stop markers for this family.
    pub(super) stop_markers: &'static [&'static str],
    /// Backend loaders keyed by backend name (`native`, `vulkan`), in
    /// declared order — the order `registered_backend_names` reports.
    pub(super) backends: &'static [(&'static str, BackendLoader)],
    /// Family constants for the native backend's parameter fallback chain;
    /// `None` for families the native backend does not serve.
    pub(super) family: Option<ModelFamily>,
}

impl ModelRegistration {
    /// Whether this registration serves the architecture name (normalized:
    /// case-insensitive, `ForCausalLM` suffix stripped on both sides).
    pub(super) fn serves(&self, architecture: &str) -> bool {
        let normalized = registry::normalize_architecture(architecture);
        self.architectures
            .iter()
            .any(|name| registry::normalize_architecture(name) == normalized)
    }

    /// Backend loader by name, case-insensitive.
    pub(super) fn loader_for(&self, backend: &str) -> Option<BackendLoader> {
        self.backends
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(backend))
            .map(|(_, loader)| *loader)
    }

    /// Registered backend names, in declared order.
    pub(super) fn backend_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backends.iter().map(|(name, _)| *name)
    }
}

// Generated by build.rs from src/chat/models/registrations/: the
// `<family>_registration` modules plus `REGISTRY` / `ALL_ARCHITECTURES`.
include!(concat!(env!("OUT_DIR"), "/model_registry.rs"));

/// Chat prompt renderer resolved from the architecture name through the
/// declarative registrations (init-time metadata keys prompt families by
/// architecture).
pub(crate) fn chat_template_for_architecture(architecture: &str) -> Option<ChatTemplate> {
    registry::registration_for_architecture(architecture)
        .ok()
        .map(|registration| registration.chat_template)
}

/// Stop marker token names resolved from the architecture name through the
/// declarative registrations; an unknown architecture stops on nothing.
pub(crate) fn stop_markers_for_architecture(architecture: &str) -> &'static [&'static str] {
    registry::registration_for_architecture(architecture)
        .map(|registration| registration.stop_markers)
        .unwrap_or(&[])
}

pub(crate) use registry::{
    backend_device, backend_loader_for_architecture, family_for_architecture, load_model,
    load_model_for_device, registered_backends as registered_backend_names,
};
