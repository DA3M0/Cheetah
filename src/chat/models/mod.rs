mod candle;
pub(super) mod native;
mod registry;
mod templates;

// Model runtime implementations. Their declarative registrations live in
// src/chat/models/registrations/ and are wired into REGISTRY by build.rs.
mod gemma;
mod llama;
mod mistral;
mod mixtral;
mod phi;
mod qwen2;
mod qwen3;

use candle_core::Device;
use candle_nn::VarBuilder;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub(super) struct ChatTemplate {
    pub(super) id: &'static str,
    pub(super) render: PromptRenderer,
    pub(super) header_renderer: Option<PromptRenderer>,
}

pub(super) type PromptRenderer = fn(&[super::ChatMessage], Option<&str>) -> String;

pub(super) trait ModelRuntime: ModelRuntimeClone + Send {
    /// Run the model over `tokens` starting at absolute position `offset` and
    /// return the logits of the final position. Backend-agnostic: no Candle
    /// types cross this boundary.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String>;
    fn clear_cache(&mut self);
    fn device_name(&self) -> &'static str;
    /// Ant: real tier transitions on backends that own mmap-backed weights
    /// (the native CPU backend); candle-backed models report None.
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

/// One inference backend registered for an architecture. Each variant carries
/// the loader signature natural to that backend; selection happens by backend
/// name through `ModelRegistration::loader_for`.
#[derive(Clone, Copy)]
pub(super) enum BackendLoader {
    /// Candle: builds tensors from mapped shards through a VarBuilder.
    Candle(
        for<'a> fn(&[u8], Value, VarBuilder<'a>, &Device) -> Result<Box<dyn ModelRuntime>, String>,
    ),
    /// Vulkan compute path: receives the aligned shard paths.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    Vulkan(fn(
        &[PathBuf],
        &[u8],
        Value,
        &std::collections::BTreeMap<String, String>,
    ) -> Result<Box<dyn ModelRuntime>, String>),
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

impl BackendLoader {
    /// Test/registry diagnostic: the backend name this loader serves.
    #[cfg(test)]
    pub(super) fn name(&self) -> &'static str {
        match self {
            BackendLoader::Candle(_) => "candle",
            BackendLoader::Vulkan(_) => "vulkan",
            BackendLoader::Native(_) => "native",
        }
    }
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
    /// Family-specific config.json fields the Candle transformers structs
    /// require (e.g. hidden_act / sliding_window), merged into a synthesized
    /// config when the model directory has none. Empty JSON object or "" = none.
    pub(super) extra_config_json: &'static str,
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

pub(super) struct ModelRegistration {
    pub(super) architectures: &'static [&'static str],
    pub(super) chat_template: ChatTemplate,
    pub(super) stop_markers: &'static [&'static str],
    /// Backends registered for this architecture, each as `(name, loader)`.
    pub(super) backends: &'static [(&'static str, BackendLoader)],
    /// Family constants for backends that need fallback parameters.
    pub(super) family: Option<ModelFamily>,
    pub(super) shaders: &'static [&'static str],
}

impl ModelRegistration {
    /// The registered loader for one backend name (e.g. "candle", "vulkan",
    /// "native"); None when the architecture does not register that backend.
    pub(super) fn loader_for(&self, backend: &str) -> Option<BackendLoader> {
        self.backends
            .iter()
            .find(|(name, _)| *name == backend)
            .map(|(_, loader)| *loader)
    }

    /// Registered backend names, in declaration order.
    pub(super) fn backend_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backends.iter().map(|(name, _)| *name)
    }
}

pub(crate) struct ShaderRegistration {
    name: &'static str,
    path: Option<&'static str>,
}

include!(concat!(env!("OUT_DIR"), "/model_registry.rs"));
include!(concat!(env!("OUT_DIR"), "/shader_registry.rs"));

pub(crate) fn registration_for_architecture(
    architecture: &str,
) -> Result<&'static ModelRegistration, String> {
    REGISTRY
        .iter()
        .copied()
        .find(|registration| registration.architectures.contains(&architecture))
        .ok_or_else(|| format!("Unsupported model architecture '{architecture}'."))
}

/// The loader registered for one architecture/backend pair, if any.
pub(crate) fn backend_loader_for_architecture(
    architecture: &str,
    backend: &str,
) -> Option<BackendLoader> {
    registration_for_architecture(architecture)
        .ok()
        .and_then(|registration| registration.loader_for(backend))
}

/// Backend names registered for an architecture (e.g. `["candle", "native"]`).
pub(crate) fn registered_backend_names(architecture: &str) -> Vec<&'static str> {
    registration_for_architecture(architecture)
        .map(|registration| registration.backend_names().collect())
        .unwrap_or_default()
}

/// Family constants registered for an architecture, if any.
pub(crate) fn family_for_architecture(architecture: &str) -> Option<ModelFamily> {
    registration_for_architecture(architecture)
        .ok()
        .and_then(|registration| registration.family)
}

pub(crate) fn shader_path(name: &str) -> Option<&'static str> {
    SHADERS
        .iter()
        .find(|shader| shader.name == name)
        .and_then(|shader| shader.path)
}

pub(super) fn shader_is_registered(name: &str) -> bool {
    shader_path(name).is_some() || SHADERS.iter().any(|shader| shader.name == name)
}

pub(super) use registry::{backend_device, load_model, load_model_for_device};
