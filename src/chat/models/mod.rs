pub(super) mod native;
mod registry;
mod templates;

// Model runtime implementations. Backend capability is resolved from the
// architecture at load time; there is no build-time model registration.
mod gemma;
mod llama;
mod mistral;
mod mixtral;
mod phi;
mod qwen2;
mod qwen3;

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
    /// Vulkan compute path: receives the aligned shard paths.
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
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

impl BackendLoader {
    /// Test/registry diagnostic: the backend name this loader serves.
    #[cfg(test)]
    pub(super) fn name(&self) -> &'static str {
        match self {
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
    /// Family-specific config.json fields
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

pub(crate) struct ShaderRegistration {
    name: &'static str,
    path: Option<&'static str>,
}

include!(concat!(env!("OUT_DIR"), "/shader_registry.rs"));

pub(crate) fn shader_path(name: &str) -> Option<&'static str> {
    SHADERS
        .iter()
        .find(|shader| shader.name == name)
        .and_then(|shader| shader.path)
}

pub(super) fn shader_is_registered(name: &str) -> bool {
    shader_path(name).is_some() || SHADERS.iter().any(|shader| shader.name == name)
}

/// Chat prompt renderer resolved from the architecture name (init-time
/// metadata keys prompt families by architecture).
pub(crate) fn chat_template_for_architecture(architecture: &str) -> Option<ChatTemplate> {
    let template = match architecture {
        "llama" | "LlamaForCausalLM" => ChatTemplate {
            id: "llama",
            render: templates::llama2,
            header_renderer: Some(templates::llama3),
        },
        "qwen2" | "Qwen2ForCausalLM" | "qwen3" | "Qwen3ForCausalLM" => ChatTemplate {
            id: "qwen",
            render: templates::qwen,
            header_renderer: None,
        },
        "gemma" | "GemmaForCausalLM" => ChatTemplate {
            id: "gemma",
            render: templates::gemma,
            header_renderer: None,
        },
        "mistral" | "MistralForCausalLM" => ChatTemplate {
            id: "mistral",
            render: templates::mistral,
            header_renderer: None,
        },
        "mixtral" | "MixtralForCausalLM" => ChatTemplate {
            id: "mixtral",
            render: templates::mixtral,
            header_renderer: None,
        },
        "phi" | "PhiForCausalLM" => ChatTemplate {
            id: "phi",
            render: templates::phi,
            header_renderer: None,
        },
        _ => return None,
    };
    Some(template)
}

/// Stop marker token names resolved from the architecture name.
pub(crate) fn stop_markers_for_architecture(architecture: &str) -> &'static [&'static str] {
    match architecture {
        "llama" | "LlamaForCausalLM" => &["<|eot_id|>", "</s>"],
        "qwen2" | "Qwen2ForCausalLM" | "qwen3" | "Qwen3ForCausalLM" => &["<|im_end|>"],
        "gemma" | "GemmaForCausalLM" => &["<end_of_turn>"],
        "mistral" | "MistralForCausalLM" => &["</s>"],
        "mixtral" | "MixtralForCausalLM" => &["</s>"],
        "phi" | "PhiForCausalLM" => &["</s>"],
        _ => &[],
    }
}

pub(crate) use registry::{
    backend_device, backend_loader_for_architecture, family_for_architecture, load_model,
    load_model_for_device, registered_backends as registered_backend_names,
};
