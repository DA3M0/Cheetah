//! Declarative registration for `llama`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::llama`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::{llama2, llama3};
use super::llama::load;
#[cfg(feature = "vulkan")]
use super::llama::load_vulkan;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["LlamaForCausalLM", "llama"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "llama",
        render: llama2,
        header_renderer: Some(llama3),
    },
    stop_markers: &["<|eot_id|>", "</s>"],
    backends: model_backends(),
    family: Some(llama_family()),
    shaders: &["shared_ops"],
};

/// Backends registered for this architecture: the Candle loader always, the
/// native mmap backend (real Ant tiering), and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("candle", BackendLoader::Candle(load)),
            ("native", BackendLoader::Native(super::native::native_family_loader)),
            ("vulkan", BackendLoader::Vulkan(load_vulkan)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[
            ("candle", BackendLoader::Candle(load)),
            ("native", BackendLoader::Native(super::native::native_family_loader)),
        ]
    }
}

/// Family constants for the Llama family: fallback inference parameters and
/// the known hidden/vocab profiles of published sizes (llama2 is MHA with a
/// 32k vocab, llama3 is GQA with a 128k vocab — same hidden size).
const fn llama_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (2048, 128256, 32, 8, 64),  // Llama-3.2-1B (tied)
            (3072, 128256, 24, 8, 128), // Llama-3.2-3B (tied)
            (4096, 128256, 32, 8, 128), // Llama-3-8B / 3.1-8B
            (4096, 32000, 32, 32, 128), // Llama-2-7B (MHA)
        ],
        extra_config_json: "",
    }
}
