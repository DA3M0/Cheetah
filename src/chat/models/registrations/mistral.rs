//! Declarative registration for `mistral`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::mistral`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::mistral;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["MistralForCausalLM", "mistral"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "mistral",
        render: mistral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(mistral_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(super::native::native_family_loader)),
            ("vulkan", BackendLoader::Vulkan(super::mistral::load_vulkan)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(super::native::native_family_loader))]
    }
}

/// Family constants for the Mistral family: the llama-shaped baseline with
/// the decoding-time sliding window (the HF Mistral modeling masks keys
/// older than `sliding_window` during cached decoding, which the native
/// decoder implements as the uniform window).
const fn mistral_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32000, 32, 8, 128), // Mistral-7B / 7B-Instruct-v0.2 (GQA)
            (4096, 32768, 32, 8, 128), // Mistral-7B-v0.3 (extended vocab)
        ],
        extra_config_json: "{\"hidden_act\":\"silu\",\"sliding_window\":4096}",
        max_position_embeddings: Some(32768),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}
