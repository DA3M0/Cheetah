//! Declarative registration for `qwen3`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::qwen3`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::qwen;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["Qwen3ForCausalLM", "qwen3"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "qwen",
        render: qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    backends: model_backends(),
    family: Some(qwen3_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(super::native::native_family_loader)),
            ("vulkan", BackendLoader::Vulkan(super::vulkan_executors::load)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(super::native::native_family_loader))]
    }
}

/// Family constants for the Qwen3 family: per-head q/k RMSNorm units are
/// detected from the prepared manifest, not declared here.
const fn qwen3_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (8192, 32768, 64, 8, 128), // Qwen3-32B GQA config
        ],
        extra_config_json: "{\"hidden_act\":\"silu\"}",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}
