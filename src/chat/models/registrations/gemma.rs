//! Declarative registration for `gemma`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::gemma`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::gemma;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["GemmaForCausalLM", "gemma"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "gemma",
        render: gemma,
        header_renderer: None,
    },
    stop_markers: &["<end_of_turn>"],
    backends: model_backends(),
    family: Some(gemma_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(super::native::native_family_loader)),
            ("vulkan", BackendLoader::Vulkan(super::gemma::load_vulkan)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(super::native::native_family_loader))]
    }
}

/// Family constants for the Gemma family: offset RMSNorms `x/rms · (1 + w)`,
/// a sqrt(hidden)-scaled embedding, GeGLU (tanh-GELU gated) feed-forward, and
/// tied embeddings. gemma-1 applies full attention on every layer (the
/// config's `sliding_window` is not enforced by the HF Gemma modeling), so no
/// window is declared here.
const fn gemma_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (2048, 256128, 8, 1, 256),  // gemma-2b / 2b-it
            (2048, 256000, 8, 1, 256),  // gemma-2b mirrors with padded-down vocab
            (3072, 256000, 16, 1, 256), // gemma-7b / 7b-it
            (3072, 256128, 16, 1, 256), // gemma-1.1-7b
        ],
        extra_config_json: "{\"hidden_act\":\"gelu_pytorch_tanh\"}",
        max_position_embeddings: Some(8192),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::GEMMA_FEATURES,
    }
}
