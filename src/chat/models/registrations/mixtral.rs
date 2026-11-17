//! Declarative registration for `mixtral`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::mixtral`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::mixtral;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["MixtralForCausalLM", "mixtral"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "mixtral",
        render: mixtral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(mixtral_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering; the mixture-of-experts forward routes and mixes experts per
/// token) always, and Vulkan when compiled in.
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

/// Family constants for the Mixtral family: the sparse feed-forward resolves
/// expert counts from the prepared manifest; the MIXTRAL layout names the
/// router (`block_sparse_moe.gate`) and expert projections (`w1/w3/w2`).
const fn mixtral_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 8_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32768, 32, 8, 128), // Mixtral-8x7B (GQA: 32 q heads / 8 kv heads)
        ],
        extra_config_json: "",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}
