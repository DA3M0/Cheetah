//! Declarative registration for `phi`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::phi`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::phi;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["PhiForCausalLM", "phi"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "phi",
        render: phi,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(phi_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and the dedicated `VulkanPhiExecutor` when compiled
/// in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(super::native::native_family_loader)),
            ("vulkan", BackendLoader::Vulkan(crate::chat::vulkan::load)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(super::native::native_family_loader))]
    }
}

/// Family constants for the Phi family: biased LayerNorms, a parallel
/// attention/MLP residual, fused `qkv_proj` + `dense` attention, `fc1`/`fc2`
/// MLP, and partial rotary embeddings (the exact rotary dim resolves per
/// model from `partial_rotary_factor * head_dim`; phi-1.5 carries 0.5 and
/// phi-2 carries 0.4 in their configs, so the family constant is the phi-2
/// fallback).
const fn phi_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (2048, 50257, 32, 32, 64), // phi-1.5 (head_dim 2048/32)
            (2048, 51200, 32, 32, 64), // phi-1.5 padded vocab
            (2560, 51200, 32, 32, 80), // phi-2 (head_dim 2560/32)
        ],
        extra_config_json: "{\"hidden_act\":\"gelu_new\"}",
        max_position_embeddings: Some(2048),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::PHI_FEATURES,
    }
}
