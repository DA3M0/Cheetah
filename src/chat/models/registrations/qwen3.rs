//! Declarative registration for `qwen3`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::qwen3`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::qwen;
use super::qwen3::load;
#[cfg(feature = "vulkan")]
use super::qwen3::load_vulkan;

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

/// Family constants for the Qwen3 family: fallback inference parameters and
/// the known hidden/vocab profiles of published sizes (q_norm/k_norm are
/// carried by every published size; per-head RMSNorm precedes rope).
const fn qwen3_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (1024, 151936, 16, 8, 64),  // Qwen3-0.6B (tied)
            (2048, 151936, 16, 8, 128), // Qwen3-1.7B (tied)
            (2560, 151936, 32, 8, 80),  // Qwen3-4B (tied)
            (4096, 151936, 32, 8, 128), // Qwen3-8B (untied)
        ],
        extra_config_json: "",
    }
}
