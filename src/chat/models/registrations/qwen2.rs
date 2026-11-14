//! Declarative registration for `qwen2`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::qwen2`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::qwen;
use super::qwen2::load;
#[cfg(feature = "vulkan")]
use super::qwen2::load_vulkan;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["Qwen2ForCausalLM", "qwen2"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "qwen",
        render: qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    backends: model_backends(),
    family: Some(sha_family()),
    shaders: &["shared_ops"],
};

/// Family constants for the Qwen2 family: fallback inference
/// parameters and the known hidden-size profiles of published sizes
/// (the q/kv/head-dim split is not derivable from tensor shapes).
const fn sha_family() -> super::ModelFamily {
    super::ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (896, 151936, 14, 2, 64),   // Qwen2-0.5B
            (1536, 151936, 12, 2, 128), // Qwen2-1.5B
            (3584, 152064, 28, 4, 128), // Qwen2-7B
        ],
        extra_config_json:
            "{\"hidden_act\":\"silu\",\"sliding_window\":4096,\"use_sliding_window\":false,\"max_window_layers\":21}",
    }
}

/// Backends registered for this architecture: the Candle loader always, the
/// native mmap backend (real Ant tiering) for supported architectures, and
/// Vulkan when compiled in.
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
