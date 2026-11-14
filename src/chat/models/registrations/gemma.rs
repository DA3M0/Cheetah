//! Declarative registration for `gemma`.
//!
//! build.rs collects every file in `src/chat/models/registrations/` into the
//! generated `REGISTRY`; the runtime implementations live in `super::gemma`.
//! This file intentionally contains no runtime code.

use super::{BackendLoader, ChatTemplate, ModelRegistration};
use super::templates::gemma;
use super::gemma::load;
#[cfg(feature = "vulkan")]
use super::gemma::load_vulkan;

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
    family: None,
    shaders: &["shared_ops"],
};

/// Backends registered for this architecture: the Candle loader always, the
/// native mmap backend (real Ant tiering) for supported architectures, and
/// Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("candle", BackendLoader::Candle(load)),            ("vulkan", BackendLoader::Vulkan(load_vulkan)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[
            ("candle", BackendLoader::Candle(load)),        ]
    }
}
