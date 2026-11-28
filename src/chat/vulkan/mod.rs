//! The Vulkan backend: shared executor machinery for the registered
//! families. There is no central `model_type` dispatch here — every
//! registration file in `src/chat/registrations/` owns its own Vulkan
//! loader entry (the generic transformer families build a
//! `VulkanTransformerExecutor` around their declared feature set; phi builds
//! its dedicated executor). A family only touches this module when it needs
//! executor features no existing piece covers.
//!
//! The split is structural, not accidental: the llama-shaped families share
//! one block topology (RMSNorm, sequential residuals, gate/up activations),
//! so they parameterize the single generic executor through their declared
//! `TransformerFeatures`. Phi runs a different topology (biased LayerNorms,
//! parallel residual, partial rotary, plain `fc1`/`fc2` MLP) — expressing it
//! as more feature flags inside the generic executor would reintroduce the
//! family branching in core code that the registration mode removed, so phi
//! keeps its own compute graph in its registration file.
//!
//! - `context` — the family-free forward machinery (expert residency, GPU
//!   slot paging, KV cache container): the `OpsToolbox` analogue every
//!   executor composes
//! - `transformer` — the generic transformer executor (llama-shaped families)
//! - `weights` — the shared weight layer + precision-registry dispatch
//!
//! Phi has no module here: its executor lives in its registration file
//! (`src/chat/registrations/phi.rs`), the same one-family-one-file rule the
//! native side follows.

#[cfg(feature = "vulkan")]
pub(crate) mod context;
#[cfg(feature = "vulkan")]
#[allow(dead_code)] // additive catalog: callers arrive with model-forward adoption
pub(crate) mod ops;
#[cfg(feature = "vulkan")]
pub(crate) mod transformer;
pub(crate) mod weights;

#[cfg(feature = "vulkan")]
use crate::color::{self, Color};
use serde_json::Value;

/// Read a required usize field from a model config.
pub(crate) fn required_usize(config: &Value, name: &str) -> Result<usize, String> {
    config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Model config is missing a valid {name}."))
        .and_then(|value| usize::try_from(value).map_err(|_| format!("{name} does not fit usize.")))
}

/// Bootstrap the Vulkan substrate for one executor load and guarantee
/// symmetric teardown: initialize the backend, validate the shader
/// pipelines, run the caller's `load_initialized`, and shut the substrate
/// back down whenever the load fails — reporting a second failure if the
/// cleanup itself errors (a leak here would poison every later load attempt).
/// Executors call this from their `load` entry with a closure that captures
/// their own paths/config/plan.
#[cfg(feature = "vulkan")]
pub(crate) fn load_with_substrate<T>(
    load_initialized: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let result = (|| {
        crate::native::vulkan::initialize()?;
        crate::native::vulkan::prepare()?;
        load_initialized()
    })();
    if result.is_err()
        && let Err(shutdown_error) = crate::native::vulkan::shutdown()
    {
        eprintln!(
            "{}",
            color::stderr(
                &format!(
                    "cheetah: Vulkan cleanup after model startup failure also failed ({shutdown_error})."
                ),
                Color::Red
            )
        );
    }
    result
}
