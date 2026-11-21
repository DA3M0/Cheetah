//! The Vulkan backend: shared executor machinery for the registered
//! families. There is no central `model_type` dispatch here — every
//! registration file in `src/chat/registrations/` owns its own Vulkan
//! loader entry (the generic transformer families build a
//! `VulkanTransformerExecutor` around their declared feature set; phi builds
//! its dedicated executor). A family only touches this module when it needs
//! executor features no existing piece covers.
//!
//! - `transformer` — the generic transformer executor (llama-shaped families)
//! - `phi` — the phi executor (different block topology)
//! - `weights` — the shared weight layer + precision-registry dispatch

#[cfg(feature = "vulkan")]
pub(crate) mod phi;
#[cfg(feature = "vulkan")]
pub(crate) mod transformer;
pub(crate) mod weights;

use serde_json::Value;

/// Read a required usize field from a model config.
pub(crate) fn required_usize(config: &Value, name: &str) -> Result<usize, String> {
    config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Model config is missing a valid {name}."))
        .and_then(|value| usize::try_from(value).map_err(|_| format!("{name} does not fit usize.")))
}
