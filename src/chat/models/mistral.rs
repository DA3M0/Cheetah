#[cfg(feature = "vulkan")]
use super::ModelRuntime;
#[cfg(feature = "vulkan")]
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<Box<dyn ModelRuntime>, String> {
    use super::super::vulkan_transformer::{
        TransformerFeatures, VulkanTransformerConfig, VulkanTransformerExecutor,
    };

    if raw_config
        .get("num_local_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        != 0
        || raw_config
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("Mistral config must not define MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Mistral");
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.attention_output_bias = features.attention_bias;
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let config = VulkanTransformerConfig::from_raw(&raw_config, features)?;
    Ok(Box::new(VulkanTransformerExecutor::load(
        weight_paths,
        &config,
        plan,
    )?))
}
