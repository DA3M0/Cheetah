#[cfg(feature = "vulkan")]
use super::ModelRuntime;
#[cfg(feature = "vulkan")]
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = vulkan_config(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load(
            weight_paths,
            &config,
            plan,
        )?,
    ))
}

#[cfg(feature = "vulkan")]
fn vulkan_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{TransformerFeatures, VulkanTransformerConfig};

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
        return Err("Gemma does not support MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Gemma");
    features.norm_weight_offset = true;
    features.embedding_scale = true;
    features.tie_word_embeddings = true;
    features.max_position_embeddings_default = Some(4096);
    features.rms_norm_eps_default = Some(1e-6);
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.attention_output_bias = features.attention_bias;
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    #[test]
    fn gemma_vulkan_features_configure_norm_and_embedding_behavior() {
        let config = vulkan_config(&serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "gelu",
        }))
        .unwrap();
        assert!(config.features.norm_weight_offset);
        assert!(config.features.embedding_scale);
        assert!(config.features.tie_word_embeddings);
        assert_eq!(config.features.max_position_embeddings_default, Some(4096));
        assert_eq!(config.features.rms_norm_eps_default, Some(1e-6));
    }
}
