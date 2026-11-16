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
    use super::super::vulkan_transformer::{
        TransformerFeatures, VulkanTransformerConfig, required_usize,
    };

    let hidden_size = required_usize(raw_config, "hidden_size")?;
    let num_attention_heads = required_usize(raw_config, "num_attention_heads")?;
    let head_dim = raw_config
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|_| "head_dim does not fit usize.".to_string())?
        .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
    let mut features = TransformerFeatures::standard("Qwen3");
    features.head_dim_override = Some(head_dim);
    features.qk_norm = true;
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if raw_config
        .get("use_sliding_window")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || raw_config
            .get("rope_scaling")
            .is_some_and(|scaling| !scaling.is_null())
    {
        return Err("Vulkan Qwen3 does not support sliding windows or scaled RoPE.".into());
    }
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    #[test]
    fn qwen3_vulkan_features_use_head_dim_and_qk_norms() {
        let config = vulkan_config(&serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "attention_bias": false,
            "hidden_act": "silu",
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true
        }))
        .unwrap();
        assert_eq!(config.features.head_dim_override, Some(8));
        assert!(config.features.qk_norm);
        assert!(!config.features.attention_bias);
        assert!(config.features.tie_word_embeddings);
    }

    #[test]
    fn qwen3_vulkan_rejects_scaled_rope() {
        assert!(
            vulkan_config(&serde_json::json!({
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "rope_scaling": {"type": "linear", "factor": 2.0}
            }))
            .is_err()
        );
    }
}
