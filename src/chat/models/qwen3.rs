//! Qwen3 model implementation for the Vulkan compute backend.
//!
//! Qwen3 is a dense transformer architecture with:
//! - Standard GQA attention (Grouped Query Attention) with per-head q_norm /
//!   k_norm tensors
//! - Sliding window attention declared by `use_sliding_window` / `sliding_window`
//! - Tied word embeddings
//! - SiLU activation function
//!
//! Unlike MoE variants, Qwen3 uses standard dense feed-forward layers
//! without expert routing, providing consistent performance characteristics.

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
        TransformerFeatures, TransformerFeedForwardFeatures, VulkanTransformerConfig,
    };

    let mut features = TransformerFeatures::standard("Qwen3");
    // Qwen3 carries per-head q_norm/k_norm tensors on every attention layer.
    features.qk_norm = true;
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    features.feed_forward = TransformerFeedForwardFeatures::Dense;

    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;
    use crate::chat::vulkan_transformer::TransformerFeedForwardFeatures;

    fn raw_config() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "sliding_window": 4096
        })
    }

    #[test]
    fn qwen3_vulkan_features_configure_dense_tensor_layout() {
        let raw = raw_config();
        let config = vulkan_config(&raw).unwrap();
        assert!(matches!(
            &config.features.feed_forward,
            TransformerFeedForwardFeatures::Dense
        ));
        assert!(config.features.tie_word_embeddings);
        assert!(config.features.qk_norm);
        assert!(!config.features.attention_bias);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.sliding_window, 4096);
    }

    /// Qwen3 ships `use_sliding_window: false` (full attention); the sparse
    /// window must not be applied when the config disables it.
    #[test]
    fn qwen3_vulkan_disables_the_sliding_window_when_the_config_asks() {
        let mut raw = raw_config();
        raw["use_sliding_window"] = serde_json::json!(false);
        let config = vulkan_config(&raw).unwrap();
        assert_eq!(config.sliding_window, 0);
    }

    #[test]
    fn qwen3_vulkan_rejects_scaled_rope() {
        let mut raw = raw_config();
        raw["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
        assert!(vulkan_config(&raw).is_err());
    }

    #[test]
    fn qwen3_vulkan_honors_a_configured_attention_bias() {
        let mut raw = raw_config();
        raw["attention_bias"] = serde_json::json!(true);
        assert!(vulkan_config(&raw).unwrap().features.attention_bias);
    }
}
