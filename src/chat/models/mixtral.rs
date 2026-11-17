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
        required_usize,
    };

    let num_local_experts = required_usize(raw_config, "num_local_experts")?;
    let num_experts_per_tok = required_usize(raw_config, "num_experts_per_tok")?;
    if num_experts_per_tok > num_local_experts {
        return Err("Mixtral expert counts are inconsistent.".into());
    }
    let mut features = TransformerFeatures::standard("Mixtral");
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
        num_local_experts,
        num_experts_per_tok,
    };
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;
    use crate::chat::vulkan_transformer::TransformerFeedForwardFeatures;

    #[test]
    fn mixtral_vulkan_features_configure_moe_tensor_layout() {
        let config = vulkan_config(&serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap();
        assert!(matches!(
            config.features.feed_forward,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 8,
                num_experts_per_tok: 2,
            }
        ));
    }

    #[test]
    fn mixtral_vulkan_features_reject_invalid_expert_counts() {
        let raw = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 2,
            "num_experts_per_tok": 3
        });
        assert!(vulkan_config(&raw).is_err());
    }
}
