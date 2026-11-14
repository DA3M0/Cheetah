use super::{ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::mixtral;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;


#[derive(Clone)]
struct Runtime {
    model: mixtral::Model,
    device: Device,
}

impl ModelRuntime for Runtime {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        // Bridge the backend-agnostic runtime boundary to the Candle model:
        // tokens in, final-position logits out.
        let input = candle_core::Tensor::from_vec(
            tokens.to_vec(),
            (1, tokens.len()),
            &self.device,
        )
        .map_err(|error| error.to_string())?;
        let logits = self
            .model
            .forward(&input, offset)
            .map_err(|error| error.to_string())?;
        logits
            .reshape((logits.elem_count(),))
            .and_then(|logits| logits.to_vec1::<f32>())
            .map_err(|error| error.to_string())
    }

    fn clear_cache(&mut self) {}

    fn device_name(&self) -> &'static str {
        match &self.device {
            Device::Cpu => "CPU",
            Device::Cuda(_) => "CUDA",
            Device::Metal(_) => "Metal",
        }
    }
}

pub(super) fn load(
    _config_bytes: &[u8],
    raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = candle_config(raw_config)?;
    let model = mixtral::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Mixtral weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}

fn candle_config(mut raw_config: Value) -> Result<mixtral::Config, String> {
    let object = raw_config
        .as_object_mut()
        .ok_or_else(|| "Mixtral configuration must be a JSON object.".to_string())?;
    object.insert("use_flash_attn".into(), Value::Bool(false));
    serde_json::from_value(raw_config)
        .map_err(|error| format!("Invalid Mixtral configuration: {error}"))
}


#[cfg(feature = "vulkan")]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = vulkan_config(&raw_config)?;
    Ok(Box::new(super::super::vulkan_transformer::VulkanTransformerExecutor::load(
        weight_paths,
        &config,
    )?))
}

#[cfg(feature = "vulkan")]
fn vulkan_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{
        Activation, TransformerFeatures, TransformerFeedForwardFeatures,
        VulkanTransformerConfig, required_usize,
    };

    let num_experts = required_usize(raw_config, "num_local_experts")?;
    let top_k = required_usize(raw_config, "num_experts_per_tok")?;
    if top_k > num_experts {
        return Err("Mixtral expert counts are inconsistent.".into());
    }
    let mut features = TransformerFeatures::standard("Mixtral");
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
        num_experts,
        top_k,
        activation: Activation::Silu,
        router: "block_sparse_moe.gate",
        experts_prefix: "block_sparse_moe.experts",
        gate: "w1",
        up: "w3",
        down: "w2",
    };
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_config_disables_flash_attention_without_changing_checkpoint_fields() {
        let config = candle_config(serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_act": "silu",
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "sliding_window": 4096,
            "num_experts_per_tok": 2,
            "num_local_experts": 8
        }))
        .unwrap();
        assert_eq!(config, mixtral::Config::v0_1_8x7b(false));
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
                    num_experts: 8,
                    top_k: 2,
                    router: "block_sparse_moe.gate",
                    experts_prefix: "block_sparse_moe.experts",
                    ..
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

    #[test]
    fn candle_config_rejects_non_object_configs() {
        assert!(candle_config(Value::Null).is_err());
    }
}
