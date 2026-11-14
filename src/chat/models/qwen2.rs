use super::{ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;


#[derive(Clone)]
struct Runtime {
    model: qwen2::Model,
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

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }

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
    let config: qwen2::Config = serde_json::from_value(raw_config)
        .map_err(|error| format!("Invalid Qwen2 configuration: {error}"))?;
    let model = qwen2::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Qwen2 weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}


#[cfg(feature = "vulkan")]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = vulkan_config(&raw_config)?;
    Ok(Box::new(super::super::vulkan_transformer::VulkanTransformerExecutor::load(
        weight_paths,
        &config,
        plan,
    )?))
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
    if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
        return Err("Qwen2 attention dimensions are inconsistent.".into());
    }
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
        return Err("Qwen2 config must not define MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Qwen2");
    features.attention_bias = true;
    features.head_dim_override = Some(hidden_size / num_attention_heads);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    #[test]
    fn qwen2_features_include_attention_bias_and_tied_embeddings() {
        let config = vulkan_config(&serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "tie_word_embeddings": true
        }))
        .unwrap();
        assert!(config.features.attention_bias);
        assert!(!config.features.attention_output_bias);
        assert!(config.features.tie_word_embeddings);
    }

    #[test]
    fn qwen2_features_reject_inconsistent_attention_dimensions() {
        assert!(
            vulkan_config(&serde_json::json!({
                "vocab_size": 16,
                "hidden_size": 9,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "hidden_act": "silu"
            }))
            .is_err()
        );
    }
}
