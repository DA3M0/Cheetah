use super::{ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::gemma;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;


#[derive(Clone)]
struct Runtime {
    model: gemma::Model,
    device: Device,
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
        TransformerFeatures, VulkanTransformerConfig,
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
    config_bytes: &[u8],
    _raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config: gemma::Config = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Gemma configuration: {error}"))?;
    let model = gemma::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Gemma weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}
