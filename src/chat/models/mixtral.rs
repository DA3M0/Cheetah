use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::mixtral;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["MixtralForCausalLM", "mixtral"],
    chat_template: ChatTemplate {
        id: "mixtral",
        render: super::templates::mixtral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["qwen_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: mixtral::Model,
    device: Device,
}

impl ModelRuntime for Runtime {
    fn forward(
        &mut self,
        input: &candle_core::Tensor,
        offset: usize,
    ) -> Result<candle_core::Tensor, String> {
        self.model
            .forward(input, offset)
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

fn load(
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

const fn vulkan_loader() -> Option<super::VulkanModelLoader> {
    #[cfg(feature = "vulkan")]
    {
        Some(load_vulkan)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
}

#[cfg(feature = "vulkan")]
fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config =
        super::super::vulkan_transformer::VulkanTransformerConfig::mixtral(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load_mixtral(
            weight_paths,
            &config,
        )?,
    ))
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

    #[test]
    fn candle_config_rejects_non_object_configs() {
        assert!(candle_config(Value::Null).is_err());
    }
}
