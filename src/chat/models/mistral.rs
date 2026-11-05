use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::mistral;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["MistralForCausalLM", "mistral"],
    chat_template: ChatTemplate {
        id: "mistral",
        render: super::templates::mistral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["shared_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: mistral::Model,
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

fn load(
    config_bytes: &[u8],
    _raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config_json: Value = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Mistral configuration: {error}"))?;
    let config = candle_config(&config_json)?;
    let model = mistral::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Mistral weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}

fn candle_config(config: &Value) -> Result<mistral::Config, String> {
    let vocab_size = config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| "Mistral config is missing a valid vocab_size.".to_string())?;
    let fields = [
        ("hidden_size", 4096u64),
        ("intermediate_size", 14336),
        ("num_hidden_layers", 32),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
        ("max_position_embeddings", 32768),
        ("sliding_window", 4096),
    ];
    for (name, expected) in fields {
        if config.get(name).and_then(Value::as_u64) != Some(expected) {
            return Err(format!(
                "This Candle Mistral adapter supports the standard 7B config only ({name} must be {expected})."
            ));
        }
    }
    if config.get("hidden_act").and_then(Value::as_str) != Some("silu")
        || config.get("rope_theta").and_then(Value::as_f64) != Some(10_000.0)
        || config.get("rms_norm_eps").and_then(Value::as_f64) != Some(1e-5)
    {
        return Err(
            "This Candle Mistral adapter supports the standard 7B SiLU/RoPE/RMSNorm config only."
                .into(),
        );
    }
    match vocab_size {
        32000 => Ok(mistral::Config::config_7b_v0_1(false)),
        32002 => Ok(mistral::Config::config_chat_ml(false)),
        32003 => Ok(mistral::Config::config_amazon_mistral_lite(false)),
        _ => Err(format!(
            "Unsupported Mistral vocabulary size {vocab_size}; supported sizes are 32000, 32002, and 32003."
        )),
    }
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
    candle_config(&raw_config)?;
    let config =
        super::super::vulkan_transformer::VulkanTransformerConfig::mistral(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load_mistral(
            weight_paths,
            &config,
        )?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_config(vocab_size: usize) -> Value {
        serde_json::json!({
            "vocab_size": vocab_size,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "hidden_act": "silu"
        })
    }

    #[test]
    fn selects_only_candle_supported_mistral_7b_presets() {
        assert!(candle_config(&standard_config(32000)).is_ok());
        assert!(candle_config(&standard_config(32002)).is_ok());
        assert!(candle_config(&standard_config(32003)).is_ok());
        assert!(candle_config(&standard_config(32768)).is_err());
        let mut unsupported = standard_config(32000);
        unsupported["num_key_value_heads"] = Value::from(4);
        assert!(candle_config(&unsupported).is_err());
    }
}
