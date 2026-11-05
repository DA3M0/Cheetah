use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["LlamaForCausalLM", "llama"],
    chat_template: ChatTemplate {
        id: "llama",
        render: super::templates::llama2,
        header_renderer: Some(super::templates::llama3),
    },
    stop_markers: &["<|eot_id|>", "</s>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["shared_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: llama::Llama,
    cache: llama::Cache,
    device: Device,
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
    use super::super::vulkan_transformer::{
        TransformerFeatures, VulkanTransformerConfig, VulkanTransformerExecutor,
    };

    if raw_config
        .get("rope_scaling")
        .is_some_and(|value| !value.is_null())
    {
        return Err("Vulkan Llama does not support RoPE scaling.".into());
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
        return Err("Vulkan Llama does not support MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Llama");
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
    Ok(Box::new(
        VulkanTransformerExecutor::load(weight_paths, &config)?,
    ))
}

impl ModelRuntime for Runtime {
    fn forward(
        &mut self,
        input: &candle_core::Tensor,
        offset: usize,
    ) -> Result<candle_core::Tensor, String> {
        self.model
            .forward(input, offset, &mut self.cache)
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
    config_bytes: &[u8],
    _raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let raw: llama::LlamaConfig = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Llama configuration: {error}"))?;
    let config = raw.into_config(false);
    let model = llama::Llama::load(builder, &config)
        .map_err(|error| format!("Failed to load Llama weights: {error}"))?;
    let cache = llama::Cache::new(true, DType::F32, &config, device)
        .map_err(|error| format!("Failed to initialize Llama KV cache: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        cache,
        device: device.clone(),
    }))
}
