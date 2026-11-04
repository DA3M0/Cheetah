use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama;
use serde_json::Value;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["LlamaForCausalLM", "llama"],
    chat_template: ChatTemplate {
        id: "llama",
        render: super::templates::llama2,
        header_renderer: Some(super::templates::llama3),
    },
    stop_markers: &["<|eot_id|>", "</s>"],
    loader: load,
    vulkan_loader: None,
    shaders: &[],
};

#[derive(Clone)]
struct Runtime {
    model: llama::Llama,
    cache: llama::Cache,
    device: Device,
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
