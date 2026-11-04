use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::phi;
use serde_json::Value;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["PhiForCausalLM", "phi"],
    chat_template: ChatTemplate {
        id: "phi",
        render: super::templates::phi,
        header_renderer: None,
    },
    stop_markers: &["<|end|>", "<|assistant|>"],
    loader: load,
    vulkan_loader: None,
    shaders: &[],
};

#[derive(Clone)]
struct Runtime {
    model: phi::Model,
    device: Device,
}

impl ModelRuntime for Runtime {
    fn forward(
        &mut self,
        input: &candle_core::Tensor,
        _offset: usize,
    ) -> Result<candle_core::Tensor, String> {
        self.model
            .forward(input)
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
    let config: phi::Config = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Phi configuration: {error}"))?;
    let model = phi::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Phi weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}
