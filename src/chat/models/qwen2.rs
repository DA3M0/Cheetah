use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["Qwen2ForCausalLM", "qwen2"],
    chat_template: ChatTemplate {
        id: "qwen",
        render: super::templates::qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["qwen_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: qwen2::Model,
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
    let config: qwen2::Config = serde_json::from_value(raw_config)
        .map_err(|error| format!("Invalid Qwen2 configuration: {error}"))?;
    Ok(Box::new(super::super::vulkan_qwen::VulkanQwenExecutor::load(
        weight_paths,
        &config,
    )?))
}
