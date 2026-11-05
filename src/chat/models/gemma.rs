use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::gemma;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["GemmaForCausalLM", "gemma"],
    chat_template: ChatTemplate {
        id: "gemma",
        render: super::templates::gemma,
        header_renderer: None,
    },
    stop_markers: &["<end_of_turn>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["shared_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: gemma::Model,
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
    let config =
        super::super::vulkan_transformer::VulkanTransformerConfig::gemma(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load_gemma(
            weight_paths,
            &config,
        )?,
    ))
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
    let config: gemma::Config = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Gemma configuration: {error}"))?;
    let model = gemma::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Gemma weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}
