use super::{ModelRuntime};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::phi;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;


#[derive(Clone)]
struct Runtime {
    model: phi::Model,
    device: Device,
}


#[cfg(feature = "vulkan")]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config =
        super::super::vulkan_transformer::VulkanPhiConfig::parse(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanPhiExecutor::load(weight_paths, &config)?,
    ))
}

impl ModelRuntime for Runtime {
    fn forward(&mut self, tokens: &[u32], _offset: usize) -> Result<Vec<f32>, String> {
        // Bridge the backend-agnostic runtime boundary to the Candle model:
        // tokens in, final-position logits out (Phi has no KV cache).
        let input = candle_core::Tensor::from_vec(
            tokens.to_vec(),
            (1, tokens.len()),
            &self.device,
        )
        .map_err(|error| error.to_string())?;
        let logits = self
            .model
            .forward(&input)
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
    let config: phi::Config = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Phi configuration: {error}"))?;
    let model = phi::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Phi weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}
