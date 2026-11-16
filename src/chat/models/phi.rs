#[cfg(feature = "vulkan")]
use super::ModelRuntime;
#[cfg(feature = "vulkan")]
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
#[allow(dead_code)]
pub(super) fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    _plan: &std::collections::BTreeMap<String, String>,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = super::super::vulkan_transformer::VulkanPhiConfig::parse(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanPhiExecutor::load(weight_paths, &config)?,
    ))
}
