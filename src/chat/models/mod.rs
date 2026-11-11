mod candle;
mod registry;
mod templates;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Clone, Copy)]
pub(super) struct ChatTemplate {
    pub(super) id: &'static str,
    pub(super) render: PromptRenderer,
    pub(super) header_renderer: Option<PromptRenderer>,
}

pub(super) type PromptRenderer = fn(&[super::ChatMessage], Option<&str>) -> String;

pub(super) trait ModelRuntime: ModelRuntimeClone + Send {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String>;
    fn clear_cache(&mut self);
    fn device_name(&self) -> &'static str;
}

pub(super) trait ModelRuntimeClone {
    fn clone_box(&self) -> Box<dyn ModelRuntime>;
}

impl<T> ModelRuntimeClone for T
where
    T: ModelRuntime + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ModelRuntime> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelRuntime> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub(super) type ModelLoader =
    for<'a> fn(&[u8], Value, VarBuilder<'a>, &Device) -> Result<Box<dyn ModelRuntime>, String>;

pub(super) type VulkanModelLoader =
    fn(&[PathBuf], &[u8], Value) -> Result<Box<dyn ModelRuntime>, String>;

pub(super) struct ModelRegistration {
    pub(super) architectures: &'static [&'static str],
    pub(super) chat_template: ChatTemplate,
    pub(super) stop_markers: &'static [&'static str],
    pub(super) loader: ModelLoader,
    pub(super) vulkan_loader: Option<VulkanModelLoader>,
    pub(super) shaders: &'static [&'static str],
}

pub(crate) struct ShaderRegistration {
    name: &'static str,
    path: Option<&'static str>,
}

include!(concat!(env!("OUT_DIR"), "/model_registry.rs"));
include!(concat!(env!("OUT_DIR"), "/shader_registry.rs"));

pub(crate) fn registration_for_architecture(
    architecture: &str,
) -> Result<&'static ModelRegistration, String> {
    REGISTRY
        .iter()
        .copied()
        .find(|registration| registration.architectures.contains(&architecture))
        .ok_or_else(|| format!("Unsupported model architecture '{architecture}'."))
}

pub(crate) fn shader_path(name: &str) -> Option<&'static str> {
    SHADERS
        .iter()
        .find(|shader| shader.name == name)
        .and_then(|shader| shader.path)
}

pub(super) fn shader_is_registered(name: &str) -> bool {
    shader_path(name).is_some() || SHADERS.iter().any(|shader| shader.name == name)
}

pub(super) use registry::{backend_device, load_model, load_model_for_device};
