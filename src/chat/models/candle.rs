use super::{ModelRegistration, ModelRuntime};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};
use serde_json::Value;
use std::path::PathBuf;

struct TiedEmbeddingBackend<'a> {
    source: VarBuilder<'a>,
}

impl SimpleBackend for TiedEmbeddingBackend<'_> {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        hints: Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let name = if name == "lm_head.weight" {
            "model.embed_tokens.weight"
        } else {
            name
        };
        self.source
            .get_with_hints(shape, name, hints)
            .and_then(|tensor| tensor.to_dtype(dtype))
            .and_then(|tensor| tensor.to_device(device))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.source.contains_tensor(name)
            || name == "lm_head.weight" && self.source.contains_tensor("model.embed_tokens.weight")
    }
}

pub(super) fn load_model(
    weight_paths: &[PathBuf],
    config_bytes: &[u8],
    raw_config: Value,
    registration: &'static ModelRegistration,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let mut builder =
        unsafe { VarBuilder::from_mmaped_safetensors(weight_paths, DType::F32, device) }
            .map_err(|error| format!("Failed to map Safetensors weights: {error}"))?;
    if raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && !builder.contains_tensor("lm_head.weight")
    {
        builder = VarBuilder::new_with_args(
            Box::new(TiedEmbeddingBackend { source: builder }),
            DType::F32,
            device,
        );
    }

    (registration.loader)(config_bytes, raw_config, builder, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn registers_architecture_names_and_aliases() {
        for architecture in [
            "Qwen2ForCausalLM",
            "qwen2",
            "GemmaForCausalLM",
            "gemma",
            "LlamaForCausalLM",
            "llama",
            "PhiForCausalLM",
            "phi",
            "MistralForCausalLM",
            "mistral",
            "MixtralForCausalLM",
            "mixtral",
        ] {
            assert!(
                super::super::REGISTRY
                    .iter()
                    .any(|registration| registration.architectures.contains(&architecture))
            );
        }
    }

    #[test]
    fn registrations_include_chat_metadata() {
        assert!(super::super::REGISTRY.iter().all(|registration| {
            !registration.architectures.is_empty() && !registration.stop_markers.is_empty()
        }));
        assert_eq!(super::super::qwen2::REGISTRATION.chat_template.id, "qwen");
        assert_eq!(
            super::super::mistral::REGISTRATION.chat_template.id,
            "mistral"
        );
        assert_eq!(
            super::super::mixtral::REGISTRATION.chat_template.id,
            "mixtral"
        );
        assert!(super::super::shader_is_registered("qwen_ops"));
    }

    #[test]
    fn vulkan_loader_registration_matches_compiled_support() {
        #[cfg(feature = "vulkan")]
        assert!(
            super::super::REGISTRY
                .iter()
                .all(|registration| registration.vulkan_loader.is_some()
                    && registration.shaders.contains(&"qwen_ops"))
        );
        #[cfg(not(feature = "vulkan"))]
        assert!(
            super::super::REGISTRY
                .iter()
                .all(|registration| registration.vulkan_loader.is_none())
        );
    }

    #[test]
    fn tied_lm_head_uses_embedding_weights() {
        let device = Device::Cpu;
        let embeddings = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        let source = VarBuilder::from_tensors(
            HashMap::from([("model.embed_tokens.weight".to_owned(), embeddings)]),
            DType::F32,
            &device,
        );
        let tied = VarBuilder::new_with_args(
            Box::new(TiedEmbeddingBackend { source }),
            DType::F32,
            &device,
        );
        let lm_head = tied
            .pp("lm_head")
            .get::<(usize, usize)>((2, 2), "weight")
            .unwrap();
        assert_eq!(
            lm_head.to_vec2::<f32>().unwrap(),
            vec![vec![1., 2.], vec![3., 4.]]
        );
    }
}
