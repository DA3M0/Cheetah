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

    match registration.loader_for("candle") {
        Some(super::BackendLoader::Candle(load)) => load(config_bytes, raw_config, builder, device),
        _ => Err(format!(
            "Architecture '{}' does not register a candle loader.",
            registration.chat_template.id
        )),
    }
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
            "Qwen3ForCausalLM",
            "qwen3",
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
        let template_of = |architecture: &str| {
            super::super::registration_for_architecture(architecture)
                .unwrap()
                .chat_template
                .id
        };
        assert_eq!(template_of("qwen2"), "qwen");
        assert_eq!(template_of("qwen3"), "qwen");
        assert_eq!(template_of("mistral"), "mistral");
        assert_eq!(template_of("mixtral"), "mixtral");
        assert!(super::super::shader_is_registered("shared_ops"));
    }

    #[test]
    fn vulkan_loader_registration_matches_compiled_support() {
        for registration in super::super::REGISTRY {
            let names: Vec<_> = registration.backend_names().collect();
            assert!(
                names.contains(&"candle"),
                "architecture '{}' must always register a candle backend",
                registration.chat_template.id
            );
            #[cfg(feature = "vulkan")]
            assert!(
                names.contains(&"vulkan"),
                "architecture '{}' must register a vulkan backend when compiled in",
                registration.chat_template.id
            );
            #[cfg(not(feature = "vulkan"))]
            assert!(
                !names.contains(&"vulkan"),
                "architecture '{}' registers vulkan without the feature",
                registration.chat_template.id
            );
        }
    }

    #[test]
    fn backend_registration_is_explicit_per_architecture() {
        // The native mmap backend is registered per architecture through the
        // same registry as the Candle and Vulkan loaders. The decoder-family
        // runner currently serves qwen2 and llama; the other families keep
        // the Candle-only CPU path until their native migrations land.
        for architecture in ["qwen2", "llama", "qwen3"] {
            let names = super::super::registered_backend_names(architecture);
            assert!(
                names.contains(&"native"),
                "architecture '{architecture}' must register a native backend"
            );
            assert!(super::super::family_for_architecture(architecture).is_some());
        }
        for architecture in ["gemma", "phi", "mistral", "mixtral"] {
            let names = super::super::registered_backend_names(architecture);
            assert!(
                !names.contains(&"native"),
                "architecture '{architecture}' unexpectedly registers a native backend"
            );
        }
        assert!(super::super::registered_backend_names("qwen2").contains(&"candle"));
        assert!(super::super::backend_loader_for_architecture("qwen2", "native").is_some());
        assert!(super::super::backend_loader_for_architecture("llama", "native").is_some());
        assert!(super::super::backend_loader_for_architecture("gemma", "native").is_none());
        assert!(super::super::backend_loader_for_architecture("qwen2", "candle").is_some());
        assert!(super::super::backend_loader_for_architecture("nonexistent", "candle").is_none());
    }

    #[test]
    fn all_architectures_index_covers_the_registry() {
        // The generated build-time index must list every architecture in the
        // registry, and every listed name must be resolvable.
        let mut from_index: Vec<&str> = Vec::new();
        for architectures in super::super::ALL_ARCHITECTURES {
            from_index.extend_from_slice(architectures);
        }
        assert!(from_index.contains(&"Qwen2ForCausalLM"));
        assert!(from_index.contains(&"Qwen3ForCausalLM"));
        assert!(from_index.contains(&"mixtral"));
        for architecture in &from_index {
            assert!(
                super::super::registration_for_architecture(architecture).is_ok(),
                "architecture '{architecture}' is indexed but not registered"
            );
        }
        for registration in super::super::REGISTRY {
            for architecture in registration.architectures {
                assert!(
                    from_index.contains(architecture),
                    "architecture '{architecture}' is registered but missing from ALL_ARCHITECTURES"
                );
            }
        }
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
