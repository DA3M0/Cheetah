//! Candle-backed chat inference over Safetensors checkpoints.

mod engine;
mod models;
#[cfg(feature = "vulkan")]
mod vulkan_transformer;

pub(crate) use engine::{ChatMessage, ChatParams, Engine, Generation, TextSink, ToolChoice};

pub fn chat(
    path: &std::path::Path,
    options: crate::options::ChatOptions,
    archive: crate::context::ContextArchive,
    store: crate::context::ContextStore,
) -> Result<(), String> {
    engine::chat(path, options, archive, store)
}

pub(crate) fn model_metadata_for_init(
    source: &std::path::Path,
) -> Result<Option<cheetah::native::ModelMetadata>, String> {
    let model_dir = if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or_else(|| std::path::Path::new("."))
    };
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        return Ok(None);
    }
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .map_err(|error| format!("Failed to read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("Invalid model configuration: {error}"))?;
    let architecture = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str)
        .or_else(|| config.get("model_type").and_then(serde_json::Value::as_str));
    let Some(architecture) = architecture else {
        return Ok(None);
    };
    let Ok(registration) = models::registration_for_architecture(architecture) else {
        return Ok(None);
    };
    Ok(Some(cheetah::native::ModelMetadata {
        chat_template: registration.chat_template.id,
        stop_markers: registration.stop_markers,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn init_metadata_comes_from_generated_model_registration() {
        let directory = std::env::temp_dir().join(format!(
            "cheetah-init-registry-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"new_model","architectures":["qwen2"]}"#,
        )
        .unwrap();

        let metadata = model_metadata_for_init(&directory).unwrap().unwrap();
        assert_eq!(metadata.chat_template, "qwen");
        assert_eq!(metadata.stop_markers, ["<|im_end|>"]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn init_metadata_is_absent_for_unregistered_architecture() {
        let directory = std::env::temp_dir().join(format!(
            "cheetah-init-unregistered-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("config.json"),
            r#"{"model_type":"unknown-model"}"#,
        )
        .unwrap();
        assert!(model_metadata_for_init(&directory).unwrap().is_none());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn init_metadata_comes_from_mistral_and_mixtral_registrations() {
        let directory = std::env::temp_dir().join(format!(
            "cheetah-init-registered-architectures-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        for (architecture, template) in [
            ("MistralForCausalLM", "mistral"),
            ("MixtralForCausalLM", "mixtral"),
        ] {
            std::fs::write(
                directory.join("config.json"),
                format!(r#"{{"architectures":["{architecture}"]}}"#),
            )
            .unwrap();
            let metadata = model_metadata_for_init(&directory).unwrap().unwrap();
            assert_eq!(metadata.chat_template, template);
            assert_eq!(metadata.stop_markers, ["</s>"]);
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
