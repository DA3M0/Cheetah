//! Generic Vulkan executor loader for every registered family. All backend
//! tables reference the one `load` entry point; the per-family executor
//! features derive from the model's own `config.json` (`model_type`), so a
//! registration file needs nothing beyond that reference. A family only
//! touches this module when it needs executor features no existing arm
//! covers.

#[cfg(feature = "vulkan")]
use super::ModelRuntime;
#[cfg(feature = "vulkan")]
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

/// The one Vulkan entry point every registration references. The model's
/// `model_type` selects the executor feature set; the declared architecture
/// names in the registration and the `model_type` here describe the same
/// family (HF configs carry both).
#[cfg(feature = "vulkan")]
pub(super) fn load(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &std::collections::BTreeMap<String, String>,
) -> Result<Box<dyn ModelRuntime>, String> {
    let model_type = raw_config
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            "Vulkan executor: the model config carries no 'model_type' to select executor features."
                .to_string()
        })?;
    match model_type {
        "llama" => transformer(weight_paths, plan, llama_config(&raw_config)?),
        "qwen2" => transformer(weight_paths, plan, qwen2_config(&raw_config)?),
        "qwen3" => transformer(weight_paths, plan, qwen3_config(&raw_config)?),
        "mistral" => transformer(weight_paths, plan, mistral_config(&raw_config)?),
        "gemma" => transformer(weight_paths, plan, gemma_config(&raw_config)?),
        "mixtral" => transformer(weight_paths, plan, mixtral_config(&raw_config)?),
        // Phi runs its own executor: a different block topology (biased
        // LayerNorms, parallel residual, fused projections), so it does not
        // share the transformer config builder.
        "phi" => {
            let config = super::super::vulkan_transformer::VulkanPhiConfig::parse(&raw_config)?;
            Ok(Box::new(
                super::super::vulkan_transformer::VulkanPhiExecutor::load(weight_paths, &config)?,
            ))
        }
        other => Err(format!(
            "Vulkan executor: no executor features for model_type '{other}'."
        )),
    }
}

/// Build the transformer executor around an already-derived feature set.
#[cfg(feature = "vulkan")]
fn transformer(
    weight_paths: &[PathBuf],
    plan: &std::collections::BTreeMap<String, String>,
    config: super::super::vulkan_transformer::VulkanTransformerConfig,
) -> Result<Box<dyn ModelRuntime>, String> {
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load(
            weight_paths,
            &config,
            plan,
        )?,
    ))
}

#[cfg(feature = "vulkan")]
fn llama_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{TransformerFeatures, VulkanTransformerConfig};

    if raw_config
        .get("rope_scaling")
        .is_some_and(|value| !value.is_null())
    {
        return Err("Vulkan Llama does not support RoPE scaling.".into());
    }
    if raw_config
        .get("num_local_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        != 0
        || raw_config
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("Vulkan Llama does not support MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Llama");
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.attention_output_bias = features.attention_bias;
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(feature = "vulkan")]
fn qwen2_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{
        TransformerFeatures, VulkanTransformerConfig, required_usize,
    };

    let hidden_size = required_usize(raw_config, "hidden_size")?;
    let num_attention_heads = required_usize(raw_config, "num_attention_heads")?;
    if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
        return Err("Qwen2 attention dimensions are inconsistent.".into());
    }
    if raw_config
        .get("num_local_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        != 0
        || raw_config
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("Qwen2 config must not define MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Qwen2");
    features.attention_bias = true;
    features.head_dim_override = Some(hidden_size / num_attention_heads);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(feature = "vulkan")]
fn qwen3_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{
        TransformerFeatures, TransformerFeedForwardFeatures, VulkanTransformerConfig,
    };

    let mut features = TransformerFeatures::standard("Qwen3");
    // Qwen3 carries per-head q_norm/k_norm tensors on every attention layer.
    features.qk_norm = true;
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    features.feed_forward = TransformerFeedForwardFeatures::Dense;

    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(feature = "vulkan")]
fn mistral_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{TransformerFeatures, VulkanTransformerConfig};

    if raw_config
        .get("num_local_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        != 0
        || raw_config
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("Mistral config must not define MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Mistral");
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.attention_output_bias = features.attention_bias;
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(feature = "vulkan")]
fn gemma_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{TransformerFeatures, VulkanTransformerConfig};

    if raw_config
        .get("num_local_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        != 0
        || raw_config
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            != 0
    {
        return Err("Gemma does not support MoE experts.".into());
    }
    let mut features = TransformerFeatures::standard("Gemma");
    features.norm_weight_offset = true;
    features.embedding_scale = true;
    features.tie_word_embeddings = true;
    features.max_position_embeddings_default = Some(4096);
    features.rms_norm_eps_default = Some(1e-6);
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.attention_output_bias = features.attention_bias;
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(feature = "vulkan")]
fn mixtral_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{
        TransformerFeatures, TransformerFeedForwardFeatures, VulkanTransformerConfig,
        required_usize,
    };

    let num_local_experts = required_usize(raw_config, "num_local_experts")?;
    let num_experts_per_tok = required_usize(raw_config, "num_experts_per_tok")?;
    if num_experts_per_tok > num_local_experts {
        return Err("Mixtral expert counts are inconsistent.".into());
    }
    let mut features = TransformerFeatures::standard("Mixtral");
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
        num_local_experts,
        num_experts_per_tok,
    };
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::chat::vulkan_transformer::TransformerFeedForwardFeatures;

    /// Load error without `Debug` on the runtime handle.
    fn load_error(raw: serde_json::Value) -> String {
        match load(&[], &[], raw, &Default::default()) {
            Err(error) => error,
            Ok(_) => panic!("empty weight paths cannot load a model"),
        }
    }

    #[test]
    fn load_dispatches_by_model_type() {
        // Each family's config fn must be reachable through its model_type —
        // a registration file references only `load`.
        for model_type in ["llama", "qwen2", "qwen3", "mistral", "gemma", "mixtral"] {
            let raw = serde_json::json!({ "model_type": model_type });
            let error = load_error(raw);
            // The load fails on weight paths, never on feature selection.
            assert!(
                !error.contains("no executor features"),
                "{model_type} must dispatch to its family features ({error})"
            );
        }
        let unknown = load_error(serde_json::json!({"model_type": "bloom"}));
        assert!(unknown.contains("no executor features"), "{unknown}");
    }

    #[test]
    fn qwen2_features_include_attention_bias_and_tied_embeddings() {
        let config = qwen2_config(&serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "tie_word_embeddings": true
        }))
        .unwrap();
        assert!(config.features.attention_bias);
        assert!(!config.features.attention_output_bias);
        assert!(config.features.tie_word_embeddings);
    }

    #[test]
    fn qwen2_features_reject_inconsistent_attention_dimensions() {
        assert!(
            qwen2_config(&serde_json::json!({
                "vocab_size": 16,
                "hidden_size": 9,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "hidden_act": "silu"
            }))
            .is_err()
        );
    }

    fn qwen3_raw() -> serde_json::Value {
        serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "sliding_window": 4096
        })
    }

    #[test]
    fn qwen3_vulkan_features_configure_dense_tensor_layout() {
        let config = qwen3_config(&qwen3_raw()).unwrap();
        assert!(matches!(
            &config.features.feed_forward,
            TransformerFeedForwardFeatures::Dense
        ));
        assert!(config.features.tie_word_embeddings);
        assert!(config.features.qk_norm);
        assert!(!config.features.attention_bias);
        assert_eq!(config.head_dim, 4);
        assert_eq!(config.sliding_window, 4096);
    }

    /// Qwen3 ships `use_sliding_window: false` (full attention); the sparse
    /// window must not be applied when the config disables it.
    #[test]
    fn qwen3_vulkan_disables_the_sliding_window_when_the_config_asks() {
        let mut raw = qwen3_raw();
        raw["use_sliding_window"] = serde_json::json!(false);
        let config = qwen3_config(&raw).unwrap();
        assert_eq!(config.sliding_window, 0);
    }

    #[test]
    fn qwen3_vulkan_rejects_scaled_rope() {
        let mut raw = qwen3_raw();
        raw["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
        assert!(qwen3_config(&raw).is_err());
    }

    #[test]
    fn qwen3_vulkan_honors_a_configured_attention_bias() {
        let mut raw = qwen3_raw();
        raw["attention_bias"] = serde_json::json!(true);
        assert!(qwen3_config(&raw).unwrap().features.attention_bias);
    }

    #[test]
    fn gemma_vulkan_features_configure_norm_and_embedding_behavior() {
        let config = gemma_config(&serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "gelu",
        }))
        .unwrap();
        assert!(config.features.norm_weight_offset);
        assert!(config.features.embedding_scale);
        assert!(config.features.tie_word_embeddings);
        assert_eq!(config.features.max_position_embeddings_default, Some(4096));
        assert_eq!(config.features.rms_norm_eps_default, Some(1e-6));
    }

    #[test]
    fn mixtral_vulkan_features_configure_moe_tensor_layout() {
        let config = mixtral_config(&serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap();
        assert!(matches!(
            config.features.feed_forward,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 8,
                num_experts_per_tok: 2,
            }
        ));
    }

    #[test]
    fn mixtral_vulkan_features_reject_invalid_expert_counts() {
        let raw = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 2,
            "num_experts_per_tok": 3
        });
        assert!(mixtral_config(&raw).is_err());
    }
}
