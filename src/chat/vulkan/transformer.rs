//! The transformer config surface: `VulkanTransformerConfig` parsing and the
//! per-family `TransformerFeatures` / `TransformerWeightLayout` /
//! `TransformerFeedForwardFeatures` declarations the registration files build
//! their own executors around. Part B retired the former generic executor —
//! every family now owns its decode graph in its registration file; the
//! family-free machinery lives in `super::context`, weight loading and the
//! operator dispatch in `super::weights` (the precision registry).

use serde_json::Value;

use super::context::Activation;
use super::required_usize;

#[derive(Clone, Debug)]
pub(crate) struct VulkanTransformerConfig {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) sliding_window: usize,
    pub(crate) activation: Activation,
    pub(crate) features: TransformerFeatures,
}

#[derive(Clone, Debug)]
pub(crate) struct TransformerFeatures {
    pub(crate) model_name: &'static str,
    pub(crate) attention_bias: bool,
    pub(crate) attention_output_bias: bool,
    pub(crate) qk_norm: bool,
    pub(crate) head_dim_override: Option<usize>,
    pub(crate) norm_weight_offset: bool,
    pub(crate) embedding_scale: bool,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) max_position_embeddings_default: Option<usize>,
    pub(crate) rms_norm_eps_default: Option<f32>,
    pub(crate) layout: TransformerWeightLayout,
    pub(crate) feed_forward: TransformerFeedForwardFeatures,
}

impl TransformerFeatures {
    /// The baseline executor feature set for one family name: no attention
    /// biases, no qk-norm, full rope, dense feed-forward, and the standard
    /// llama-shaped tensor layout — families override pieces in their
    /// registration files.
    pub(crate) fn standard(model_name: &'static str) -> Self {
        Self {
            model_name,
            attention_bias: false,
            attention_output_bias: false,
            qk_norm: false,
            head_dim_override: None,
            norm_weight_offset: false,
            embedding_scale: false,
            tie_word_embeddings: false,
            max_position_embeddings_default: None,
            rms_norm_eps_default: None,
            layout: TransformerWeightLayout::default(),
            feed_forward: TransformerFeedForwardFeatures::Dense,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TransformerWeightLayout {
    pub(crate) embedding: &'static str,
    pub(crate) final_norm: &'static str,
    pub(crate) lm_head: &'static str,
    pub(crate) layer_prefix: &'static str,
    pub(crate) input_norm: &'static str,
    pub(crate) post_attention_norm: &'static str,
    pub(crate) attention_prefix: &'static str,
    pub(crate) q_proj: &'static str,
    pub(crate) k_proj: &'static str,
    pub(crate) v_proj: &'static str,
    pub(crate) o_proj: &'static str,
    pub(crate) q_norm: &'static str,
    pub(crate) k_norm: &'static str,
    pub(crate) dense_gate: &'static str,
    pub(crate) dense_up: &'static str,
    pub(crate) dense_down: &'static str,
}

impl Default for TransformerWeightLayout {
    /// The standard llama-shaped tensor naming (embedding, norms, projections,
    /// MLP suffixes) shared by every family that does not override pieces of it.
    fn default() -> Self {
        Self {
            embedding: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers",
            input_norm: "input_layernorm.weight",
            post_attention_norm: "post_attention_layernorm.weight",
            attention_prefix: "self_attn",
            q_proj: "q_proj",
            k_proj: "k_proj",
            v_proj: "v_proj",
            o_proj: "o_proj",
            q_norm: "q_norm",
            k_norm: "k_norm",
            dense_gate: "mlp.gate_proj",
            dense_up: "mlp.up_proj",
            dense_down: "mlp.down_proj",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TransformerFeedForwardFeatures {
    Dense,
    MixtureOfExperts {
        /// Total experts in the layer (router output width).
        num_local_experts: usize,
        /// Experts routed per token; `0` disables routing (dense).
        num_experts_per_tok: usize,
    },
}

impl VulkanTransformerConfig {
    /// Config parsing + validation: derives `head_dim` and the kv-head count,
    /// applies family defaults, honors `use_sliding_window`, maps `hidden_act`,
    /// and fails loudly on rope scaling, inconsistent MoE counts, or any
    /// non-positive / non-finite dimension.
    pub(crate) fn from_raw(raw: &Value, features: TransformerFeatures) -> Result<Self, String> {
        let model_name = features.model_name;
        let vocab_size = required_usize(raw, "vocab_size")?;
        let hidden_size = required_usize(raw, "hidden_size")?;
        let intermediate_size = required_usize(raw, "intermediate_size")?;
        let num_hidden_layers = required_usize(raw, "num_hidden_layers")?;
        let num_attention_heads = required_usize(raw, "num_attention_heads")?;
        let num_key_value_heads = raw
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "num_key_value_heads does not fit usize.".to_string())?
            .unwrap_or(num_attention_heads);
        let head_dim = if let Some(head_dim) = features.head_dim_override {
            head_dim
        } else {
            raw.get("head_dim")
                .and_then(Value::as_u64)
                .map(usize::try_from)
                .transpose()
                .map_err(|_| "head_dim does not fit usize.".to_string())?
                .unwrap_or_else(|| hidden_size / num_attention_heads.max(1))
        };
        let max_position_embeddings = raw
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "max_position_embeddings does not fit usize.".to_string())?
            .or(features.max_position_embeddings_default)
            .unwrap_or(0);
        let rms_norm_eps = raw
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .or(features.rms_norm_eps_default)
            .unwrap_or(1e-5);
        let rope_theta = raw
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        let sliding_window = raw
            .get("sliding_window")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "sliding_window does not fit usize.".to_string())?
            .unwrap_or(max_position_embeddings);
        // Sparse (sliding-window) attention is gated by `use_sliding_window`
        // (default true, matching the Hugging Face configs). A disabled window
        // resolves to 0, which the attention shader reads as "attend to every
        // cached position" — the same full-attention path a missing
        // `sliding_window` takes above.
        let sliding_window = if raw
            .get("use_sliding_window")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            sliding_window
        } else {
            0
        };
        let activation = match raw.get("hidden_act").and_then(Value::as_str) {
            Some("silu") => Activation::Silu,
            Some("gelu") => Activation::Gelu,
            Some("gelu_new" | "gelu_pytorch_tanh") => Activation::NewGelu,
            _ => {
                return Err(format!(
                    "Vulkan {model_name} does not support the configured activation."
                ));
            }
        };
        // Rope scaling (linear / dynamic / llama3 / ...) is not implemented by
        // the rotary operator: a checkpoint that requires it must fail loudly
        // instead of rotating every position with the base theta.
        if raw
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return Err(format!(
                "Vulkan {model_name} does not support RoPE scaling."
            ));
        }
        if let TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts,
            num_experts_per_tok,
        } = &features.feed_forward
            && (*num_local_experts == 0
                || *num_experts_per_tok == 0
                || *num_experts_per_tok > *num_local_experts)
        {
            return Err(format!(
                "Vulkan {model_name} MoE expert counts are inconsistent."
            ));
        }
        if vocab_size == 0
            || hidden_size == 0
            || intermediate_size == 0
            || num_hidden_layers == 0
            || num_attention_heads == 0
            || num_key_value_heads == 0
            || head_dim == 0
            || max_position_embeddings == 0
            || head_dim % 2 != 0
            || num_attention_heads % num_key_value_heads != 0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(format!(
                "Vulkan {model_name} model dimensions are inconsistent."
            ));
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            sliding_window,
            activation,
            features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::collections::BTreeMap;
    use super::super::weights::WeightData;

    #[test]
    fn parses_common_and_moe_transformer_configurations() {
        let raw = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "sliding_window": 32
        });
        let config =
            VulkanTransformerConfig::from_raw(&raw, TransformerFeatures::standard("Dense"))
                .unwrap();
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.sliding_window, 32);

        let mut features = TransformerFeatures::standard("Sparse");
        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts: 8,
            num_experts_per_tok: 2,
        };
        let config = VulkanTransformerConfig::from_raw(&raw, features.clone()).unwrap();
        assert!(matches!(
            config.features.feed_forward,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 8,
                num_experts_per_tok: 2,
            }
        ));

        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts: 2,
            num_experts_per_tok: 3,
        };
        assert!(VulkanTransformerConfig::from_raw(&raw, features).is_err());
    }

    #[test]
    fn feature_values_override_common_configuration() {
        let raw = serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true
        });
        let mut features = TransformerFeatures::standard("Configured");
        features.attention_bias = true;
        features.tie_word_embeddings = true;
        let config = VulkanTransformerConfig::from_raw(&raw, features).unwrap();
        assert_eq!(config.head_dim, 2);
        assert_eq!(config.sliding_window, 64);
        assert!(config.features.tie_word_embeddings);
        assert!(config.features.attention_bias);
        assert!(!config.features.attention_output_bias);
    }

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn linear_splits_large_sequences_into_bounded_row_tiles() {
        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let weight = WeightData::F32(vec![2.0, -1.0].into());
        let output = weight.linear(&vec![3.0; 65], 65, 1, 2).unwrap();
        cheetah::native::vulkan::shutdown().unwrap();
        assert_eq!(output.len(), 130);
        for row in output.chunks_exact(2) {
            assert_eq!(row, &[6.0, -3.0]);
        }
    }
}

#[cfg(test)]
mod q8_tests {
    use super::*;

    /// CHEETAH_QUANT_Q8=1 + a synthetic BF16 decoder: the Vulkan executor's
    /// q8-quantized output must stay finite and terminate (a hang here means
    /// the resident q8 matmul deadlocks). Set CHEETAH_TEST_VULKAN_Q8=1 to run.
    #[test]
    fn vulkan_q8_quantized_forward_terminates() {
        if std::env::var("CHEETAH_TEST_VULKAN_Q8").is_err() {
            eprintln!("skipping q8 forward test (set CHEETAH_TEST_VULKAN_Q8=1)");
            return;
        }
        // SAFETY: test-scoped process env mutation; single-threaded test.
        unsafe { std::env::set_var("CHEETAH_QUANT_Q8", "1") };
        // Native forward tests live in the native module; rebuild
        // the tiny model here via the same helper through the native tests is
        // not reachable, so prepare a minimal directory directly.
        let base = std::env::temp_dir().join(format!(
            "cheetah-synth-q8-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("config.json"),
            r#"{
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "max_position_embeddings": 512
            }"#,
        )
        .unwrap();

        let mut shape_f32: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        shape_f32.push((
            "model.embed_tokens.weight".into(),
            vec![0.0; 128 * 64],
            vec![128, 64],
        ));
        shape_f32.push(("lm_head.weight".into(), vec![0.0; 128 * 64], vec![128, 64]));
        shape_f32.push(("model.norm.weight".into(), vec![1.0; 64], vec![64]));
        for layer in 0..2usize {
            let prefix = format!("model.layers.{layer}.");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shape_f32.push((format!("{prefix}{name}"), vec![1.0; 64], vec![64]));
            }
            shape_f32.push((
                format!("{prefix}self_attn.q_proj.weight"),
                vec![0.5; 4096],
                vec![64, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.k_proj.weight"),
                vec![0.25; 2048],
                vec![32, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.v_proj.weight"),
                vec![0.25; 2048],
                vec![32, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.o_proj.weight"),
                vec![0.5; 4096],
                vec![64, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.gate_proj.weight"),
                vec![0.5; 8192],
                vec![128, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.up_proj.weight"),
                vec![0.5; 8192],
                vec![128, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.down_proj.weight"),
                vec![0.5; 8192],
                vec![64, 128],
            ));
        }

        let tensors: Vec<(String, safetensors::tensor::TensorView<'static>)> = shape_f32
            .into_iter()
            .map(|(name, data, shape)| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for value in &data {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                (
                    name,
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, leaked)
                        .unwrap(),
                )
            })
            .collect();
        std::fs::write(
            src.join("model.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();

        cheetah::native::init_model(&src, &dst).unwrap();

        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dst.join(".cheetah/weights"))
            .unwrap()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
                    .then_some(path)
            })
            .collect();
        let plan = std::collections::BTreeMap::new(); // all units resident
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join("config.json")).unwrap()).unwrap();
        // Resolve the llama family's own executor through the registration
        // table — the same path a real `CHEETAH_DEVICE=vulkan` load takes.
        let crate::chat::registry::BackendLoader::Vulkan(load) =
            crate::chat::registry::backend_loader_for_architecture("llama", "vulkan")
                .expect("llama registers a vulkan backend")
        else {
            unreachable!()
        };
        let mut model = load(&paths, &[], raw, &plan).unwrap();

        let tokens: Vec<u32> = vec![3, 41, 17];
        let started = std::time::Instant::now();
        let logits = model.forward(&tokens, 0).unwrap();
        eprintln!(
            "q8 forward finished in {:?}; finite={}",
            started.elapsed(),
            logits.iter().all(|value| f32::is_finite(*value))
        );
        assert!(logits.iter().all(|value| f32::is_finite(*value)));

        let _ = std::fs::remove_dir_all(&base);
        // SAFETY: test-scoped process env mutation.
        unsafe { std::env::remove_var("CHEETAH_QUANT_Q8") };
        cheetah::native::vulkan::shutdown().unwrap();
    }
}
