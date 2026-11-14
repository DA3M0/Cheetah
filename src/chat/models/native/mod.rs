//! Native (Candle-free) CPU inference backend. Weights stay mmap-backed from
//! the prepared Ant shards and every Ant tier acts on real pages via madvise,
//! so the resident set only holds what the plan keeps resident.

pub(super) mod kernels;
pub(super) mod qwen2;
pub(super) mod store;

use super::{AntTierControl, ModelRuntime};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Registry-facing loader for the native decoder-family CPU backend (llama /
/// qwen2 shape: GQA attention, rotate-half rope, silu MLP; attention biases
/// and tied embeddings are per-family). Registrations expose this as their
/// native backend loader, so selecting the native backend goes through the
/// same registry as the Candle and Vulkan loaders.
pub(super) fn native_family_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &serde_json::Value,
    plan: &BTreeMap<String, String>,
    family: super::ModelFamily,
) -> Result<Box<dyn ModelRuntime>, String> {
    let store = Arc::new(store::MmapWeightStore::open(model_dir)?);
    let cfg = qwen2::Qwen2Config::resolve(Some(raw_config), &store, Some(family))?;
    let mut qwen = qwen2::Qwen2Native::with_store(cfg, Arc::clone(&store));
    qwen.apply_tiers(plan);
    Ok(Box::new(NativeCpuRuntime { qwen }))
}

pub(super) struct NativeCpuRuntime {
    qwen: qwen2::Qwen2Native,
}

impl Clone for NativeCpuRuntime {
    fn clone(&self) -> Self {
        Self {
            qwen: self.qwen.clone(),
        }
    }
}

impl ModelRuntime for NativeCpuRuntime {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.qwen.forward(tokens, offset)
    }

    fn clear_cache(&mut self) {
        self.qwen.clear_kv();
    }

    fn device_name(&self) -> &'static str {
        "native-cpu"
    }

    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

impl AntTierControl for NativeCpuRuntime {
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.qwen.page_in_unit(unit_id)
    }

    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.qwen.evict_unit(unit_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 fill in [-1, 1).
    fn fill(name: &str, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let hash = (name.len() as u32)
                    .wrapping_mul(31)
                    .wrapping_add(index as u32)
                    .wrapping_mul(2654435761);
                ((hash % 2000) as f32 / 1000.0) - 1.0
            })
            .collect()
    }

    fn tensor_view(
        name: &str,
        shape: Vec<usize>,
    ) -> (String, safetensors::tensor::TensorView<'static>) {
        let elements: usize = shape.iter().product();
        let data_f32 = fill(name, elements);
        let mut data = Vec::with_capacity(elements * 4);
        for value in &data_f32 {
            data.extend_from_slice(&value.to_le_bytes());
        }
        // SAFETY: the bytes stay alive for the whole test via a leak; the
        // synthetic model is tiny (~tens of KB).
        let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
        (
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, leaked).unwrap(),
        )
    }

    /// Write a tiny decoder-family model (2 layers, hidden 64, GQA 4/2,
    /// head_dim 16, vocab 128) and run `cheetah init` over it. `qk_norm` adds
    /// qwen3-style per-head q/k RMSNorm units; `tie` controls embedding/lm_head
    /// sharing.
    fn prepare_synthetic_decoder(
        label: &str,
        model_type: &str,
        architecture: &str,
        tie: bool,
        qk_norm: bool,
    ) -> std::path::PathBuf {
        let base = std::env::temp_dir().join(format!(
            "cheetah-synth-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();

        let mut config = serde_json::json!({
            "model_type": model_type,
            "architectures": [architecture],
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": tie,
            "max_position_embeddings": 512,
        });
        if qk_norm {
            config["head_dim"] = serde_json::json!(16);
            config["attention_bias"] = serde_json::json!(false);
        }
        std::fs::write(
            src.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();

        let hidden = 64usize;
        let head_dim = 16usize;
        let mut shape_f32: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        shape_f32.push((
            "model.embed_tokens.weight".into(),
            fill("embed", 128 * hidden),
            vec![128, hidden],
        ));
        if !tie {
            shape_f32.push((
                "lm_head.weight".into(),
                fill("lm_head", 128 * hidden),
                vec![128, hidden],
            ));
        }
        shape_f32.push(("model.norm.weight".into(), fill("norm", hidden), vec![hidden]));
        for layer in 0..2usize {
            let prefix = format!("model.layers.{layer}.");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shape_f32.push((format!("{prefix}{name}"), fill(name, hidden), vec![hidden]));
            }
            shape_f32.push((
                format!("{prefix}self_attn.q_proj.weight"),
                fill("q", hidden * hidden),
                vec![hidden, hidden],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.k_proj.weight"),
                fill("k", 32 * hidden),
                vec![32, hidden],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.v_proj.weight"),
                fill("v", 32 * hidden),
                vec![32, hidden],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.o_proj.weight"),
                fill("o", hidden * hidden),
                vec![hidden, hidden],
            ));
            shape_f32.push((
                format!("{prefix}mlp.gate_proj.weight"),
                fill("gate", 128 * hidden),
                vec![128, hidden],
            ));
            shape_f32.push((
                format!("{prefix}mlp.up_proj.weight"),
                fill("up", 128 * hidden),
                vec![128, hidden],
            ));
            shape_f32.push((
                format!("{prefix}mlp.down_proj.weight"),
                fill("down", hidden * 128),
                vec![hidden, 128],
            ));
            if qk_norm {
                shape_f32.push((
                    format!("{prefix}self_attn.q_norm.weight"),
                    fill("qn", head_dim),
                    vec![head_dim],
                ));
                shape_f32.push((
                    format!("{prefix}self_attn.k_norm.weight"),
                    fill("kn", head_dim),
                    vec![head_dim],
                ));
            }
        }

        let tensors: Vec<(String, safetensors::tensor::TensorView<'static>)> = shape_f32
            .into_iter()
            .map(|(name, _, shape)| tensor_view(&name, shape))
            .collect();
        let serialized = safetensors::serialize(tensors, &None).unwrap();
        std::fs::write(src.join("model.safetensors"), serialized).unwrap();

        cheetah::native::init_model(&src, &dst).expect("synthetic decoder init");
        dst
    }

    /// The native decoder runner must reproduce the Candle runtime's logits
    /// for a synthetic model: prefill argmax, prefill max diff, decode max diff.
    fn compare_native_vs_candle(model_dir: &std::path::Path, architecture: &str, label: &str) {
        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();

        let store = Arc::new(store::MmapWeightStore::open(model_dir).unwrap());
        let family = crate::chat::models::family_for_architecture(architecture).unwrap();
        let cfg = qwen2::Qwen2Config::resolve(Some(&raw), &store, Some(family)).unwrap();
        let mut native = qwen2::Qwen2Native::with_store(cfg, Arc::clone(&store));

        let registration = crate::chat::models::registration_for_architecture(architecture).unwrap();
        let mut weight_paths: Vec<std::path::PathBuf> =
            std::fs::read_dir(model_dir.join(".cheetah/weights"))
                .unwrap()
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    (path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
                        .then_some(path)
                })
                .collect();
        weight_paths.sort();
        let mut candle_model = crate::chat::models::candle::load_model(
            &weight_paths,
            &config_bytes,
            raw.clone(),
            registration,
            &candle_core::Device::Cpu,
        )
        .unwrap();

        let tokens: Vec<u32> = vec![3, 41, 17, 88, 5];
        let native_prefill = native.forward(&tokens, 0).unwrap();
        let candle_prefill = candle_model.forward(&tokens, 0).unwrap();
        let prefill_diff = native_prefill
            .iter()
            .zip(&candle_prefill)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(
            native_prefill
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0,
            candle_prefill
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0,
            "prefill argmax diverges"
        );

        let native_decode = native.forward(&[7], tokens.len()).unwrap();
        let candle_decode = candle_model.forward(&[7], tokens.len()).unwrap();
        let decode_diff = native_decode
            .iter()
            .zip(&candle_decode)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("synthetic {label}: prefill max|diff|={prefill_diff:.5} decode max|diff|={decode_diff:.5}");
        assert!(prefill_diff < 1e-3, "prefill logits diverge ({prefill_diff})");
        assert!(decode_diff < 1e-3, "decode logits diverge ({decode_diff})");

        let _ = std::fs::remove_dir_all(&model_dir.parent().unwrap());
    }

    /// Covers the no-bias and separate-lm_head paths that the qwen2
    /// comparisons do not exercise.
    #[test]
    fn native_llama_matches_candle_on_synthetic_model() {
        let model_dir = prepare_synthetic_decoder("llama", "llama", "LlamaForCausalLM", false, false);
        compare_native_vs_candle(&model_dir, "llama", "llama");
    }

    /// Covers the qwen3-specific per-head q/k RMSNorm path (before rope) plus
    /// explicit head_dim resolution.
    #[test]
    fn native_qwen3_matches_candle_on_synthetic_model() {
        let model_dir = prepare_synthetic_decoder("qwen3", "qwen3", "Qwen3ForCausalLM", false, true);
        compare_native_vs_candle(&model_dir, "qwen3", "qwen3");
    }
}
