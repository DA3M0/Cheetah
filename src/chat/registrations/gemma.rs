//! Registration for `gemma` — declarative capability plus the family's
//! native forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the gemma order — sqrt(hidden)-scaled embedding → per layer
//! (offset RMSNorm → attention → residual → offset RMSNorm → GeGLU →
//! residual) → final offset RMSNorm → lm head. Gemma specifics: RMSNorm
//! applies the affine weight with an implicit `(1 + weight)` offset, the
//! embedding output is scaled by `sqrt(hidden_size)` (the lm head later
//! reads the same unscaled table), the feed-forward activation is the tanh
//! GELU (GeGLU: `down(gelu(gate·x) ⊙ up·x)`), and the embeddings are tied.
//! gemma-1 applies full attention on every layer (the config's
//! `sliding_window` is not enforced by the HF Gemma modeling), so no window
//! is declared here.

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, OpsToolbox, Scratch};
#[cfg(test)]
use super::native::store::MmapWeightStore;
use super::templates::gemma;
use super::{AntTierControl, BackendLoader, ChatTemplate, ModelFamily, ModelForward, ModelRegistration};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
#[cfg(test)]
use std::sync::Arc;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["GemmaForCausalLM", "gemma"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "gemma",
        render: gemma,
        header_renderer: None,
    },
    stop_markers: &["<end_of_turn>"],
    backends: model_backends(),
    family: Some(gemma_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(gemma_native_loader)),
            ("vulkan", BackendLoader::Vulkan(gemma_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(gemma_native_loader))]
    }
}

/// Family constants for the Gemma family: offset RMSNorms `x/rms · (1 + w)`,
/// a sqrt(hidden)-scaled embedding, GeGLU (tanh-GELU gated) feed-forward, and
/// tied embeddings.
const fn gemma_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (2048, 256128, 8, 1, 256),  // gemma-2b / 2b-it
            (2048, 256000, 8, 1, 256),  // gemma-2b mirrors with padded-down vocab
            (3072, 256000, 16, 1, 256), // gemma-7b / 7b-it
            (3072, 256128, 16, 1, 256), // gemma-1.1-7b
        ],
        extra_config_json: "{\"hidden_act\":\"gelu_pytorch_tanh\"}",
        max_position_embeddings: Some(8192),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::GEMMA_FEATURES,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The gemma family's native runtime: the llama-shaped block topology with
/// the family's own norm flavor, embedding scale, and GeGLU activation.
#[derive(Clone)]
pub(crate) struct GemmaNative {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
}

impl GemmaNative {
    /// Fixture/test constructor around an already-resolved config and store.
    #[cfg(test)]
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        let ctx = OpsToolbox::with_store(cfg.clone(), store);
        Self {
            scratch: Scratch::new(&cfg),
            cfg,
            ctx,
        }
    }

    /// Loader-path constructor: the toolbox already carries the resolved
    /// config and the installed Ant residency plan.
    fn new(ctx: OpsToolbox) -> Self {
        let cfg = ctx.cfg.clone();
        Self {
            scratch: Scratch::new(&cfg),
            cfg,
            ctx,
        }
    }

    /// Run the full forward for tokens starting at `offset`; returns the
    /// logits of the final position only (matching the engine's sampling
    /// contract).
    fn run_forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.ctx.validate_tokens(tokens)?;
        for (index, &token) in tokens.iter().enumerate() {
            self.forward_position(token as usize, offset + index)?;
        }
        // Gemma ties the lm head to the (unscaled) embedding table; the
        // untied branch reads a separate `lm_head.weight` tensor instead.
        let head_unit = if self.cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        self.ctx
            .lm_head_into(head_unit, &self.scratch.hidden, &mut self.scratch.logits)?;
        // Ant: layer 0's bucket also holds the embedding / final norm, whose
        // pages lm_head just swept; drop the disk-tier ones now. The pages
        // re-fault (from the OS page cache) on their next access.
        self.ctx.drop_disk_units_of_layer(0);
        Ok(std::mem::take(&mut self.scratch.logits))
    }

    /// One token position: scaled embedding row lookup, then every layer,
    /// then the final norm.
    fn forward_position(&mut self, token: usize, position: usize) -> Result<(), String> {
        self.ctx
            .embedding_row_into(token, &mut self.scratch.hidden)?;
        // gemma scales the embedding output by sqrt(hidden_size); the lm_head
        // later reads the same (unscaled) table directly.
        let scale = (self.cfg.hidden_size as f32).sqrt();
        for value in self.scratch.hidden.iter_mut() {
            *value *= scale;
        }
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.ctx.drop_disk_units_of_layer(layer as i32);
        }
        self.ctx
            .rmsnorm_offset("model.norm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer: offset input norm → attention → residual add →
    /// offset post-attention norm → GeGLU → residual add.
    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let prefix = format!("model.layers.{layer}.");

        // Input norm: the attention reads it.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx.rmsnorm_offset(
            &format!("{prefix}input_layernorm"),
            &mut self.scratch.normed,
        )?;

        self.forward_attention(&prefix, layer, position)?;

        // Residual add: hidden += attn_projected.
        for (value, residual) in self
            .scratch
            .attn_projected
            .iter()
            .zip(self.scratch.hidden.iter_mut())
        {
            *residual += *value;
        }

        // Post-attention norm, then the feed-forward block, then its residual.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx.rmsnorm_offset(
            &format!("{prefix}post_attention_layernorm"),
            &mut self.scratch.normed,
        )?;
        self.forward_mlp(&prefix)?;
        let mlp_out = std::mem::take(&mut self.scratch.mlp_out);
        for (value, residual) in mlp_out.iter().zip(self.scratch.hidden.iter_mut()) {
            *residual += value;
        }
        self.scratch.mlp_out = mlp_out;
        Ok(())
    }

    /// Attention block over `scratch.normed`, written to
    /// `scratch.attn_projected` (the caller owns the residual add). Split
    /// `q/k/v_proj` projections, full-span rotate-half rope, GQA attention,
    /// then `o_proj`. Biases are family-optional — the adds are unit-optional
    /// no-ops (gemma carries none).
    fn forward_attention(
        &mut self,
        prefix: &str,
        layer: usize,
        position: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_dim = cfg.q_heads * cfg.head_dim;
        let kv_dim = cfg.kv_heads * cfg.head_dim;

        self.scratch.q.clear();
        self.scratch.q.resize(q_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.q_proj.weight"),
            q_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.q,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.q, &format!("{prefix}self_attn.q_proj"));

        self.scratch.k.clear();
        self.scratch.k.resize(kv_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.k_proj.weight"),
            kv_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.k,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.k, &format!("{prefix}self_attn.k_proj"));

        self.scratch.v.clear();
        self.scratch.v.resize(kv_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.v_proj.weight"),
            kv_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.v,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.v, &format!("{prefix}self_attn.v_proj"));

        // Rotate-half rope over the full head span of every head.
        ops::rope_heads(
            &mut self.scratch.q,
            cfg.q_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        );
        ops::rope_heads(
            &mut self.scratch.k,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        );

        // Append K/V to the cache for this position, then GQA attention over
        // the visible cached keys (the resolved sliding window applies —
        // gemma-1 resolves it to full attention).
        self.ctx.cache_append(layer, &self.scratch.k, &self.scratch.v);
        self.scratch.attn_out.clear();
        self.scratch.attn_out.resize(q_dim, 0.0);
        self.ctx
            .attention(layer, position, &self.scratch.q, &mut self.scratch.attn_out);

        // Output projection (plus optional bias).
        self.scratch.attn_projected.clear();
        self.scratch
            .attn_projected
            .resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.o_proj.weight"),
            cfg.hidden_size,
            q_dim,
            &self.scratch.attn_out,
            &mut self.scratch.attn_projected,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.attn_projected, &format!("{prefix}self_attn.o_proj"));
        Ok(())
    }

    /// Feed-forward over `scratch.normed`, written to `scratch.mlp_out` —
    /// the caller owns the residual add. GeGLU: `down(gelu(gate·x) ⊙ up·x)`
    /// with the tanh-approximation GELU.
    fn forward_mlp(&mut self, prefix: &str) -> Result<(), String> {
        let cfg = &self.cfg;

        self.scratch.gate.clear();
        self.scratch.gate.resize(cfg.intermediate_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.gate_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.gate,
        )?;

        self.scratch.up.clear();
        self.scratch.up.resize(cfg.intermediate_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.up_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.up,
        )?;

        ops::gelu_multiply(&mut self.scratch.gate, &self.scratch.up);

        self.scratch.mlp_out.clear();
        self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.down_proj.weight"),
            cfg.hidden_size,
            cfg.intermediate_size,
            &self.scratch.gate,
            &mut self.scratch.mlp_out,
        )?;
        Ok(())
    }
}

impl ModelForward for GemmaNative {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.run_forward(tokens, offset)
    }

    fn clear_cache(&mut self) {
        self.ctx.clear_kv();
    }

    fn device_name(&self) -> &'static str {
        "native-cpu"
    }

    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

impl AntTierControl for GemmaNative {
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.ctx.page_in_unit(unit_id)
    }

    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.ctx.evict_unit(unit_id)
    }
}

/// Native backend loader for this registration: opens the mmap store,
/// resolves the decoder config through the parameter fallback chain, and
/// installs the Ant residency plan before the first forward.
fn gemma_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(GemmaNative::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Vulkan transformer features for the Gemma family: offset norms, embedding
/// scale, tied embeddings.
#[cfg(feature = "vulkan")]
fn gemma_vulkan_config(
    raw_config: &Value,
) -> Result<crate::chat::vulkan::transformer::VulkanTransformerConfig, String> {
    use crate::chat::vulkan::transformer::TransformerFeatures;

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
    crate::chat::vulkan::transformer::VulkanTransformerConfig::from_raw(raw_config, features)
}

/// Vulkan loader for this registration: build the generic transformer
/// executor around the gemma feature set.
#[cfg(feature = "vulkan")]
fn gemma_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = gemma_vulkan_config(&raw_config)?;
    Ok(Box::new(
        crate::chat::vulkan::transformer::VulkanTransformerExecutor::load(
            weight_paths, &config, plan,
        )?,
    ))
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_feature_tests {
    use super::*;

    #[test]
    fn gemma_vulkan_features_configure_norm_and_embedding_behavior() {
        let config = gemma_vulkan_config(&serde_json::json!({
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
}
