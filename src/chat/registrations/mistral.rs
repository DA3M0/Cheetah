//! Registration for `mistral` — declarative capability plus the family's
//! native forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the llama-shaped order — embedding → per layer (input
//! RMSNorm → attention → residual → post-attention RMSNorm → SiLU gated MLP
//! → residual) → final RMSNorm → lm head. Mistral specifics: the
//! decoding-time sliding window (the HF Mistral modeling masks keys older
//! than `sliding_window` during cached decoding — the resolved window flows
//! through the toolbox attention op), untied lm head, and optional attention
//! biases.

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, OpsToolbox, Scratch};
use super::templates::mistral;
use super::{AntTierControl, BackendLoader, ChatTemplate, ModelFamily, ModelForward, ModelRegistration};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["MistralForCausalLM", "mistral"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "mistral",
        render: mistral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(mistral_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(mistral_native_loader)),
            ("vulkan", BackendLoader::Vulkan(mistral_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(mistral_native_loader))]
    }
}

/// Family constants for the Mistral family: the llama-shaped baseline with
/// the decoding-time sliding window (the HF Mistral modeling masks keys
/// older than `sliding_window` during cached decoding, which the native
/// decoder implements as the uniform window).
const fn mistral_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32000, 32, 8, 128), // Mistral-7B / 7B-Instruct-v0.2 (GQA)
            (4096, 32768, 32, 8, 128), // Mistral-7B-v0.3 (extended vocab)
        ],
        extra_config_json: "{\"hidden_act\":\"silu\",\"sliding_window\":4096}",
        max_position_embeddings: Some(32768),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The mistral family's native runtime: the llama-shaped
/// sequential-residual block with the decoding-time sliding window (the
/// window itself is resolved config data the attention op applies).
#[derive(Clone)]
pub(crate) struct MistralNative {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
}

impl MistralNative {
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
        // Mistral keeps a separate lm_head tensor; the tied branch reads the
        // embedding table instead.
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

    /// One token position: embedding row lookup, then every layer, then the
    /// final norm.
    fn forward_position(&mut self, token: usize, position: usize) -> Result<(), String> {
        self.ctx
            .embedding_row_into(token, &mut self.scratch.hidden)?;
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.ctx.drop_disk_units_of_layer(layer as i32);
        }
        self.ctx.rmsnorm("model.norm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer: input norm → attention → residual add →
    /// post-attention norm → MLP → residual add.
    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let prefix = format!("model.layers.{layer}.");

        // Input norm: the attention reads it.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx
            .rmsnorm(&format!("{prefix}input_layernorm"), &mut self.scratch.normed)?;

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
        self.ctx.rmsnorm(
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
    /// `scratch.attn_projected` (the caller owns the residual add). The
    /// mistral delta over the llama baseline is the decoding-time sliding
    /// window, which the toolbox attention op resolves from the model config.
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
        // the visible cached keys — `sliding_window` masks keys older than
        // the window out of the softmax (0 = full attention).
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
    /// the caller owns the residual add. SwiGLU: `down(silu(gate·x) ⊙ up·x)`.
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

        ops::silu_multiply(&mut self.scratch.gate, &self.scratch.up);

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

impl ModelForward for MistralNative {
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

impl AntTierControl for MistralNative {
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
fn mistral_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(MistralNative::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Vulkan transformer features for the Mistral family.
#[cfg(feature = "vulkan")]
fn mistral_vulkan_config(
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
    crate::chat::vulkan::transformer::VulkanTransformerConfig::from_raw(raw_config, features)
}

/// Vulkan loader for this registration: build the generic transformer
/// executor around the mistral feature set.
#[cfg(feature = "vulkan")]
fn mistral_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = mistral_vulkan_config(&raw_config)?;
    Ok(Box::new(
        crate::chat::vulkan::transformer::VulkanTransformerExecutor::load(
            weight_paths, &config, plan,
        )?,
    ))
}
