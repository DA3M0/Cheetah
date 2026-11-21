//! Registration for `phi` — declarative capability plus the family's native
//! forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the phi order — embedding → per layer (input LayerNorm →
//! attention and MLP in parallel off the same normed state → single
//! `x + attn_out + mlp_out` residual update) → final LayerNorm → lm head.
//! Phi specifics: biased LayerNorms (not RMSNorm), fused `qkv_proj`
//! attention input + `dense` output projection, a plain `fc1`/`fc2` GELU
//! MLP, partial rotary embeddings (rope touches only the leading
//! `rotary_dim` share of each head), and an untied lm head.

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, OpsToolbox, Scratch};
#[cfg(test)]
use super::native::store::MmapWeightStore;
use super::templates::phi;
use super::{AntTierControl, BackendLoader, ChatTemplate, ModelFamily, ModelForward, ModelRegistration};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
#[cfg(test)]
use std::sync::Arc;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["PhiForCausalLM", "phi"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "phi",
        render: phi,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(phi_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and the dedicated `VulkanPhiExecutor` when compiled
/// in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(phi_native_loader)),
            ("vulkan", BackendLoader::Vulkan(phi_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(phi_native_loader))]
    }
}

/// Family constants for the Phi family: biased LayerNorms, a parallel
/// attention/MLP residual, fused `qkv_proj` + `dense` attention, `fc1`/`fc2`
/// MLP, and partial rotary embeddings (the exact rotary dim resolves per
/// model from `partial_rotary_factor * head_dim`; phi-1.5 carries 0.5 and
/// phi-2 carries 0.4 in their configs, so the family constant is the phi-2
/// fallback).
const fn phi_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (2048, 50257, 32, 32, 64), // phi-1.5 (head_dim 2048/32)
            (2048, 51200, 32, 32, 64), // phi-1.5 padded vocab
            (2560, 51200, 32, 32, 80), // phi-2 (head_dim 2560/32)
        ],
        extra_config_json: "{\"hidden_act\":\"gelu_new\"}",
        max_position_embeddings: Some(2048),
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::PHI_FEATURES,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The phi family's native runtime: the parallel-residual block topology
/// with biased LayerNorms, fused attention projections, and the plain GELU
/// MLP.
#[derive(Clone)]
pub(crate) struct PhiNative {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
}

impl PhiNative {
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
        // Phi keeps a separate lm_head tensor; the tied branch reads the
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

    /// One token position: embedding row lookup (unscaled — phi has no
    /// embedding scale), then every layer, then the final norm. Phi's
    /// GPT-NeoX heritage names it `final_layernorm`.
    fn forward_position(&mut self, token: usize, position: usize) -> Result<(), String> {
        self.ctx
            .embedding_row_into(token, &mut self.scratch.hidden)?;
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.ctx.drop_disk_units_of_layer(layer as i32);
        }
        self.ctx
            .layernorm("model.final_layernorm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer, phi shape: one input LayerNorm feeds both the
    /// attention and the MLP (parallel residual), and the residual is the
    /// single update `x + attn_out + mlp_out` — no post-attention norm unit.
    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let prefix = format!("model.layers.{layer}.");

        // Input norm: attention AND MLP read the same normed state.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx
            .layernorm(&format!("{prefix}input_layernorm"), &mut self.scratch.normed)?;

        self.forward_attention(&prefix, layer, position)?;
        self.forward_mlp(&prefix)?;

        // Parallel residual: one update sums attention and MLP outputs.
        let mlp_out = std::mem::take(&mut self.scratch.mlp_out);
        for ((residual, attn), mlp) in self
            .scratch
            .hidden
            .iter_mut()
            .zip(self.scratch.attn_projected.iter())
            .zip(mlp_out.iter())
        {
            *residual += attn + mlp;
        }
        self.scratch.mlp_out = mlp_out;
        Ok(())
    }

    /// Attention block over `scratch.normed`, written to
    /// `scratch.attn_projected` (the caller owns the residual add). The fused
    /// `qkv_proj` stacks q/k/v rows — split after the projection (+ fused
    /// bias) — rope touches only the leading `rotary_dim` share of each head,
    /// and the output projection is named `dense`.
    fn forward_attention(
        &mut self,
        prefix: &str,
        layer: usize,
        position: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_dim = cfg.q_heads * cfg.head_dim;
        let kv_dim = cfg.kv_heads * cfg.head_dim;

        // One `qkv_proj` unit stacks q/k/v rows; split after the projection.
        let fused_rows = (cfg.q_heads + 2 * cfg.kv_heads) * cfg.head_dim;
        self.scratch.q.clear();
        self.scratch.q.resize(fused_rows, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.qkv_proj.weight"),
            fused_rows,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.q,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.q, &format!("{prefix}self_attn.qkv_proj"));
        self.scratch.k.clear();
        self.scratch
            .k
            .extend_from_slice(&self.scratch.q[q_dim..q_dim + kv_dim]);
        self.scratch.v.clear();
        self.scratch
            .v
            .extend_from_slice(&self.scratch.q[q_dim + kv_dim..q_dim + 2 * kv_dim]);
        self.scratch.q.truncate(q_dim);

        // Partial rotary: rope over the leading `rotary_dim` dims of each
        // head (remaining dims pass through untouched).
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

        // Append K/V to the cache for this position, then attention over the
        // visible cached keys (the resolved sliding window applies).
        self.ctx.cache_append(layer, &self.scratch.k, &self.scratch.v);
        self.scratch.attn_out.clear();
        self.scratch.attn_out.resize(q_dim, 0.0);
        self.ctx
            .attention(layer, position, &self.scratch.q, &mut self.scratch.attn_out);

        // Output projection: phi names it `dense` (plus its bias).
        self.scratch.attn_projected.clear();
        self.scratch
            .attn_projected
            .resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.dense.weight"),
            cfg.hidden_size,
            q_dim,
            &self.scratch.attn_out,
            &mut self.scratch.attn_projected,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.attn_projected, &format!("{prefix}self_attn.dense"));
        Ok(())
    }

    /// Feed-forward over `scratch.normed`, written to `scratch.mlp_out` —
    /// the caller owns the residual add. Phi plain MLP: `fc2(gelu(fc1·x + b)
    /// ) + b` with the tanh-approximation GELU.
    fn forward_mlp(&mut self, prefix: &str) -> Result<(), String> {
        let cfg = &self.cfg;

        self.scratch.gate.clear();
        self.scratch.gate.resize(cfg.intermediate_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.fc1.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.gate,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.gate, &format!("{prefix}mlp.fc1"));
        ops::gelu(&mut self.scratch.gate);

        self.scratch.mlp_out.clear();
        self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.fc2.weight"),
            cfg.hidden_size,
            cfg.intermediate_size,
            &self.scratch.gate,
            &mut self.scratch.mlp_out,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.mlp_out, &format!("{prefix}mlp.fc2"));
        Ok(())
    }
}

impl ModelForward for PhiNative {
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

impl AntTierControl for PhiNative {
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
fn phi_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(PhiNative::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Phi runs its own executor: a different block topology (biased LayerNorms,
/// parallel residual, fused projections), so it does not share the generic
/// transformer config builder.
#[cfg(feature = "vulkan")]
fn phi_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    _plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = crate::chat::vulkan::phi::VulkanPhiConfig::parse(&raw_config)?;
    Ok(Box::new(crate::chat::vulkan::phi::VulkanPhiExecutor::load(
        weight_paths,
        &config,
    )?))
}
