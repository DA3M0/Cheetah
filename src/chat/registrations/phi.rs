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
use super::native::ops::{self, ChunkScratch, OpsToolbox, Scratch};
#[cfg(test)]
use super::native::store::MmapWeightStore;
use super::templates::phi;
use super::{
    AntTierControl, BackendLoader, ChatTemplate, KvSnapshot, ModelFamily, ModelForward,
    ModelRegistration,
};
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
    /// Chunked-prefill activation buffers (`[rows, width]` stacks).
    pub(crate) chunk: ChunkScratch,
}

impl PhiNative {
    /// Fixture/test constructor around an already-resolved config and store.
    #[cfg(test)]
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        let ctx = OpsToolbox::with_store(cfg.clone(), store);
        Self {
            scratch: Scratch::new(&cfg),
            chunk: ChunkScratch::new(),
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
            chunk: ChunkScratch::new(),
            cfg,
            ctx,
        }
    }

    /// Run the full forward for tokens starting at `offset`; returns the
    /// logits of the final position only (matching the engine's sampling
    /// contract). Single tokens (decode steps) run the per-position path;
    /// longer batches (prefill) split into chunked batched-GEMM passes.
    fn run_forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.forward_with_chunk(tokens, offset, ops::prefill_chunk_size())
    }

    /// [`Self::run_forward`] with an explicit chunk size — tests force
    /// `chunk_rows = 1` as the per-position reference.
    pub(crate) fn forward_with_chunk(
        &mut self,
        tokens: &[u32],
        offset: usize,
        chunk_rows: usize,
    ) -> Result<Vec<f32>, String> {
        self.ctx.validate_tokens(tokens)?;
        // Reserve the KV span for the whole prefill up front: growth into
        // reserved capacity instead of doubling copies mid-prefill.
        self.ctx.reserve_kv(offset + tokens.len());
        if tokens.len() == 1 {
            self.forward_position(tokens[0] as usize, offset)?;
        } else {
            self.forward_chunked(tokens, offset, chunk_rows)?;
        }
        self.lm_head()
    }

    /// Phi keeps a separate lm_head tensor; the tied branch reads the
    /// embedding table instead. Ant: layer 0's bucket also holds the
    /// embedding / final norm, whose pages lm_head just swept; the disk-tier
    /// ones drop now and re-fault from the OS page cache.
    fn lm_head(&mut self) -> Result<Vec<f32>, String> {
        let head_unit = if self.cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        self.ctx
            .lm_head_into(head_unit, &self.scratch.hidden, &mut self.scratch.logits)?;
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

    /// Chunked prefill: splits `tokens` into `chunk_rows`-sized batches and
    /// processes each through [`Self::forward_chunk`]. The projections run as
    /// batched GEMMs (the per-token path is a GEMV per token, re-streaming
    /// every weight once per position); norms, rope, KV append and attention
    /// stay per-row, which keeps the result bit-exact against the
    /// per-position path.
    pub(crate) fn forward_chunked(
        &mut self,
        tokens: &[u32],
        offset: usize,
        chunk_rows: usize,
    ) -> Result<(), String> {
        for start in (0..tokens.len()).step_by(chunk_rows.max(1)) {
            let end = (start + chunk_rows).min(tokens.len());
            self.forward_chunk(&tokens[start..end], offset + start)?;
        }
        Ok(())
    }

    /// One chunk of prefill positions as batched GEMMs, ending with the final
    /// norm of the last position (the only one whose logits the caller reads).
    fn forward_chunk(&mut self, tokens: &[u32], offset: usize) -> Result<(), String> {
        let rows = tokens.len();
        let hidden = self.cfg.hidden_size;
        self.chunk.resize_for(&self.cfg, rows);
        // Stack the embedding rows (unscaled — phi has no embedding scale).
        self.chunk.hidden.clear();
        let mut row = std::mem::take(&mut self.scratch.hidden);
        for &token in tokens {
            self.ctx.embedding_row_into(token as usize, &mut row)?;
            self.chunk.hidden.extend_from_slice(&row);
        }
        self.scratch.hidden = row;
        for layer in 0..self.cfg.num_layers {
            self.forward_layer_chunk(layer, offset, rows)?;
            // Ant: same per-layer page release as the per-position path.
            self.ctx.drop_disk_units_of_layer(layer as i32);
        }
        // Only the final position feeds the LM head: norm its residual row
        // into `scratch.hidden` (the caller's logits input). Phi's GPT-NeoX
        // heritage names it `final_layernorm`.
        let last_row_start = (rows - 1) * hidden;
        self.scratch.hidden.clear();
        self.scratch
            .hidden
            .extend_from_slice(&self.chunk.hidden[last_row_start..last_row_start + hidden]);
        self.ctx
            .layernorm("model.final_layernorm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer over a whole chunk — the phi parallel-residual block
    /// with the projections batched: one input LayerNorm feeds attention AND
    /// MLP, the fused `qkv_proj` runs as one GEMM, and the residual is the
    /// single update `x + attn_out + mlp_out`.
    fn forward_layer_chunk(
        &mut self,
        layer: usize,
        offset: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_dim = cfg.q_heads * cfg.head_dim;
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        let fused_rows = (cfg.q_heads + 2 * cfg.kv_heads) * cfg.head_dim;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let prefix = format!("model.layers.{layer}.");

        // Input norm per row: attention AND MLP read the same normed state.
        self.chunk.normed.clear();
        self.chunk.normed.resize(rows * hidden, 0.0);
        for row in 0..rows {
            let range = row * hidden..(row + 1) * hidden;
            self.chunk.normed[range.clone()].copy_from_slice(&self.chunk.hidden[range.clone()]);
            self.ctx
                .layernorm(&format!("{prefix}input_layernorm"), &mut self.chunk.normed[range])?;
        }

        // Fused qkv projection as one batched GEMM (plus its fused bias per
        // row), then split into q/k/v the way the per-position path does.
        self.chunk.q.clear();
        self.chunk.q.resize(rows * fused_rows, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.qkv_proj.weight"),
            fused_rows,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.q,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.q[row * fused_rows..(row + 1) * fused_rows],
                &format!("{prefix}self_attn.qkv_proj"),
            );
        }
        self.chunk.k.clear();
        self.chunk.k.resize(rows * kv_dim, 0.0);
        self.chunk.v.clear();
        self.chunk.v.resize(rows * kv_dim, 0.0);
        for row in 0..rows {
            let fused_start = row * fused_rows;
            self.chunk.k[row * kv_dim..(row + 1) * kv_dim]
                .copy_from_slice(&self.chunk.q[fused_start + q_dim..fused_start + q_dim + kv_dim]);
            self.chunk.v[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(
                &self.chunk.q[fused_start + q_dim + kv_dim..fused_start + fused_rows],
            );
        }
        for row in (0..rows).rev() {
            self.chunk
                .q
                .copy_within(row * fused_rows..row * fused_rows + q_dim, row * q_dim);
        }
        self.chunk.q.truncate(rows * q_dim);

        // Partial rotary, cache append and attention per row, in order:
        // appending row r's K/V before attending position r keeps the causal
        // prefix exactly what the per-position path sees at that point.
        self.chunk.attn_out.clear();
        self.chunk.attn_out.resize(rows * q_dim, 0.0);
        for row in 0..rows {
            let position = offset + row;
            ops::rope_heads_scaled(
                &mut self.chunk.q[row * q_dim..(row + 1) * q_dim],
                cfg.q_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                position,
                cfg.rope_theta,
            &cfg.rope_scaling,
            );
            ops::rope_heads_scaled(
                &mut self.chunk.k[row * kv_dim..(row + 1) * kv_dim],
                cfg.kv_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                position,
                cfg.rope_theta,
            &cfg.rope_scaling,
            );
            self.ctx.cache_append(
                layer,
                &self.chunk.k[row * kv_dim..(row + 1) * kv_dim],
                &self.chunk.v[row * kv_dim..(row + 1) * kv_dim],
            );
            self.ctx.attention(
                layer,
                position,
                &self.chunk.q[row * q_dim..(row + 1) * q_dim],
                &mut self.chunk.attn_out[row * q_dim..(row + 1) * q_dim],
            );
        }

        // Output projection `dense` as a batched GEMM (plus its bias).
        self.chunk.attn_projected.clear();
        self.chunk.attn_projected.resize(rows * hidden, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.dense.weight"),
            hidden,
            q_dim,
            &self.chunk.attn_out,
            &mut self.chunk.attn_projected,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.attn_projected[row * hidden..(row + 1) * hidden],
                &format!("{prefix}self_attn.dense"),
            );
        }

        // MLP off the same normed rows: fc1 batched + bias + per-row GELU,
        // then fc2 batched + bias.
        self.chunk.gate.clear();
        self.chunk.gate.resize(rows * intermediate, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}mlp.fc1.weight"),
            intermediate,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.gate,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.gate[row * intermediate..(row + 1) * intermediate],
                &format!("{prefix}mlp.fc1"),
            );
            ops::gelu(&mut self.chunk.gate[row * intermediate..(row + 1) * intermediate]);
        }
        self.chunk.mlp_out.clear();
        self.chunk.mlp_out.resize(rows * hidden, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}mlp.fc2.weight"),
            hidden,
            intermediate,
            &self.chunk.gate,
            &mut self.chunk.mlp_out,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.mlp_out[row * hidden..(row + 1) * hidden],
                &format!("{prefix}mlp.fc2"),
            );
        }

        // Parallel residual: one update sums attention and MLP outputs.
        for index in 0..rows * hidden {
            self.chunk.hidden[index] +=
                self.chunk.attn_projected[index] + self.chunk.mlp_out[index];
        }
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
        ops::rope_heads_scaled(
            &mut self.scratch.q,
            cfg.q_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        &cfg.rope_scaling,
        );
        ops::rope_heads_scaled(
            &mut self.scratch.k,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        &cfg.rope_scaling,
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
/// Runs the Phi forward pass over `tokens` starting at absolute
/// position `offset` and returns the final-position logits — the
/// backend-agnostic [`ModelForward`] entry delegating to `run_forward`.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.run_forward(tokens, offset)
    }

/// Clears the toolbox KV caches between generations.
    fn clear_cache(&mut self) {
        self.ctx.clear_kv();
    }

    /// Captures the toolbox KV state for cross-generation prefix reuse.
    fn kv_snapshot(&self) -> Option<KvSnapshot> {
        Some(self.ctx.kv_snapshot())
    }

    /// Restores a cached prefix into the (freshly cleared) toolbox KV —
    /// native caches restore in their own storage mode (f32 or Q8), so a
    /// prefix captured by the same runtime restores bit-exact.
    fn restore_kv_prefix(&mut self, snapshot: &KvSnapshot, positions: usize) -> bool {
        self.ctx.restore_kv_prefix(snapshot, positions);
        true
    }

/// Reports the constant backend label `"native-cpu"`.
    fn device_name(&self) -> &'static str {
        "native-cpu"
    }

/// Exposes the family runtime for Ant page-level tier transitions — the
/// native backend owns mmap-backed weights, so its tiers act on real pages.
    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

impl AntTierControl for PhiNative {
/// Ant page-in: faults one unit's pages back into the resident set; the
/// bool reports whether the units manifest knows the unit.
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.ctx.page_in_unit(unit_id)
    }

/// Ant evict: drops one unit's pages to the disk tier; the bool reports
/// whether the units manifest knows the unit.
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
    let config = VulkanPhiConfig::parse(&raw_config)?;
    Ok(Box::new(VulkanPhiExecutor::load(weight_paths, &config)?))
}

// ---------------------------------------------------------------------------
// Vulkan executor (the family's GPU compute graph — phi runs its own block
// topology, so unlike the llama-shaped families it does not parameterize the
// generic transformer executor; the substrate bootstrap and the operator
// dispatch come from `crate::chat::vulkan`).
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
use crate::chat::vulkan::required_usize;
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::context::{
    Activation, GpuKvCache, apply_activation, validate_sequence,
};
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::weights::{
    Linear, NamedWeight, WeightData, load_weights, take_linear, take_weight,
};
#[cfg(feature = "vulkan")]
use std::collections::HashMap;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;

#[cfg(feature = "vulkan")]
#[derive(Clone, Debug)]
#[cfg(feature = "vulkan")]
pub(crate) struct VulkanPhiConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f32,
    rope_theta: f32,
    qk_layernorm: bool,
    tie_word_embeddings: bool,
    activation: Activation,
}

#[cfg(feature = "vulkan")]
impl VulkanPhiConfig {
    /// Phi config parsing: derives `head_dim` and `rotary_dim` from
    /// `partial_rotary_factor`, maps `hidden_act`, reads `layer_norm_eps` /
    /// `qk_layernorm` / tie choice, and rejects inconsistent dimensions, odd
    /// rotary spans and rope scaling.
#[cfg(feature = "vulkan")]
    pub(crate) fn parse(raw: &Value) -> Result<Self, String> {
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
        let max_position_embeddings = required_usize(raw, "max_position_embeddings")?;
        let layer_norm_eps = raw
            .get("layer_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5) as f32;
        let rope_theta = raw
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        let partial_rotary_factor = raw
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(0.5);
        let activation = match raw.get("hidden_act").and_then(Value::as_str) {
            Some("gelu") => Activation::Gelu,
            Some("gelu_new") => Activation::NewGelu,
            Some("silu") => Activation::Silu,
            _ => return Err("Vulkan Phi does not support the configured activation.".into()),
        };
        let head_dim = if num_attention_heads == 0 {
            0
        } else {
            hidden_size / num_attention_heads
        };
        let rotary_dim = (partial_rotary_factor * head_dim as f64) as usize;
        if vocab_size == 0
            || hidden_size == 0
            || intermediate_size == 0
            || num_hidden_layers == 0
            || num_attention_heads == 0
            || num_key_value_heads == 0
            || hidden_size % num_attention_heads.max(1) != 0
            || num_attention_heads % num_key_value_heads.max(1) != 0
            || head_dim == 0
            || rotary_dim == 0
            || rotary_dim > head_dim
            || rotary_dim % 2 != 0
            || max_position_embeddings == 0
            || !partial_rotary_factor.is_finite()
            || !layer_norm_eps.is_finite()
            || layer_norm_eps <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
            || raw
                .get("rope_scaling")
                .is_some_and(|value| !value.is_null())
        {
            return Err("Vulkan Phi model dimensions or RoPE settings are inconsistent.".into());
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            max_position_embeddings,
            layer_norm_eps,
            rope_theta,
            qk_layernorm: raw
                .get("qk_layernorm")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            tie_word_embeddings: raw
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            activation,
        })
    }
}

#[cfg(feature = "vulkan")]
#[derive(Clone)]
#[cfg(feature = "vulkan")]
struct PhiNorm {
    weight: WeightData,
    bias: WeightData,
}

#[cfg(feature = "vulkan")]
impl PhiNorm {
    /// One biased LayerNorm pass (weight + bias pair) through the shared weight
    /// layer — phi's norms are LayerNorms, not RMSNorms.
#[cfg(feature = "vulkan")]
    fn forward(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        self.weight
            .layer_norm(&self.bias, input, rows, width, epsilon)
    }
}

/// Extracts a `{prefix}.weight` + `{prefix}.bias` pair with shape validation.
#[cfg(feature = "vulkan")]
fn take_phi_norm(
    weights: &mut HashMap<String, NamedWeight>,
    prefix: &str,
    width: usize,
) -> Result<PhiNorm, String> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    Ok(PhiNorm {
        weight: take_weight(weights, &weight_name, &[width])?.data,
        bias: take_weight(weights, &bias_name, &[width])?.data,
    })
}

#[cfg(feature = "vulkan")]
#[derive(Clone)]
#[cfg(feature = "vulkan")]
struct PhiLayer {
    input_norm: PhiNorm,
    q_norm: Option<PhiNorm>,
    k_norm: Option<PhiNorm>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    fc1: Linear,
    fc2: Linear,
    kv: GpuKvCache,
}

#[cfg(feature = "vulkan")]
#[derive(Clone)]
#[cfg(feature = "vulkan")]
pub(crate) struct VulkanPhiExecutor {
    config: VulkanPhiConfig,
    embedding: WeightData,
    layers: Vec<PhiLayer>,
    final_norm: PhiNorm,
    lm_head: Linear,
}

#[cfg(feature = "vulkan")]
impl crate::chat::registry::ModelForward for VulkanPhiExecutor {
    /// Trait entry delegating to the inherent executor forward.
#[cfg(feature = "vulkan")]
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        VulkanPhiExecutor::forward(self, tokens, offset)
    }

    /// Clears every phi layer's KV cache between generations.
#[cfg(feature = "vulkan")]
    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv.clear();
        }
    }

    /// Reports the constant backend label `"Vulkan"`.
#[cfg(feature = "vulkan")]
    fn device_name(&self) -> &'static str {
        "Vulkan"
    }
}

#[cfg(feature = "vulkan")]
impl VulkanPhiExecutor {
    /// Public load entry: bootstraps the substrate and guarantees teardown on
    /// failure via [`load_with_substrate`].
#[cfg(feature = "vulkan")]
    pub(crate) fn load(paths: &[PathBuf], config: &VulkanPhiConfig) -> Result<Self, String> {
        crate::chat::vulkan::load_with_substrate(|| Self::load_initialized(paths, config))
    }

    /// The phi load graph: embedding, final LayerNorm pair, tied-or-separate lm
    /// head, then per layer the input norm, optional q/k layernorms, split
    /// q/k/v projections, `dense` output projection and `fc1`/`fc2` MLP — every
    /// phi linear carries its bias.
#[cfg(feature = "vulkan")]
    fn load_initialized(paths: &[PathBuf], config: &VulkanPhiConfig) -> Result<Self, String> {
        let mut weights = load_weights(paths)?;
        let embedding = take_weight(
            &mut weights,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?
        .data;
        let final_norm = take_phi_norm(&mut weights, "model.final_layernorm", config.hidden_size)?;
        let lm_head = if weights.contains_key("lm_head.weight") {
            take_linear(
                &mut weights,
                &BTreeMap::new(),
                "lm_head",
                config.hidden_size,
                config.vocab_size,
                true,
            )?
        } else if config.tie_word_embeddings {
            Linear {
                name: "lm_head".into(),
                input_size: config.hidden_size,
                output_size: config.vocab_size,
                weight: embedding.clone(),
                slot: None,
                bias: None,
            }
        } else {
            return Err("Vulkan Phi checkpoint is missing tensor 'lm_head.weight'.".into());
        };
        let qk_norm_size = config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            let attention = format!("{prefix}.self_attn");
            let input_norm = take_phi_norm(
                &mut weights,
                &format!("{prefix}.input_layernorm"),
                config.hidden_size,
            )?;
            let (q_norm, k_norm) = if config.qk_layernorm {
                (
                    Some(take_phi_norm(
                        &mut weights,
                        &format!("{attention}.q_layernorm"),
                        qk_norm_size,
                    )?),
                    Some(take_phi_norm(
                        &mut weights,
                        &format!("{attention}.k_layernorm"),
                        qk_norm_size,
                    )?),
                )
            } else {
                (None, None)
            };
            layers.push(PhiLayer {
                input_norm,
                q_norm,
                k_norm,
                q_proj: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.q_proj"),
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                    true,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.k_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.v_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                dense: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.dense"),
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                    true,
                )?,
                fc1: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{prefix}.mlp.fc1"),
                    config.hidden_size,
                    config.intermediate_size,
                    true,
                )?,
                fc2: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{prefix}.mlp.fc2"),
                    config.intermediate_size,
                    config.hidden_size,
                    true,
                )?,
                kv: GpuKvCache::default(),
            });
        }
        Ok(Self {
            config: config.clone(),
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    /// The phi graph: host-side embedding gather, then per layer input
    /// LayerNorm → q/k/v (+ optional qk LayerNorm) → partial rotary → cache →
    /// attention → dense → feed-forward (fc1 → activation → fc2) → the single
    /// parallel-residual update `x + attn + mlp`; final LayerNorm + lm head.
#[cfg(feature = "vulkan")]
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        let cached_elements = self
            .layers
            .first()
            .map(|layer| layer.kv.keys.len())
            .unwrap_or(0);
        let sequence = validate_sequence(
            tokens,
            offset,
            self.config.vocab_size,
            self.config.max_position_embeddings,
            self.config.num_key_value_heads,
            self.config.head_dim,
            cached_elements,
        )?;
        let hidden = self.config.hidden_size;
        let mut activations = Vec::with_capacity(sequence * hidden);
        for token_id in tokens {
            let token = *token_id as usize;
            if token >= self.config.vocab_size {
                return Err(format!("Token ID {token} exceeds the Phi vocabulary."));
            }
            let start = token * hidden;
            activations.extend((0..hidden).map(|index| self.embedding.value(start + index)));
        }

        for layer in &mut self.layers {
            let residual = activations.clone();
            let normalized = layer.input_norm.forward(
                &activations,
                sequence,
                hidden,
                self.config.layer_norm_eps,
            )?;
            let mut query = layer.q_proj.forward(&normalized, sequence)?;
            let mut keys = layer.k_proj.forward(&normalized, sequence)?;
            let values = layer.v_proj.forward(&normalized, sequence)?;
            if let Some(norm) = &layer.q_norm {
                query = norm.forward(
                    &query,
                    sequence * self.config.num_attention_heads,
                    self.config.head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            if let Some(norm) = &layer.k_norm {
                keys = norm.forward(
                    &keys,
                    sequence * self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.layer_norm_eps,
                )?;
            }
            (query, keys) = cheetah::native::vulkan::rotary(
                &query,
                &keys,
                sequence,
                offset,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rotary_dim,
                &crate::chat::native::ops::scaled_freqs(
                    self.config.rope_theta,
                    self.config.rotary_dim,
                    &crate::chat::native::ops::RopeScaling::None,
                ),
            )?;
            layer.kv.append(&keys, &values);
            let attention = cheetah::native::vulkan::attention(
                &query,
                &layer.kv.keys,
                &layer.kv.values,
                sequence,
                offset,
                offset + sequence,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                0,
            )?;
            let attention = layer.dense.forward(&attention, sequence)?;
            let mut activated = layer.fc1.forward(&normalized, sequence)?;
            let ones = vec![1.0; activated.len()];
            activated = apply_activation(self.config.activation, &activated, &ones)?;
            let feed_forward = layer.fc2.forward(&activated, sequence)?;
            let residual_attention = cheetah::native::vulkan::add(&residual, &attention)?;
            activations = cheetah::native::vulkan::add(&residual_attention, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .forward(last, 1, hidden, self.config.layer_norm_eps)?;
        let logits = self.lm_head.forward(&normalized, 1)?;
        Ok(logits)
    }
}
