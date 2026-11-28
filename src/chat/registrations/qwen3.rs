//! Registration for `qwen3` — declarative capability plus the family's
//! native forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the llama-shaped order — embedding → per layer (input
//! RMSNorm → attention → residual → post-attention RMSNorm → SiLU gated MLP
//! → residual) → final RMSNorm → lm head. Qwen3 specifics: per-head RMSNorm
//! on q and k (`q_norm`/`k_norm` units, applied before rope — a no-op add
//! when a variant omits the units), tied embeddings, optional attention
//! biases, and full-rope rotate-half attention over split `q/k/v_proj`.

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, ChunkScratch, OpsToolbox, Scratch};
use super::templates::qwen;
use super::{
    AntTierControl, BackendLoader, ChatTemplate, KvSnapshot, ModelFamily, ModelForward,
    ModelRegistration,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[cfg(test)]
use super::native::store::MmapWeightStore;
#[cfg(test)]
use std::sync::Arc;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["Qwen3ForCausalLM", "qwen3"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "qwen",
        render: qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    backends: model_backends(),
    family: Some(qwen3_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(qwen3_native_loader)),
            ("vulkan", BackendLoader::Vulkan(qwen3_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(qwen3_native_loader))]
    }
}

/// Family constants for the Qwen3 family: per-head q/k RMSNorm units are
/// detected from the prepared manifest, not declared here.
const fn qwen3_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[
            (8192, 32768, 64, 8, 128), // Qwen3-32B GQA config
        ],
        extra_config_json: "{\"hidden_act\":\"silu\"}",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The qwen3 family's native runtime: the llama-shaped sequential-residual
/// block with the family's per-head q/k norms.
#[derive(Clone)]
pub(crate) struct Qwen3Native {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
    /// Chunked-prefill activation buffers (`[rows, width]` stacks).
    pub(crate) chunk: ChunkScratch,
}

impl Qwen3Native {
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

    /// Qwen3 ties the lm head to the embedding table; the untied branch reads
    /// a separate `lm_head.weight` tensor instead. Ant: layer 0's bucket also
    /// holds the embedding / final norm, whose pages lm_head just swept; the
    /// disk-tier ones drop now and re-fault from the OS page cache.
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
        // Stack the embedding rows into `chunk.hidden` = [rows, hidden].
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
        // into `scratch.hidden` (the caller's logits input).
        let last_row_start = (rows - 1) * hidden;
        self.scratch.hidden.clear();
        self.scratch
            .hidden
            .extend_from_slice(&self.chunk.hidden[last_row_start..last_row_start + hidden]);
        self.ctx.rmsnorm("model.norm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer over a whole chunk — the qwen3 block shape (llama's
    /// sequential residual plus per-head q/k norms) with the projections
    /// batched. Biases are family-optional no-ops.
    fn forward_layer_chunk(
        &mut self,
        layer: usize,
        offset: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_dim = cfg.q_heads * cfg.head_dim;
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let prefix = format!("model.layers.{layer}.");

        // Input norm per row.
        self.chunk.normed.clear();
        self.chunk.normed.resize(rows * hidden, 0.0);
        for row in 0..rows {
            let range = row * hidden..(row + 1) * hidden;
            self.chunk.normed[range.clone()].copy_from_slice(&self.chunk.hidden[range.clone()]);
            self.ctx.rmsnorm(
                &format!("{prefix}input_layernorm"),
                &mut self.chunk.normed[range],
            )?;
        }

        // Q/K/V projections as batched GEMMs, then the (unit-optional)
        // biases per row.
        self.chunk.q.clear();
        self.chunk.q.resize(rows * q_dim, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.q_proj.weight"),
            q_dim,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.q,
        )?;
        self.chunk.k.clear();
        self.chunk.k.resize(rows * kv_dim, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.k_proj.weight"),
            kv_dim,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.k,
        )?;
        self.chunk.v.clear();
        self.chunk.v.resize(rows * kv_dim, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.v_proj.weight"),
            kv_dim,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.v,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.q[row * q_dim..(row + 1) * q_dim],
                &format!("{prefix}self_attn.q_proj"),
            );
            self.ctx.add_bias(
                &mut self.chunk.k[row * kv_dim..(row + 1) * kv_dim],
                &format!("{prefix}self_attn.k_proj"),
            );
            self.ctx.add_bias(
                &mut self.chunk.v[row * kv_dim..(row + 1) * kv_dim],
                &format!("{prefix}self_attn.v_proj"),
            );
        }

        // Per-head RMSNorm on q and k, applied before rope (a no-op when
        // the variant carries no q_norm/k_norm unit) — the same placement the
        // per-position path uses between the bias adds and rope.
        for row in 0..rows {
            self.ctx.qk_norm(
                &mut self.chunk.q[row * q_dim..(row + 1) * q_dim],
                cfg.q_heads,
                cfg.head_dim,
                &format!("{prefix}self_attn.q_norm"),
            )?;
            self.ctx.qk_norm(
                &mut self.chunk.k[row * kv_dim..(row + 1) * kv_dim],
                cfg.kv_heads,
                cfg.head_dim,
                &format!("{prefix}self_attn.k_norm"),
            )?;
        }

        // Rope, cache append and attention per row, in order: appending row
        // r's K/V before attending position r keeps the causal prefix exactly
        // what the per-position path sees at that point.
        self.chunk.attn_out.clear();
        self.chunk.attn_out.resize(rows * q_dim, 0.0);
        for row in 0..rows {
            let position = offset + row;
            ops::rope_heads(
                &mut self.chunk.q[row * q_dim..(row + 1) * q_dim],
                cfg.q_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                position,
                cfg.rope_theta,
            );
            ops::rope_heads(
                &mut self.chunk.k[row * kv_dim..(row + 1) * kv_dim],
                cfg.kv_heads,
                cfg.head_dim,
                cfg.rotary_dim,
                position,
                cfg.rope_theta,
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

        // Output projection as a batched GEMM (plus optional bias), then the
        // residual add.
        self.chunk.attn_projected.clear();
        self.chunk.attn_projected.resize(rows * hidden, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}self_attn.o_proj.weight"),
            hidden,
            q_dim,
            &self.chunk.attn_out,
            &mut self.chunk.attn_projected,
        )?;
        for row in 0..rows {
            self.ctx.add_bias(
                &mut self.chunk.attn_projected[row * hidden..(row + 1) * hidden],
                &format!("{prefix}self_attn.o_proj"),
            );
        }
        for (value, residual) in self
            .chunk
            .attn_projected
            .iter()
            .zip(self.chunk.hidden.iter_mut())
        {
            *residual += *value;
        }

        // Post-attention norm per row.
        self.chunk.normed.clear();
        self.chunk.normed.resize(rows * hidden, 0.0);
        for row in 0..rows {
            let range = row * hidden..(row + 1) * hidden;
            self.chunk.normed[range.clone()].copy_from_slice(&self.chunk.hidden[range.clone()]);
            self.ctx.rmsnorm(
                &format!("{prefix}post_attention_layernorm"),
                &mut self.chunk.normed[range],
            )?;
        }

        // Feed-forward as batched GEMMs: gate/up, per-row SwiGLU, down.
        self.chunk.gate.clear();
        self.chunk.gate.resize(rows * intermediate, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}mlp.gate_proj.weight"),
            intermediate,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.gate,
        )?;
        self.chunk.up.clear();
        self.chunk.up.resize(rows * intermediate, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}mlp.up_proj.weight"),
            intermediate,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.up,
        )?;
        for row in 0..rows {
            ops::silu_multiply(
                &mut self.chunk.gate[row * intermediate..(row + 1) * intermediate],
                &self.chunk.up[row * intermediate..(row + 1) * intermediate],
            );
        }
        self.chunk.mlp_out.clear();
        self.chunk.mlp_out.resize(rows * hidden, 0.0);
        self.ctx.linear_batch(
            &format!("{prefix}mlp.down_proj.weight"),
            hidden,
            intermediate,
            &self.chunk.gate,
            &mut self.chunk.mlp_out,
        )?;
        for (value, residual) in self
            .chunk
            .mlp_out
            .iter()
            .zip(self.chunk.hidden.iter_mut())
        {
            *residual += *value;
        }
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
    /// qwen3 delta over the llama baseline: per-head RMSNorm on q and k
    /// before rope (a unit-optional no-op), plus optional attention biases.
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
        // Per-head RMSNorm on q, applied before rope (a no-op when the
        // variant carries no q_norm unit).
        self.ctx.qk_norm(
            &mut self.scratch.q,
            cfg.q_heads,
            cfg.head_dim,
            &format!("{prefix}self_attn.q_norm"),
        )?;

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
        self.ctx.qk_norm(
            &mut self.scratch.k,
            cfg.kv_heads,
            cfg.head_dim,
            &format!("{prefix}self_attn.k_norm"),
        )?;

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
        // the visible cached keys (the resolved sliding window applies).
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

impl ModelForward for Qwen3Native {
/// Runs the Qwen3 forward pass over `tokens` starting at absolute
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

impl AntTierControl for Qwen3Native {
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
fn qwen3_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(Qwen3Native::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Vulkan transformer features for the Qwen3 family: per-head q/k norms and
/// a dense feed-forward.
#[cfg(feature = "vulkan")]
fn qwen3_vulkan_config(
    raw_config: &Value,
) -> Result<crate::chat::vulkan::transformer::VulkanTransformerConfig, String> {
    use crate::chat::vulkan::transformer::{
        TransformerFeatures, TransformerFeedForwardFeatures,
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

    crate::chat::vulkan::transformer::VulkanTransformerConfig::from_raw(raw_config, features)
}

/// Vulkan loader for this registration: build the qwen3 family's own
/// executor (the decode graph below) around the qwen3 feature set.
#[cfg(feature = "vulkan")]
fn qwen3_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = qwen3_vulkan_config(&raw_config)?;
    Ok(Box::new(Qwen3VulkanExecutor::load(weight_paths, &config, plan)?))
}

// ---------------------------------------------------------------------------
// The family's own Vulkan executor (Part B Phase 2): the decode graph lives
// in this registration file. Qwen3's block: RMSNorm → split q/k/v →
// per-head q/k RMSNorm → full-span rotate-half rope → GQA attention →
// o_proj → residual → RMSNorm → gate/up activation/down → residual. Tied
// embeddings by default (config-driven), no attention biases by default,
// no embedding scale, dense feed-forward.
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
use crate::chat::vulkan::context::{
    GpuKvCache, apply_activation, page_linear, validate_sequence,
};
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::transformer::VulkanTransformerConfig;
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::weights::{
    Linear, WeightData, load_weights, take_linear, take_weight, validate_shape,
};

/// One qwen3 decoder layer on the GPU, carrying the per-head q/k norms.
#[cfg(feature = "vulkan")]
#[derive(Clone)]
struct Qwen3VulkanLayer {
    input_norm: WeightData,
    post_attention_norm: WeightData,
    q_norm: Option<WeightData>,
    k_norm: Option<WeightData>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    kv: GpuKvCache,
}

/// The qwen3 family's Vulkan executor.
#[cfg(feature = "vulkan")]
#[derive(Clone)]
pub(crate) struct Qwen3VulkanExecutor {
    config: VulkanTransformerConfig,
    embedding: WeightData,
    layers: Vec<Qwen3VulkanLayer>,
    final_norm: WeightData,
    lm_head: WeightData,
}

#[cfg(feature = "vulkan")]
impl Qwen3VulkanExecutor {
    /// Load entry with guaranteed substrate teardown on failure.
    fn load(
        paths: &[std::path::PathBuf],
        config: &VulkanTransformerConfig,
        plan: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        crate::chat::vulkan::load_with_substrate(|| Self::load_initialized(paths, config, plan))
    }

    /// The load graph. Per-head QK norms are required (the family declares
    /// them on every attention layer).
    fn load_initialized(
        paths: &[std::path::PathBuf],
        config: &VulkanTransformerConfig,
        plan: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim;
        let kv_size = config.num_key_value_heads * head_dim;
        let layout = &config.features.layout;
        let mut weights = load_weights(paths)?;
        let embedding =
            take_weight(&mut weights, layout.embedding, &[config.vocab_size, config.hidden_size])?
                .data;
        let final_norm = take_weight(&mut weights, layout.final_norm, &[config.hidden_size])?.data;
        let lm_head = match weights.remove(layout.lm_head) {
            Some(head) => {
                validate_shape(
                    layout.lm_head,
                    &head.shape,
                    &[config.vocab_size, config.hidden_size],
                )?;
                head.data
            }
            None if config.features.tie_word_embeddings => embedding.clone(),
            None => {
                return Err(format!(
                    "Vulkan {} checkpoint is missing tensor '{}'.",
                    config.features.model_name, layout.lm_head
                ));
            }
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("{}.{index}", layout.layer_prefix);
            let input_norm = take_weight(
                &mut weights,
                &format!("{prefix}.{}", layout.input_norm),
                &[config.hidden_size],
            )?
            .data;
            let post_attention_norm = take_weight(
                &mut weights,
                &format!("{prefix}.{}", layout.post_attention_norm),
                &[config.hidden_size],
            )?
            .data;
            let attention = format!("{prefix}.{}", layout.attention_prefix);
            let q_norm = take_weight(
                &mut weights,
                &format!("{attention}.{}.weight", layout.q_norm),
                &[head_dim],
            )?
            .data;
            let k_norm = take_weight(
                &mut weights,
                &format!("{attention}.{}.weight", layout.k_norm),
                &[head_dim],
            )?
            .data;
            let gate = take_linear(
                &mut weights,
                plan,
                &format!("{prefix}.{}", layout.dense_gate),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?;
            let up = take_linear(
                &mut weights,
                plan,
                &format!("{prefix}.{}", layout.dense_up),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?;
            let down = take_linear(
                &mut weights,
                plan,
                &format!("{prefix}.{}", layout.dense_down),
                config.intermediate_size,
                config.hidden_size,
                false,
            )?;
            layers.push(Qwen3VulkanLayer {
                input_norm,
                post_attention_norm,
                q_norm: Some(q_norm),
                k_norm: Some(k_norm),
                q_proj: take_linear(
                    &mut weights,
                    plan,
                    &format!("{attention}.{}", layout.q_proj),
                    config.hidden_size,
                    config.num_attention_heads * head_dim,
                    config.features.attention_bias,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    plan,
                    &format!("{attention}.{}", layout.k_proj),
                    config.hidden_size,
                    kv_size,
                    config.features.attention_bias,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    plan,
                    &format!("{attention}.{}", layout.v_proj),
                    config.hidden_size,
                    kv_size,
                    config.features.attention_bias,
                )?,
                o_proj: take_linear(
                    &mut weights,
                    plan,
                    &format!("{attention}.{}", layout.o_proj),
                    config.num_attention_heads * head_dim,
                    config.hidden_size,
                    config.features.attention_output_bias,
                )?,
                gate,
                up,
                down,
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

    /// Per-layer KV reset backing the trait impl.
    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv.clear();
        }
    }

    /// The qwen3 decode graph: validate, gather embeddings as a shader, then
    /// per layer the block above (per-head q/k norm between projection and
    /// rope); final norm + lm head on the last position.
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
        // Ant op 8: embedding gather runs as a Vulkan shader (no embedding
        // scale — qwen3 has none).
        let (table_ptr, table_format, table_elements) = self.embedding.ffi_parts();
        let mut activations = cheetah::native::vulkan::embedding_gather(
            tokens,
            table_ptr,
            table_format,
            table_elements,
            hidden,
            false,
        )?;

        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let residual = activations.clone();
            let normalized =
                layer
                    .input_norm
                    .rms_norm(&activations, sequence, hidden, self.config.rms_norm_eps)?;
            let mut query = layer.q_proj.forward(&normalized, sequence)?;
            let mut keys = layer.k_proj.forward(&normalized, sequence)?;
            let values = layer.v_proj.forward(&normalized, sequence)?;
            if let Some(norm) = &layer.q_norm {
                query = norm.rms_norm(
                    &query,
                    sequence * self.config.num_attention_heads,
                    self.config.head_dim,
                    self.config.rms_norm_eps,
                )?;
            }
            if let Some(norm) = &layer.k_norm {
                keys = norm.rms_norm(
                    &keys,
                    sequence * self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.rms_norm_eps,
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
                self.config.head_dim,
                self.config.rope_theta,
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
                self.config.window_for_layer(layer_index),
            )?;
            let attention = layer.o_proj.forward(&attention, sequence)?;
            activations = cheetah::native::vulkan::add(&residual, &attention)?;

            let residual = activations.clone();
            let normalized = layer.post_attention_norm.rms_norm(
                &activations,
                sequence,
                hidden,
                self.config.rms_norm_eps,
            )?;
            let gate = layer.gate.forward(&normalized, sequence)?;
            let up = layer.up.forward(&normalized, sequence)?;
            let activated = apply_activation(self.config.activation, &gate, &up)?;
            let feed_forward = layer.down.forward(&activated, sequence)?;
            activations = cheetah::native::vulkan::add(&residual, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .rms_norm(last, 1, hidden, self.config.rms_norm_eps)?;
        self.lm_head
            .linear(&normalized, 1, hidden, self.config.vocab_size)
    }

    /// Ant bucketing: page one weight tensor between the GPU bucket and the
    /// host-streaming bucket through the shared paging primitive.
    fn page_weight(&mut self, unit_id: &str, resident: bool) -> bool {
        let mut handled = false;
        for layer in &mut self.layers {
            for linear in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.down,
            ] {
                if linear.name != unit_id {
                    continue;
                }
                handled = true;
                page_linear(linear, resident);
            }
        }
        handled
    }
}

#[cfg(feature = "vulkan")]
impl ModelForward for Qwen3VulkanExecutor {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        Qwen3VulkanExecutor::forward(self, tokens, offset)
    }

    fn clear_cache(&mut self) {
        Qwen3VulkanExecutor::clear_cache(self);
    }

    fn device_name(&self) -> &'static str {
        "Vulkan"
    }

    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

#[cfg(feature = "vulkan")]
impl AntTierControl for Qwen3VulkanExecutor {
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.page_weight(unit_id, true)
    }

    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.page_weight(unit_id, false)
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_feature_tests {
    use super::*;

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
        let config = qwen3_vulkan_config(&qwen3_raw()).unwrap();
        assert!(matches!(
            &config.features.feed_forward,
            crate::chat::vulkan::transformer::TransformerFeedForwardFeatures::Dense
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
        let config = qwen3_vulkan_config(&raw).unwrap();
        assert_eq!(config.sliding_window, 0);
    }

    #[test]
    fn qwen3_vulkan_rejects_scaled_rope() {
        let mut raw = qwen3_raw();
        raw["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
        assert!(qwen3_vulkan_config(&raw).is_err());
    }

    #[test]
    fn qwen3_vulkan_honors_a_configured_attention_bias() {
        let mut raw = qwen3_raw();
        raw["attention_bias"] = serde_json::json!(true);
        assert!(qwen3_vulkan_config(&raw).unwrap().features.attention_bias);
    }
}
