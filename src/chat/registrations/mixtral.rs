//! Registration for `mixtral` — declarative capability plus the family's
//! native forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the llama-shaped order with the sparse feed-forward: router
//! → softmax top-k expert selection (the shared `chat::moe` rule) →
//! route-weighted expert mix. Mixtral specifics: untied lm head, optional
//! attention biases, full-rope rotate-half attention over split
//! `q/k/v_proj`, and the routing-aware per-expert Ant residency (selected
//! experts page in, the rest evict to disk).

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, OpsToolbox, Scratch};
#[cfg(test)]
use super::native::store::MmapWeightStore;
use super::templates::mixtral;
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
pub(crate) const ARCHITECTURES: &[&str] = &["MixtralForCausalLM", "mixtral"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "mixtral",
        render: mixtral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(mixtral_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering; the mixture-of-experts forward routes and mixes experts per
/// token) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(mixtral_native_loader)),
            ("vulkan", BackendLoader::Vulkan(mixtral_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(mixtral_native_loader))]
    }
}

/// Family constants for the Mixtral family: the sparse feed-forward resolves
/// expert counts from the prepared manifest; the MIXTRAL layout names the
/// router (`block_sparse_moe.gate`) and expert projections (`w1/w3/w2`).
const fn mixtral_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 8_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32768, 32, 8, 128), // Mixtral-8x7B (GQA: 32 q heads / 8 kv heads)
        ],
        extra_config_json: "",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The mixtral family's native runtime: the llama-shaped
/// sequential-residual block with the sparse (routed) feed-forward.
#[derive(Clone)]
pub(crate) struct MixtralNative {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
    /// Chunked-prefill buffers (the llama-shaped stack plus the MoE extras).
    pub(crate) chunk: MoEChunk,
}

/// Chunked-prefill buffers for the sparse feed-forward: the llama-shaped
/// activation stack plus the MoE extras (router logits, per-row per-expert
/// outputs, expert bucket workspaces). Sized on first use and grown only when
/// a later chunk is larger than anything seen before.
#[derive(Clone)]
pub(crate) struct MoEChunk {
    /// Residual stream rows, `[rows, hidden_size]`.
    hidden: Vec<f32>,
    /// Most recent norm output per row, `[rows, hidden_size]`.
    normed: Vec<f32>,
    /// Query rows after projection + rope, `[rows, q_dim]`.
    q: Vec<f32>,
    /// Key rows, `[rows, kv_dim]`.
    k: Vec<f32>,
    /// Value rows, `[rows, kv_dim]`.
    v: Vec<f32>,
    /// Attention output rows before `o_proj`, `[rows, q_dim]`.
    attn_out: Vec<f32>,
    /// Attention output rows after `o_proj`, `[rows, hidden_size]`.
    attn_projected: Vec<f32>,
    /// Router logits per row, `[rows, num_local_experts]`.
    router: Vec<f32>,
    /// Each row's per-expert outputs parked by selection slot,
    /// `[rows, k, hidden_size]` — the per-row score-order accumulation reads
    /// them back after every expert's bucket GEMMs have run.
    expert_out: Vec<f32>,
    /// Gathered rows of the expert currently being computed, `[b, hidden]`.
    bucket_x: Vec<f32>,
    /// Its gate projection / activated intermediate, `[b, intermediate]`.
    bucket_gate: Vec<f32>,
    /// Its up projection, `[b, intermediate]`.
    bucket_up: Vec<f32>,
    /// Its down projection, `[b, hidden]`.
    bucket_out: Vec<f32>,
    /// Routed feed-forward output rows, `[rows, hidden_size]`.
    mlp_out: Vec<f32>,
    /// Rows the buffers were last sized for.
    capacity_rows: usize,
}

impl MoEChunk {
    fn new() -> Self {
        Self {
            hidden: Vec::new(),
            normed: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            attn_projected: Vec::new(),
            router: Vec::new(),
            expert_out: Vec::new(),
            bucket_x: Vec::new(),
            bucket_gate: Vec::new(),
            bucket_up: Vec::new(),
            bucket_out: Vec::new(),
            mlp_out: Vec::new(),
            capacity_rows: 0,
        }
    }

    fn resize_for(&mut self, cfg: &DecoderConfig, rows: usize) {
        if rows <= self.capacity_rows {
            return;
        }
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        let k = cfg.num_experts_per_tok.min(cfg.num_local_experts);
        self.hidden = vec![0.0; rows * cfg.hidden_size];
        self.normed = vec![0.0; rows * cfg.hidden_size];
        self.q = vec![0.0; rows * cfg.q_heads * cfg.head_dim];
        self.k = vec![0.0; rows * kv_dim];
        self.v = vec![0.0; rows * kv_dim];
        self.attn_out = vec![0.0; rows * cfg.q_heads * cfg.head_dim];
        self.attn_projected = vec![0.0; rows * cfg.hidden_size];
        self.router = vec![0.0; rows * cfg.num_local_experts];
        self.expert_out = vec![0.0; rows * k * cfg.hidden_size];
        self.bucket_x = vec![0.0; rows * cfg.hidden_size];
        self.bucket_gate = vec![0.0; rows * cfg.intermediate_size];
        self.bucket_up = vec![0.0; rows * cfg.intermediate_size];
        self.bucket_out = vec![0.0; rows * cfg.hidden_size];
        self.mlp_out = vec![0.0; rows * cfg.hidden_size];
        self.capacity_rows = rows;
    }
}

impl MixtralNative {
    /// Fixture/test constructor around an already-resolved config and store.
    #[cfg(test)]
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        let ctx = OpsToolbox::with_store(cfg.clone(), store);
        Self {
            scratch: Scratch::new(&cfg),
            chunk: MoEChunk::new(),
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
            chunk: MoEChunk::new(),
            cfg,
            ctx,
        }
    }

    /// Run the full forward for tokens starting at `offset`; returns the
    /// logits of the final position only (matching the engine's sampling
    /// contract). Single tokens (decode steps) run the per-position path;
    /// longer batches (prefill) split into chunked batched-GEMM passes with
    /// the sparse feed-forward bucketed per expert.
    fn run_forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.forward_with_chunk(tokens, offset, ops::prefill_chunk_size())
    }

    /// [`Self::run_forward`] with an explicit chunk size — tests force
    /// uneven chunks and compare against the per-position reference.
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
        // Mixtral keeps a separate lm_head tensor; the tied branch reads the
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
    /// post-attention norm → routed (or dense) feed-forward → residual add.
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
        self.forward_mlp(&prefix, layer)?;
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
    /// then `o_proj`. Biases are family-optional — the adds are
    /// unit-optional no-ops.
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

    /// Feed-forward dispatch over `scratch.normed`, written to
    /// `scratch.mlp_out` — the caller owns the residual add. Dense vs MoE
    /// branch: detected by expert counts from config. This runtime check is
    /// negligible overhead and enables clean code paths for both shapes
    /// without compile-time specialization.
    fn forward_mlp(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        if self.cfg.num_local_experts > 0 && self.cfg.num_experts_per_tok > 0 {
            self.forward_moe_block(prefix, layer)
        } else {
            self.forward_dense_mlp(prefix)
        }
    }

    /// Forward pass for MoE blocks: router selection + top-k expert
    /// computation, each selected expert's output accumulated into
    /// `scratch.mlp_out`, weighted by its route weight.
    pub(crate) fn forward_moe_block(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        let cfg = &self.cfg;

        // Router: compute expert scores from normalized hidden states
        self.scratch.router.clear();
        self.scratch.router.resize(cfg.num_local_experts, 0.0);
        self.ctx.linear(
            &cfg.moe_layout.router_unit(prefix),
            cfg.num_local_experts,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.router,
        )?;

        // Router selection: softmax over the router logits, top-k by score,
        // selected weights renormalized to one — the shared routing rule
        // every backend uses (`chat::moe::select_experts`).
        let num_experts_per_tok = cfg.num_experts_per_tok.min(cfg.num_local_experts);
        let selected = crate::chat::moe::select_experts(&self.scratch.router, num_experts_per_tok)?;
        let picked: Vec<usize> = selected.iter().map(|(expert, _)| *expert).collect();

        // RC2-B: routing-aware per-expert residency management. Evict unused
        // experts, page in newly selected ones. State keyed by (layer,
        // expert) persists across positions.
        self.ctx.update_expert_residency(layer, &picked);

        // Compute selected experts: gate*up activation, then down projection,
        // scaled by the router's mixing weight for this token. The weighted
        // accumulator starts at zero every call (it swaps into `mlp_out` at
        // the end).
        self.scratch.mlp_out_weighted.clear();
        self.scratch
            .mlp_out_weighted
            .resize(cfg.hidden_size, 0.0);
        for (expert, route_weight) in selected {
            let gate_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.gate);
            let up_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.up);
            let down_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.down);

            self.scratch.gate.clear();
            self.scratch.gate.resize(cfg.intermediate_size, 0.0);
            self.ctx.linear(
                &gate_unit,
                cfg.intermediate_size,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.gate,
            )?;

            self.scratch.up.clear();
            self.scratch.up.resize(cfg.intermediate_size, 0.0);
            self.ctx.linear(
                &up_unit,
                cfg.intermediate_size,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.up,
            )?;

            ops::silu_multiply(&mut self.scratch.gate, &self.scratch.up);

            self.scratch.mlp_out.clear();
            self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
            self.ctx.linear(
                &down_unit,
                cfg.hidden_size,
                cfg.intermediate_size,
                &self.scratch.gate,
                &mut self.scratch.mlp_out,
            )?;

            // Accumulate this expert's output, weighted by the router's
            // probability for the token; the caller owns the residual add.
            for (value, out) in self
                .scratch
                .mlp_out
                .iter()
                .zip(self.scratch.mlp_out_weighted.iter_mut())
            {
                *out += value * route_weight;
            }
        }
        std::mem::swap(
            &mut self.scratch.mlp_out,
            &mut self.scratch.mlp_out_weighted,
        );
        Ok(())
    }

    /// Dense feed-forward over `scratch.normed`, written to `scratch.mlp_out`
    /// (the dense fallback for a mixtral-shaped manifest without routed
    /// experts). SwiGLU: `down(silu(gate·x) ⊙ up·x)`.
    fn forward_dense_mlp(&mut self, prefix: &str) -> Result<(), String> {
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

    /// Chunked prefill: splits `tokens` into `chunk_rows`-sized batches and
    /// processes each through [`Self::forward_chunk`]. The projections and
    /// the routed feed-forward run as batched GEMMs (per-expert over the
    /// rows that picked it); norms, rope, KV append and attention stay
    /// per-row, which keeps the result bit-exact against the per-position
    /// path.
    fn forward_chunked(
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

    /// One decoder layer over a whole chunk — the llama block shape with the
    /// projections batched and the feed-forward routed per expert. Biases are
    /// family-optional: mixtral carries none, so the adds stay unit-optional
    /// no-ops.
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

        // Rope, cache append and attention per row, in order: appending row
        // r's K/V before attending position r keeps the causal prefix exactly
        // what the per-position path sees at that point.
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

        // Post-attention norm per row, then the routed feed-forward, then its
        // residual add.
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
        self.forward_moe_block_chunk(&prefix, layer, rows)?;
        for (value, residual) in self.chunk.mlp_out.iter().zip(self.chunk.hidden.iter_mut()) {
            *residual += *value;
        }
        Ok(())
    }

    /// Routed feed-forward over a whole chunk: the router runs as one
    /// batched GEMM, every row picks its top-k experts through the shared
    /// `chat::moe` rule, and each expert computes one batched gate/up/down
    /// GEMM triple over the rows that picked it (its bucket). Each row's
    /// expert outputs are parked by selection slot and accumulated in the
    /// row's score order afterwards, so the mix is bit-exact against the
    /// per-position path. Residency: the chunk's union of picked experts
    /// drives one `update_expert_residency` call with the same per-`(layer,
    /// expert)` contract.
    fn forward_moe_block_chunk(
        &mut self,
        prefix: &str,
        layer: usize,
        rows: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let num_experts = cfg.num_local_experts;
        let k = cfg.num_experts_per_tok.min(num_experts);
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;

        // Router logits as one batched GEMM.
        self.chunk.router.clear();
        self.chunk.router.resize(rows * num_experts, 0.0);
        self.ctx.linear_batch(
            &cfg.moe_layout.router_unit(prefix),
            num_experts,
            hidden,
            &self.chunk.normed,
            &mut self.chunk.router,
        )?;

        // Per-row selection in score order, plus the chunk-wide union that
        // drives the routing-aware residency in one call.
        let mut selections: Vec<Vec<(usize, f32)>> = Vec::with_capacity(rows);
        let mut picked: Vec<usize> = Vec::new();
        for row in 0..rows {
            let logits = &self.chunk.router[row * num_experts..(row + 1) * num_experts];
            let selected = crate::chat::moe::select_experts(logits, k)?;
            for (expert, _) in &selected {
                if !picked.contains(expert) {
                    picked.push(*expert);
                }
            }
            selections.push(selected);
        }
        self.ctx.update_expert_residency(layer, &picked);

        // Per-expert buckets, expert-index order. A row's expert output lands
        // in `expert_out[row][slot]` where `slot` is the expert's rank in
        // that row's score-ordered selection.
        self.chunk.expert_out.clear();
        self.chunk.expert_out.resize(rows * k * hidden, 0.0);
        for expert in 0..num_experts {
            let members: Vec<usize> = (0..rows)
                .filter(|&row| selections[row].iter().any(|(e, _)| *e == expert))
                .collect();
            if members.is_empty() {
                continue;
            }
            let bucket = members.len();
            for (index, &row) in members.iter().enumerate() {
                self.chunk.bucket_x[index * hidden..(index + 1) * hidden]
                    .copy_from_slice(&self.chunk.normed[row * hidden..(row + 1) * hidden]);
            }
            self.chunk.bucket_gate.clear();
            self.chunk.bucket_gate.resize(bucket * intermediate, 0.0);
            self.ctx.linear_batch(
                &cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.gate),
                intermediate,
                hidden,
                &self.chunk.bucket_x[..bucket * hidden],
                &mut self.chunk.bucket_gate,
            )?;
            self.chunk.bucket_up.clear();
            self.chunk.bucket_up.resize(bucket * intermediate, 0.0);
            self.ctx.linear_batch(
                &cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.up),
                intermediate,
                hidden,
                &self.chunk.bucket_x[..bucket * hidden],
                &mut self.chunk.bucket_up,
            )?;
            for index in 0..bucket {
                ops::silu_multiply(
                    &mut self.chunk.bucket_gate[index * intermediate..(index + 1) * intermediate],
                    &self.chunk.bucket_up[index * intermediate..(index + 1) * intermediate],
                );
            }
            self.chunk.bucket_out.clear();
            self.chunk.bucket_out.resize(bucket * hidden, 0.0);
            self.ctx.linear_batch(
                &cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.down),
                hidden,
                intermediate,
                &self.chunk.bucket_gate[..bucket * intermediate],
                &mut self.chunk.bucket_out,
            )?;
            for (index, &row) in members.iter().enumerate() {
                let slot = selections[row]
                    .iter()
                    .position(|(e, _)| *e == expert)
                    .expect("bucket member must have selected this expert");
                self.chunk.expert_out
                    [row * k * hidden + slot * hidden..row * k * hidden + (slot + 1) * hidden]
                    .copy_from_slice(&self.chunk.bucket_out[index * hidden..(index + 1) * hidden]);
            }
        }

        // Per-row mix in score order — the exact accumulation the
        // per-position path performs (mlp_out starts at zero, adds
        // `expert_out · route_weight` per selected expert in descending
        // score order).
        self.chunk.mlp_out.clear();
        self.chunk.mlp_out.resize(rows * hidden, 0.0);
        for row in 0..rows {
            let acc = &mut self.chunk.mlp_out[row * hidden..(row + 1) * hidden];
            for (slot, (_, route_weight)) in selections[row].iter().enumerate() {
                let source = &self.chunk.expert_out[row * k * hidden + slot * hidden..][..hidden];
                for (out, value) in acc.iter_mut().zip(source) {
                    *out += *value * route_weight;
                }
            }
        }
        Ok(())
    }
}

impl ModelForward for MixtralNative {
/// Runs the Mixtral forward pass over `tokens` starting at absolute
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

impl AntTierControl for MixtralNative {
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
fn mixtral_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(MixtralNative::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Vulkan transformer features for the Mixtral family: the sparse
/// mixture-of-experts feed-forward.
#[cfg(feature = "vulkan")]
fn mixtral_vulkan_config(
    raw_config: &Value,
) -> Result<crate::chat::vulkan::transformer::VulkanTransformerConfig, String> {
    use crate::chat::vulkan::required_usize;
    use crate::chat::vulkan::transformer::{
        TransformerFeatures, TransformerFeedForwardFeatures,
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
    crate::chat::vulkan::transformer::VulkanTransformerConfig::from_raw(raw_config, features)
}

/// Vulkan loader for this registration: build the mixtral family's own
/// executor (the decode graph below) around the mixtral feature set.
#[cfg(feature = "vulkan")]
fn mixtral_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = mixtral_vulkan_config(&raw_config)?;
    Ok(Box::new(MixtralVulkanExecutor::load(weight_paths, &config, plan)?))
}

// ---------------------------------------------------------------------------
// The family's own Vulkan executor (Part B Phase 2): the decode graph lives
// in this registration file. Mixtral's block is the llama baseline shape
// with the sparse feed-forward: RMSNorm → split q/k/v → full-span
// rotate-half rope → GQA attention → o_proj → residual → RMSNorm → routed
// experts → residual. The router runs as one batched GEMM, rows are
// bucketed BY EXPERT (one batched gate/up/activation/down per picked
// expert), and the routing-aware per-expert residency is driven through the
// shared `VulkanForwardContext`.
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan")]
use crate::chat::moe::select_experts;
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::context::{Expert, GpuKvCache, VulkanForwardContext, apply_activation, page_linear, validate_sequence};
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::transformer::{TransformerFeedForwardFeatures, VulkanTransformerConfig};
#[cfg(feature = "vulkan")]
use crate::chat::vulkan::weights::{
    Linear, WeightData, load_weights, take_linear, take_weight, validate_shape,
};

/// Mixtral-style sparse-MoE tensor naming: router + expert projections.
#[cfg(feature = "vulkan")]
const MOE_ROUTER_LAYOUT: &str = "block_sparse_moe.gate";
#[cfg(feature = "vulkan")]
const MOE_EXPERTS_LAYOUT: &str = "block_sparse_moe.experts";
#[cfg(feature = "vulkan")]
const MOE_GATE_LAYOUT: &str = "w1";
#[cfg(feature = "vulkan")]
const MOE_UP_LAYOUT: &str = "w3";
#[cfg(feature = "vulkan")]
const MOE_DOWN_LAYOUT: &str = "w2";

/// The mixtral sparse feed-forward: the router plus one `Expert` per expert
/// (three projections paged together through the context).
#[cfg(feature = "vulkan")]
#[derive(Clone)]
struct MixtralFeedForward {
    router: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
}

/// One mixtral decoder layer on the GPU.
#[cfg(feature = "vulkan")]
#[derive(Clone)]
struct MixtralVulkanLayer {
    input_norm: WeightData,
    post_attention_norm: WeightData,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    feed_forward: MixtralFeedForward,
    kv: GpuKvCache,
}

/// The mixtral family's Vulkan executor.
#[cfg(feature = "vulkan")]
#[derive(Clone)]
pub(crate) struct MixtralVulkanExecutor {
    config: VulkanTransformerConfig,
    embedding: WeightData,
    layers: Vec<MixtralVulkanLayer>,
    final_norm: WeightData,
    lm_head: WeightData,
    /// Expert-residency state (keyed (layer, expert)) the MoE block drives.
    context: VulkanForwardContext,
}

#[cfg(feature = "vulkan")]
impl MixtralVulkanExecutor {
    /// Load entry with guaranteed substrate teardown on failure.
    fn load(
        paths: &[std::path::PathBuf],
        config: &VulkanTransformerConfig,
        plan: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        crate::chat::vulkan::load_with_substrate(|| Self::load_initialized(paths, config, plan))
    }

    /// The load graph (tied-or-untied lm head per the config; the sparse
    /// feed-forward under the fixed MOE_* layout).
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
        let TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts,
            num_experts_per_tok,
        } = &config.features.feed_forward
        else {
            unreachable!("mixtral_vulkan_config only resolves the MoE feature set");
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
            let router = take_linear(
                &mut weights,
                plan,
                &format!("{prefix}.{MOE_ROUTER_LAYOUT}"),
                config.hidden_size,
                *num_local_experts,
                false,
            )?;
            let mut experts = Vec::with_capacity(*num_local_experts);
            for expert_index in 0..*num_local_experts {
                let expert_prefix = format!("{prefix}.{MOE_EXPERTS_LAYOUT}.{expert_index}");
                experts.push(Expert {
                    gate: take_linear(
                        &mut weights,
                        plan,
                        &format!("{expert_prefix}.{MOE_GATE_LAYOUT}"),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    up: take_linear(
                        &mut weights,
                        plan,
                        &format!("{expert_prefix}.{MOE_UP_LAYOUT}"),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    down: take_linear(
                        &mut weights,
                        plan,
                        &format!("{expert_prefix}.{MOE_DOWN_LAYOUT}"),
                        config.intermediate_size,
                        config.hidden_size,
                        false,
                    )?,
                });
            }
            layers.push(MixtralVulkanLayer {
                input_norm,
                post_attention_norm,
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
                feed_forward: MixtralFeedForward {
                    router,
                    experts,
                    num_experts_per_tok: *num_experts_per_tok,
                },
                kv: GpuKvCache::default(),
            });
        }
        Ok(Self {
            config: config.clone(),
            embedding,
            layers,
            final_norm,
            lm_head,
            context: VulkanForwardContext::new(),
        })
    }

    /// Per-layer KV reset backing the trait impl.
    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv.clear();
        }
    }

    /// The mixtral decode graph: validate, gather embeddings as a shader, then
    /// per layer the block above; final norm + lm head on the last position.
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
            let rope_freqs = crate::chat::native::ops::scaled_freqs(
                self.config.rope_theta,
                self.config.head_dim,
                &self.config.rope_scaling,
            );
            (query, keys) = cheetah::native::vulkan::rotary(
                &query,
                &keys,
                sequence,
                offset,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.head_dim,
                &rope_freqs,
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
            let feed_forward = Self::forward_moe(
                &mut self.context,
                self.config.activation,
                layer,
                layer_index,
                &normalized,
                sequence,
                hidden,
            )?;
            activations = cheetah::native::vulkan::add(&residual, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .rms_norm(last, 1, hidden, self.config.rms_norm_eps)?;
        self.lm_head
            .linear(&normalized, 1, hidden, self.config.vocab_size)
    }

    /// The sparse feed-forward over `rows` positions: route every row through
    /// `select_experts`, drive the residency through the shared context, run
    /// one batched matmul per picked expert over its bucket rows, and scatter
    /// the route-weighted outputs back.
    fn forward_moe(
        context: &mut VulkanForwardContext,
        activation: crate::chat::vulkan::context::Activation,
        layer: &mut MixtralVulkanLayer,
        layer_index: usize,
        input: &[f32],
        rows: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        let feed_forward = &mut layer.feed_forward;
        let router_logits = feed_forward.router.forward(input, rows)?;
        let mut output = vec![0.0; rows * hidden_size];
        let mut selected_by_expert: Vec<Vec<(usize, f32)>> =
            vec![Vec::new(); feed_forward.experts.len()];
        for row in 0..rows {
            let logits = &router_logits[row * feed_forward.experts.len()..(row + 1) * feed_forward.experts.len()];
            for (expert_index, route_weight) in
                select_experts(logits, feed_forward.num_experts_per_tok)?
            {
                selected_by_expert[expert_index].push((row, route_weight));
            }
        }
        let resident: Vec<bool> = selected_by_expert
            .iter()
            .map(|selected_rows| !selected_rows.is_empty())
            .collect();
        context.sync_expert_residency(layer_index, &mut feed_forward.experts, &resident);

        for (expert, selected_rows) in feed_forward
            .experts
            .iter_mut()
            .zip(selected_by_expert.into_iter())
        {
            if selected_rows.is_empty() {
                continue;
            }
            let mut expert_input = Vec::with_capacity(selected_rows.len() * hidden_size);
            for (row, _) in &selected_rows {
                expert_input.extend_from_slice(&input[row * hidden_size..(row + 1) * hidden_size]);
            }
            let gate = expert.gate.forward(&expert_input, selected_rows.len())?;
            let up = expert.up.forward(&expert_input, selected_rows.len())?;
            let activated = apply_activation(activation, &gate, &up)?;
            let expert_output = expert.down.forward(&activated, selected_rows.len())?;
            for (selected_index, (row, route_weight)) in selected_rows.into_iter().enumerate() {
                let destination = &mut output[row * hidden_size..(row + 1) * hidden_size];
                let start = selected_index * hidden_size;
                for (out, value) in destination
                    .iter_mut()
                    .zip(&expert_output[start..start + hidden_size])
                {
                    *out += value * route_weight;
                }
            }
        }
        Ok(output)
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
                &mut layer.feed_forward.router,
            ] {
                if linear.name != unit_id {
                    continue;
                }
                handled = true;
                page_linear(linear, resident);
            }
            for expert in &mut layer.feed_forward.experts {
                for linear in [&mut expert.gate, &mut expert.up, &mut expert.down] {
                    if linear.name != unit_id {
                        continue;
                    }
                    handled = true;
                    page_linear(linear, resident);
                }
            }
        }
        handled
    }
}

#[cfg(feature = "vulkan")]
impl ModelForward for MixtralVulkanExecutor {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        MixtralVulkanExecutor::forward(self, tokens, offset)
    }

    fn clear_cache(&mut self) {
        MixtralVulkanExecutor::clear_cache(self);
    }

    fn device_name(&self) -> &'static str {
        "Vulkan"
    }

    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

#[cfg(feature = "vulkan")]
impl AntTierControl for MixtralVulkanExecutor {
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

    #[test]
    fn mixtral_vulkan_features_configure_moe_tensor_layout() {
        let config = mixtral_vulkan_config(&serde_json::json!({
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
            crate::chat::vulkan::transformer::TransformerFeedForwardFeatures::MixtureOfExperts {
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
        assert!(mixtral_vulkan_config(&raw).is_err());
    }
}
