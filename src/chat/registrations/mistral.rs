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
use super::native::ops::{self, ChunkScratch, OpsToolbox, Scratch};
use super::templates::mistral;
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
    /// Chunked-prefill activation buffers (`[rows, width]` stacks).
    pub(crate) chunk: ChunkScratch,
}

impl MistralNative {
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

    /// Mistral keeps a separate lm_head tensor; the tied branch reads the
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

    /// One decoder layer over a whole chunk — the mistral block shape (llama's
    /// sequential residual; the sliding window resolves inside
    /// `OpsToolbox::attention`) with the projections batched. Biases are
    /// family-optional no-ops.
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
/// Runs the Mistral forward pass over `tokens` starting at absolute
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

impl AntTierControl for MistralNative {
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
