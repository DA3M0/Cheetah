//! Native decoder forward pass over the mmap-backed weight store. Activations
//! are f32; weight bytes are read in their stored dtype straight from the
//! mapping (zero copies). Disk-tier units are dropped from the resident set at
//! the end of every step so a generation holds only the planned resident bytes.

use super::kernels;
use super::store::{MmapWeightStore, Tier, WeightDType};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    /// Mixture-of-experts layout: number of experts (0 => dense feed-forward)
    /// and experts routed per token (0 => dense).
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    /// Sparse-MoE expert naming layout (Mixtral-compatible defaults). Drives
    /// router + expert weight slicing so any architecture can declare a
    /// sparse feed-forward without Mixtral-only hardcoding.
    pub moe_layout: MoEExpertLayout,
}

/// Sparse-MoE expert naming layout. Mixtral-compatible defaults; other
/// architectures supply their own expert naming via prepared configs.
#[derive(Clone, Copy, Debug)]
pub struct MoEExpertLayout {
    /// Router weight tensor name suffix (Mixtral: `block_sparse_moe.gate`).
    pub router: &'static str,
    /// Expert weight name prefix (Mixtral: `block_sparse_moe.experts`).
    pub experts_prefix: &'static str,
    /// Expert gate/up/down file suffixes (Mixtral: `w1` / `w3` / `w2`).
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
}

impl MoEExpertLayout {
    pub const MIXTRAL: Self = Self {
        router: "block_sparse_moe.gate",
        experts_prefix: "block_sparse_moe.experts",
        gate: "w1",
        up: "w3",
        down: "w2",
    };
}

impl DecoderConfig {
    /// Resolve the configuration through the parameter fallback chain:
    /// explicit config fields, then derivation from the units manifest's
    /// tensor shapes, then the registered family constants. This keeps the
    /// native backend working when the model directory's config.json is
    /// missing or incomplete.
    pub(super) fn resolve(
        raw: Option<&Value>,
        store: &MmapWeightStore,
        family: Option<crate::chat::models::ModelFamily>,
    ) -> Result<Self, String> {
        let field_u = |name: &str| {
            raw.and_then(|value| value.get(name))
                .and_then(Value::as_u64)
                .map(|value| value as usize)
        };
        let field_f = |name: &str| {
            raw.and_then(|value| value.get(name))
                .and_then(Value::as_f64)
        };
        let field_b = |name: &str| {
            raw.and_then(|value| value.get(name))
                .and_then(Value::as_bool)
        };
        let family = family.unwrap_or(crate::chat::models::ModelFamily {
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            profiles: &[],
            extra_config_json: "",
        });

        let embed_shape = store
            .unit_shape("model.embed_tokens.weight")
            .ok_or_else(|| "Native backend: units manifest has no embedding tensor.".to_string())?;
        if embed_shape.len() != 2 {
            return Err("Native backend: unexpected embedding shape in the units manifest.".into());
        }
        let hidden_size = field_u("hidden_size").unwrap_or(embed_shape[1]);
        let vocab_size = field_u("vocab_size").unwrap_or(embed_shape[0]);
        let num_layers =
            field_u("num_hidden_layers").unwrap_or((store.max_layer_id() + 1).max(1) as usize);
        let q_out = store
            .unit_shape("model.layers.0.self_attn.q_proj.weight")
            .ok_or_else(|| "Native backend: units manifest has no q_proj tensor.".to_string())?[0];
        let k_out = store
            .unit_shape("model.layers.0.self_attn.k_proj.weight")
            .ok_or_else(|| "Native backend: units manifest has no k_proj tensor.".to_string())?[0];
        let intermediate_size = field_u("intermediate_size").unwrap_or(
            store
                .unit_shape("model.layers.0.mlp.gate_proj.weight")
                .map(|shape| shape[0])
                .unwrap_or(0),
        );
        let num_local_experts = field_u("num_local_experts").unwrap_or(0);
        let num_experts_per_tok = field_u("num_experts_per_tok").unwrap_or(0);

        let (q_heads, kv_heads, head_dim) = match field_u("num_attention_heads") {
            Some(q_heads) => {
                let head_dim = field_u("head_dim").unwrap_or_else(|| {
                    family
                        .profile_for(hidden_size, vocab_size)
                        .map(|(_, _, head_dim)| head_dim)
                        .unwrap_or(q_out / q_heads.max(1))
                });
                let kv_heads = field_u("num_key_value_heads").unwrap_or_else(|| {
                    if k_out % head_dim == 0 {
                        k_out / head_dim
                    } else {
                        q_heads
                    }
                });
                (q_heads, kv_heads, head_dim)
            }
            None => family
                .profile_for(hidden_size, vocab_size)
                .ok_or_else(|| {
                    format!(
                        "Native backend: cannot determine the attention head layout for hidden size {hidden_size}; provide config.json or extend the family profiles."
                    )
                })?,
        };

        Ok(Self {
            hidden_size,
            num_layers,
            q_heads,
            kv_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_eps: field_f("rms_norm_eps").unwrap_or(family.rms_norm_eps) as f32,
            rope_theta: field_f("rope_theta").unwrap_or(family.rope_theta) as f32,
            tie_word_embeddings: field_b("tie_word_embeddings")
                .unwrap_or(family.tie_word_embeddings),
            num_local_experts,
            num_experts_per_tok,
            moe_layout: MoEExpertLayout::MIXTRAL,
        })
    }
}

/// Slice borrows come from `store` (a field) while activations live in
/// `scratch` (another field) — the borrow checker accepts the disjoint paths.
struct Scratch {
    hidden: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    attn_projected: Vec<f32>,
    scores: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mlp_out: Vec<f32>,
    router: Vec<f32>,
    logits: Vec<f32>,
}

pub struct DecoderNative {
    pub(super) cfg: DecoderConfig,
    store: Arc<MmapWeightStore>,
    /// Per-layer KV cache, flattened as [position * kv_heads * head_dim + index].
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    scratch: Scratch,
    /// Test hook: capture layer-0's attention o_proj output (pre-residual).
    debug_capture_layer0_attn: bool,
    debug_capture_buf: Option<DebugCapture>,
    /// Scratch copy of head-0 scores used by the debug capture.
    head0_scores: Vec<f32>,
    /// Pre-rope capture slots (only written while the debug flag is set).
    q_raw_capture: Vec<f32>,
    k_raw_capture: Vec<f32>,
    normed_capture: Vec<f32>,
    /// RC2-B: routing-aware per-expert residency, keyed by
    /// (layer_index, expert_index) -> resident. Kept in sync with the MoE
    /// router's top-k selections while a generation walks the layers.
    expert_residency: HashMap<(usize, usize), bool>,
}

/// Stage-wise debug capture of layer 0 at the last input position.
/// Test instrumentation only; individual stages are consumed by the
/// native attention capture tests.
#[allow(dead_code)]
pub struct DebugCapture {
    /// Input layernorm output feeding q/k/v.
    pub normed: Vec<f32>,
    /// Query states after projection+bias, BEFORE rope.
    pub q_raw: Vec<f32>,
    /// Key states after projection+bias, BEFORE rope.
    pub k_raw: Vec<f32>,
    /// Query states after rope, flattened [q_heads * head_dim].
    pub q: Vec<f32>,
    /// Key states after rope, flattened [kv_heads * head_dim].
    pub k: Vec<f32>,
    /// Softmax scores of query head 0 over all cached keys.
    pub scores: Vec<f32>,
    /// Attention output before o_proj, flattened [q_heads * head_dim].
    pub attn_out: Vec<f32>,
}

impl Clone for DecoderNative {
    fn clone(&self) -> Self {
        // Weights are shared through the Arc'd store; only a fresh KV cache is
        // created (the engine clears the cache before every generation anyway).
        Self::with_store(self.cfg.clone(), Arc::clone(&self.store))
    }
}

impl DecoderNative {
    pub fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        Self {
            kv_k: vec![Vec::new(); cfg.num_layers],
            kv_v: vec![Vec::new(); cfg.num_layers],
            scratch: Scratch {
                router: Vec::new(),
                hidden: Vec::with_capacity(cfg.hidden_size),
                normed: Vec::with_capacity(cfg.hidden_size),
                q: Vec::with_capacity(cfg.q_heads * cfg.head_dim),
                k: Vec::with_capacity(kv_dim),
                v: Vec::with_capacity(kv_dim),
                attn_out: Vec::with_capacity(cfg.q_heads * cfg.head_dim),
                attn_projected: Vec::with_capacity(cfg.hidden_size),
                scores: Vec::with_capacity(4096),
                gate: Vec::with_capacity(cfg.intermediate_size),
                up: Vec::with_capacity(cfg.intermediate_size),
                mlp_out: Vec::with_capacity(cfg.hidden_size),
                logits: Vec::with_capacity(cfg.vocab_size),
            },
            cfg,
            store,
            debug_capture_layer0_attn: false,
            debug_capture_buf: None,
            head0_scores: Vec::new(),
            q_raw_capture: Vec::new(),
            k_raw_capture: Vec::new(),
            normed_capture: Vec::new(),
            expert_residency: HashMap::new(),
        }
    }

    pub fn clear_kv(&mut self) {
        for cache in self.kv_k.iter_mut().chain(self.kv_v.iter_mut()) {
            cache.clear();
        }
    }

    /// Run the full forward for tokens starting at `offset`; returns the logits
    /// of the final position only (matching the engine's sampling contract).
    pub fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("Native backend: empty token input.".into());
        }
        for &token in tokens {
            if (token as usize) >= self.cfg.vocab_size {
                return Err(format!(
                    "Native backend: token id {token} exceeds the vocabulary size {}.",
                    self.cfg.vocab_size
                ));
            }
        }
        for (index, &token) in tokens.iter().enumerate() {
            self.forward_position(token as usize, offset + index)?;
        }
        self.run_lm_head()?;
        // Ant: layer 0's bucket also holds the embedding / final norm, whose
        // pages lm_head just swept; drop the disk-tier ones now. The pages
        // re-fault (from the OS page cache) on their next access.
        self.store.drop_disk_units_of_layer(0);
        Ok(std::mem::take(&mut self.scratch.logits))
    }

    fn forward_position(&mut self, token: usize, position: usize) -> Result<(), String> {
        self.scratch.hidden.clear();
        {
            let (bytes, dtype, shape) = self
                .store
                .slice("model.embed_tokens.weight")
                .map_err(|error| format!("Native backend: {error}"))?;
            if shape.len() != 2 || shape[1] != self.cfg.hidden_size {
                return Err("Native backend: unexpected embedding shape.".into());
            }
            let base = token * self.cfg.hidden_size;
            self.scratch.hidden.extend(
                (0..self.cfg.hidden_size)
                    .map(|index| kernels::scalar_at(bytes, base + index, dtype)),
            );
        }
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.store.drop_disk_units_of_layer(layer as i32);
        }
        let (weight, dtype, _) = self
            .store
            .slice("model.norm.weight")
            .map_err(|error| format!("Native backend: {error}"))?;
        kernels::rmsnorm(&mut self.scratch.hidden, weight, dtype, self.cfg.rms_eps);
        Ok(())
    }

    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let prefix = format!("model.layers.{layer}.");

        // --- attention block ---
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        {
            let (weight, dtype, _) = self
                .store
                .slice(&format!("{prefix}input_layernorm.weight"))?;
            kernels::rmsnorm(&mut self.scratch.normed, weight, dtype, cfg.rms_eps);
        }
        if self.debug_capture_layer0_attn && layer == 0 {
            self.normed_capture = self.scratch.normed.clone();
        }

        let (q_weight, q_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.q_proj.weight"))?;
        // Biases are family-optional (qwen2 has them, llama does not).
        let (q_bias, q_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.q_proj.bias"))
            .unwrap_or((&[], q_dtype, &[]));
        let (k_weight, k_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.k_proj.weight"))?;
        let (k_bias, k_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.k_proj.bias"))
            .unwrap_or((&[], k_dtype, &[]));
        let (v_weight, v_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.v_proj.weight"))?;
        let (v_bias, v_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.v_proj.bias"))
            .unwrap_or((&[], v_dtype, &[]));
        let (o_weight, o_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.o_proj.weight"))?;
        let (gate_weight, gate_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.gate_proj.weight"))?;
        let (up_weight, up_dtype, _) = self.store.slice(&format!("{prefix}mlp.up_proj.weight"))?;
        let (down_weight, down_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.down_proj.weight"))?;

        self.scratch.q.clear();
        self.scratch.q.resize(cfg.q_heads * cfg.head_dim, 0.0);
        kernels::gemv(
            q_weight,
            q_dtype,
            cfg.q_heads * cfg.head_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.q,
        );
        add_bias(&mut self.scratch.q, q_bias, q_bias_dtype);
        // qwen3-style per-head RMSNorm on q, applied before rope when the
        // family carries a q_norm unit.
        if self
            .store
            .contains(&format!("{prefix}self_attn.q_norm.weight"))
        {
            let (weight, dtype, _) = self
                .store
                .slice(&format!("{prefix}self_attn.q_norm.weight"))?;
            for head in 0..cfg.q_heads {
                let start = head * cfg.head_dim;
                kernels::rmsnorm(
                    &mut self.scratch.q[start..start + cfg.head_dim],
                    weight,
                    dtype,
                    cfg.rms_eps,
                );
            }
        }
        if self.debug_capture_layer0_attn && layer == 0 {
            self.q_raw_capture = self.scratch.q.clone();
        }
        self.scratch.k.clear();
        self.scratch.k.resize(cfg.kv_heads * cfg.head_dim, 0.0);
        kernels::gemv(
            k_weight,
            k_dtype,
            cfg.kv_heads * cfg.head_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.k,
        );
        add_bias(&mut self.scratch.k, k_bias, k_bias_dtype);
        if self
            .store
            .contains(&format!("{prefix}self_attn.k_norm.weight"))
        {
            let (weight, dtype, _) = self
                .store
                .slice(&format!("{prefix}self_attn.k_norm.weight"))?;
            for head in 0..cfg.kv_heads {
                let start = head * cfg.head_dim;
                kernels::rmsnorm(
                    &mut self.scratch.k[start..start + cfg.head_dim],
                    weight,
                    dtype,
                    cfg.rms_eps,
                );
            }
        }
        if self.debug_capture_layer0_attn && layer == 0 {
            self.k_raw_capture = self.scratch.k.clone();
        }
        self.scratch.v.clear();
        self.scratch.v.resize(cfg.kv_heads * cfg.head_dim, 0.0);
        kernels::gemv(
            v_weight,
            v_dtype,
            cfg.kv_heads * cfg.head_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.v,
        );
        add_bias(&mut self.scratch.v, v_bias, v_bias_dtype);

        for head in 0..cfg.q_heads {
            let start = head * cfg.head_dim;
            kernels::rope_in_place(
                &mut self.scratch.q[start..start + cfg.head_dim],
                1,
                cfg.head_dim,
                position,
                cfg.rope_theta,
            );
        }
        for head in 0..cfg.kv_heads {
            let start = head * cfg.head_dim;
            kernels::rope_in_place(
                &mut self.scratch.k[start..start + cfg.head_dim],
                1,
                cfg.head_dim,
                position,
                cfg.rope_theta,
            );
        }

        // Append K/V to the cache for this position.
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        self.kv_k[layer].extend_from_slice(&self.scratch.k);
        self.kv_v[layer].extend_from_slice(&self.scratch.v);
        let cache_len = self.kv_k[layer].len() / kv_dim;

        // Attention: GQA — every q_heads/kv_heads query heads share one kv head.
        let head_group = cfg.q_heads / cfg.kv_heads;
        self.scratch.attn_out.clear();
        self.scratch
            .attn_out
            .resize(cfg.q_heads * cfg.head_dim, 0.0);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        for head in 0..cfg.q_heads {
            let kv_head = head / head_group;
            let q_base = head * cfg.head_dim;
            let kv_base = kv_head * cfg.head_dim;
            self.scratch.scores.clear();
            self.scratch.scores.resize(cache_len, f32::NEG_INFINITY);
            for pos in 0..cache_len {
                let k_base = pos * kv_dim + kv_base;
                let score: f32 = self.kv_k[layer][k_base..k_base + cfg.head_dim]
                    .iter()
                    .zip(&self.scratch.q[q_base..q_base + cfg.head_dim])
                    .map(|(k, q)| k * q)
                    .sum();
                self.scratch.scores[pos] = score * scale;
            }
            kernels::softmax(&mut self.scratch.scores);
            if head == 0 && self.debug_capture_layer0_attn {
                self.head0_scores = self.scratch.scores.clone();
            }
            for pos in 0..cache_len {
                let weight = self.scratch.scores[pos];
                if weight == 0.0 {
                    continue;
                }
                let v_base = pos * kv_dim + kv_base;
                for (index, out) in self.scratch.attn_out[q_base..q_base + cfg.head_dim]
                    .iter_mut()
                    .enumerate()
                {
                    *out += weight * self.kv_v[layer][v_base + index];
                }
            }
        }

        self.scratch.attn_projected.clear();
        self.scratch.attn_projected.resize(cfg.hidden_size, 0.0);
        kernels::gemv(
            o_weight,
            o_dtype,
            cfg.hidden_size,
            cfg.q_heads * cfg.head_dim,
            &self.scratch.attn_out,
            &mut self.scratch.attn_projected,
        );
        if self.debug_capture_layer0_attn && layer == 0 {
            self.debug_capture_layer0_attn = false;
            self.debug_capture_buf = Some(DebugCapture {
                normed: self.normed_capture.clone(),
                q_raw: self.q_raw_capture.clone(),
                k_raw: self.k_raw_capture.clone(),
                q: self.scratch.q.clone(),
                k: self.scratch.k.clone(),
                scores: self.head0_scores.clone(),
                attn_out: self.scratch.attn_out.clone(),
            });
        }
        for (value, residual) in self
            .scratch
            .attn_projected
            .iter()
            .zip(self.scratch.hidden.iter_mut())
        {
            *residual += *value;
        }

        // --- MLP block ---
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        {
            let (weight, dtype, _) = self
                .store
                .slice(&format!("{prefix}post_attention_layernorm.weight"))?;
            kernels::rmsnorm(&mut self.scratch.normed, weight, dtype, cfg.rms_eps);
        }
        if cfg.num_local_experts > 0 && cfg.num_experts_per_tok > 0 {
            // --- Mixture-of-experts block ---
            self.scratch.router.clear();
            self.scratch.router.resize(cfg.num_local_experts, 0.0);
            let (router_weight, router_dtype, _) = self
                .store
                .slice(&format!("{}{}", prefix, cfg.moe_layout.router))?;
            kernels::gemv(
                router_weight,
                router_dtype,
                cfg.num_local_experts,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.router,
            );
            let mut order: Vec<usize> = (0..cfg.num_local_experts).collect();
            order.sort_by(|&a, &b| {
                let left = self.scratch.router.get(b).copied().unwrap_or(0.0);
                let right = self.scratch.router.get(a).copied().unwrap_or(0.0);
                right.partial_cmp(&left).unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_k = cfg.num_experts_per_tok.min(cfg.num_local_experts);
            let selected: Vec<usize> = order.into_iter().take(top_k).collect();

            // RC2-B: routing-aware per-expert residency. Experts the current
            // router selection no longer picks for this layer are evicted;
            // freshly picked experts are paged in. State is keyed per
            // (layer, expert) so it holds across positions and generation steps.
            for expert in 0..cfg.num_local_experts {
                let key = (layer, expert);
                let was_resident = self.expert_residency.get(&key).copied().unwrap_or(false);
                if was_resident && !selected.contains(&expert) {
                    self.set_expert_resident(layer, expert, false);
                    self.expert_residency.insert(key, false);
                }
            }
            for &expert in &selected {
                let key = (layer, expert);
                if !self.expert_residency.get(&key).copied().unwrap_or(false) {
                    self.set_expert_resident(layer, expert, true);
                    self.expert_residency.insert(key, true);
                }
            }

            for expert in selected {
                let (gate_weight, gate_dtype, _) = self
                    .store
                    .slice(&format!(
                        "{}{}.{}.{}",
                        prefix,
                        cfg.moe_layout.experts_prefix,
                        expert,
                        cfg.moe_layout.gate
                    ))?;
                let (up_weight, up_dtype, _) = self
                    .store
                    .slice(&format!(
                        "{}{}.{}.{}",
                        prefix,
                        cfg.moe_layout.experts_prefix,
                        expert,
                        cfg.moe_layout.up
                    ))?;
                let (down_weight, down_dtype, _) = self
                    .store
                    .slice(&format!(
                        "{}{}.{}.{}",
                        prefix,
                        cfg.moe_layout.experts_prefix,
                        expert,
                        cfg.moe_layout.down
                    ))?;
                self.scratch.gate.clear();
                self.scratch.gate.resize(cfg.intermediate_size, 0.0);
                kernels::gemv(
                    gate_weight,
                    gate_dtype,
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    &self.scratch.normed,
                    &mut self.scratch.gate,
                );
                self.scratch.up.clear();
                self.scratch.up.resize(cfg.intermediate_size, 0.0);
                kernels::gemv(
                    up_weight,
                    up_dtype,
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    &self.scratch.normed,
                    &mut self.scratch.up,
                );
                kernels::silu_multiply(&mut self.scratch.gate, &self.scratch.up);
                self.scratch.mlp_out.clear();
                self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
                kernels::gemv(
                    down_weight,
                    down_dtype,
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    &self.scratch.gate,
                    &mut self.scratch.mlp_out,
                );
                for (value, residual) in self
                    .scratch
                    .mlp_out
                    .iter()
                    .zip(self.scratch.hidden.iter_mut())
                {
                    *residual += *value;
                }
            }
            return Ok(());
        }

        self.scratch.gate.clear();
        self.scratch.gate.resize(cfg.intermediate_size, 0.0);
        kernels::gemv(
            gate_weight,
            gate_dtype,
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.gate,
        );
        self.scratch.up.clear();
        self.scratch.up.resize(cfg.intermediate_size, 0.0);
        kernels::gemv(
            up_weight,
            up_dtype,
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.up,
        );
        kernels::silu_multiply(&mut self.scratch.gate, &self.scratch.up);
        self.scratch.mlp_out.clear();
        self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
        kernels::gemv(
            down_weight,
            down_dtype,
            cfg.hidden_size,
            cfg.intermediate_size,
            &self.scratch.gate,
            &mut self.scratch.mlp_out,
        );
        for (value, residual) in self
            .scratch
            .mlp_out
            .iter()
            .zip(self.scratch.hidden.iter_mut())
        {
            *residual += *value;
        }
        Ok(())
    }

    fn run_lm_head(&mut self) -> Result<(), String> {
        let cfg = &self.cfg;
        // Tied families reuse the embedding as the head; untied families
        // (e.g. llama) carry a separate lm_head.weight tensor.
        let head_unit = if cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        if !self.store.contains(head_unit) {
            return Err(format!(
                "Native backend: neither tied embeddings nor a separate '{head_unit}' tensor are present."
            ));
        }
        let (bytes, dtype, shape) = self
            .store
            .slice(head_unit)
            .map_err(|error| format!("Native backend: {error}"))?;
        if shape.len() != 2 || shape[1] != cfg.hidden_size {
            return Err("Native backend: unexpected lm_head shape.".into());
        }
        self.scratch.logits.clear();
        self.scratch.logits.resize(cfg.vocab_size, 0.0);
        // Ant: sweep the embedding (the largest tensor) in row chunks and
        // release each chunk's pages right away when it sits on the disk tier,
        // so the lm_head pass does not transiently hold all 272 MB resident.
        let chunk_rows = 16384usize;
        let row_stride = cfg.hidden_size * dtype.element_size();
        let mut start_row = 0usize;
        while start_row < cfg.vocab_size {
            let end_row = (start_row + chunk_rows).min(cfg.vocab_size);
            let byte_start = start_row * row_stride;
            let byte_len = (end_row - start_row) * row_stride;
            kernels::gemv(
                &bytes[byte_start..byte_start + byte_len],
                dtype,
                end_row - start_row,
                cfg.hidden_size,
                &self.scratch.hidden,
                &mut self.scratch.logits[start_row..end_row],
            );
            if self.store.tier(head_unit) == Tier::Disk {
                self.store.drop_range(head_unit, byte_start, byte_len);
            }
            start_row = end_row;
        }
        Ok(())
    }

    /// Test helper: stage-wise layer-0 attention capture for the last input
    /// position, used by the native attention capture test.
    #[allow(dead_code)]
    pub fn debug_attention_out(&mut self, tokens: &[u32]) -> Result<DebugCapture, String> {
        for (index, &token) in tokens.iter().enumerate().take(tokens.len() - 1) {
            self.forward_position(token as usize, index)?;
        }
        let last = tokens.len() - 1;
        self.debug_capture_layer0_attn = true;
        self.forward_position(tokens[last] as usize, last)?;
        self.debug_capture_buf
            .take()
            .ok_or_else(|| "Native backend: layer-0 attention capture did not run.".into())
    }

    /// Apply the Ant residency plan to the mapped pages: resident tiers are
    /// faulted in, disk tiers are dropped.
    pub fn apply_tiers(&mut self, plan: &BTreeMap<String, String>) {
        for unit_id in self.store.unit_ids().map(str::to_owned).collect::<Vec<_>>() {
            let tier = match plan.get(&unit_id).map(String::as_str) {
                Some("disk") => Tier::Disk,
                Some(_) | None => Tier::Resident,
            };
            self.store.set_tier(&unit_id, tier);
        }
    }

    /// RC2-B: apply one expert's residency transition. The router-selected
    /// expert's gate/up/down units move together between the resident and disk
    /// tiers; unit ids the store does not know are ignored.
    fn set_expert_resident(&self, layer: usize, expert: usize, resident: bool) {
        let prefix = format!("model.layers.{layer}.");
        let gate_unit = format!(
            "{}{}.{}.{}",
            prefix, self.cfg.moe_layout.experts_prefix, expert, self.cfg.moe_layout.gate
        );
        let up_unit = format!(
            "{}{}.{}.{}",
            prefix, self.cfg.moe_layout.experts_prefix, expert, self.cfg.moe_layout.up
        );
        let down_unit = format!(
            "{}{}.{}.{}",
            prefix, self.cfg.moe_layout.experts_prefix, expert, self.cfg.moe_layout.down
        );
        if resident {
            self.store.set_tier(&gate_unit, Tier::Resident);
            self.store.set_tier(&up_unit, Tier::Resident);
            self.store.set_tier(&down_unit, Tier::Resident);
        } else {
            self.store.set_tier(&gate_unit, Tier::Disk);
            self.store.set_tier(&up_unit, Tier::Disk);
            self.store.set_tier(&down_unit, Tier::Disk);
        }
    }

    pub fn page_in_unit(&mut self, unit_id: &str) -> bool {
        if self.store.unit_ids().any(|id| id == unit_id) {
            self.store.set_tier(unit_id, Tier::Resident);
            true
        } else {
            false
        }
    }

    pub fn evict_unit(&mut self, unit_id: &str) -> bool {
        if self.store.unit_ids().any(|id| id == unit_id) {
            self.store.set_tier(unit_id, Tier::Disk);
            true
        } else {
            false
        }
    }
}

fn add_bias(values: &mut [f32], bias_bytes: &[u8], dtype: WeightDType) {
    if bias_bytes.is_empty() {
        return; // families without attention biases (e.g. llama)
    }
    for (value, index) in values.iter_mut().zip(0..) {
        *value += kernels::scalar_at(bias_bytes, index, dtype);
    }
}

