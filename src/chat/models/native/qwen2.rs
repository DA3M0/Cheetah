//! Native Qwen2 forward pass over the mmap-backed weight store. Activations
//! are f32; weight bytes are read in their stored dtype straight from the
//! mapping (zero copies). Disk-tier units are dropped from the resident set at
//! the end of every step so a generation holds only the planned resident bytes.

use super::kernels;
use super::store::{MmapWeightStore, Tier, WeightDType};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Qwen2Config {
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
}

impl Qwen2Config {
    pub fn from_config_json(raw: &Value) -> Result<Self, String> {
        let field = |name: &str| -> Result<f64, String> {
            raw.get(name)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("config.json is missing '{name}' for the native backend."))
        };
        let hidden_size = field("hidden_size")? as usize;
        let q_heads = field("num_attention_heads")? as usize;
        Ok(Self {
            hidden_size,
            num_layers: field("num_hidden_layers")? as usize,
            q_heads,
            kv_heads: field("num_key_value_heads")? as usize,
            head_dim: hidden_size / q_heads,
            intermediate_size: field("intermediate_size")? as usize,
            vocab_size: field("vocab_size")? as usize,
            rms_eps: field("rms_norm_eps")? as f32,
            rope_theta: field("rope_theta")? as f32,
            tie_word_embeddings: raw
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

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
        let field_f = |name: &str| raw.and_then(|value| value.get(name)).and_then(Value::as_f64);
        let field_b = |name: &str| raw.and_then(|value| value.get(name)).and_then(Value::as_bool);
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
        let num_layers = field_u("num_hidden_layers")
            .unwrap_or((store.max_layer_id() + 1).max(1) as usize);
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
            rms_eps: field_f("rms_norm_eps")
                .unwrap_or(family.rms_norm_eps) as f32,
            rope_theta: field_f("rope_theta").unwrap_or(family.rope_theta) as f32,
            tie_word_embeddings: field_b("tie_word_embeddings")
                .unwrap_or(family.tie_word_embeddings),
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
    logits: Vec<f32>,
}

pub struct Qwen2Native {
    pub(super) cfg: Qwen2Config,
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
}

/// Stage-wise debug capture of layer 0 at the last input position.
/// Test instrumentation only; individual stages are consumed by the
/// candle-reference comparison tests.
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

impl Clone for Qwen2Native {
    fn clone(&self) -> Self {
        // Weights are shared through the Arc'd store; only a fresh KV cache is
        // created (the engine clears the cache before every generation anyway).
        Self::with_store(self.cfg.clone(), Arc::clone(&self.store))
    }
}

impl Qwen2Native {
    pub fn with_store(cfg: Qwen2Config, store: Arc<MmapWeightStore>) -> Self {
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        Self {
            kv_k: vec![Vec::new(); cfg.num_layers],
            kv_v: vec![Vec::new(); cfg.num_layers],
            scratch: Scratch {
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
            let (weight, dtype, _) =
                self.store
                    .slice(&format!("{prefix}input_layernorm.weight"))?;
            kernels::rmsnorm(&mut self.scratch.normed, weight, dtype, cfg.rms_eps);
        }
        if self.debug_capture_layer0_attn && layer == 0 {
            self.normed_capture = self.scratch.normed.clone();
        }

        let (q_weight, q_dtype, _) =
            self.store.slice(&format!("{prefix}self_attn.q_proj.weight"))?;
        // Biases are family-optional (qwen2 has them, llama does not).
        let (q_bias, q_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.q_proj.bias"))
            .unwrap_or((&[], q_dtype, &[]));
        let (k_weight, k_dtype, _) =
            self.store.slice(&format!("{prefix}self_attn.k_proj.weight"))?;
        let (k_bias, k_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.k_proj.bias"))
            .unwrap_or((&[], k_dtype, &[]));
        let (v_weight, v_dtype, _) =
            self.store.slice(&format!("{prefix}self_attn.v_proj.weight"))?;
        let (v_bias, v_bias_dtype, _) = self
            .store
            .slice(&format!("{prefix}self_attn.v_proj.bias"))
            .unwrap_or((&[], v_dtype, &[]));
        let (o_weight, o_dtype, _) =
            self.store.slice(&format!("{prefix}self_attn.o_proj.weight"))?;
        let (gate_weight, gate_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.gate_proj.weight"))?;
        let (up_weight, up_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.up_proj.weight"))?;
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
        if self.store.contains(&format!("{prefix}self_attn.q_norm.weight")) {
            let (weight, dtype, _) =
                self.store.slice(&format!("{prefix}self_attn.q_norm.weight"))?;
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
        if self.store.contains(&format!("{prefix}self_attn.k_norm.weight")) {
            let (weight, dtype, _) =
                self.store.slice(&format!("{prefix}self_attn.k_norm.weight"))?;
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
        self.scratch.attn_out.resize(cfg.q_heads * cfg.head_dim, 0.0);
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
        self.scratch
            .attn_projected
            .resize(cfg.hidden_size, 0.0);
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
            let (weight, dtype, _) =
                self.store
                    .slice(&format!("{prefix}post_attention_layernorm.weight"))?;
            kernels::rmsnorm(&mut self.scratch.normed, weight, dtype, cfg.rms_eps);
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
    /// position, used by the candle-reference comparison test.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared_model_dir() -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new("/home/benny/models/qwen-0.5b-prepared");
        let units = dir.join(crate::ant::CHEETAH_UNITS_FILENAME);
        if dir.join("config.json").is_file() && units.is_file() {
            Some(dir.to_path_buf())
        } else {
            None
        }
    }

    fn shard_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir.join(".cheetah/weights"))
            .expect("weights dir")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
                    .then_some(path)
            })
            .collect();
        paths.sort();
        paths
    }

    /// Cross-check the native logits against the candle path on the same
    /// prepared model. Requires a locally prepared model; skips otherwise.
    #[test]
    fn native_logits_match_candle() {
        let Some(model_dir) = prepared_model_dir() else {
            eprintln!("skipping native/candle logits comparison: model not prepared");
            return;
        };
        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let cfg = Qwen2Config::from_config_json(&raw).unwrap();

        let store = Arc::new(MmapWeightStore::open(&model_dir).unwrap());
        let mut native = Qwen2Native::with_store(cfg.clone(), Arc::clone(&store));
        // A multi-token prompt exercises the causal attention path; the single
        // token case degenerates (softmax over one entry) and cannot catch
        // attention bugs.
        let tokens: Vec<u32> = vec![9707, 11, 2624, 358, 1457, 311, 358, 1614]; // "Hi! I want to know"
        let native_logits = native.forward(&tokens, 0).unwrap();

        let registration =
            crate::chat::models::registration_for_architecture("Qwen2ForCausalLM").unwrap();
        let mut candle_model = crate::chat::models::candle::load_model(
            &shard_paths(&model_dir),
            &config_bytes,
            raw.clone(),
            registration,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let candle_logits = candle_model.forward(&tokens, 0).unwrap();

        let mut diffs: Vec<f32> = native_logits
            .iter()
            .zip(&candle_logits)
            .map(|(a, b)| (a - b).abs())
            .collect();
        diffs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let max_diff = diffs[0];
        let native_top = native_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let candle_top = candle_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        eprintln!(
            "native top={native_top} candle top={candle_top} max|diff|={max_diff:.4} mean|diff|={:.4}",
            diffs.iter().sum::<f32>() / diffs.len() as f32
        );
        assert_eq!(native_top, candle_top, "argmax logits diverge");
        assert!(
            max_diff < 0.1,
            "native/candle logits diverge by more than 0.1 (max {max_diff})"
        );
    }

    /// Layer-0 reference computed with candle tensor ops (replicating
    /// candle-transformers' qwen2 math exactly) compared against the native
    /// kernels — isolates the attention/KV path on real weights.
    #[test]
    fn native_layer0_matches_candle_reference() {
        use candle_core::{DType, Device, Tensor};
        let Some(model_dir) = prepared_model_dir() else {
            eprintln!("skipping layer0 comparison: model not prepared");
            return;
        };
        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let cfg = Qwen2Config::from_config_json(&raw).unwrap();
        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &shard_paths(&model_dir),
                DType::F32,
                &device,
            )
            .unwrap()
        };
        use candle_nn::Module;
        let prefix = "model.layers.0.";
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;

        // Two-token input hidden states after embedding.
        let tokens: Vec<u32> = vec![9707, 11];
        let embed = vb.get((cfg.vocab_size, hidden), "model.embed_tokens.weight").unwrap();
        let xs = embed
            .index_select(&Tensor::from_slice(&tokens, (tokens.len(),), &device).unwrap(), 0)
            .unwrap(); // (2, hidden)

        // input_layernorm
        let normed = candle_nn::rms_norm(hidden, 1e-6, vb.pp(&format!("{prefix}input_layernorm")))
            .unwrap()
            .forward(&xs)
            .unwrap();

        // q/k/v projections (with bias)
        let linear = |out: usize, name: &str, bias: bool| candle_nn::Linear::new(
            vb.get((out, hidden), &format!("{prefix}{name}.weight")).unwrap(),
            if bias { Some(vb.get((out,), &format!("{prefix}{name}.bias")).unwrap()) } else { None },
        );
        let q = linear(cfg.q_heads * head_dim, "self_attn.q_proj", true).forward(&normed).unwrap();
        let k = linear(cfg.kv_heads * head_dim, "self_attn.k_proj", true).forward(&normed).unwrap();
        let v = linear(cfg.kv_heads * head_dim, "self_attn.v_proj", true).forward(&normed).unwrap();

        // rope (replicating candle qwen2: duplicated-frequency rotate-half)
        let rotate = |xs: &Tensor| -> Tensor {
            let d = xs.dim(2).unwrap();
            let x1 = xs.narrow(2, 0, d / 2).unwrap();
            let x2 = xs.narrow(2, d / 2, d - d / 2).unwrap();
            Tensor::cat(&[&x2.neg().unwrap(), &x1], 2).unwrap()
        };
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta.powf(i as f32 / head_dim as f32)) as f32)
            .collect();
        let rope_for = |states: &Tensor, heads: usize| -> Tensor {
            // states: (2, heads * head_dim) -> (2, heads, head_dim); cos/sin are
            // (2, 1, head_dim) with each frequency duplicated across both halves.
            let states = states.reshape((2, heads, head_dim)).unwrap();
            let mut cos_rows = Vec::new();
            let mut sin_rows = Vec::new();
            for pos in 0..2usize {
                let mut cos_row = Vec::with_capacity(head_dim);
                let mut sin_row = Vec::with_capacity(head_dim);
                for _ in 0..2usize {
                    for f in &inv_freq {
                        cos_row.push((pos as f32 * f).cos());
                        sin_row.push((pos as f32 * f).sin());
                    }
                }
                cos_rows.push(cos_row);
                sin_rows.push(sin_row);
            }
            let cos = Tensor::new(cos_rows, &device)
                .unwrap()
                .unsqueeze(1)
                .unwrap();
            let sin = Tensor::new(sin_rows, &device)
                .unwrap()
                .unsqueeze(1)
                .unwrap();
            (states.broadcast_mul(&cos).unwrap()
                + rotate(&states).broadcast_mul(&sin).unwrap())
            .unwrap()
        };
        let q_rope = rope_for(&q, cfg.q_heads);
        let k_rope = rope_for(&k, cfg.kv_heads);

        // Attention for query position 1 (the last token), head h, over both keys.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_vec = q_rope.get(1).unwrap(); // (q_heads, head_dim)
        let mut attn_ref = vec![0f32; cfg.q_heads * head_dim];
        let mut head0_ref = Vec::new();
        for head in 0..cfg.q_heads {
            let kv_head = head / (cfg.q_heads / cfg.kv_heads);
            let scores: Vec<f32> = (0..2usize)
                .map(|pos| {
                    let k_head = k_rope.get(pos).unwrap().get(kv_head).unwrap(); // (head_dim,)
                    let dot: f32 = (0..head_dim)
                        .map(|i| {
                            q_vec.get(head).unwrap().get(i).unwrap().to_scalar::<f32>().unwrap()
                                * k_head.get(i).unwrap().to_scalar::<f32>().unwrap()
                        })
                        .sum();
                    dot * scale
                })
                .collect();
            let mut weights = scores.clone();
            let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for w in weights.iter_mut() {
                *w = (*w - max).exp();
                sum += *w;
            }
            for w in weights.iter_mut() {
                *w /= sum;
            }
            if head == 0 {
                head0_ref = weights.clone();
            }
            for pos in 0..2usize {
                let v_row = v.get(pos).unwrap();
                let base = kv_head * head_dim;
                for i in 0..head_dim {
                    attn_ref[head * head_dim + i] +=
                        weights[pos] * v_row.get(base + i).unwrap().to_scalar::<f32>().unwrap();
                }
            }
        }

        // o_proj reference for position 1.
        let o = linear(cfg.hidden_size, "self_attn.o_proj", false)
            .forward(&Tensor::from_vec(attn_ref.clone(), (1, attn_ref.len()), &device).unwrap())
            .unwrap()
            .get(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // Native stage-wise capture.
        let store = Arc::new(MmapWeightStore::open(&model_dir).unwrap());
        let mut native = Qwen2Native::with_store(cfg.clone(), Arc::clone(&store));
        let cap = native.debug_attention_out(&tokens).unwrap();

        let q_candle_row = q_rope.get(1).unwrap().to_vec2::<f32>().unwrap().concat();
        let k_candle_row = k_rope.get(1).unwrap().to_vec2::<f32>().unwrap().concat();
        let diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        eprintln!(
            "normed max|diff|={:.5} q_raw max|diff|={:.5} k_raw max|diff|={:.5} q_rope max|diff|={:.5} k_rope max|diff|={:.5} head0 scores max|diff|={:.5} attn_out max|diff|={:.5} o max|diff|={:.5}",
            diff(&cap.normed, &normed.get(1).unwrap().to_vec1::<f32>().unwrap()),
            diff(&cap.q_raw, &q_candle_row),
            diff(&cap.k_raw, &k_candle_row),
            diff(&cap.q, &q_candle_row),
            diff(&cap.k, &k_candle_row),
            diff(&cap.scores, &head0_ref),
            diff(&cap.attn_out, &attn_ref),
            {
                let o_native = linear(cfg.hidden_size, "self_attn.o_proj", false)
                    .forward(&Tensor::from_vec(cap.attn_out.clone(), (1, cap.attn_out.len()), &device).unwrap())
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                diff(&o, &o_native)
            }
        );
        eprintln!(
            "q_raw native[32..40]={:?}\nq_pre-rope candle[32..40]={:?}\nq_post-rope native[32..40]={:?}\nq_post-rope candle[32..40]={:?}\nq_raw native[0..8]={:?}\nq_pre-rope candle[0..8]={:?}\nq_post-rope native[0..8]={:?}\nq_post-rope candle[0..8]={:?}",
            &cap.q_raw[32..40],
            &q.get(1).unwrap().to_vec1::<f32>().unwrap()[32..40],
            &cap.q[32..40],
            &q_candle_row[32..40],
            &cap.q_raw[..8],
            &q.get(1).unwrap().to_vec1::<f32>().unwrap()[..8],
            &cap.q[..8],
            &q_candle_row[..8],
        );
        eprintln!(
            "k_raw native[0..8]={:?}\nk_pre-rope candle[0..8]={:?}",
            &cap.k_raw[..8],
            &k.get(1).unwrap().to_vec1::<f32>().unwrap()[..8]
        );
        assert!(
            diff(&cap.attn_out, &attn_ref) < 1e-3,
            "layer-0 attention output diverges from the candle reference"
        );
    }

    /// Isolate the gemv path: naive sequential dot products over the store's
    /// q_proj bytes must reproduce the runtime capture exactly.
    #[test]
    fn native_q_proj_gemv_matches_naive_reference() {
        let Some(model_dir) = prepared_model_dir() else {
            eprintln!("skipping gemv isolation: model not prepared");
            return;
        };
        let config_bytes = std::fs::read(model_dir.join("config.json")).unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&config_bytes).unwrap();
        let cfg = Qwen2Config::from_config_json(&raw).unwrap();
        let store = Arc::new(MmapWeightStore::open(&model_dir).unwrap());
        let mut native = Qwen2Native::with_store(cfg.clone(), Arc::clone(&store));
        let tokens: Vec<u32> = vec![9707, 11];
        let cap = native.debug_attention_out(&tokens).unwrap();

        let (w_bytes, w_dtype, w_shape) = store.slice("model.layers.0.self_attn.q_proj.weight").unwrap();
        let (b_bytes, b_dtype, _) = store.slice("model.layers.0.self_attn.q_proj.bias").unwrap();
        assert_eq!(w_shape, &[896, 896]);
        let rows = cfg.q_heads * cfg.head_dim;
        let mut naive = vec![0f32; rows];
        for (row, out) in naive.iter_mut().enumerate() {
            let mut acc = 0f32;
            for col in 0..cfg.hidden_size {
                acc += kernels::scalar_at(w_bytes, row * cfg.hidden_size + col, w_dtype)
                    * cap.normed[col];
            }
            *out = acc + kernels::scalar_at(b_bytes, row, b_dtype);
        }
        let max_diff = diff_fn(&naive, &cap.q_raw);
        eprintln!("q_proj naive vs runtime gemv max|diff|={max_diff:.5}");
        assert!(max_diff < 1e-3, "gemv diverges from naive reference ({max_diff})");
    }

    fn diff_fn(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }
}
