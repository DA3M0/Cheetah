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
    /// Sparse (sliding-window) attention: keys older than this many positions
    /// are masked out of the softmax. 0 disables the window (full attention),
    /// which is what an absent `sliding_window` or a config that sets
    /// `use_sliding_window: false` resolves to.
    pub sliding_window: usize,
    /// Mixture-of-experts layout: number of experts (0 => dense feed-forward)
    /// and experts routed per token (0 => dense).
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    /// Resolved rotary span per head: `head_dim` for full rope (llama-shaped
    /// families), the `partial_rotary_factor * head_dim` share for phi-style
    /// partial rotary. Rope applies to the leading `rotary_dim` dims only.
    pub rotary_dim: usize,
    /// Sparse-MoE expert naming layout (Mixtral-compatible defaults). Drives
    /// router + expert weight slicing so any architecture can declare a
    /// sparse feed-forward without Mixtral-only hardcoding.
    pub moe_layout: MoEExpertLayout,
    /// Dense-architecture deltas over the llama-shaped baseline (norm kind,
    /// activation, projection naming, residual topology, partial rope).
    pub features: DenseFeatures,
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

    /// Router weight unit id for one layer. The units manifest stores full
    /// tensor names, so the `.weight` suffix belongs to the unit id — the same
    /// suffix the Vulkan loader appends to its layout strings.
    pub fn router_unit(&self, prefix: &str) -> String {
        format!("{prefix}{}.weight", self.router)
    }

    /// One expert projection's unit id for one layer (`projection` is `gate` /
    /// `up` / `down` from this layout).
    pub fn expert_unit(&self, prefix: &str, expert: usize, projection: &str) -> String {
        format!(
            "{prefix}{}.{}.{projection}.weight",
            self.experts_prefix, expert
        )
    }
}

/// Dense-architecture deltas over the llama-shaped baseline (llama / qwen2 /
/// qwen3 / mixtral): RMSNorm without weight offset, rotate-half rope over the
/// full head, SwiGLU MLP over `gate/up/down_proj`, sequential residuals, and
/// split `q/k/v_proj` projections with an `o_proj` output projection. Every
/// dense family declares its deltas from this baseline through one struct, so
/// the forward pass branches on features instead of architecture names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenseFeatures {
    /// gemma: RMSNorm output is `x / rms * (1 + weight)` — the affine weight
    /// is stored as-is but applied with an implicit +1 offset.
    pub norm_weight_offset: bool,
    /// gemma: the embedding output is scaled by `sqrt(hidden_size)` before the
    /// first layer (and the lm_head reads the unscaled embedding table).
    pub embedding_scale: bool,
    /// gemma / phi: the feed-forward activation is GELU (tanh approximation —
    /// `gelu_pytorch_tanh` / `gelu_new`) instead of SiLU.
    pub mlp_gelu: bool,
    /// phi: norms are LayerNorms (mean/variance + affine weight + bias)
    /// carrying a `.bias` unit, instead of RMSNorm.
    pub layer_norm: bool,
    /// phi: attention and MLP both read the same input norm and the residual
    /// is `x + attn_out + mlp_out` (one residual per layer, no post-attention
    /// norm unit).
    pub parallel_residual: bool,
    /// phi: the attention input projection is fused into one `qkv_proj` unit
    /// (q/k/v rows stacked) and the output projection is named `dense`
    /// instead of `o_proj`.
    pub fused_qkv: bool,
    /// phi: the MLP projections are plain `fc1`/`fc2` units (no gate/up split);
    /// activation applies to `fc1(x)` and `fc2` projects back to hidden.
    pub fc_mlp: bool,
    /// phi: rope applies only to the first `round(factor * head_dim)` head
    /// dims (GPT-NeoX partial rotary); 1.0 = full rope over `head_dim`.
    pub partial_rotary_factor: f32,
}

impl Default for DenseFeatures {
    fn default() -> Self {
        Self {
            norm_weight_offset: false,
            embedding_scale: false,
            mlp_gelu: false,
            layer_norm: false,
            parallel_residual: false,
            fused_qkv: false,
            fc_mlp: false,
            partial_rotary_factor: 1.0,
        }
    }
}

/// gemma family deltas: llama tensor naming with offset RMSNorms, a
/// sqrt(hidden)-scaled embedding, and GeGLU (tanh-GELU gated) feed-forward.
pub const GEMMA_FEATURES: DenseFeatures = DenseFeatures {
    norm_weight_offset: true,
    embedding_scale: true,
    mlp_gelu: true,
    layer_norm: false,
    parallel_residual: false,
    fused_qkv: false,
    fc_mlp: false,
    partial_rotary_factor: 1.0,
};

/// phi family deltas: LayerNorms with biases, a parallel attention/MLP
/// residual, fused `qkv_proj` + `dense` attention, `fc1`/`fc2` MLP, and
/// partial rotary embeddings (the exact rotary dim resolves per model from
/// `partial_rotary_factor * head_dim`).
pub const PHI_FEATURES: DenseFeatures = DenseFeatures {
    norm_weight_offset: false,
    embedding_scale: false,
    mlp_gelu: true,
    layer_norm: true,
    parallel_residual: true,
    fused_qkv: true,
    fc_mlp: true,
    partial_rotary_factor: 0.4,
};

impl DecoderConfig {
    /// Resolve the configuration through the parameter fallback chain:
    /// explicit config fields, then derivation from the units manifest's
    /// tensor shapes, then the registered family constants. This keeps the
    /// native backend working when the model directory's config.json is
    /// missing or incomplete.
    pub(crate) fn resolve(
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
            max_position_embeddings: None,
            moe_layout: MoEExpertLayout::MIXTRAL,
            features: DenseFeatures::default(),
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
        // Attention projection naming: phi fuses q/k/v into one `qkv_proj`
        // tensor, the llama-shaped families carry split `q/k_proj` units.
        let features = family.features;
        let q_unit = if features.fused_qkv {
            "model.layers.0.self_attn.qkv_proj.weight"
        } else {
            "model.layers.0.self_attn.q_proj.weight"
        };
        let q_out = store
            .unit_shape(q_unit)
            .ok_or_else(|| format!("Native backend: units manifest has no {q_unit} tensor."))?[0];
        let k_out = if features.fused_qkv {
            q_out
        } else {
            store
                .unit_shape("model.layers.0.self_attn.k_proj.weight")
                .ok_or_else(|| "Native backend: units manifest has no k_proj tensor.".to_string())?[0]
        };
        let moe_layout = family.moe_layout;
        let expert_gate_unit = moe_layout.expert_unit("model.layers.0.", 0, moe_layout.gate);
        let intermediate_size = field_u("intermediate_size").unwrap_or_else(|| {
            store
                .unit_shape("model.layers.0.mlp.gate_proj.weight")
                .map(|shape| shape[0])
                // phi names its MLP projections fc1/fc2; fc1's rows carry the
                // intermediate size.
                .or_else(|| store.unit_shape("model.layers.0.mlp.fc1.weight").map(|shape| shape[0]))
                // Sparse feed-forward: a sparse layer has no `mlp.*` units, so
                // the first expert's gate row carries the intermediate size.
                .or_else(|| store.unit_shape(&expert_gate_unit).map(|shape| shape[0]))
                .unwrap_or(0)
        });
        if intermediate_size == 0 {
            return Err(
                "Native backend: cannot resolve the intermediate size from config.json or the units manifest."
                    .into(),
            );
        }
        // Expert counts: config first, then the layer-0 units manifest for the
        // layer-wide count. Per-token `num_experts_per_tok` is config-only —
        // tensor shapes cannot carry it — so a sparse manifest without it fails
        // loudly instead of silently taking the dense feed-forward path.
        let num_local_experts =
            field_u("num_local_experts").unwrap_or_else(|| count_layer_experts(store, &moe_layout));
        let num_experts_per_tok = field_u("num_experts_per_tok").unwrap_or(0);
        if num_local_experts > 0 && num_experts_per_tok == 0 {
            return Err(format!(
                "Native backend: the units manifest carries {num_local_experts} experts but num_experts_per_tok is not declared; keep config.json in the prepared directory so expert routing can be resolved."
            ));
        }
        if family.features.parallel_residual && num_local_experts > 0 {
            return Err(
                "Native backend: a parallel-residual family cannot also carry routed experts; the two residual topologies are mutually exclusive."
                    .into(),
            );
        }
        // Sparse attention: honor `use_sliding_window` (default true, the
        // Hugging Face convention). A disabled window — or a family that never
        // declares one — resolves to 0, i.e. attend to the whole cache.
        let sliding_window = if field_b("use_sliding_window").unwrap_or(true) {
            field_u("sliding_window").unwrap_or(0)
        } else {
            0
        };

        let (q_heads, kv_heads, head_dim) = match field_u("num_attention_heads") {
            Some(q_heads) => {
                let head_dim = field_u("head_dim").unwrap_or_else(|| {
                    family
                        .profile_for(hidden_size, vocab_size)
                        .map(|(_, _, head_dim)| head_dim)
                        .unwrap_or(if features.fused_qkv {
                            // Fused `qkv_proj` stacks q/k/v rows; the fallback
                            // assumes the phi MHA shape (kv mirrors q), so
                            // q_out = 3 * q_heads * head_dim.
                            q_out / (q_heads.max(1) * 3)
                        } else {
                            q_out / q_heads.max(1)
                        })
                });
                let kv_heads = field_u("num_key_value_heads").unwrap_or_else(|| {
                    if features.fused_qkv {
                        q_heads // phi is MHA: the fused tensor mirrors q rows
                    } else if k_out % head_dim == 0 {
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

        // Partial rotary (phi): the model config's `partial_rotary_factor`
        // wins over the family constant; the resolved rotary dim rounds to the
        // nearest head-dim share. 1.0 (or an absent factor) = full rope.
        let rotary_factor = field_f("partial_rotary_factor")
            .unwrap_or(features.partial_rotary_factor as f64)
            as f32;
        let rotary_dim = if rotary_factor >= 1.0 {
            head_dim
        } else {
            ((rotary_factor * head_dim as f32).round() as usize).clamp(1, head_dim)
        };

        // Feed-forward activation: the config's `hidden_act` wins over the
        // family delta (`gelu` / `gelu_new` / `gelu_pytorch_tanh` are the tanh
        // approximation; `silu` is the llama-shaped baseline).
        let mlp_gelu = match raw.and_then(|value| value.get("hidden_act")).and_then(Value::as_str)
        {
            Some("gelu") | Some("gelu_new") | Some("gelu_pytorch_tanh") => true,
            Some("silu") => false,
            _ => features.mlp_gelu,
        };
        let features = DenseFeatures {
            mlp_gelu,
            ..features
        };

        Ok(Self {
            hidden_size,
            num_layers,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            intermediate_size,
            vocab_size,
            rms_eps: field_f("rms_norm_eps").unwrap_or(family.rms_norm_eps) as f32,
            rope_theta: field_f("rope_theta").unwrap_or(family.rope_theta) as f32,
            tie_word_embeddings: field_b("tie_word_embeddings")
                .unwrap_or(family.tie_word_embeddings),
            sliding_window,
            num_local_experts,
            num_experts_per_tok,
            moe_layout,
            features,
        })
    }
}

/// Layer-0 expert count read back from the units manifest (sparse manifests
/// name experts `<experts_prefix>.<index>.<projection>.weight`).
fn count_layer_experts(store: &MmapWeightStore, layout: &MoEExpertLayout) -> usize {
    let prefix = format!("model.layers.0.{}.", layout.experts_prefix);
    let suffix = format!(".{}.weight", layout.gate);
    store
        .unit_ids()
        .filter(|unit_id| unit_id.starts_with(&prefix) && unit_id.ends_with(&suffix))
        .filter_map(|unit_id| {
            unit_id[prefix.len()..unit_id.len() - suffix.len()]
                .parse::<usize>()
                .ok()
        })
        .max()
        .map(|highest| highest + 1)
        .unwrap_or(0)
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
    /// MoE accumulator: each selected expert's scaled output lands here
    /// before swapping into `mlp_out`.
    mlp_out_weighted: Vec<f32>,
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
                mlp_out_weighted: Vec::with_capacity(cfg.hidden_size),
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
            // gemma scales the embedding output by sqrt(hidden_size); the
            // lm_head later reads the same (unscaled) table directly.
            if self.cfg.features.embedding_scale {
                let scale = (self.cfg.hidden_size as f32).sqrt();
                for value in self.scratch.hidden.iter_mut() {
                    *value *= scale;
                }
            }
        }
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.store.drop_disk_units_of_layer(layer as i32);
        }
        // Final norm: phi's GPT-NeoX heritage names it `final_layernorm`; the
        // llama-shaped families use `model.norm`. The parallel-residual flag
        // selects the phi block shape, and the naming travels with it.
        let final_norm_unit = if self.cfg.features.parallel_residual {
            "model.final_layernorm"
        } else {
            "model.norm"
        };
        Self::apply_norm_to(
            &self.store,
            &self.cfg.features,
            self.cfg.rms_eps,
            final_norm_unit,
            &mut self.scratch.hidden,
        )?;
        Ok(())
    }

    /// Apply one norm unit (`{unit}.weight`, plus `{unit}.bias` for
    /// LayerNorm families) to `target`: LayerNorm for phi-shaped families,
    /// otherwise RMSNorm with the gemma `(1 + weight)` offset when declared.
    /// An associated function so callers can borrow `self.store` and
    /// `self.scratch` disjointly.
    fn apply_norm_to(
        store: &MmapWeightStore,
        features: &DenseFeatures,
        eps: f32,
        unit: &str,
        target: &mut [f32],
    ) -> Result<(), String> {
        let (weight, dtype, _) = store.slice(&format!("{unit}.weight"))?;
        match (features.layer_norm, features.norm_weight_offset) {
            (true, _) => {
                let (bias, bias_dtype, _) = store
                    .slice(&format!("{unit}.bias"))
                    .unwrap_or((&[], dtype, &[]));
                kernels::layernorm(target, weight, dtype, bias, bias_dtype, eps);
            }
            (false, true) => kernels::rmsnorm_offset(target, weight, dtype, eps),
            (false, false) => kernels::rmsnorm(target, weight, dtype, eps),
        }
        Ok(())
    }

    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        // Copy the config values this fn needs across mutable calls (a
        // long-lived `&self.cfg` borrow would fight the `&mut self` calls).
        let features = self.cfg.features;
        let rms_eps = self.cfg.rms_eps;
        let prefix = format!("model.layers.{layer}.");

        // Input norm: the attention reads it, and phi's parallel residual
        // feeds the MLP from the same normed state.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        Self::apply_norm_to(
            &self.store,
            &features,
            rms_eps,
            &format!("{prefix}input_layernorm"),
            &mut self.scratch.normed,
        )?;
        if self.debug_capture_layer0_attn && layer == 0 {
            self.normed_capture = self.scratch.normed.clone();
        }

        self.forward_attention(&prefix, layer, position)?;

        if features.parallel_residual {
            // phi: attention and MLP share the input norm; the residual is
            // `x + attn_out + mlp_out` — one update, no post-attention norm.
            self.forward_mlp(&prefix, layer)?;
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
        } else {
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
            Self::apply_norm_to(
                &self.store,
                &features,
                rms_eps,
                &format!("{prefix}post_attention_layernorm"),
                &mut self.scratch.normed,
            )?;
            self.forward_mlp(&prefix, layer)?;
            let mlp_out = std::mem::take(&mut self.scratch.mlp_out);
            for (value, residual) in mlp_out.iter().zip(self.scratch.hidden.iter_mut()) {
                *residual += value;
            }
        }
        Ok(())
    }

    /// Attention block over `scratch.normed`, written to
    /// `scratch.attn_projected` (the caller owns the residual add). Handles
    /// the family deltas: fused `qkv_proj` + `dense` output naming (phi),
    /// partial rotary (phi), and family-optional q/k/v/o biases.
    fn forward_attention(&mut self, prefix: &str, layer: usize, position: usize) -> Result<(), String> {
        let cfg = &self.cfg;

        if cfg.features.fused_qkv {
            // phi: one `qkv_proj` unit stacks q/k/v rows; split after the
            // projection (+ fused bias).
            let (qkv_weight, qkv_dtype, _) = self
                .store
                .slice(&format!("{prefix}self_attn.qkv_proj.weight"))?;
            let (qkv_bias, qkv_bias_dtype, _) = self
                .store
                .slice(&format!("{prefix}self_attn.qkv_proj.bias"))
                .unwrap_or((&[], qkv_dtype, &[]));
            let fused_rows = (cfg.q_heads + 2 * cfg.kv_heads) * cfg.head_dim;
            self.scratch.q.clear();
            self.scratch.q.resize(fused_rows, 0.0);
            kernels::gemv(
                qkv_weight,
                qkv_dtype,
                fused_rows,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.q,
            );
            add_bias(&mut self.scratch.q, qkv_bias, qkv_bias_dtype);
            let q_dim = cfg.q_heads * cfg.head_dim;
            let kv_dim = cfg.kv_heads * cfg.head_dim;
            self.scratch.k.clear();
            self.scratch
                .k
                .extend_from_slice(&self.scratch.q[q_dim..q_dim + kv_dim]);
            self.scratch.v.clear();
            self.scratch
                .v
                .extend_from_slice(&self.scratch.q[q_dim + kv_dim..q_dim + 2 * kv_dim]);
            self.scratch.q.truncate(q_dim);
        } else {
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
        }

        // Rope over the leading `rotary_dim` dims of each head: head_dim for
        // the full-rotation families, the partial share for phi (remaining
        // dims pass through untouched).
        let rotary_dim = cfg.rotary_dim;
        for head in 0..cfg.q_heads {
            let start = head * cfg.head_dim;
            kernels::rope_in_place(
                &mut self.scratch.q[start..start + rotary_dim],
                1,
                rotary_dim,
                position,
                cfg.rope_theta,
            );
        }
        for head in 0..cfg.kv_heads {
            let start = head * cfg.head_dim;
            kernels::rope_in_place(
                &mut self.scratch.k[start..start + rotary_dim],
                1,
                rotary_dim,
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
        // Sparse (sliding-window) attention: the query at `position` only sees
        // keys from `end - window` onward (`end` is one past this position),
        // matching the Vulkan attention shader. window == 0 attends to the
        // whole cache.
        let window_start = if cfg.sliding_window == 0 {
            0
        } else {
            (position + 1).saturating_sub(cfg.sliding_window)
        };
        // Never start past the cache: an all-masked softmax would be NaN.
        let window_start = window_start.min(cache_len.saturating_sub(1));
        for head in 0..cfg.q_heads {
            let kv_head = head / head_group;
            let q_base = head * cfg.head_dim;
            let kv_base = kv_head * cfg.head_dim;
            self.scratch.scores.clear();
            self.scratch.scores.resize(cache_len, f32::NEG_INFINITY);
            for pos in window_start..cache_len {
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
            for pos in window_start..cache_len {
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

        // Output projection: phi names it `dense`, the llama-shaped families
        // `o_proj`. The bias is family-optional (qwen2 carries one on o_proj,
        // llama / gemma do not) — an absent unit skips the add.
        self.scratch.attn_projected.clear();
        self.scratch.attn_projected.resize(cfg.hidden_size, 0.0);
        let output_unit = if cfg.features.fused_qkv {
            format!("{prefix}self_attn.dense")
        } else {
            format!("{prefix}self_attn.o_proj")
        };
        let (o_weight, o_dtype, _) = self.store.slice(&format!("{output_unit}.weight"))?;
        let (o_bias, o_bias_dtype, _) = self
            .store
            .slice(&format!("{output_unit}.bias"))
            .unwrap_or((&[], o_dtype, &[]));
        kernels::gemv(
            o_weight,
            o_dtype,
            cfg.hidden_size,
            cfg.q_heads * cfg.head_dim,
            &self.scratch.attn_out,
            &mut self.scratch.attn_projected,
        );
        add_bias(&mut self.scratch.attn_projected, o_bias, o_bias_dtype);
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
        Ok(())
    }

    /// Feed-forward dispatch over `scratch.normed`, written to
    /// `scratch.mlp_out` — the caller owns the residual add, so phi's parallel
    /// path can sum attention and MLP outputs in one update.
    fn forward_mlp(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        // Dense vs MoE branch: detected by expert counts from config.
        // This runtime check is negligible overhead and enables clean code paths
        // for both architectures without compile-time specialization.
        if cfg.num_local_experts > 0 && cfg.num_experts_per_tok > 0 {
            self.forward_moe_block(prefix, layer)
        } else {
            self.forward_dense_mlp(prefix)
        }
    }

    /// Forward pass for MoE blocks: router selection + top-k expert computation,
    /// each selected expert's output accumulated into `scratch.mlp_out`,
    /// weighted by its route weight.
    fn forward_moe_block(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        let cfg = &self.cfg;

        // Router: compute expert scores from normalized hidden states
        self.scratch.router.clear();
        self.scratch.router.resize(cfg.num_local_experts, 0.0);
        let (router_weight, router_dtype, _) =
            self.store.slice(&cfg.moe_layout.router_unit(prefix))?;
        kernels::gemv(
            router_weight,
            router_dtype,
            cfg.num_local_experts,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.router,
        );

        // Router selection: softmax over the router logits, top-k by score,
        // selected weights renormalized to one — the shared routing rule every
        // backend uses (`chat::moe::select_experts`).
        let num_experts_per_tok = cfg.num_experts_per_tok.min(cfg.num_local_experts);
        let selected = crate::chat::moe::select_experts(&self.scratch.router, num_experts_per_tok)?;
        let picked: Vec<usize> = selected.iter().map(|(expert, _)| *expert).collect();

        // RC2-B: routing-aware per-expert residency management. Evict unused experts,
        // page in newly selected ones. State keyed by (layer, expert) persists across positions.
        for expert in 0..cfg.num_local_experts {
            let key = (layer, expert);
            let was_resident = self.expert_residency.get(&key).copied().unwrap_or(false);
            if was_resident && !picked.contains(&expert) {
                self.set_expert_resident(layer, expert, false);
                self.expert_residency.insert(key, false);
            }
        }
        for &expert in &picked {
            let key = (layer, expert);
            if !self.expert_residency.get(&key).copied().unwrap_or(false) {
                self.set_expert_resident(layer, expert, true);
                self.expert_residency.insert(key, true);
            }
        }

        // Compute selected experts: gate*up activation, then down projection,
        // scaled by the router's mixing weight for this token. The weighted
        // accumulator starts at zero every call (it swaps into `mlp_out` at
        // the end).
        self.scratch.mlp_out_weighted.clear();
        self.scratch
            .mlp_out_weighted
            .resize(cfg.hidden_size, 0.0);
        for (expert, route_weight) in selected {
            let gate_unit = cfg
                .moe_layout
                .expert_unit(prefix, expert, cfg.moe_layout.gate);
            let up_unit = cfg
                .moe_layout
                .expert_unit(prefix, expert, cfg.moe_layout.up);
            let down_unit = cfg
                .moe_layout
                .expert_unit(prefix, expert, cfg.moe_layout.down);

            let (gate_weight, gate_dtype, _) = self.store.slice(&gate_unit)?;
            let (up_weight, up_dtype, _) = self.store.slice(&up_unit)?;
            let (down_weight, down_dtype, _) = self.store.slice(&down_unit)?;

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
        std::mem::swap(&mut self.scratch.mlp_out, &mut self.scratch.mlp_out_weighted);
        Ok(())
    }

    /// Dense feed-forward over `scratch.normed`, written to `scratch.mlp_out`.
    /// Gated (llama-shaped baseline / gemma): `down(act(gate·x) ⊙ up·x)` with
    /// `act` = SiLU or the tanh GELU. Plain (phi): `fc2(gelu(fc1·x + b)) + b`.
    fn forward_dense_mlp(&mut self, prefix: &str) -> Result<(), String> {
        let cfg = &self.cfg;

        if cfg.features.fc_mlp {
            // phi plain MLP: fc1 (hidden -> intermediate) with bias, tanh
            // GELU, fc2 (intermediate -> hidden) with bias.
            let (fc1_weight, fc1_dtype, _) =
                self.store.slice(&format!("{prefix}mlp.fc1.weight"))?;
            let (fc1_bias, fc1_bias_dtype, _) = self
                .store
                .slice(&format!("{prefix}mlp.fc1.bias"))
                .unwrap_or((&[], fc1_dtype, &[]));
            self.scratch.gate.clear();
            self.scratch.gate.resize(cfg.intermediate_size, 0.0);
            kernels::gemv(
                fc1_weight,
                fc1_dtype,
                cfg.intermediate_size,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.gate,
            );
            add_bias(&mut self.scratch.gate, fc1_bias, fc1_bias_dtype);
            kernels::gelu(&mut self.scratch.gate);

            let (fc2_weight, fc2_dtype, _) =
                self.store.slice(&format!("{prefix}mlp.fc2.weight"))?;
            let (fc2_bias, fc2_bias_dtype, _) = self
                .store
                .slice(&format!("{prefix}mlp.fc2.bias"))
                .unwrap_or((&[], fc2_dtype, &[]));
            self.scratch.mlp_out.clear();
            self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
            kernels::gemv(
                fc2_weight,
                fc2_dtype,
                cfg.hidden_size,
                cfg.intermediate_size,
                &self.scratch.gate,
                &mut self.scratch.mlp_out,
            );
            add_bias(&mut self.scratch.mlp_out, fc2_bias, fc2_bias_dtype);
            return Ok(());
        }

        // Gate projection
        let (gate_weight, gate_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.gate_proj.weight"))?;
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

        // Up projection
        let (up_weight, up_dtype, _) = self.store.slice(&format!("{prefix}mlp.up_proj.weight"))?;
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

        // SwiGLU (baseline) or GeGLU (gemma) activation: act(gate) ⊙ up
        if cfg.features.mlp_gelu {
            kernels::gelu_multiply(&mut self.scratch.gate, &self.scratch.up);
        } else {
            kernels::silu_multiply(&mut self.scratch.gate, &self.scratch.up);
        }

        // Down projection
        let (down_weight, down_dtype, _) =
            self.store.slice(&format!("{prefix}mlp.down_proj.weight"))?;
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
        let layout = &self.cfg.moe_layout;
        let gate_unit = layout.expert_unit(&prefix, expert, layout.gate);
        let up_unit = layout.expert_unit(&prefix, expert, layout.up);
        let down_unit = layout.expert_unit(&prefix, expert, layout.down);
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

#[cfg(test)]
mod forward_tests {
    use super::*;
    use crate::ant::CHEETAH_UNITS_FILENAME;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Feed-forward flavor of the fixture model.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FeedForward {
        Dense,
        /// Sparse block feed-forward: router + per-expert gate/up/down, no `mlp.*`.
        SparseMoE,
    }

    /// Deterministic value pattern (no RNG): every tensor — and every token
    /// embedding row — differs, so attention scores actually vary between
    /// positions. Constant fixtures make every position look identical, which
    /// would hide both a masking bug and a routing bug.
    fn pattern(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|index| (index as f32 * 1.7 + seed).sin() * 0.35)
            .collect()
    }

    struct FixtureUnit {
        unit_id: &'static str,
        shape: Vec<usize>,
    }

    impl FixtureUnit {
        fn new(unit_id: &'static str, shape: Vec<usize>) -> Self {
            Self { unit_id, shape }
        }
    }

    fn units(feed_forward: FeedForward) -> Vec<FixtureUnit> {
        let mut units = vec![
            FixtureUnit::new("model.embed_tokens.weight", vec![32, 8]),
            FixtureUnit::new("model.norm.weight", vec![8]),
            FixtureUnit::new("model.layers.0.input_layernorm.weight", vec![8]),
            FixtureUnit::new("model.layers.0.post_attention_layernorm.weight", vec![8]),
            FixtureUnit::new("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            FixtureUnit::new("model.layers.0.self_attn.k_proj.weight", vec![8, 8]),
            FixtureUnit::new("model.layers.0.self_attn.v_proj.weight", vec![8, 8]),
            FixtureUnit::new("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
        ];
        match feed_forward {
            FeedForward::Dense => {
                units.push(FixtureUnit::new(
                    "model.layers.0.mlp.gate_proj.weight",
                    vec![12, 8],
                ));
                units.push(FixtureUnit::new(
                    "model.layers.0.mlp.up_proj.weight",
                    vec![12, 8],
                ));
                units.push(FixtureUnit::new(
                    "model.layers.0.mlp.down_proj.weight",
                    vec![8, 12],
                ));
            }
            FeedForward::SparseMoE => {
                units.push(FixtureUnit::new(
                    "model.layers.0.block_sparse_moe.gate.weight",
                    vec![2, 8],
                ));
                for expert in 0..2 {
                    units.push(FixtureUnit::new(MOE_EXPERT_UNITS[expert].0, vec![12, 8]));
                    units.push(FixtureUnit::new(MOE_EXPERT_UNITS[expert].1, vec![12, 8]));
                    units.push(FixtureUnit::new(MOE_EXPERT_UNITS[expert].2, vec![8, 12]));
                }
            }
        }
        units
    }

    /// `(w1, w3, w2)` unit ids for the two fixture experts.
    const MOE_EXPERT_UNITS: [(&str, &str, &str); 2] = [
        (
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        ),
        (
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w3.weight",
            "model.layers.0.block_sparse_moe.experts.1.w2.weight",
        ),
    ];

    fn tensors(feed_forward: FeedForward, patterned: bool) -> Vec<(String, Vec<f32>, Vec<usize>)> {
        let values = |len: usize, seed: f32, constant: f32| {
            if patterned {
                pattern(len, seed)
            } else {
                vec![constant; len]
            }
        };
        let mut tensors: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        tensors.push((
            "model.embed_tokens.weight".into(),
            values(256, 0.5, 0.1),
            vec![32, 8],
        ));
        tensors.push(("model.norm.weight".into(), vec![1.0; 8], vec![8]));
        tensors.push((
            "model.layers.0.input_layernorm.weight".into(),
            vec![1.0; 8],
            vec![8],
        ));
        tensors.push((
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0; 8],
            vec![8],
        ));
        for (index, name) in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ]
        .into_iter()
        .enumerate()
        {
            tensors.push((name.into(), values(64, 0.7 + index as f32, 0.2), vec![8, 8]));
        }
        match feed_forward {
            FeedForward::Dense => {
                for name in [
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.0.mlp.up_proj.weight",
                ] {
                    tensors.push((name.into(), values(96, 1.3, 0.2), vec![12, 8]));
                }
                tensors.push((
                    "model.layers.0.mlp.down_proj.weight".into(),
                    values(96, 1.9, 0.2),
                    vec![8, 12],
                ));
            }
            FeedForward::SparseMoE => {
                tensors.push((
                    "model.layers.0.block_sparse_moe.gate.weight".into(),
                    values(16, 2.5, 0.4),
                    vec![2, 8],
                ));
                for (expert, names) in MOE_EXPERT_UNITS.into_iter().enumerate() {
                    let specs: [(&'static str, [usize; 2], f32); 3] = [
                        (names.0, [12, 8], 3.1),
                        (names.1, [12, 8], 3.7),
                        (names.2, [8, 12], 4.3),
                    ];
                    for (projection, (name, shape, seed)) in specs.into_iter().enumerate() {
                        let len = shape[0] * shape[1];
                        tensors.push((
                            name.into(),
                            values(len, seed + expert as f32 + projection as f32, 0.3),
                            shape.to_vec(),
                        ));
                    }
                }
            }
        }
        tensors
    }

    /// Prepared-directory fixture: aligned safetensors shard + units manifest.
    /// Dropping it removes the directory (including on a failed assertion).
    struct Fixture {
        dir: std::path::PathBuf,
        store: Arc<MmapWeightStore>,
        unit_ids: Vec<&'static str>,
    }

    impl Fixture {
        fn new(name: &str, feed_forward: FeedForward, patterned: bool) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "cheetah_{name}_{}_{}",
                std::process::id(),
                nonce
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let fixture_units = units(feed_forward);
            let shard_name = write_shard(&dir, tensors(feed_forward, patterned));
            write_manifest(&dir, &shard_name, &fixture_units);
            let store = Arc::new(MmapWeightStore::open(&dir).unwrap());
            Self {
                unit_ids: fixture_units.iter().map(|unit| unit.unit_id).collect(),
                dir,
                store,
            }
        }

        /// phi-shaped fixture: the Dense builder's llama naming would not
        /// resolve a fused-projection config, so phi builds its own tensor set.
        fn phi(name: &str, patterned: bool) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "cheetah_{name}_{}_{}",
                std::process::id(),
                nonce
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let fixture_units = phi_units();
            let shard_name = write_shard(&dir, phi_tensors(patterned));
            write_manifest(&dir, &shard_name, &fixture_units);
            let store = Arc::new(MmapWeightStore::open(&dir).unwrap());
            Self {
                unit_ids: fixture_units.iter().map(|unit| unit.unit_id).collect(),
                dir,
                store,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn write_shard(dir: &std::path::Path, tensors: Vec<(String, Vec<f32>, Vec<usize>)>) -> String {
        let views: Vec<(String, safetensors::tensor::TensorView<'static>)> = tensors
            .into_iter()
            .map(|(name, data, shape)| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for value in &data {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
                (
                    name,
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, leaked)
                        .unwrap(),
                )
            })
            .collect();
        let shard_dir = dir.join(".cheetah/weights");
        std::fs::create_dir_all(&shard_dir).unwrap();
        let shard_name = "model-00001-of-00002.safetensors";
        std::fs::write(
            shard_dir.join(shard_name),
            safetensors::serialize(views, None).unwrap(),
        )
        .unwrap();
        shard_name.to_string()
    }

    fn write_manifest(dir: &std::path::Path, shard_name: &str, units: &[FixtureUnit]) {
        let units_json: Vec<serde_json::Value> = units
            .iter()
            .map(|unit| {
                let bytes = unit.shape.iter().product::<usize>() * 4;
                serde_json::json!({
                    "unit_id": unit.unit_id,
                    "name": "layers_0",
                    "dtype": "F32",
                    "shape": unit.shape,
                    "bytes": bytes,
                    "source_file": shard_name,
                    "source_offset": 0,
                    "shard": shard_name,
                    "layer_id": 0,
                })
            })
            .collect();
        std::fs::write(
            dir.join(CHEETAH_UNITS_FILENAME),
            serde_json::to_string(&units_json).unwrap(),
        )
        .unwrap();
    }

    fn dense_config(sliding_window: usize) -> DecoderConfig {
        DecoderConfig {
            hidden_size: 8,
            num_layers: 1,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
            rotary_dim: 4,
            intermediate_size: 12,
            vocab_size: 32,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            sliding_window,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            moe_layout: MoEExpertLayout::MIXTRAL,
            features: DenseFeatures::default(),
        }
    }

    fn moe_config(num_experts_per_tok: usize) -> DecoderConfig {
        DecoderConfig {
            sliding_window: 0,
            num_local_experts: 2,
            num_experts_per_tok,
            ..dense_config(0)
        }
    }

    fn logits_of(cfg: DecoderConfig, store: &Arc<MmapWeightStore>, tokens: &[u32]) -> Vec<f32> {
        let mut decoder = DecoderNative::with_store(cfg, Arc::clone(store));
        decoder.forward(tokens, 0).unwrap()
    }

    /// RC3 dense completion: the dense forward path must be parity-clean across
    /// residency transitions — resident (fresh) -> evict to disk -> page back in
    /// via the Ant tier control — and produce finite, deterministic logits.
    #[test]
    fn dense_forward_is_parity_clean_across_residency_transitions() {
        let fixture = Fixture::new("dense_residency", FeedForward::Dense, false);
        let mut decoder = DecoderNative::with_store(dense_config(0), Arc::clone(&fixture.store));
        let tokens: Vec<u32> = vec![3];
        let logits_first = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first.len(), 32);
        assert!(logits_first.iter().all(|value| value.is_finite()));

        for unit in &fixture.unit_ids {
            decoder.evict_unit(unit);
        }
        decoder.clear_kv();
        let logits_evicted = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_evicted);

        for unit in &fixture.unit_ids {
            decoder.page_in_unit(unit);
        }
        decoder.clear_kv();
        let logits_paged_in = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_paged_in);
    }

    /// RC3 sparse/MoE completion: the router-selected experts follow the same
    /// residency contract as the dense path — evict to disk, page back in —
    /// with bit-identical logits before and after.
    #[test]
    fn moe_forward_is_parity_clean_across_residency_transitions() {
        let fixture = Fixture::new("moe_residency_parity", FeedForward::SparseMoE, false);
        let mut decoder = DecoderNative::with_store(moe_config(2), Arc::clone(&fixture.store));
        let tokens: Vec<u32> = vec![3];
        let logits_first = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first.len(), 32);
        assert!(logits_first.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.expert_residency.get(&(0, 0)), Some(&true));
        assert_eq!(decoder.expert_residency.get(&(0, 1)), Some(&true));

        for unit in &fixture.unit_ids {
            decoder.evict_unit(unit);
        }
        decoder.clear_kv();
        assert_eq!(logits_first, decoder.forward(&tokens, 0).unwrap());

        for unit in &fixture.unit_ids {
            decoder.page_in_unit(unit);
        }
        decoder.clear_kv();
        assert_eq!(logits_first, decoder.forward(&tokens, 0).unwrap());
    }

    /// The MoE block must mix the selected experts with the router's
    /// probabilities: `residual + Σ softmax(router_logits)[top-k] * expert(x)`.
    /// The reference below is written straight from the tensor math so a
    /// missing or dropped route weight cannot agree with it.
    #[test]
    fn moe_block_mixes_selected_experts_by_router_weight() {
        let fixture = Fixture::new("moe_block", FeedForward::SparseMoE, true);
        let cfg = moe_config(2);
        let mut decoder = DecoderNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let input: Vec<f32> = pattern(8, 3.25);
        let base: Vec<f32> = pattern(8, -1.5);
        decoder.scratch.normed.clear();
        decoder.scratch.normed.extend_from_slice(&input);
        decoder.scratch.hidden.clear();
        decoder.scratch.hidden.extend_from_slice(&base);
        decoder
            .forward_moe_block("model.layers.0.", 0)
            .expect("moe block forward");

        let layout = &cfg.moe_layout;
        let prefix = "model.layers.0.";
        let (router, router_dtype, _) = fixture
            .store
            .slice(&layout.router_unit(prefix))
            .expect("router weight");
        let mut router_logits = vec![0.0f32; cfg.num_local_experts];
        for expert in 0..cfg.num_local_experts {
            for (column, value) in input.iter().enumerate() {
                router_logits[expert] +=
                    kernels::scalar_at(router, expert * 8 + column, router_dtype) * value;
            }
        }
        let max = router_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = router_logits
            .iter()
            .map(|logit| (logit - max).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|value| value / sum).collect();
        assert!(
            weights.iter().any(|weight| (weight - 1.0).abs() > 0.1),
            "fixture router must not be one-hot or uniform; weights {weights:?} cannot prove weighting"
        );

        let mut expected = base.clone();
        let mut expected_unweighted = base.clone();
        for (expert, weight) in weights.iter().enumerate() {
            let gate_bytes = fixture
                .store
                .slice(&layout.expert_unit(prefix, expert, layout.gate))
                .unwrap()
                .0;
            let up_bytes = fixture
                .store
                .slice(&layout.expert_unit(prefix, expert, layout.up))
                .unwrap()
                .0;
            let down_bytes = fixture
                .store
                .slice(&layout.expert_unit(prefix, expert, layout.down))
                .unwrap()
                .0;
            let mut gate = vec![0.0f32; 12];
            let mut up = vec![0.0f32; 12];
            for row in 0..12 {
                for (column, value) in input.iter().enumerate() {
                    gate[row] += f32::from_le_bytes(
                        gate_bytes[(row * 8 + column) * 4..][..4]
                            .try_into()
                            .unwrap(),
                    ) * value;
                    up[row] += f32::from_le_bytes(
                        up_bytes[(row * 8 + column) * 4..][..4].try_into().unwrap(),
                    ) * value;
                }
            }
            let activated: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(gate, up)| gate * (up / (1.0 + (-gate).exp())))
                .collect();
            let mut expert_output = vec![0.0f32; 8];
            for row in 0..8 {
                for (column, value) in activated.iter().enumerate() {
                    expert_output[row] += f32::from_le_bytes(
                        down_bytes[(row * 12 + column) * 4..][..4]
                            .try_into()
                            .unwrap(),
                    ) * value;
                }
            }
            for (value, (weighted, plain)) in expert_output
                .iter()
                .zip(expected.iter_mut().zip(expected_unweighted.iter_mut()))
            {
                *weighted += value * weight;
                *plain += value;
            }
        }

        // The block contract: the weighted expert mix lands in
        // `scratch.mlp_out`; the residual stream (`scratch.hidden`) is left
        // untouched for the caller to update.
        let mix = expected
            .iter()
            .zip(base.iter())
            .map(|(value, residual)| value - residual)
            .collect::<Vec<_>>();
        for (index, (actual, reference)) in decoder
            .scratch
            .mlp_out
            .iter()
            .zip(mix.iter())
            .enumerate()
        {
            assert!(
                (actual - reference).abs() < 1e-5,
                "expert mix mismatch at {index}: {actual} vs {reference}"
            );
        }
        let contribution = mix
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        assert!(
            contribution > 1e-6,
            "fixture experts must actually contribute to the residual"
        );
        let unweighted_delta = mix
            .iter()
            .zip(
                expected_unweighted
                    .iter()
                    .zip(base.iter())
                    .map(|(value, residual)| value - residual)
                    .collect::<Vec<_>>()
                    .iter(),
            )
            .map(|(actual, reference)| (actual - reference).abs())
            .fold(0.0f32, f32::max);
        assert!(
            unweighted_delta > contribution * 0.1,
            "route weights were not applied: delta {unweighted_delta} against contribution {contribution}"
        );
    }

    /// RC2-B contract on the native backend: whoever the router picks becomes
    /// resident, and the expert it moved away from is evicted to disk.
    #[test]
    fn moe_router_drives_per_expert_residency_transitions() {
        let fixture = Fixture::new("moe_router_residency", FeedForward::SparseMoE, true);
        let mut decoder = DecoderNative::with_store(moe_config(1), Arc::clone(&fixture.store));
        let mut token_for_expert: [Option<u32>; 2] = [None, None];
        for token in 0..fixture.unit_ids.len() as u32 {
            decoder.clear_kv();
            decoder.forward(&[token], 0).unwrap();
            for expert in 0..2 {
                if decoder.expert_residency.get(&(0, expert)) == Some(&true) {
                    token_for_expert[expert] = Some(token);
                }
            }
            if token_for_expert.iter().all(Option::is_some) {
                break;
            }
        }
        let token_expert0 = token_for_expert[0].expect("fixture router never selected expert 0");
        let token_expert1 = token_for_expert[1].expect("fixture router never selected expert 1");

        let layout = &decoder.cfg.moe_layout;
        let expert0 = layout.expert_unit("model.layers.0.", 0, layout.gate);
        let expert1 = layout.expert_unit("model.layers.0.", 1, layout.gate);

        decoder.clear_kv();
        decoder.forward(&[token_expert0], 0).unwrap();
        assert_eq!(decoder.expert_residency.get(&(0, 0)), Some(&true));
        assert_eq!(fixture.store.tier(&expert0), Tier::Resident);

        decoder.clear_kv();
        decoder.forward(&[token_expert1], 0).unwrap();
        assert_eq!(
            decoder.expert_residency.get(&(0, 0)),
            Some(&false),
            "expert 0 must be evicted once the router moves away from it"
        );
        assert_eq!(decoder.expert_residency.get(&(0, 1)), Some(&true));
        assert_eq!(fixture.store.tier(&expert0), Tier::Disk);
        assert_eq!(fixture.store.tier(&expert1), Tier::Resident);

        // The evicted expert still computes: pages re-fault on read.
        decoder.clear_kv();
        let logits = decoder.forward(&[token_expert0], 0).unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.expert_residency.get(&(0, 0)), Some(&true));
    }

    /// Sparse attention: a query only attends inside its window. At position 4
    /// with window 4 the key at position 0 (distance 4) is masked out of the
    /// softmax entirely; with the window disabled every key contributes.
    #[test]
    fn sparse_attention_masks_positions_outside_the_window() {
        let fixture = Fixture::new("sparse_window", FeedForward::Dense, true);
        let tokens: Vec<u32> = (0..5).collect();

        let mut windowed = DecoderNative::with_store(dense_config(4), Arc::clone(&fixture.store));
        let capture = windowed.debug_attention_out(&tokens).unwrap();
        assert_eq!(capture.scores.len(), tokens.len());
        assert_eq!(capture.scores[0], 0.0, "position 0 sits outside the window");
        assert!(capture.scores[1..].iter().all(|score| *score > 0.0));
        let total: f32 = capture.scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);

        let mut full = DecoderNative::with_store(dense_config(0), fixture.store.clone());
        let capture = full.debug_attention_out(&tokens).unwrap();
        assert!(capture.scores[0] > 0.0, "no window means full attention");
    }

    /// A window at least as wide as the sequence is full attention; a narrower
    /// window changes the emitted logits.
    #[test]
    fn sparse_window_changes_the_logits_only_when_it_cuts_the_cache() {
        let fixture = Fixture::new("sparse_logits", FeedForward::Dense, true);
        let tokens: Vec<u32> = (0..8).collect();
        let full = logits_of(dense_config(0), &fixture.store, &tokens);
        let wide = logits_of(dense_config(64), &fixture.store, &tokens);
        let narrow = logits_of(dense_config(3), &fixture.store, &tokens);
        assert!(full.iter().all(|value| value.is_finite()));
        assert_eq!(full, wide, "a window beyond the sequence must not mask");
        assert_ne!(full, narrow, "a 3-wide window must mask older keys");
    }

    /// Parameter chain for the sparse/MoE fields: config first, then the units
    /// manifest, and a sparse manifest without `num_experts_per_tok` fails
    /// loudly instead of silently taking the dense feed-forward path.
    #[test]
    fn decoder_config_resolves_sparse_window_and_moe_expert_fields() {
        let fixture = Fixture::new("resolve_moe", FeedForward::SparseMoE, false);
        let base = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "sliding_window": 4096,
            "use_sliding_window": false
        });
        let cfg = DecoderConfig::resolve(Some(&base), &fixture.store, None).unwrap();
        assert_eq!(cfg.num_local_experts, 2);
        assert_eq!(cfg.num_experts_per_tok, 1);
        assert_eq!(cfg.intermediate_size, 12);
        assert_eq!(cfg.head_dim, 4);
        assert_eq!(
            cfg.sliding_window, 0,
            "use_sliding_window: false disables the sparse window"
        );

        let mut enabled = base.clone();
        enabled["use_sliding_window"] = serde_json::json!(true);
        enabled["sliding_window"] = serde_json::json!(4);
        let cfg = DecoderConfig::resolve(Some(&enabled), &fixture.store, None).unwrap();
        assert_eq!(cfg.sliding_window, 4);

        // Top-k is config-only, so a sparse manifest without it must error.
        let mut routed = base.clone();
        routed
            .as_object_mut()
            .unwrap()
            .remove("num_experts_per_tok");
        let error = DecoderConfig::resolve(Some(&routed), &fixture.store, None)
            .err()
            .expect("sparse manifest without top-k must not resolve");
        assert!(error.contains("num_experts_per_tok"), "{error}");

        // Without any expert fields the layer-wide count comes from the units
        // manifest (and the intermediate size from an expert's gate rows).
        let mut derived = base.clone();
        derived.as_object_mut().unwrap().remove("num_local_experts");
        derived.as_object_mut().unwrap().remove("intermediate_size");
        let cfg = DecoderConfig::resolve(Some(&derived), &fixture.store, None).unwrap();
        assert_eq!(cfg.num_local_experts, 2);
        assert_eq!(cfg.intermediate_size, 12);
    }

    // --- dense delta families (mistral / gemma / phi) ---

    /// phi-shaped units: fused `qkv_proj` + `dense` attention, `fc1`/`fc2`
    /// MLP, biased LayerNorms, GPT-NeoX `final_layernorm`, untied lm_head.
    fn phi_units() -> Vec<FixtureUnit> {
        vec![
            FixtureUnit::new("model.embed_tokens.weight", vec![32, 8]),
            FixtureUnit::new("model.final_layernorm.weight", vec![8]),
            FixtureUnit::new("model.final_layernorm.bias", vec![8]),
            FixtureUnit::new("model.layers.0.input_layernorm.weight", vec![8]),
            FixtureUnit::new("model.layers.0.input_layernorm.bias", vec![8]),
            FixtureUnit::new("model.layers.0.self_attn.qkv_proj.weight", vec![24, 8]),
            FixtureUnit::new("model.layers.0.self_attn.qkv_proj.bias", vec![24]),
            FixtureUnit::new("model.layers.0.self_attn.dense.weight", vec![8, 8]),
            FixtureUnit::new("model.layers.0.self_attn.dense.bias", vec![8]),
            FixtureUnit::new("model.layers.0.mlp.fc1.weight", vec![12, 8]),
            FixtureUnit::new("model.layers.0.mlp.fc1.bias", vec![12]),
            FixtureUnit::new("model.layers.0.mlp.fc2.weight", vec![8, 12]),
            FixtureUnit::new("model.layers.0.mlp.fc2.bias", vec![8]),
            FixtureUnit::new("lm_head.weight", vec![32, 8]),
        ]
    }

    /// Tensor values matching [`phi_units`]; biases are nonzero so a dropped
    /// bias add cannot pass the reference comparison.
    fn phi_tensors(patterned: bool) -> Vec<(String, Vec<f32>, Vec<usize>)> {
        let values = |len: usize, seed: f32, constant: f32| {
            if patterned {
                pattern(len, seed)
            } else {
                vec![constant; len]
            }
        };
        vec![
            (
                "model.embed_tokens.weight".into(),
                values(256, 0.5, 0.1),
                vec![32, 8],
            ),
            (
                "model.final_layernorm.weight".into(),
                vec![1.0; 8],
                vec![8],
            ),
            (
                "model.final_layernorm.bias".into(),
                values(8, 6.1, 0.05),
                vec![8],
            ),
            (
                "model.layers.0.input_layernorm.weight".into(),
                vec![1.0; 8],
                vec![8],
            ),
            (
                "model.layers.0.input_layernorm.bias".into(),
                values(8, 6.3, 0.05),
                vec![8],
            ),
            (
                "model.layers.0.self_attn.qkv_proj.weight".into(),
                values(192, 7.1, 0.2),
                vec![24, 8],
            ),
            (
                "model.layers.0.self_attn.qkv_proj.bias".into(),
                values(24, 7.7, 0.05),
                vec![24],
            ),
            (
                "model.layers.0.self_attn.dense.weight".into(),
                values(64, 8.1, 0.2),
                vec![8, 8],
            ),
            (
                "model.layers.0.self_attn.dense.bias".into(),
                values(8, 8.5, 0.05),
                vec![8],
            ),
            (
                "model.layers.0.mlp.fc1.weight".into(),
                values(96, 9.1, 0.2),
                vec![12, 8],
            ),
            (
                "model.layers.0.mlp.fc1.bias".into(),
                values(12, 9.5, 0.05),
                vec![12],
            ),
            (
                "model.layers.0.mlp.fc2.weight".into(),
                values(96, 9.9, 0.2),
                vec![8, 12],
            ),
            (
                "model.layers.0.mlp.fc2.bias".into(),
                values(8, 10.3, 0.05),
                vec![8],
            ),
            ("lm_head.weight".into(), values(256, 10.7, 0.1), vec![32, 8]),
        ]
    }

    /// gemma reuses the Dense (llama-named) fixture — its deltas live in the
    /// config, not the tensor names.
    fn gemma_config() -> DecoderConfig {
        DecoderConfig {
            features: GEMMA_FEATURES,
            ..dense_config(0)
        }
    }

    fn phi_config() -> DecoderConfig {
        DecoderConfig {
            hidden_size: 8,
            num_layers: 1,
            q_heads: 2,
            kv_heads: 2,
            head_dim: 4,
            rotary_dim: 2, // partial rotary: half the fixture head_dim
            intermediate_size: 12,
            vocab_size: 32,
            rms_eps: 1e-5, // phi's layer_norm_eps
            rope_theta: 10_000.0,
            tie_word_embeddings: false,
            sliding_window: 0,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            moe_layout: MoEExpertLayout::MIXTRAL,
            features: DenseFeatures {
                layer_norm: true,
                parallel_residual: true,
                fused_qkv: true,
                fc_mlp: true,
                mlp_gelu: true,
                partial_rotary_factor: 0.5,
                ..DenseFeatures::default()
            },
        }
    }

    // Reference math helpers: written straight from the tensor math so a
    // dropped delta (offset norm, bias, partial rope, embedding scale)
    // cannot agree with them.

    /// Flatten one fixture unit's F32 bytes into a value vector.
    fn ref_vec(store: &MmapWeightStore, unit: &str, len: usize) -> Vec<f32> {
        let (bytes, _, _) = store.slice(unit).expect(unit);
        (0..len)
            .map(|index| {
                f32::from_le_bytes(bytes[index * 4..][..4].try_into().unwrap())
            })
            .collect()
    }

    /// Reference matvec: y = W · x over a row-major flattened weight.
    fn ref_mat(weight: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|column| weight[row * cols + column])
                    .zip(x.iter())
                    .map(|(w, value)| w * value)
                    .sum()
            })
            .collect()
    }

    /// Reference NeoX rotate-half rope over the leading `dim` dims of one
    /// head slice.
    fn ref_rope(values: &mut [f32], dim: usize, position: usize, theta: f32) {
        let half = dim / 2;
        for i in 0..half {
            let freq = theta.powf(-2.0 * i as f32 / dim as f32);
            let angle = position as f32 * freq;
            // f32::sin_cos returns (sin, cos).
            let (sin, cos) = angle.sin_cos();
            let (a, b) = (values[i], values[half + i]);
            values[i] = a * cos - b * sin;
            values[half + i] = b * cos + a * sin;
        }
    }

    /// Reference tanh-approximation GELU in place.
    fn ref_gelu(values: &mut [f32]) {
        const SQRT_2_OVER_PI: f32 = 0.797_884_56;
        for value in values.iter_mut() {
            let x = *value;
            *value = 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh());
        }
    }

    /// Reference RMSNorm with the gemma `(1 + weight)` offset.
    fn ref_rmsnorm_offset(x: &mut [f32], weight: &[f32], eps: f32) {
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_square + eps).sqrt();
        for (value, w) in x.iter_mut().zip(weight.iter()) {
            *value *= inv * (1.0 + w);
        }
    }

    /// Reference LayerNorm with affine parameters.
    fn ref_layernorm(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32) {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance =
            x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (variance + eps).sqrt();
        for (index, value) in x.iter_mut().enumerate() {
            *value = (*value - mean) * inv * weight[index] + bias[index];
        }
    }

    /// gemma family: offset RMSNorms `x/rms · (1 + w)`, a sqrt(hidden)-scaled
    /// embedding, and GeGLU (tanh-GELU gated) feed-forward, compared against
    /// the full forward written straight from the tensor math — over two
    /// positions, so rope actually rotates on the second token.
    #[test]
    fn gemma_forward_matches_reference_math() {
        let fixture = Fixture::new("gemma_reference", FeedForward::Dense, true);
        let cfg = gemma_config();
        let mut decoder = DecoderNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let tokens: [u32; 2] = [3, 11];
        let logits = decoder.forward(&tokens, 0).unwrap();

        let store = &fixture.store;
        let e = ref_vec(store, "model.embed_tokens.weight", 32 * 8);
        let ln1 = ref_vec(store, "model.layers.0.input_layernorm.weight", 8);
        let ln2 = ref_vec(store, "model.layers.0.post_attention_layernorm.weight", 8);
        let norm_w = ref_vec(store, "model.norm.weight", 8);
        let wq = ref_vec(store, "model.layers.0.self_attn.q_proj.weight", 64);
        let wk = ref_vec(store, "model.layers.0.self_attn.k_proj.weight", 64);
        let wv = ref_vec(store, "model.layers.0.self_attn.v_proj.weight", 64);
        let wo = ref_vec(store, "model.layers.0.self_attn.o_proj.weight", 64);
        let wg = ref_vec(store, "model.layers.0.mlp.gate_proj.weight", 12 * 8);
        let wu = ref_vec(store, "model.layers.0.mlp.up_proj.weight", 12 * 8);
        let wd = ref_vec(store, "model.layers.0.mlp.down_proj.weight", 8 * 12);

        let embed_scale = (cfg.hidden_size as f32).sqrt();
        let mut k_cache: Vec<f32> = Vec::new();
        let mut v_cache: Vec<f32> = Vec::new();
        let mut x: Vec<f32> = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            // gemma scales the embedding output by sqrt(hidden_size).
            x = e[*token as usize * 8..][..8].to_vec();
            for value in x.iter_mut() {
                *value *= embed_scale;
            }
            let mut n1 = x.clone();
            ref_rmsnorm_offset(&mut n1, &ln1, cfg.rms_eps);
            let mut q = ref_mat(&wq, 8, 8, &n1);
            let mut k = ref_mat(&wk, 8, 8, &n1);
            let v = ref_mat(&wv, 8, 8, &n1);
            for head in 0..cfg.q_heads {
                ref_rope(&mut q[head * 4..][..4], 4, position, cfg.rope_theta);
            }
            for head in 0..cfg.kv_heads {
                ref_rope(&mut k[head * 4..][..4], 4, position, cfg.rope_theta);
            }
            k_cache.extend_from_slice(&k);
            v_cache.extend_from_slice(&v);
            let cache_len = position + 1;
            let mut attn_out = vec![0f32; 8];
            for head in 0..cfg.q_heads {
                let mut scores = vec![0f32; cache_len];
                for pos in 0..cache_len {
                    let dot: f32 = q[head * 4..][..4]
                        .iter()
                        .zip(k_cache[pos * 8 + head * 4..][..4].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[pos] = dot * (cfg.head_dim as f32).sqrt().recip();
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                for (pos, score) in scores.iter().enumerate() {
                    let weight = (score - max).exp() / sum;
                    for (index, out) in attn_out[head * 4..][..4].iter_mut().enumerate() {
                        *out += weight * v_cache[pos * 8 + head * 4 + index];
                    }
                }
            }
            let a = ref_mat(&wo, 8, 8, &attn_out);
            for (residual, value) in x.iter_mut().zip(a.iter()) {
                *residual += value;
            }
            let mut n2 = x.clone();
            ref_rmsnorm_offset(&mut n2, &ln2, cfg.rms_eps);
            let mut gate = ref_mat(&wg, 12, 8, &n2);
            let up = ref_mat(&wu, 12, 8, &n2);
            ref_gelu(&mut gate);
            for (gate, up) in gate.iter_mut().zip(up.iter()) {
                *gate *= up;
            }
            let m = ref_mat(&wd, 8, 12, &gate);
            for (residual, value) in x.iter_mut().zip(m.iter()) {
                *residual += value;
            }
        }
        ref_rmsnorm_offset(&mut x, &norm_w, cfg.rms_eps);
        let logits_ref = ref_mat(&e, 32, 8, &x); // tied head reads the raw table

        for (index, (actual, reference)) in logits.iter().zip(logits_ref.iter()).enumerate() {
            assert!(
                (actual - reference).abs() < 1e-4,
                "gemma logits mismatch at {index}: {actual} vs {reference}"
            );
        }
    }

    /// phi family: biased LayerNorms, fused `qkv_proj` + `dense` attention,
    /// partial rotary (rope touches only the leading half of each head), a
    /// plain `fc1`/`fc2` GELU MLP, and the parallel residual
    /// `x + attn_out + mlp_out` — against the full tensor-math reference.
    #[test]
    fn phi_forward_matches_reference_math() {
        let fixture = Fixture::phi("phi_reference", true);
        let cfg = phi_config();
        let mut decoder = DecoderNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let tokens: [u32; 2] = [3, 11];
        let logits = decoder.forward(&tokens, 0).unwrap();

        let store = &fixture.store;
        let e = ref_vec(store, "model.embed_tokens.weight", 32 * 8);
        let ln_w = ref_vec(store, "model.layers.0.input_layernorm.weight", 8);
        let ln_b = ref_vec(store, "model.layers.0.input_layernorm.bias", 8);
        let fn_w = ref_vec(store, "model.final_layernorm.weight", 8);
        let fn_b = ref_vec(store, "model.final_layernorm.bias", 8);
        let wqkv = ref_vec(store, "model.layers.0.self_attn.qkv_proj.weight", 24 * 8);
        let bqkv = ref_vec(store, "model.layers.0.self_attn.qkv_proj.bias", 24);
        let wdense = ref_vec(store, "model.layers.0.self_attn.dense.weight", 64);
        let bdense = ref_vec(store, "model.layers.0.self_attn.dense.bias", 8);
        let wfc1 = ref_vec(store, "model.layers.0.mlp.fc1.weight", 12 * 8);
        let bfc1 = ref_vec(store, "model.layers.0.mlp.fc1.bias", 12);
        let wfc2 = ref_vec(store, "model.layers.0.mlp.fc2.weight", 8 * 12);
        let bfc2 = ref_vec(store, "model.layers.0.mlp.fc2.bias", 8);
        let lmh = ref_vec(store, "lm_head.weight", 32 * 8);

        let mut k_cache: Vec<f32> = Vec::new();
        let mut v_cache: Vec<f32> = Vec::new();
        let mut x: Vec<f32> = Vec::new();
        for (position, token) in tokens.iter().enumerate() {
            x = e[*token as usize * 8..][..8].to_vec(); // no embedding scale
            let mut n = x.clone();
            ref_layernorm(&mut n, &ln_w, &ln_b, cfg.rms_eps);
            let mut qkv = ref_mat(&wqkv, 24, 8, &n);
            for (value, bias) in qkv.iter_mut().zip(bqkv.iter()) {
                *value += bias;
            }
            let mut q = qkv[0..8].to_vec();
            let mut k = qkv[8..16].to_vec();
            let v = qkv[16..24].to_vec();
            // Partial rotary: rope touches only the leading rotary_dim dims.
            for head in 0..cfg.q_heads {
                ref_rope(
                    &mut q[head * 4..][..cfg.rotary_dim],
                    cfg.rotary_dim,
                    position,
                    cfg.rope_theta,
                );
            }
            for head in 0..cfg.kv_heads {
                ref_rope(
                    &mut k[head * 4..][..cfg.rotary_dim],
                    cfg.rotary_dim,
                    position,
                    cfg.rope_theta,
                );
            }
            k_cache.extend_from_slice(&k);
            v_cache.extend_from_slice(&v);
            let cache_len = position + 1;
            let mut attn_out = vec![0f32; 8];
            for head in 0..cfg.q_heads {
                let mut scores = vec![0f32; cache_len];
                for pos in 0..cache_len {
                    let dot: f32 = q[head * 4..][..4]
                        .iter()
                        .zip(k_cache[pos * 8 + head * 4..][..4].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[pos] = dot * (cfg.head_dim as f32).sqrt().recip();
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
                for (pos, score) in scores.iter().enumerate() {
                    let weight = (score - max).exp() / sum;
                    for (index, out) in attn_out[head * 4..][..4].iter_mut().enumerate() {
                        *out += weight * v_cache[pos * 8 + head * 4 + index];
                    }
                }
            }
            let mut a = ref_mat(&wdense, 8, 8, &attn_out);
            for (value, bias) in a.iter_mut().zip(bdense.iter()) {
                *value += bias;
            }
            // Parallel residual: the MLP reads the same normed state.
            let mut hidden_fc1 = ref_mat(&wfc1, 12, 8, &n);
            for (value, bias) in hidden_fc1.iter_mut().zip(bfc1.iter()) {
                *value += bias;
            }
            ref_gelu(&mut hidden_fc1);
            let mut m = ref_mat(&wfc2, 8, 12, &hidden_fc1);
            for (value, bias) in m.iter_mut().zip(bfc2.iter()) {
                *value += bias;
            }
            for ((residual, attn), mlp) in x.iter_mut().zip(a.iter()).zip(m.iter()) {
                *residual += attn + mlp;
            }
        }
        ref_layernorm(&mut x, &fn_w, &fn_b, cfg.rms_eps);
        let logits_ref = ref_mat(&lmh, 32, 8, &x); // untied head

        for (index, (actual, reference)) in logits.iter().zip(logits_ref.iter()).enumerate() {
            assert!(
                (actual - reference).abs() < 1e-4,
                "phi logits mismatch at {index}: {actual} vs {reference}"
            );
        }
    }

    /// Family plumbing: the registered dense families carry their deltas into
    /// the resolved decoder config — the rotary dim rounds from the partial
    /// factor, the activation follows the config's `hidden_act`, and the
    /// fused-projection naming follows the phi family.
    #[test]
    fn decoder_config_resolves_dense_family_features() {
        let phi_fixture = Fixture::phi("phi_resolve", true);
        let family = crate::chat::models::family_for_architecture("phi").unwrap();
        let raw = serde_json::json!({
            "hidden_size": 8,
            "vocab_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "partial_rotary_factor": 0.5,
            "hidden_act": "gelu_new",
        });
        let cfg = DecoderConfig::resolve(Some(&raw), &phi_fixture.store, Some(family)).unwrap();
        assert_eq!(cfg.rotary_dim, 2);
        assert!(cfg.features.fused_qkv);
        assert!(cfg.features.fc_mlp);
        assert!(cfg.features.layer_norm);
        assert!(cfg.features.parallel_residual);
        assert!(cfg.features.mlp_gelu);

        // gemma resolves the offset-norm + embedding-scale deltas with full
        // rope; the config's hidden_act matches the family default.
        let gemma_fixture = Fixture::new("gemma_resolve", FeedForward::Dense, true);
        let family = crate::chat::models::family_for_architecture("gemma").unwrap();
        let raw = serde_json::json!({
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "gelu_pytorch_tanh",
        });
        let cfg = DecoderConfig::resolve(Some(&raw), &gemma_fixture.store, Some(family)).unwrap();
        assert!(cfg.features.norm_weight_offset);
        assert!(cfg.features.embedding_scale);
        assert!(cfg.features.mlp_gelu);
        assert_eq!(cfg.rotary_dim, cfg.head_dim); // full rope

        // mistral stays on the llama-shaped baseline and declares the
        // decoding-time sliding window in its family extras.
        let family = crate::chat::models::family_for_architecture("mistral").unwrap();
        assert_eq!(
            family.features,
            DenseFeatures::default(),
            "mistral must carry the llama-shaped baseline features"
        );
        assert!(family.extra_config_json.contains("sliding_window"));
        let raw = serde_json::json!({
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 4096,
            "use_sliding_window": true,
            "hidden_act": "silu",
        });
        let cfg = DecoderConfig::resolve(Some(&raw), &gemma_fixture.store, Some(family)).unwrap();
        assert_eq!(cfg.sliding_window, 4096);
    }
}
