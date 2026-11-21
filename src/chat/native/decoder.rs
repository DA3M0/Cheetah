//! Decoder configuration resolution for the native backend.
//!
//! The model-specific forward assemblies live in the registration files
//! (`src/chat/registrations/<family>.rs`); the shared math and the
//! inference plumbing they compose live in [`super::ops`] (the OpsToolbox).
//! What remains here is the *resolution* layer every family's loader uses:
//!
//! - [`DecoderConfig`] — the fully resolved hyper-parameters, produced by
//!   [`DecoderConfig::resolve`] through the parameter fallback chain
//!   (config.json → units manifest → registered family constants);
//! - [`DenseFeatures`] — the declarative per-family deltas over the
//!   llama-shaped baseline (norm kind, activation, projection naming,
//!   residual topology, partial rope) that drive the resolution probes;
//! - [`MoEExpertLayout`] — the sparse expert naming layout;
//! - the native forward test suite (below), which drives the registration
//!   files' runtime types against mmap fixtures and reference math.

use super::store::MmapWeightStore;
use serde_json::Value;

/// Fully resolved decoder hyper-parameters for the CPU forward pass.
///
/// A config is produced either by `DecoderConfig::resolve` (config.json plus
/// manifest fallbacks) or built directly by tests. Every field here is already
/// concrete: there are no optional values, so the forward pass never has to
/// re-derive shapes or re-read JSON. Cloning is cheap (only the feature and
/// layout tags are copyable metadata; there are no heap allocations).
#[derive(Clone)]
pub struct DecoderConfig {
    /// Width of the residual stream; also the embedding row length and the
    /// input/output width of every linear projection that touches it.
    pub hidden_size: usize,
    /// Number of transformer blocks. The per-layer KV caches are sized to match.
    pub num_layers: usize,
    /// Number of query attention heads; q projection rows = this * `head_dim`.
    pub q_heads: usize,
    /// Number of key/value heads. With `kv_heads < q_heads` the model is GQA:
    /// each KV head serves `q_heads / kv_heads` consecutive query heads.
    pub kv_heads: usize,
    /// Per-head width. q/k/v row slices are `head_dim` wide, and the attention
    /// scale is `1 / sqrt(head_dim)`.
    pub head_dim: usize,
    /// Hidden width of the feed-forward block: gate/up/fc1 output width and
    /// down/fc2 input width. For sparse models a single expert's gate rows
    /// carry the same value.
    pub intermediate_size: usize,
    /// Vocabulary size: embedding and lm_head row count. Token ids must be
    /// strictly smaller than this.
    pub vocab_size: usize,
    /// Epsilon added to the mean-square/variance denominator of RMSNorm and
    /// LayerNorm to avoid division by zero.
    pub rms_eps: f32,
    /// Base frequency of the rotary embedding (`rope_theta`).
    pub rope_theta: f32,
    /// Whether the lm head reuses the embedding table instead of reading a
    /// separate `lm_head.weight` unit.
    pub tie_word_embeddings: bool,
    /// Sparse (sliding-window) attention: keys older than this many positions
    /// are masked out of the softmax. 0 disables the window (full attention),
    /// which is what an absent `sliding_window` or a config that sets
    /// `use_sliding_window: false` resolves to.
    pub sliding_window: usize,
    /// Number of experts in every sparse feed-forward block. Zero means the
    /// model is dense and the family's dense MLP path runs.
    pub num_local_experts: usize,
    /// Experts routed per token by the sparse router (top-k). Zero means dense;
    /// it is required to be nonzero whenever `num_local_experts` is nonzero,
    /// because tensor shapes cannot carry this value.
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
    /// Resolution metadata: the forward assemblies read their family's
    /// constants directly, so this is consumed by the tests (and any
    /// inspector) rather than by the hot path.
    #[allow(dead_code)]
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
    /// Expert gate projection suffix (Mixtral: `w1`), used to build the first
    /// of the three per-expert unit ids.
    pub gate: &'static str,
    /// Expert up projection suffix (Mixtral: `w3`). Pairing it with `gate`
    /// gives the two-branch gated activation before the down projection.
    pub up: &'static str,
    /// Expert down projection suffix (Mixtral: `w2`), projecting the activated
    /// intermediate back to the hidden width.
    pub down: &'static str,
}

/// Naming helpers and the Mixtral-compatible default layout for sparse blocks.
///
/// The unit ids produced here are the exact strings the units manifest and the
/// Vulkan layout metadata use, including the `.weight` suffix, so the CPU and
/// GPU loaders always agree on which file backs which projection.
impl MoEExpertLayout {
    /// Mixtral's expert naming: `block_sparse_moe.gate` plus
    /// `block_sparse_moe.experts.<i>.{w1,w3,w2}`. This is also the fallback
    /// layout for any family whose prepared config does not override it.
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
    /// Build the router's unit id for a layer prefix such as
    /// `model.layers.3.`. The stored unit id includes the trailing `.weight`
    /// because the manifest names whole tensors, not parameter groups.
    pub fn router_unit(&self, prefix: &str) -> String {
        format!("{prefix}{}.weight", self.router)
    }

    /// One expert projection's unit id for one layer (`projection` is `gate` /
    /// `up` / `down` from this layout).
    ///
    /// This produces e.g. `model.layers.3.block_sparse_moe.experts.1.w3.weight`:
    /// layer prefix, experts collection, numeric expert index, projection
    /// suffix, and the tensor suffix. The index is formatted in decimal with no
    /// padding, matching the manifest naming convention.
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
/// dense family declares its deltas from this baseline through one struct,
/// which the config resolution reads to probe the right unit names.
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

/// Baseline llama-shaped values: every structural delta disabled and a full
/// rotary span, i.e. exactly the `LLAMA_BASELINE` constant written as an impl.
impl Default for DenseFeatures {
    /// Return the llama-shaped baseline feature set.
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

/// The llama-shaped baseline: full rope, RMSNorm without offset, unscaled
/// embeddings, SiLU gated MLP, sequential residuals, split projections. The
/// constant form lets declarative registration files (const fns) reference
/// the default feature set.
pub const LLAMA_BASELINE: DenseFeatures = DenseFeatures {
    norm_weight_offset: false,
    embedding_scale: false,
    mlp_gelu: false,
    layer_norm: false,
    parallel_residual: false,
    fused_qkv: false,
    fc_mlp: false,
    partial_rotary_factor: 1.0,
};

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

/// Config resolution: turn the optional model config.json plus the units
/// manifest into a complete `DecoderConfig`.
impl DecoderConfig {
    /// Resolve the configuration through the parameter fallback chain:
    /// explicit config fields, then derivation from the units manifest's
    /// tensor shapes, then the registered family constants. This keeps the
    /// native backend working when the model directory's config.json is
    /// missing or incomplete.
    ///
    /// The fallbacks are deliberately ordered most-specific first: a declared
    /// `hidden_size` always wins, an embedding shape discovered in the
    /// manifest is next, and family constants (rope theta, epsilon, tied
    /// embeddings, MoE layout, dense features) fill the rest. Resolution fails
    /// loudly when a value cannot be recovered at all, because silently
    /// guessing a shape would corrupt every downstream matvec.
    pub(crate) fn resolve(
        raw: Option<&Value>,
        store: &MmapWeightStore,
        family: Option<crate::chat::registry::ModelFamily>,
    ) -> Result<Self, String> {
        // JSON accessors. Each returns `None` for an absent config object, a
        // missing key, or a value of the wrong JSON type, letting `unwrap_or`
        // chains fall through to manifest derivation. `field_u` narrows to
        // usize, `field_f` to f64 (later cast to f32), `field_b` to bool.
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
        // Unknown architecture: fall back to the llama-shaped family defaults
        // (rope theta 1e6, epsilon 1e-6, untied head, Mixtral expert naming,
        // baseline dense features). This is the same shape as a family whose
        // config lists no extras.
        let family = family.unwrap_or(crate::chat::registry::ModelFamily {
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            profiles: &[],
            extra_config_json: "",
            max_position_embeddings: None,
            moe_layout: MoEExpertLayout::MIXTRAL,
            features: DenseFeatures::default(),
        });

        // The embedding is the one tensor every decoder must have: its shape
        // anchors the derived hidden size (columns) and vocabulary (rows) when
        // config.json omits them.
        let embed_shape = store
            .unit_shape("model.embed_tokens.weight")
            .ok_or_else(|| "Native backend: units manifest has no embedding tensor.".to_string())?;
        if embed_shape.len() != 2 {
            return Err("Native backend: unexpected embedding shape in the units manifest.".into());
        }
        let hidden_size = field_u("hidden_size").unwrap_or(embed_shape[1]);
        let vocab_size = field_u("vocab_size").unwrap_or(embed_shape[0]);
        // Layer count: config first, otherwise one past the highest layer id
        // seen in the manifest (clamped to at least 1 for a degenerate empty
        // manifest).
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
        // q projection row count: the full query width for split families, or
        // the stacked q+k+v width for phi's fused tensor.
        let q_out = store
            .unit_shape(q_unit)
            .ok_or_else(|| format!("Native backend: units manifest has no {q_unit} tensor."))?[0];
        // k projection width. Fused tensors carry it inside `q_out`, so the
        // same row count is used as a stand-in; otherwise the manifest's
        // k_proj shape is authoritative and distinguishes MHA from GQA.
        let k_out = if features.fused_qkv {
            q_out
        } else {
            store
                .unit_shape("model.layers.0.self_attn.k_proj.weight")
                .ok_or_else(|| "Native backend: units manifest has no k_proj tensor.".to_string())?
                [0]
        };
        let moe_layout = family.moe_layout;
        // Probe unit id for expert 0's gate on layer 0; only consulted when the
        // dense MLP units are absent, i.e. for a sparse manifest.
        let expert_gate_unit = moe_layout.expert_unit("model.layers.0.", 0, moe_layout.gate);
        // Intermediate width fallback chain: config, dense gate rows, phi fc1
        // rows, then a sparse expert's gate rows. Zero means nothing matched
        // and resolution aborts below.
        let intermediate_size = field_u("intermediate_size").unwrap_or_else(|| {
            store
                .unit_shape("model.layers.0.mlp.gate_proj.weight")
                .map(|shape| shape[0])
                // phi names its MLP projections fc1/fc2; fc1's rows carry the
                // intermediate size.
                .or_else(|| {
                    store
                        .unit_shape("model.layers.0.mlp.fc1.weight")
                        .map(|shape| shape[0])
                })
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

        // Head layout. When the config declares the query head count, derive
        // head_dim and kv_heads around it; otherwise consult the family's
        // (hidden_size, vocab_size) profiles and fail if none matches.
        let (q_heads, kv_heads, head_dim) = match field_u("num_attention_heads") {
            Some(q_heads) => {
                // head_dim fallback order: explicit field, family profile,
                // then width division. The `max(1)` guards against a malformed
                // zero-head config turning this into a division by zero.
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
                // kv head count fallback: phi mirrors q (MHA); otherwise the
                // k_proj rows split evenly by head_dim (GQA/MHA). If they do
                // not divide evenly the shape is inconsistent, so assume MHA
                // to keep the forward valid rather than inventing a partial
                // head.
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
            .unwrap_or(features.partial_rotary_factor as f64) as f32;
        let rotary_dim = if rotary_factor >= 1.0 {
            head_dim
        } else {
            ((rotary_factor * head_dim as f32).round() as usize).clamp(1, head_dim)
        };

        // Feed-forward activation: the config's `hidden_act` wins over the
        // family delta (`gelu` / `gelu_new` / `gelu_pytorch_tanh` are the tanh
        // approximation; `silu` is the llama-shaped baseline).
        let mlp_gelu = match raw
            .and_then(|value| value.get("hidden_act"))
            .and_then(Value::as_str)
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
/// Count layer-0 experts by scanning the units manifest.
///
/// Only the gate projection is counted (each expert has exactly one), and the
/// expert index is recovered by trimming the shared prefix/suffix and parsing
/// the numeric infix. The result is the highest index plus one, which assumes
/// expert indices are contiguous from zero — true for every prepared
/// Mixtral-shaped model. A manifest with no matching units reports zero,
/// selecting the dense feed-forward path.
fn count_layer_experts(store: &MmapWeightStore, layout: &MoEExpertLayout) -> usize {
    let prefix = format!("model.layers.0.{}.", layout.experts_prefix);
    let suffix = format!(".{}.weight", layout.gate);
    store
        .unit_ids()
        .filter(|unit_id| unit_id.starts_with(&prefix) && unit_id.ends_with(&suffix))
        .filter_map(|unit_id| {
            // Slice out the infix between prefix and suffix: the expert index.
            // Non-numeric infixes (other layout variants) are skipped.
            unit_id[prefix.len()..unit_id.len() - suffix.len()]
                .parse::<usize>()
                .ok()
        })
        .max()
        .map(|highest| highest + 1)
        .unwrap_or(0)
}

#[cfg(test)]
mod forward_tests {
    use super::*;
    use crate::ant::CHEETAH_UNITS_FILENAME;
    use crate::chat::gemma_registration::GemmaNative;
    use crate::chat::llama_registration::LlamaNative;
    use crate::chat::mixtral_registration::MixtralNative;
    use crate::chat::native::ops;
    use crate::chat::native::store::Tier;
    use crate::chat::phi_registration::PhiNative;
    use crate::chat::registry::ModelForward;
    use std::sync::Arc;
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
        let mut decoder = LlamaNative::with_store(cfg, Arc::clone(store));
        decoder.forward(tokens, 0).unwrap()
    }

    /// RC3 dense completion: the dense forward path must be parity-clean across
    /// residency transitions — resident (fresh) -> evict to disk -> page back in
    /// via the Ant tier control — and produce finite, deterministic logits.
    #[test]
    fn dense_forward_is_parity_clean_across_residency_transitions() {
        let fixture = Fixture::new("dense_residency", FeedForward::Dense, false);
        let mut decoder = LlamaNative::with_store(dense_config(0), Arc::clone(&fixture.store));
        let tokens: Vec<u32> = vec![3];
        let logits_first = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first.len(), 32);
        assert!(logits_first.iter().all(|value| value.is_finite()));

        for unit in &fixture.unit_ids {
            decoder.ctx.evict_unit(unit);
        }
        decoder.clear_cache();
        let logits_evicted = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_evicted);

        for unit in &fixture.unit_ids {
            decoder.ctx.page_in_unit(unit);
        }
        decoder.clear_cache();
        let logits_paged_in = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_paged_in);
    }

    /// CHEETAH_QUANT_Q8=1 (the registered q8 selector) + the dense fixture:
    /// every linear unit quantizes at first use through the precision
    /// registry's native operator table and the forward computes from the
    /// quantized buffers — finite, deterministic across a residency cycle,
    /// with the quantized-unit table populated. Norms/biases/embedding never
    /// quantize, so the embedding unit stays absent from the table.
    /// Opt-in via CHEETAH_TEST_NATIVE_Q8=1 (the env mutation is
    /// process-global; same gating as the Vulkan q8 forward test).
    #[test]
    fn q8_quantized_forward_terminates_and_populates_quantized_units() {
        if std::env::var("CHEETAH_TEST_NATIVE_Q8").is_err() {
            eprintln!("skipping native q8 forward test (set CHEETAH_TEST_NATIVE_Q8=1)");
            return;
        }
        // SAFETY: test-scoped process env mutation; single-threaded within
        // this test (same pattern as the Vulkan q8 forward test).
        unsafe { std::env::set_var("CHEETAH_QUANT_Q8", "1") };
        let fixture = Fixture::new("native_q8_forward", FeedForward::Dense, false);
        let mut decoder = LlamaNative::with_store(dense_config(0), Arc::clone(&fixture.store));
        let tokens: Vec<u32> = vec![3];
        let logits_first = decoder.forward(&tokens, 0).unwrap();
        assert!(logits_first.iter().all(|value| value.is_finite()));
        assert!(
            !decoder.ctx.quant.buffers.is_empty(),
            "linear units must quantize at first use"
        );
        assert!(
            !decoder
                .ctx
                .quant
                .buffers
                .contains_key("model.embed_tokens.weight"),
            "the embedding stays in its stored precision"
        );
        // Deterministic across evict -> page-in: the quantized buffers are
        // owned; tier transitions only touch the source pages.
        for unit in &fixture.unit_ids {
            decoder.ctx.evict_unit(unit);
        }
        decoder.clear_cache();
        let logits_evicted = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_evicted);
        for unit in &fixture.unit_ids {
            decoder.ctx.page_in_unit(unit);
        }
        decoder.clear_cache();
        let logits_paged_in = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first, logits_paged_in);
        // SAFETY: restoring the process env after the opt-in test.
        unsafe { std::env::remove_var("CHEETAH_QUANT_Q8") };
    }

    /// RC3 sparse/MoE completion: the router-selected experts follow the same
    /// residency contract as the dense path — evict to disk, page back in —
    /// with bit-identical logits before and after.
    #[test]
    fn moe_forward_is_parity_clean_across_residency_transitions() {
        let fixture = Fixture::new("moe_residency_parity", FeedForward::SparseMoE, false);
        let mut decoder = MixtralNative::with_store(moe_config(2), Arc::clone(&fixture.store));
        let tokens: Vec<u32> = vec![3];
        let logits_first = decoder.forward(&tokens, 0).unwrap();
        assert_eq!(logits_first.len(), 32);
        assert!(logits_first.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.ctx.expert_residency.get(&(0, 0)), Some(&true));
        assert_eq!(decoder.ctx.expert_residency.get(&(0, 1)), Some(&true));

        for unit in &fixture.unit_ids {
            decoder.ctx.evict_unit(unit);
        }
        decoder.clear_cache();
        assert_eq!(logits_first, decoder.forward(&tokens, 0).unwrap());

        for unit in &fixture.unit_ids {
            decoder.ctx.page_in_unit(unit);
        }
        decoder.clear_cache();
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
        let mut decoder = MixtralNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
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
                    ops::scalar_at(router, expert * 8 + column, router_dtype) * value;
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
        for (index, (actual, reference)) in
            decoder.scratch.mlp_out.iter().zip(mix.iter()).enumerate()
        {
            assert!(
                (actual - reference).abs() < 1e-5,
                "expert mix mismatch at {index}: {actual} vs {reference}"
            );
        }
        let contribution = mix.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
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
        let mut decoder = MixtralNative::with_store(moe_config(1), Arc::clone(&fixture.store));
        let mut token_for_expert: [Option<u32>; 2] = [None, None];
        for token in 0..fixture.unit_ids.len() as u32 {
            decoder.clear_cache();
            decoder.forward(&[token], 0).unwrap();
            for expert in 0..2 {
                if decoder.ctx.expert_residency.get(&(0, expert)) == Some(&true) {
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

        decoder.clear_cache();
        decoder.forward(&[token_expert0], 0).unwrap();
        assert_eq!(decoder.ctx.expert_residency.get(&(0, 0)), Some(&true));
        assert_eq!(fixture.store.tier(&expert0), Tier::Resident);

        decoder.clear_cache();
        decoder.forward(&[token_expert1], 0).unwrap();
        assert_eq!(
            decoder.ctx.expert_residency.get(&(0, 0)),
            Some(&false),
            "expert 0 must be evicted once the router moves away from it"
        );
        assert_eq!(decoder.ctx.expert_residency.get(&(0, 1)), Some(&true));
        assert_eq!(fixture.store.tier(&expert0), Tier::Disk);
        assert_eq!(fixture.store.tier(&expert1), Tier::Resident);

        // The evicted expert still computes: pages re-fault on read.
        decoder.clear_cache();
        let logits = decoder.forward(&[token_expert0], 0).unwrap();
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(decoder.ctx.expert_residency.get(&(0, 0)), Some(&true));
    }

    /// Sparse attention: a query only attends inside its window. At position 4
    /// with window 4 the key at position 0 (distance 4) is masked out of the
    /// softmax entirely; with the window disabled every key contributes.
    #[test]
    fn sparse_attention_masks_positions_outside_the_window() {
        let fixture = Fixture::new("sparse_window", FeedForward::Dense, true);
        let tokens: Vec<u32> = (0..5).collect();

        let mut windowed = LlamaNative::with_store(dense_config(4), Arc::clone(&fixture.store));
        let capture = windowed.debug_attention_out(&tokens).unwrap();
        assert_eq!(capture.scores.len(), tokens.len());
        assert_eq!(capture.scores[0], 0.0, "position 0 sits outside the window");
        assert!(capture.scores[1..].iter().all(|score| *score > 0.0));
        let total: f32 = capture.scores.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);

        let mut full = LlamaNative::with_store(dense_config(0), fixture.store.clone());
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
            ("model.final_layernorm.weight".into(), vec![1.0; 8], vec![8]),
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
            .map(|index| f32::from_le_bytes(bytes[index * 4..][..4].try_into().unwrap()))
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
        let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
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
        let mut decoder = GemmaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
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
        let mut decoder = PhiNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
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
        let family = crate::chat::registry::family_for_architecture("phi").unwrap();
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
        let family = crate::chat::registry::family_for_architecture("gemma").unwrap();
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
        let family = crate::chat::registry::family_for_architecture("mistral").unwrap();
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

    // --- KV 复用基础性质（任何前缀复用优化的正确性基石） ---

    /// KV 前缀续算 parity：缓存中已有 4 个位置后，以 offset=4 续算 [5,6]，
    /// 最终位置的 logits 必须与"从头全量 prefill [1..6]"的结果**逐比特一致**。
    /// 这是任何 KV 复用/前缀缓存优化所依赖的核心正确性性质——当前引擎的
    /// `forward(tokens, offset)` 机制已满足它。
    #[test]
    fn kv_continuation_matches_full_prefill_bit_exact() {
        let fixture = Fixture::new("kv_continuation", FeedForward::Dense, true);
        let cfg = dense_config(0);

        // 增量路径：先 prefill [1,2,3,4]（KV 持有 4 个位置），再以 offset=4 续算 [5,6]。
        let mut incremental = LlamaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        incremental.forward(&[1, 2, 3, 4], 0).unwrap();
        assert_eq!(incremental.ctx.kv_len(0), 4, "KV 应持有 4 个位置");
        let logits_continued = incremental.forward(&[5, 6], 4).unwrap();

        // 全量路径：全新 decoder 一次 prefill [1..6]。
        let mut fresh = LlamaNative::with_store(cfg, Arc::clone(&fixture.store));
        let logits_full = fresh.forward(&[1, 2, 3, 4, 5, 6], 0).unwrap();

        assert_eq!(
            logits_continued, logits_full,
            "KV 前缀续算必须与全量 prefill 比特一致"
        );
    }

    /// KV 代间隔离：clear_cache 之后，同一 decoder 上跑不同 prompt 的 logits
    /// 必须与全新 decoder 的结果完全一致——上一个生成的 KV 状态绝不泄漏。
    #[test]
    fn kv_cache_does_not_leak_across_generations() {
        let fixture = Fixture::new("kv_isolation", FeedForward::Dense, true);
        let cfg = dense_config(0);

        // 复用路径：先跑一段长 prompt（A），clear 后跑短 prompt（B）。
        let mut reused = LlamaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let logits_a = reused.forward(&[1, 2, 3, 4, 5, 6, 7, 8], 0).unwrap();
        assert_eq!(reused.ctx.kv_len(0), 8);
        reused.clear_cache();
        assert_eq!(reused.ctx.kv_len(0), 0, "clear_cache 必须清空 KV");
        let logits_b_after = reused.forward(&[20, 21], 0).unwrap();

        // 全新基准：各自独立 decoder 跑 A 与 B。
        let mut fresh_a = LlamaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let logits_a_fresh = fresh_a.forward(&[1, 2, 3, 4, 5, 6, 7, 8], 0).unwrap();
        let mut fresh_b = LlamaNative::with_store(cfg, Arc::clone(&fixture.store));
        let logits_b_fresh = fresh_b.forward(&[20, 21], 0).unwrap();

        assert_eq!(logits_a, logits_a_fresh, "A 的 logits 应与全新运行一致");
        assert_eq!(
            logits_b_after, logits_b_fresh,
            "clear_cache 之后 B 的 logits 不得受 A 的 KV 残留影响"
        );
    }

    /// 克隆即新 KV：`clone()` 共享权重（Arc），但 KV 缓存必须从零开始——
    /// 这是引擎每次生成克隆 runtime 时的隔离契约。
    #[test]
    fn clone_shares_weights_but_starts_with_empty_kv() {
        let fixture = Fixture::new("kv_clone", FeedForward::Dense, true);
        let cfg = dense_config(0);
        let mut original = LlamaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        original.forward(&[3, 1, 4, 1, 5], 0).unwrap();
        assert_eq!(original.ctx.kv_len(0), 5);

        let mut cloned = original.clone();
        assert_eq!(cloned.ctx.kv_len(0), 0, "克隆必须从空 KV 开始");
        // 权重共享：克隆（空 KV、位置 0 起）与一台全新 decoder（同一 mmap
        // 权重）跑相同 prompt 的 logits 必须逐比特一致。
        let logits_cloned = cloned.forward(&[2, 7], 0).unwrap();
        let mut standalone = LlamaNative::with_store(cfg.clone(), Arc::clone(&fixture.store));
        let logits_standalone = standalone.forward(&[2, 7], 0).unwrap();
        assert_eq!(
            logits_cloned, logits_standalone,
            "克隆的权重必须与原机共享（同一 mmap 存储）"
        );
        // 原机继续在自己的 KV 上续算（位置 5-6；rope 位置不同 → logits 不同
        // 是正确行为），验证其 KV 连续性不受克隆影响。
        let logits_original = original.forward(&[2, 7], 5).unwrap();
        let mut fresh_full = LlamaNative::with_store(cfg, Arc::clone(&fixture.store));
        let logits_full = fresh_full.forward(&[3, 1, 4, 1, 5, 2, 7], 0).unwrap();
        assert_eq!(
            logits_original, logits_full,
            "原机在位置 5 续算必须等价于从头 prefill 全序列（KV 连续性）"
        );
    }
}
