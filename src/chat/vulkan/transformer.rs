//! The generic transformer executor (llama / qwen2 / qwen3 / mistral /
//! gemma / mixtral): the decoder graph (attention + dense/MoE feed-forward),
//! and the config surface the loader (`super`) builds per model_type. The
//! family-free machinery (expert residency, GPU slot paging, KV cache) lives
//! in `super::context`; weight loading and the operator dispatch in
//! `super::weights` (the precision registry).

use crate::chat::moe::select_experts;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use super::context::{Expert, GpuKvCache, VulkanForwardContext, page_linear};
use super::required_usize;
use super::weights::{
    Linear, NamedWeight, WeightData, load_weights, take_linear, take_weight, validate_shape,
};

#[derive(Clone, Debug)]
pub(crate) struct VulkanTransformerConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    pub(crate) sliding_window: usize,
    activation: Activation,
    pub(crate) features: TransformerFeatures,
}

#[derive(Clone, Debug)]
pub(crate) struct TransformerFeatures {
    pub(crate) model_name: &'static str,
    pub(crate) attention_bias: bool,
    pub(crate) attention_output_bias: bool,
    pub(crate) qk_norm: bool,
    pub(crate) head_dim_override: Option<usize>,
    pub(crate) norm_weight_offset: bool,
    pub(crate) embedding_scale: bool,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) max_position_embeddings_default: Option<usize>,
    pub(crate) rms_norm_eps_default: Option<f32>,
    pub(crate) layout: TransformerWeightLayout,
    pub(crate) feed_forward: TransformerFeedForwardFeatures,
}

impl TransformerFeatures {
    /// The baseline executor feature set for one family name: no attention
    /// biases, no qk-norm, full rope, dense feed-forward, and the standard
    /// llama-shaped tensor layout — families override pieces in their
    /// registration files.
    pub(crate) fn standard(model_name: &'static str) -> Self {
        Self {
            model_name,
            attention_bias: false,
            attention_output_bias: false,
            qk_norm: false,
            head_dim_override: None,
            norm_weight_offset: false,
            embedding_scale: false,
            tie_word_embeddings: false,
            max_position_embeddings_default: None,
            rms_norm_eps_default: None,
            layout: TransformerWeightLayout::default(),
            feed_forward: TransformerFeedForwardFeatures::Dense,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TransformerWeightLayout {
    pub(crate) embedding: &'static str,
    pub(crate) final_norm: &'static str,
    pub(crate) lm_head: &'static str,
    pub(crate) layer_prefix: &'static str,
    pub(crate) input_norm: &'static str,
    pub(crate) post_attention_norm: &'static str,
    pub(crate) attention_prefix: &'static str,
    pub(crate) q_proj: &'static str,
    pub(crate) k_proj: &'static str,
    pub(crate) v_proj: &'static str,
    pub(crate) o_proj: &'static str,
    pub(crate) q_norm: &'static str,
    pub(crate) k_norm: &'static str,
    pub(crate) dense_gate: &'static str,
    pub(crate) dense_up: &'static str,
    pub(crate) dense_down: &'static str,
}

impl Default for TransformerWeightLayout {
    /// The standard llama-shaped tensor naming (embedding, norms, projections,
    /// MLP suffixes) shared by every family that does not override pieces of it.
    fn default() -> Self {
        Self {
            embedding: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers",
            input_norm: "input_layernorm.weight",
            post_attention_norm: "post_attention_layernorm.weight",
            attention_prefix: "self_attn",
            q_proj: "q_proj",
            k_proj: "k_proj",
            v_proj: "v_proj",
            o_proj: "o_proj",
            q_norm: "q_norm",
            k_norm: "k_norm",
            dense_gate: "mlp.gate_proj",
            dense_up: "mlp.up_proj",
            dense_down: "mlp.down_proj",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum TransformerFeedForwardFeatures {
    Dense,
    MixtureOfExperts {
        /// Total experts in the layer (router output width).
        num_local_experts: usize,
        /// Experts routed per token; `0` disables routing (dense).
        num_experts_per_tok: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activation {
    Silu,
    Gelu,
    NewGelu,
}

impl VulkanTransformerConfig {
    /// Config parsing + validation: derives `head_dim` and the kv-head count,
    /// applies family defaults, honors `use_sliding_window`, maps `hidden_act`,
    /// and fails loudly on rope scaling, inconsistent MoE counts, or any
    /// non-positive / non-finite dimension.
    pub(crate) fn from_raw(raw: &Value, features: TransformerFeatures) -> Result<Self, String> {
        let model_name = features.model_name;
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
        let head_dim = if let Some(head_dim) = features.head_dim_override {
            head_dim
        } else {
            raw.get("head_dim")
                .and_then(Value::as_u64)
                .map(usize::try_from)
                .transpose()
                .map_err(|_| "head_dim does not fit usize.".to_string())?
                .unwrap_or_else(|| hidden_size / num_attention_heads.max(1))
        };
        let max_position_embeddings = raw
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "max_position_embeddings does not fit usize.".to_string())?
            .or(features.max_position_embeddings_default)
            .unwrap_or(0);
        let rms_norm_eps = raw
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .or(features.rms_norm_eps_default)
            .unwrap_or(1e-5);
        let rope_theta = raw
            .get("rope_theta")
            .and_then(Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        let sliding_window = raw
            .get("sliding_window")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "sliding_window does not fit usize.".to_string())?
            .unwrap_or(max_position_embeddings);
        // Sparse (sliding-window) attention is gated by `use_sliding_window`
        // (default true, matching the Hugging Face configs). A disabled window
        // resolves to 0, which the attention shader reads as "attend to every
        // cached position" — the same full-attention path a missing
        // `sliding_window` takes above.
        let sliding_window = if raw
            .get("use_sliding_window")
            .and_then(Value::as_bool)
            .unwrap_or(true)
        {
            sliding_window
        } else {
            0
        };
        let activation = match raw.get("hidden_act").and_then(Value::as_str) {
            Some("silu") => Activation::Silu,
            Some("gelu") => Activation::Gelu,
            Some("gelu_new" | "gelu_pytorch_tanh") => Activation::NewGelu,
            _ => {
                return Err(format!(
                    "Vulkan {model_name} does not support the configured activation."
                ));
            }
        };
        // Rope scaling (linear / dynamic / llama3 / ...) is not implemented by
        // the rotary operator: a checkpoint that requires it must fail loudly
        // instead of rotating every position with the base theta.
        if raw
            .get("rope_scaling")
            .is_some_and(|value| !value.is_null())
        {
            return Err(format!(
                "Vulkan {model_name} does not support RoPE scaling."
            ));
        }
        if let TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts,
            num_experts_per_tok,
        } = &features.feed_forward
            && (*num_local_experts == 0
                || *num_experts_per_tok == 0
                || *num_experts_per_tok > *num_local_experts)
        {
            return Err(format!(
                "Vulkan {model_name} MoE expert counts are inconsistent."
            ));
        }
        if vocab_size == 0
            || hidden_size == 0
            || intermediate_size == 0
            || num_hidden_layers == 0
            || num_attention_heads == 0
            || num_key_value_heads == 0
            || head_dim == 0
            || max_position_embeddings == 0
            || head_dim % 2 != 0
            || num_attention_heads % num_key_value_heads != 0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(format!(
                "Vulkan {model_name} model dimensions are inconsistent."
            ));
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            sliding_window,
            activation,
            features,
        })
    }
}

#[derive(Clone)]
enum FeedForward {
    Dense {
        gate: Linear,
        up: Linear,
        down: Linear,
        activation: Activation,
    },
    MixtureOfExperts {
        router: Linear,
        experts: Vec<Expert>,
        num_experts_per_tok: usize,
        activation: Activation,
    },
}

impl FeedForward {
    /// The feed-forward dispatch over `rows` positions: dense layers run
    /// gate/up/activation/down; MoE layers route every row through
    /// `select_experts`, bucket the rows BY EXPERT, run one batched matmul per
    /// picked expert and scatter the route-weighted outputs back — and drive
    /// the GPU-bucket expert residency (evict unpicked, upload picked), keyed
    /// `(layer, expert)` across positions.
    fn forward(
        &mut self,
        input: &[f32],
        rows: usize,
        hidden_size: usize,
        context: &mut VulkanForwardContext,
        layer_index: usize,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Dense {
                gate,
                up,
                down,
                activation,
            } => {
                let gate = gate.forward(input, rows)?;
                let up = up.forward(input, rows)?;
                let activated = apply_activation(*activation, &gate, &up)?;
                down.forward(&activated, rows)
            }
            Self::MixtureOfExperts {
                router,
                experts,
                num_experts_per_tok,
                activation,
            } => {
                let router_logits = router.forward(input, rows)?;
                let mut output = vec![0.0; rows * hidden_size];
                let mut selected_by_expert: Vec<Vec<(usize, f32)>> =
                    vec![Vec::new(); experts.len()];
                for row in 0..rows {
                    let logits = &router_logits[row * experts.len()..(row + 1) * experts.len()];
                    for (expert_index, route_weight) in
                        select_experts(logits, *num_experts_per_tok)?
                    {
                        selected_by_expert[expert_index].push((row, route_weight));
                    }
                }
                // RC2-B: routing-aware per-expert residency, driven through the
                // shared context: experts the current router selection no
                // longer picks are evicted (release the resident GPU bucket);
                // freshly picked experts are paged in (upload into the
                // resident bucket). State is keyed per (layer, expert) so it
                // holds across positions and generation steps.
                let resident: Vec<bool> = selected_by_expert
                    .iter()
                    .map(|selected_rows| !selected_rows.is_empty())
                    .collect();
                context.sync_expert_residency(layer_index, experts, &resident);
                for (expert, selected_rows) in experts.iter().zip(selected_by_expert) {
                    if selected_rows.is_empty() {
                        continue;
                    }
                    let mut expert_input = Vec::with_capacity(selected_rows.len() * hidden_size);
                    for (row, _) in &selected_rows {
                        expert_input
                            .extend_from_slice(&input[row * hidden_size..(row + 1) * hidden_size]);
                    }
                    let gate = expert.gate.forward(&expert_input, selected_rows.len())?;
                    let up = expert.up.forward(&expert_input, selected_rows.len())?;
                    let activated = apply_activation(*activation, &gate, &up)?;
                    let expert_output = expert.down.forward(&activated, selected_rows.len())?;
                    for (selected_index, (row, route_weight)) in
                        selected_rows.into_iter().enumerate()
                    {
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
        }
    }
}

/// Applies the configured activation (silu / gelu / tanh-gelu) to the
/// gate·up pair as one shader call.
fn apply_activation(activation: Activation, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    match activation {
        Activation::Silu => cheetah::native::vulkan::silu_mul(gate, up),
        Activation::Gelu => cheetah::native::vulkan::gelu_mul(gate, up, false),
        Activation::NewGelu => cheetah::native::vulkan::gelu_mul(gate, up, true),
    }
}

#[derive(Clone)]
struct Layer {
    input_norm: WeightData,
    post_attention_norm: WeightData,
    q_norm: Option<WeightData>,
    k_norm: Option<WeightData>,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    feed_forward: FeedForward,
    kv: GpuKvCache,
}

#[derive(Clone)]
pub(crate) struct VulkanTransformerExecutor {
    config: VulkanTransformerConfig,
    embedding: WeightData,
    layers: Vec<Layer>,
    final_norm: WeightData,
    lm_head: WeightData,
    /// Family-free forward machinery: the routing-aware per-expert residency
    /// and the substrate paging primitives it drives.
    context: VulkanForwardContext,
}

impl crate::chat::registry::ModelForward for VulkanTransformerExecutor {
    /// Trait entry delegating to the inherent executor forward.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        VulkanTransformerExecutor::forward(self, tokens, offset)
    }

    /// Clears every layer's KV cache between generations.
    fn clear_cache(&mut self) {
        VulkanTransformerExecutor::clear_cache(self);
    }

    /// Reports the constant backend label `"Vulkan"`.
    fn device_name(&self) -> &'static str {
        "Vulkan"
    }

    /// Exposes the executor's GPU-bucket page control to the engine's Ant hooks.
    fn as_ant_tier_control(&mut self) -> Option<&mut dyn crate::chat::registry::AntTierControl> {
        Some(self)
    }
}

impl VulkanTransformerExecutor {
    /// Ant bucketing: page one weight tensor between the GPU bucket
    /// (persistent upload) and the host-streaming bucket, through the shared
    /// context's paging primitive. The host copy in `WeightData` is always
    /// authoritative, so eviction only releases the device allocation and a
    /// page-in re-uploads from it.
    fn page_weight(&mut self, unit_id: &str, resident: bool) -> bool {
        let mut handled = false;
        for layer in &mut self.layers {
            for linear in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
            ] {
                if linear.name != unit_id {
                    continue;
                }
                handled = true;
                page_linear(linear, resident);
            }
            match &mut layer.feed_forward {
                FeedForward::Dense { gate, up, down, .. } => {
                    for linear in [gate, up, down] {
                        if linear.name != unit_id {
                            continue;
                        }
                        handled = true;
                        page_linear(linear, resident);
                    }
                }
                FeedForward::MixtureOfExperts {
                    router, experts, ..
                } => {
                    for linear in
                        std::iter::once(&mut *router).chain(experts.iter_mut().flat_map(|expert| {
                            [&mut expert.gate, &mut expert.up, &mut expert.down]
                        }))
                    {
                        if linear.name != unit_id {
                            continue;
                        }
                        handled = true;
                        page_linear(linear, resident);
                    }
                }
            }
        }
        handled
    }
}

impl crate::chat::registry::AntTierControl for VulkanTransformerExecutor {
    /// Ant GPU page-in: uploads one unit into the resident bucket (re-upload
    /// from the authoritative host copy); the bool reports whether a layer
    /// weight of that name was found.
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.page_weight(unit_id, true)
    }

    /// Ant GPU evict: releases one unit's resident slot (device allocation only
    /// — the host copy stays authoritative for the next page-in).
    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.page_weight(unit_id, false)
    }
}

impl VulkanTransformerExecutor {
    /// Public load entry: bootstraps the substrate, validates the shader
    /// pipelines, runs [`Self::load_initialized`], and guarantees substrate
    /// teardown on failure via [`load_with_substrate`].
    pub(crate) fn load(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
        plan: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        super::load_with_substrate(|| Self::load_initialized(paths, config, plan))
    }

    /// The load graph: take the embedding and final norm (offset applied per
    /// feature), resolve tied-vs-untied lm head, then per layer take the norms
    /// (q/k norms by declaration or checkpoint presence), the projections and
    /// the dense-or-MoE feed-forward — with Ant-bucket resident uploads for
    /// VRAM-tier units per the plan.
    fn load_initialized(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
        plan: &BTreeMap<String, String>,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim;
        let kv_size = config.num_key_value_heads * head_dim;
        let layout = &config.features.layout;
        let mut weights = load_weights(paths)?;
        let embedding = take_weight(
            &mut weights,
            layout.embedding,
            &[config.vocab_size, config.hidden_size],
        )?
        .data;
        let final_norm = take_weight(&mut weights, layout.final_norm, &[config.hidden_size])?.data;
        let final_norm = if config.features.norm_weight_offset {
            final_norm.add_one()
        } else {
            final_norm
        };
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
            let input_norm = if config.features.norm_weight_offset {
                input_norm.add_one()
            } else {
                input_norm
            };
            let post_attention_norm = take_weight(
                &mut weights,
                &format!("{prefix}.{}", layout.post_attention_norm),
                &[config.hidden_size],
            )?
            .data;
            let post_attention_norm = if config.features.norm_weight_offset {
                post_attention_norm.add_one()
            } else {
                post_attention_norm
            };
            let attention = format!("{prefix}.{}", layout.attention_prefix);
            // Per-head QK norms: required when the family declares them, and
            // otherwise picked up from the checkpoint whenever the tensors are
            // present (Qwen3 ships q_norm/k_norm without every config flagging
            // it — the native backend resolves the same way, by presence).
            let q_norm_name = format!("{attention}.{}.weight", layout.q_norm);
            let k_norm_name = format!("{attention}.{}.weight", layout.k_norm);
            let q_norm = if config.features.qk_norm || weights.contains_key(&q_norm_name) {
                Some(take_weight(&mut weights, &q_norm_name, &[head_dim])?.data)
            } else {
                None
            };
            let k_norm = if config.features.qk_norm || weights.contains_key(&k_norm_name) {
                Some(take_weight(&mut weights, &k_norm_name, &[head_dim])?.data)
            } else {
                None
            };
            let feed_forward = match &config.features.feed_forward {
                TransformerFeedForwardFeatures::Dense => FeedForward::Dense {
                    gate: take_linear(
                        &mut weights,
                        plan,
                        &format!("{prefix}.{}", layout.dense_gate),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    up: take_linear(
                        &mut weights,
                        plan,
                        &format!("{prefix}.{}", layout.dense_up),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    down: take_linear(
                        &mut weights,
                        plan,
                        &format!("{prefix}.{}", layout.dense_down),
                        config.intermediate_size,
                        config.hidden_size,
                        false,
                    )?,
                    activation: config.activation,
                },
                TransformerFeedForwardFeatures::MixtureOfExperts {
                    num_local_experts,
                    num_experts_per_tok,
                } => load_moe_feed_forward(
                    &mut weights,
                    plan,
                    &prefix,
                    config,
                    *num_local_experts,
                    *num_experts_per_tok,
                )?,
            };
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                q_norm,
                k_norm,
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
                feed_forward,
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

    /// The transformer graph: validates ids / positions / cache-offset
    /// consistency, gathers embeddings as a shader, then per layer: input
    /// RMSNorm → q/k/v (+ optional qk-norm) → rotary → cache append → attention
    /// (sliding window) → o_proj → residual → post norm → feed-forward →
    /// residual; final norm + lm head on the last position.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        let sequence = tokens.len();
        if sequence == 0 {
            return Err(format!(
                "Vulkan {} requires a non-empty token sequence.",
                self.config.features.model_name
            ));
        }
        let last_position = offset
            .checked_add(sequence)
            .ok_or_else(|| "Vulkan transformer sequence position overflow.".to_string())?;
        if last_position > self.config.max_position_embeddings {
            return Err("Vulkan transformer input exceeds max_position_embeddings.".into());
        }
        let expected_cache = offset
            .checked_mul(self.config.num_key_value_heads)
            .and_then(|count| count.checked_mul(self.config.head_dim))
            .ok_or_else(|| "Vulkan transformer cache size overflow.".to_string())?;
        if self
            .layers
            .first()
            .is_some_and(|layer| layer.kv.keys.len() != expected_cache)
        {
            return Err(
                "Vulkan transformer KV cache position does not match the input offset.".into(),
            );
        }
        let hidden = self.config.hidden_size;
        for token_id in tokens {
            let token = *token_id as usize;
            if token >= self.config.vocab_size {
                return Err(format!("Token ID {token} exceeds the model vocabulary."));
            }
        }
        // Ant op 8: embedding gather runs as a Vulkan shader — the host loop
        // only validates token ids against the vocabulary.
        let (table_ptr, table_format, table_elements) = self.embedding.ffi_parts();
        let mut activations = cheetah::native::vulkan::embedding_gather(
            tokens,
            table_ptr,
            table_format,
            table_elements,
            hidden,
            self.config.features.embedding_scale,
        )?;

        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let residual = activations.clone();
            let normalized = layer.input_norm.rms_norm(
                &activations,
                sequence,
                hidden,
                self.config.rms_norm_eps,
            )?;
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
                self.config.sliding_window,
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
            let feed_forward = layer.feed_forward.forward(
                &normalized,
                sequence,
                hidden,
                &mut self.context,
                layer_index,
            )?;
            activations = cheetah::native::vulkan::add(&residual, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .rms_norm(last, 1, hidden, self.config.rms_norm_eps)?;
        let logits = self
            .lm_head
            .linear(&normalized, 1, hidden, self.config.vocab_size)?;
        Ok(logits)
    }
}

/// Mixtral-style sparse-MoE tensor naming: router + expert projections.
const MOE_ROUTER_LAYOUT: &str = "block_sparse_moe.gate";
const MOE_EXPERTS_LAYOUT: &str = "block_sparse_moe.experts";
const MOE_GATE_LAYOUT: &str = "w1";
const MOE_UP_LAYOUT: &str = "w3";
const MOE_DOWN_LAYOUT: &str = "w2";

/// Loads the Mixtral-style sparse feed-forward (router + per-expert
/// gate/up/down) under the fixed MOE_* layout constants.
fn load_moe_feed_forward(
    weights: &mut HashMap<String, NamedWeight>,
    plan: &BTreeMap<String, String>,
    prefix: &str,
    config: &VulkanTransformerConfig,
    num_local_experts: usize,
    num_experts_per_tok: usize,
) -> Result<FeedForward, String> {
    let router = take_linear(
        weights,
        plan,
        &format!("{prefix}.{MOE_ROUTER_LAYOUT}"),
        config.hidden_size,
        num_local_experts,
        false,
    )?;
    let mut experts = Vec::with_capacity(num_local_experts);
    for index in 0..num_local_experts {
        let expert_prefix = format!("{prefix}.{MOE_EXPERTS_LAYOUT}.{index}");
        experts.push(Expert {
            gate: take_linear(
                weights,
                plan,
                &format!("{expert_prefix}.{MOE_GATE_LAYOUT}"),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?,
            up: take_linear(
                weights,
                plan,
                &format!("{expert_prefix}.{MOE_UP_LAYOUT}"),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?,
            down: take_linear(
                weights,
                plan,
                &format!("{expert_prefix}.{MOE_DOWN_LAYOUT}"),
                config.intermediate_size,
                config.hidden_size,
                false,
            )?,
        });
    }
    Ok(FeedForward::MixtureOfExperts {
        router,
        experts,
        num_experts_per_tok,
        activation: config.activation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_linear(name: &str, size: usize) -> Linear {
        let mut matrix = vec![0.0; size * size];
        for index in 0..size {
            matrix[index * size + index] = 1.0;
        }
        Linear {
            name: name.into(),
            input_size: size,
            output_size: size,
            weight: WeightData::F32(matrix.into()),
            bias: None,
            slot: None,
        }
    }

    fn tiny_executor(
        feed_forward_features: TransformerFeedForwardFeatures,
    ) -> VulkanTransformerExecutor {
        let mut features = TransformerFeatures::standard("Test");
        features.feed_forward = feed_forward_features;
        let config = VulkanTransformerConfig {
            vocab_size: 2,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            max_position_embeddings: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: 4,
            activation: Activation::Silu,
            features,
        };
        let feed_forward = match &config.features.feed_forward {
            TransformerFeedForwardFeatures::Dense => FeedForward::Dense {
                gate: identity_linear("dense_gate", 2),
                up: identity_linear("dense_up", 2),
                down: identity_linear("dense_down", 2),
                activation: Activation::Silu,
            },
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts,
                num_experts_per_tok,
            } => FeedForward::MixtureOfExperts {
                router: Linear {
                    name: "router".into(),
                    input_size: 2,
                    output_size: *num_local_experts,
                    weight: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
                    bias: None,
                    slot: None,
                },
                experts: (0..*num_local_experts)
                    .map(|index| Expert {
                        gate: identity_linear(&format!("expert_{index}_gate"), 2),
                        up: identity_linear(&format!("expert_{index}_up"), 2),
                        down: identity_linear(&format!("expert_{index}_down"), 2),
                    })
                    .collect(),
                num_experts_per_tok: *num_experts_per_tok,
                activation: Activation::Silu,
            },
        };
        VulkanTransformerExecutor {
            config,
            embedding: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
            layers: vec![Layer {
                input_norm: WeightData::F32(vec![1.0, 1.0].into()),
                post_attention_norm: WeightData::F32(vec![1.0, 1.0].into()),
                q_norm: None,
                k_norm: None,
                q_proj: identity_linear("q_proj", 2),
                k_proj: identity_linear("k_proj", 2),
                v_proj: identity_linear("v_proj", 2),
                o_proj: identity_linear("o_proj", 2),
                feed_forward,
                kv: GpuKvCache::default(),
            }],
            final_norm: WeightData::F32(vec![1.0, 1.0].into()),
            lm_head: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
            context: VulkanForwardContext::new(),
        }
    }

    #[test]
    fn parses_common_and_moe_transformer_configurations() {
        let raw = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "sliding_window": 32
        });
        let config =
            VulkanTransformerConfig::from_raw(&raw, TransformerFeatures::standard("Dense"))
                .unwrap();
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.sliding_window, 32);

        let mut features = TransformerFeatures::standard("Sparse");
        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts: 8,
            num_experts_per_tok: 2,
        };
        let config = VulkanTransformerConfig::from_raw(&raw, features.clone()).unwrap();
        assert!(matches!(
            config.features.feed_forward,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 8,
                num_experts_per_tok: 2,
            }
        ));

        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts: 2,
            num_experts_per_tok: 3,
        };
        assert!(VulkanTransformerConfig::from_raw(&raw, features).is_err());
    }

    #[test]
    fn feature_values_override_common_configuration() {
        let raw = serde_json::json!({
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "tie_word_embeddings": true
        });
        let mut features = TransformerFeatures::standard("Configured");
        features.attention_bias = true;
        features.tie_word_embeddings = true;
        let config = VulkanTransformerConfig::from_raw(&raw, features).unwrap();
        assert_eq!(config.head_dim, 2);
        assert_eq!(config.sliding_window, 64);
        assert!(config.features.tie_word_embeddings);
        assert!(config.features.attention_bias);
        assert!(!config.features.attention_output_bias);
    }

    #[test]
    fn loads_moe_weights_from_configured_tensor_layout() {
        let raw = serde_json::json!({
            "vocab_size": 2,
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "max_position_embeddings": 4,
            "hidden_act": "silu"
        });
        let mut features = TransformerFeatures::standard("Configured");
        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts: 1,
            num_experts_per_tok: 1,
        };
        let config = VulkanTransformerConfig::from_raw(&raw, features).unwrap();
        let mut weights = HashMap::new();
        for (name, shape, values) in [
            (
                "layers.0.block_sparse_moe.gate.weight",
                vec![1, 2],
                vec![1.0, 0.0],
            ),
            (
                "layers.0.block_sparse_moe.experts.0.w1.weight",
                vec![2, 2],
                vec![1.0; 4],
            ),
            (
                "layers.0.block_sparse_moe.experts.0.w3.weight",
                vec![2, 2],
                vec![1.0; 4],
            ),
            (
                "layers.0.block_sparse_moe.experts.0.w2.weight",
                vec![2, 2],
                vec![1.0; 4],
            ),
        ] {
            weights.insert(
                name.to_owned(),
                NamedWeight {
                    shape,
                    data: WeightData::F32(values.into()),
                },
            );
        }
        let TransformerFeedForwardFeatures::MixtureOfExperts {
            num_local_experts,
            num_experts_per_tok,
        } = &config.features.feed_forward
        else {
            unreachable!();
        };
        let feed_forward = load_moe_feed_forward(
            &mut weights,
            &BTreeMap::new(),
            "layers.0",
            &config,
            *num_local_experts,
            *num_experts_per_tok,
        )
        .unwrap();
        assert!(weights.is_empty());
        assert!(matches!(
            feed_forward,
            FeedForward::MixtureOfExperts {
                ref router,
                ref experts,
                num_experts_per_tok: 1,
                ..
            } if router.name == "layers.0.block_sparse_moe.gate"
                && experts[0].gate.name == "layers.0.block_sparse_moe.experts.0.w1"
                && experts[0].up.name == "layers.0.block_sparse_moe.experts.0.w3"
                && experts[0].down.name == "layers.0.block_sparse_moe.experts.0.w2"
        ));
    }

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn linear_splits_large_sequences_into_bounded_row_tiles() {
        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let weight = WeightData::F32(vec![2.0, -1.0].into());
        let output = weight.linear(&vec![3.0; 65], 65, 1, 2).unwrap();
        cheetah::native::vulkan::shutdown().unwrap();
        assert_eq!(output.len(), 130);
        for row in output.chunks_exact(2) {
            assert_eq!(row, &[6.0, -3.0]);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn tiny_dense_and_moe_models_forward_and_reset_cache() {
        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let feed_forward_features = [
            TransformerFeedForwardFeatures::Dense,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 2,
                num_experts_per_tok: 1,
            },
        ];
        for features in feed_forward_features {
            let mut model = tiny_executor(features);
            let first = [0u32];
            let logits = model.forward(&first, 0).unwrap();
            assert_eq!(logits.len(), 2);
            assert!(logits.iter().all(|value| f32::is_finite(*value)));
            let next = [1u32];
            model.forward(&next, 1).unwrap();
            model.clear_cache();
            let repeated = model.forward(&first, 0).unwrap();
            assert_eq!(logits, repeated);
        }
        cheetah::native::vulkan::shutdown().unwrap();
    }
}

#[cfg(test)]
mod q8_tests {
    use super::*;

    /// CHEETAH_QUANT_Q8=1 + a synthetic BF16 decoder: the Vulkan executor's
    /// q8-quantized output must stay finite and terminate (a hang here means
    /// the resident q8 matmul deadlocks). Set CHEETAH_TEST_VULKAN_Q8=1 to run.
    #[test]
    fn vulkan_q8_quantized_forward_terminates() {
        if std::env::var("CHEETAH_TEST_VULKAN_Q8").is_err() {
            eprintln!("skipping q8 forward test (set CHEETAH_TEST_VULKAN_Q8=1)");
            return;
        }
        // SAFETY: test-scoped process env mutation; single-threaded test.
        unsafe { std::env::set_var("CHEETAH_QUANT_Q8", "1") };
        // Native forward tests live in the native module; rebuild
        // the tiny model here via the same helper through the native tests is
        // not reachable, so prepare a minimal directory directly.
        let base = std::env::temp_dir().join(format!(
            "cheetah-synth-q8-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ));
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("config.json"),
            r#"{
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false,
                "max_position_embeddings": 512
            }"#,
        )
        .unwrap();

        let mut shape_f32: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        shape_f32.push((
            "model.embed_tokens.weight".into(),
            vec![0.0; 128 * 64],
            vec![128, 64],
        ));
        shape_f32.push(("lm_head.weight".into(), vec![0.0; 128 * 64], vec![128, 64]));
        shape_f32.push(("model.norm.weight".into(), vec![1.0; 64], vec![64]));
        for layer in 0..2usize {
            let prefix = format!("model.layers.{layer}.");
            for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                shape_f32.push((format!("{prefix}{name}"), vec![1.0; 64], vec![64]));
            }
            shape_f32.push((
                format!("{prefix}self_attn.q_proj.weight"),
                vec![0.5; 4096],
                vec![64, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.k_proj.weight"),
                vec![0.25; 2048],
                vec![32, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.v_proj.weight"),
                vec![0.25; 2048],
                vec![32, 64],
            ));
            shape_f32.push((
                format!("{prefix}self_attn.o_proj.weight"),
                vec![0.5; 4096],
                vec![64, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.gate_proj.weight"),
                vec![0.5; 8192],
                vec![128, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.up_proj.weight"),
                vec![0.5; 8192],
                vec![128, 64],
            ));
            shape_f32.push((
                format!("{prefix}mlp.down_proj.weight"),
                vec![0.5; 8192],
                vec![64, 128],
            ));
        }

        let tensors: Vec<(String, safetensors::tensor::TensorView<'static>)> = shape_f32
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
        std::fs::write(
            src.join("model.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();

        cheetah::native::init_model(&src, &dst).unwrap();

        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let paths: Vec<std::path::PathBuf> = std::fs::read_dir(dst.join(".cheetah/weights"))
            .unwrap()
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                (path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))
                    .then_some(path)
            })
            .collect();
        let plan = std::collections::BTreeMap::new(); // all units resident
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join("config.json")).unwrap()).unwrap();
        let features = TransformerFeatures::standard("Llama");
        let config = VulkanTransformerConfig::from_raw(&raw, features).unwrap();
        let mut model = VulkanTransformerExecutor::load(&paths, &config, &plan).unwrap();

        let tokens: Vec<u32> = vec![3, 41, 17];
        let started = std::time::Instant::now();
        let logits = model.forward(&tokens, 0).unwrap();
        eprintln!(
            "q8 forward finished in {:?}; finite={}",
            started.elapsed(),
            logits.iter().all(|value| f32::is_finite(*value))
        );
        assert!(logits.iter().all(|value| f32::is_finite(*value)));

        let _ = std::fs::remove_dir_all(&base);
        // SAFETY: test-scoped process env mutation.
        unsafe { std::env::remove_var("CHEETAH_QUANT_Q8") };
        cheetah::native::vulkan::shutdown().unwrap();
    }
}
