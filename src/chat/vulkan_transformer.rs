use crate::color::{self, Color};
use candle_core::{Device, Tensor};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct VulkanTransformerConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    sliding_window: usize,
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
        num_experts: usize,
        top_k: usize,
        activation: Activation,
        router: &'static str,
        experts_prefix: &'static str,
        gate: &'static str,
        up: &'static str,
        down: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Activation {
    Silu,
    Gelu,
    NewGelu,
}

impl VulkanTransformerConfig {
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
        if let TransformerFeedForwardFeatures::MixtureOfExperts {
            num_experts, top_k, ..
        } = &features.feed_forward
            && (*num_experts == 0 || *top_k == 0 || *top_k > *num_experts)
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

#[derive(Clone, Debug)]
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

impl VulkanPhiConfig {
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

pub(crate) fn required_usize(config: &Value, name: &str) -> Result<usize, String> {
    config
        .get(name)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("Model config is missing a valid {name}."))
        .and_then(|value| usize::try_from(value).map_err(|_| format!("{name} does not fit usize.")))
}

#[derive(Clone)]
enum WeightData {
    F32(Arc<[f32]>),
    F16(Arc<[u16]>),
    BF16(Arc<[u16]>),
}

impl WeightData {
    fn from_view(name: &str, view: &TensorView<'_>) -> Result<Self, String> {
        let bytes = view.data();
        match view.dtype() {
            Dtype::F32 => {
                if bytes.len() % 4 != 0 {
                    return Err(format!("Tensor '{name}' has an invalid F32 byte length."));
                }
                Ok(Self::F32(
                    bytes
                        .chunks_exact(4)
                        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                        .collect::<Vec<_>>()
                        .into(),
                ))
            }
            Dtype::F16 | Dtype::BF16 => {
                if bytes.len() % 2 != 0 {
                    return Err(format!(
                        "Tensor '{name}' has an invalid 16-bit byte length."
                    ));
                }
                let values: Arc<[u16]> = bytes
                    .chunks_exact(2)
                    .map(|value| u16::from_le_bytes([value[0], value[1]]))
                    .collect::<Vec<_>>()
                    .into();
                if view.dtype() == Dtype::F16 {
                    Ok(Self::F16(values))
                } else {
                    Ok(Self::BF16(values))
                }
            }
            dtype => Err(format!(
                "Vulkan models support F32, F16, or BF16 weights; tensor '{name}' uses {dtype:?}."
            )),
        }
    }

    fn value(&self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::F16(values) => half::f16::from_bits(values[index]).to_f32(),
            Self::BF16(values) => half::bf16::from_bits(values[index]).to_f32(),
        }
    }

    fn add_one(&self) -> Self {
        Self::F32(
            (0..self.len())
                .map(|index| self.value(index) + 1.0)
                .collect(),
        )
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) | Self::BF16(values) => values.len(),
        }
    }

    fn rms_norm(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::F32(weights) => {
                cheetah::native::vulkan::rms_norm_f32(input, weights, rows, width, epsilon)
            }
            Self::F16(weights) => {
                cheetah::native::vulkan::rms_norm_fp16(input, weights, rows, width, epsilon)
            }
            Self::BF16(weights) => {
                cheetah::native::vulkan::rms_norm_bf16(input, weights, rows, width, epsilon)
            }
        }
    }

    fn layer_norm(
        &self,
        bias: &Self,
        input: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        match (self, bias) {
            (Self::F32(weights), Self::F32(bias)) => {
                cheetah::native::vulkan::layer_norm_f32(input, weights, bias, rows, width, epsilon)
            }
            (Self::F16(weights), Self::F16(bias)) => {
                cheetah::native::vulkan::layer_norm_fp16(input, weights, bias, rows, width, epsilon)
            }
            (Self::BF16(weights), Self::BF16(bias)) => {
                cheetah::native::vulkan::layer_norm_bf16(input, weights, bias, rows, width, epsilon)
            }
            _ => Err("Layer normalization weight and bias dtypes do not match.".into()),
        }
    }

    fn linear(
        &self,
        input: &[f32],
        rows: usize,
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<f32>, String> {
        let input_len = rows
            .checked_mul(input_size)
            .ok_or_else(|| "Vulkan linear input size overflow.".to_string())?;
        let output_len = rows
            .checked_mul(output_size)
            .ok_or_else(|| "Vulkan linear output size overflow.".to_string())?;
        if rows == 0 || input_size == 0 || output_size == 0 || input.len() != input_len {
            return Err("Invalid Vulkan linear dimensions or input length.".into());
        }
        const MAX_WEIGHT_TILE_ELEMENTS: usize = 16 * 1024 * 1024;
        const MAX_ROW_TILE: usize = 64;
        let tile_size = output_size
            .min(16_384)
            .min((MAX_WEIGHT_TILE_ELEMENTS / input_size).max(1));
        let mut output = vec![0.0; output_len];
        for row_offset in (0..rows).step_by(MAX_ROW_TILE) {
            let tile_rows = (rows - row_offset).min(MAX_ROW_TILE);
            let input_start = row_offset * input_size;
            let input_tile = &input[input_start..input_start + tile_rows * input_size];
            for column_offset in (0..output_size).step_by(tile_size) {
                let columns = (output_size - column_offset).min(tile_size);
                let weight_start = column_offset * input_size;
                let weight_end = (column_offset + columns) * input_size;
                let tile = match self {
                    Self::F32(weights) => cheetah::native::vulkan::matmul_f32_transposed(
                        input_tile,
                        &weights[weight_start..weight_end],
                        tile_rows,
                        input_size,
                        columns,
                    )?,
                    Self::F16(weights) => cheetah::native::vulkan::matmul_fp16_transposed(
                        input_tile,
                        &weights[weight_start..weight_end],
                        tile_rows,
                        input_size,
                        columns,
                    )?,
                    Self::BF16(weights) => cheetah::native::vulkan::matmul_bf16_transposed(
                        input_tile,
                        &weights[weight_start..weight_end],
                        tile_rows,
                        input_size,
                        columns,
                    )?,
                };
                for row in 0..tile_rows {
                    let source = &tile[row * columns..(row + 1) * columns];
                    let destination_start = (row_offset + row) * output_size + column_offset;
                    output[destination_start..destination_start + columns].copy_from_slice(source);
                }
            }
        }
        Ok(output)
    }
}

struct NamedWeight {
    shape: Vec<usize>,
    data: WeightData,
}

#[derive(Clone)]
struct Linear {
    name: String,
    input_size: usize,
    output_size: usize,
    weight: WeightData,
    bias: Option<WeightData>,
}

impl Linear {
    fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let expected = rows
            .checked_mul(self.input_size)
            .ok_or_else(|| format!("Input size overflow in '{}'.", self.name))?;
        if rows == 0 || input.len() != expected {
            return Err(format!(
                "Invalid input for '{}': expected {expected} values, received {}.",
                self.name,
                input.len()
            ));
        }
        let mut output = self
            .weight
            .linear(input, rows, self.input_size, self.output_size)?;
        if let Some(bias) = &self.bias {
            if bias.len() != self.output_size {
                return Err(format!("Invalid bias shape for '{}'.", self.name));
            }
            for row in output.chunks_exact_mut(self.output_size) {
                for (value, index) in row.iter_mut().zip(0..self.output_size) {
                    *value += bias.value(index);
                }
            }
        }
        Ok(output)
    }
}

#[derive(Clone)]
struct Expert {
    gate: Linear,
    up: Linear,
    down: Linear,
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
        top_k: usize,
        activation: Activation,
    },
}

impl FeedForward {
    fn forward(&self, input: &[f32], rows: usize, hidden_size: usize) -> Result<Vec<f32>, String> {
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
                top_k,
                activation,
            } => {
                let router_logits = router.forward(input, rows)?;
                let mut output = vec![0.0; rows * hidden_size];
                let mut selected_by_expert = vec![Vec::new(); experts.len()];
                for row in 0..rows {
                    let logits = &router_logits[row * experts.len()..(row + 1) * experts.len()];
                    for (expert_index, route_weight) in select_experts(logits, *top_k)? {
                        selected_by_expert[expert_index].push((row, route_weight));
                    }
                }
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

fn apply_activation(activation: Activation, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    match activation {
        Activation::Silu => cheetah::native::vulkan::silu_mul(gate, up),
        Activation::Gelu => cheetah::native::vulkan::gelu_mul(gate, up, false),
        Activation::NewGelu => cheetah::native::vulkan::gelu_mul(gate, up, true),
    }
}

fn select_experts(logits: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>, String> {
    if logits.is_empty() || top_k == 0 || top_k > logits.len() {
        return Err("Invalid MoE router logits or expert count.".into());
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err("MoE router produced a non-finite score.".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probabilities = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let denominator: f32 = probabilities.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("MoE router probability normalization failed.".into());
    }
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]));
    let mut selected = order
        .into_iter()
        .take(top_k)
        .map(|index| (index, probabilities[index] / denominator))
        .collect::<Vec<_>>();
    let selected_sum: f32 = selected.iter().map(|(_, weight)| weight).sum();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err("MoE selected expert weights could not be normalized.".into());
    }
    for (_, weight) in &mut selected {
        *weight /= selected_sum;
    }
    Ok(selected)
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
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct VulkanTransformerExecutor {
    config: VulkanTransformerConfig,
    embedding: WeightData,
    layers: Vec<Layer>,
    final_norm: WeightData,
    lm_head: WeightData,
}

impl super::models::ModelRuntime for VulkanTransformerExecutor {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        VulkanTransformerExecutor::forward(self, input, offset)
    }

    fn clear_cache(&mut self) {
        VulkanTransformerExecutor::clear_cache(self);
    }

    fn device_name(&self) -> &'static str {
        "Vulkan"
    }
}

impl VulkanTransformerExecutor {
    pub(crate) fn load(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
    ) -> Result<Self, String> {
        let result = (|| {
            cheetah::native::vulkan::initialize()?;
            cheetah::native::vulkan::prepare()?;
            Self::load_initialized(paths, config)
        })();
        if result.is_err()
            && let Err(shutdown_error) = cheetah::native::vulkan::shutdown()
        {
            eprintln!(
                "{}",
                color::stderr(
                    &format!(
                        "cheetah: Vulkan cleanup after model startup failure also failed ({shutdown_error})."
                    ),
                    Color::Red
                )
            );
        }
        result
    }

    fn load_initialized(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
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
            let q_norm = if config.features.qk_norm {
                Some(
                    take_weight(
                        &mut weights,
                        &format!("{attention}.{}.weight", layout.q_norm),
                        &[head_dim],
                    )?
                    .data,
                )
            } else {
                None
            };
            let k_norm = if config.features.qk_norm {
                Some(
                    take_weight(
                        &mut weights,
                        &format!("{attention}.{}.weight", layout.k_norm),
                        &[head_dim],
                    )?
                    .data,
                )
            } else {
                None
            };
            let feed_forward = match &config.features.feed_forward {
                TransformerFeedForwardFeatures::Dense => FeedForward::Dense {
                    gate: take_linear(
                        &mut weights,
                        &format!("{prefix}.{}", layout.dense_gate),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    up: take_linear(
                        &mut weights,
                        &format!("{prefix}.{}", layout.dense_up),
                        config.hidden_size,
                        config.intermediate_size,
                        false,
                    )?,
                    down: take_linear(
                        &mut weights,
                        &format!("{prefix}.{}", layout.dense_down),
                        config.intermediate_size,
                        config.hidden_size,
                        false,
                    )?,
                    activation: config.activation,
                },
                TransformerFeedForwardFeatures::MixtureOfExperts {
                    num_experts,
                    top_k,
                    activation,
                    router,
                    experts_prefix,
                    gate,
                    up,
                    down,
                } => load_moe_feed_forward(
                    &mut weights,
                    &prefix,
                    config,
                    *num_experts,
                    *top_k,
                    *activation,
                    router,
                    experts_prefix,
                    gate,
                    up,
                    down,
                )?,
            };
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                q_norm,
                k_norm,
                q_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.{}", layout.q_proj),
                    config.hidden_size,
                    config.num_attention_heads * head_dim,
                    config.features.attention_bias,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.{}", layout.k_proj),
                    config.hidden_size,
                    kv_size,
                    config.features.attention_bias,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.{}", layout.v_proj),
                    config.hidden_size,
                    kv_size,
                    config.features.attention_bias,
                )?,
                o_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.{}", layout.o_proj),
                    config.num_attention_heads * head_dim,
                    config.hidden_size,
                    config.features.attention_output_bias,
                )?,
                feed_forward,
                keys: Vec::new(),
                values: Vec::new(),
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

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        let (batch, sequence) = input
            .dims2()
            .map_err(|error| format!("Invalid Vulkan transformer input: {error}"))?;
        if batch != 1 || sequence == 0 {
            return Err(format!(
                "Vulkan {} requires a non-empty batch of size one.",
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
            .is_some_and(|layer| layer.keys.len() != expected_cache)
        {
            return Err(
                "Vulkan transformer KV cache position does not match the input offset.".into(),
            );
        }
        let token_ids = input
            .to_vec2::<u32>()
            .map_err(|error| format!("Cannot read Vulkan transformer token IDs: {error}"))?;
        let hidden = self.config.hidden_size;
        let mut activations = Vec::with_capacity(sequence * hidden);
        for token_id in &token_ids[0] {
            let token = *token_id as usize;
            if token >= self.config.vocab_size {
                return Err(format!("Token ID {token} exceeds the model vocabulary."));
            }
            let start = token * hidden;
            for index in 0..hidden {
                let value = self.embedding.value(start + index);
                activations.push(if self.config.features.embedding_scale {
                    value * (hidden as f32).sqrt()
                } else {
                    value
                });
            }
        }

        for layer in &mut self.layers {
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
            layer.keys.extend_from_slice(&keys);
            layer.values.extend_from_slice(&values);
            let attention = cheetah::native::vulkan::attention(
                &query,
                &layer.keys,
                &layer.values,
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
            let feed_forward = layer.feed_forward.forward(&normalized, sequence, hidden)?;
            activations = cheetah::native::vulkan::add(&residual, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .rms_norm(last, 1, hidden, self.config.rms_norm_eps)?;
        let logits = self
            .lm_head
            .linear(&normalized, 1, hidden, self.config.vocab_size)?;
        Tensor::from_vec(logits, (1, 1, self.config.vocab_size), &Device::Cpu)
            .map_err(|error| format!("Cannot create Vulkan transformer logits: {error}"))
    }
}

fn load_moe_feed_forward(
    weights: &mut HashMap<String, NamedWeight>,
    prefix: &str,
    config: &VulkanTransformerConfig,
    num_experts: usize,
    top_k: usize,
    activation: Activation,
    router_layout: &str,
    experts_layout: &str,
    gate_layout: &str,
    up_layout: &str,
    down_layout: &str,
) -> Result<FeedForward, String> {
    let router = take_linear(
        weights,
        &format!("{prefix}.{router_layout}"),
        config.hidden_size,
        num_experts,
        false,
    )?;
    let mut experts = Vec::with_capacity(num_experts);
    for index in 0..num_experts {
        let expert_prefix = format!("{prefix}.{experts_layout}.{index}");
        experts.push(Expert {
            gate: take_linear(
                weights,
                &format!("{expert_prefix}.{gate_layout}"),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?,
            up: take_linear(
                weights,
                &format!("{expert_prefix}.{up_layout}"),
                config.hidden_size,
                config.intermediate_size,
                false,
            )?,
            down: take_linear(
                weights,
                &format!("{expert_prefix}.{down_layout}"),
                config.intermediate_size,
                config.hidden_size,
                false,
            )?,
        });
    }
    Ok(FeedForward::MixtureOfExperts {
        router,
        experts,
        top_k,
        activation,
    })
}

fn load_weights(paths: &[PathBuf]) -> Result<HashMap<String, NamedWeight>, String> {
    let mut weights = HashMap::new();
    for path in paths {
        let file = File::open(path).map_err(|error| {
            format!(
                "Cannot open Vulkan model shard '{}': {error}",
                path.display()
            )
        })?;
        let mapping = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|error| {
                format!(
                    "Cannot mmap Vulkan model shard '{}': {error}",
                    path.display()
                )
            })?
        };
        let tensors = SafeTensors::deserialize(&mapping)
            .map_err(|error| format!("Invalid Vulkan model shard '{}': {error}", path.display()))?;
        for name in tensors.names() {
            if name.starts_with("__cheetah_padding_") {
                continue;
            }
            let view = tensors.tensor(name).map_err(|error| {
                format!(
                    "Cannot read tensor '{name}' from '{}': {error}",
                    path.display()
                )
            })?;
            let weight = NamedWeight {
                shape: view.shape().to_vec(),
                data: WeightData::from_view(name, &view)?,
            };
            if weights.insert(name.to_owned(), weight).is_some() {
                return Err(format!("Duplicate tensor '{name}' across model shards."));
            }
        }
    }
    Ok(weights)
}

fn take_weight(
    weights: &mut HashMap<String, NamedWeight>,
    name: &str,
    expected_shape: &[usize],
) -> Result<NamedWeight, String> {
    let weight = weights
        .remove(name)
        .ok_or_else(|| format!("Vulkan model checkpoint is missing tensor '{name}'."))?;
    if weight.shape != expected_shape {
        return Err(format!(
            "Tensor '{name}' has shape {:?}; expected {expected_shape:?}.",
            weight.shape
        ));
    }
    Ok(weight)
}

fn validate_shape(name: &str, actual: &[usize], expected: &[usize]) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "Tensor '{name}' has shape {actual:?}; expected {expected:?}."
        ));
    }
    Ok(())
}

fn take_linear(
    weights: &mut HashMap<String, NamedWeight>,
    prefix: &str,
    input_size: usize,
    output_size: usize,
    with_bias: bool,
) -> Result<Linear, String> {
    let weight_name = format!("{prefix}.weight");
    let weight = take_weight(weights, &weight_name, &[output_size, input_size])?;
    let bias_name = format!("{prefix}.bias");
    let bias = match weights.remove(&bias_name) {
        Some(bias) => {
            if bias.shape != [output_size] {
                return Err(format!(
                    "Tensor '{bias_name}' has shape {:?}; expected [{output_size}].",
                    bias.shape
                ));
            }
            Some(bias.data)
        }
        None if with_bias => {
            return Err(format!(
                "Vulkan checkpoint is missing tensor '{bias_name}'."
            ));
        }
        None => None,
    };
    Ok(Linear {
        name: prefix.to_owned(),
        input_size,
        output_size,
        weight: weight.data,
        bias,
    })
}

#[derive(Clone)]
struct PhiNorm {
    weight: WeightData,
    bias: WeightData,
}

impl PhiNorm {
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

#[derive(Clone)]
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
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct VulkanPhiExecutor {
    config: VulkanPhiConfig,
    embedding: WeightData,
    layers: Vec<PhiLayer>,
    final_norm: PhiNorm,
    lm_head: Linear,
}

impl super::models::ModelRuntime for VulkanPhiExecutor {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        VulkanPhiExecutor::forward(self, input, offset)
    }

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    fn device_name(&self) -> &'static str {
        "Vulkan"
    }
}

impl VulkanPhiExecutor {
    pub(crate) fn load(paths: &[PathBuf], config: &VulkanPhiConfig) -> Result<Self, String> {
        let result = (|| {
            cheetah::native::vulkan::initialize()?;
            cheetah::native::vulkan::prepare()?;
            Self::load_initialized(paths, config)
        })();
        if result.is_err()
            && let Err(shutdown_error) = cheetah::native::vulkan::shutdown()
        {
            eprintln!(
                "{}",
                color::stderr(
                    &format!(
                        "cheetah: Vulkan cleanup after Phi startup failure also failed ({shutdown_error})."
                    ),
                    Color::Red
                )
            );
        }
        result
    }

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
                    &format!("{attention}.q_proj"),
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                    true,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.k_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.v_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                dense: take_linear(
                    &mut weights,
                    &format!("{attention}.dense"),
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                    true,
                )?,
                fc1: take_linear(
                    &mut weights,
                    &format!("{prefix}.mlp.fc1"),
                    config.hidden_size,
                    config.intermediate_size,
                    true,
                )?,
                fc2: take_linear(
                    &mut weights,
                    &format!("{prefix}.mlp.fc2"),
                    config.intermediate_size,
                    config.hidden_size,
                    true,
                )?,
                keys: Vec::new(),
                values: Vec::new(),
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

    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        let (batch, sequence) = input
            .dims2()
            .map_err(|error| format!("Invalid Vulkan Phi input: {error}"))?;
        if batch != 1 || sequence == 0 {
            return Err("Vulkan Phi currently requires a non-empty batch of size one.".into());
        }
        let last_position = offset
            .checked_add(sequence)
            .ok_or_else(|| "Vulkan Phi sequence position overflow.".to_string())?;
        if last_position > self.config.max_position_embeddings {
            return Err("Vulkan Phi input exceeds max_position_embeddings.".into());
        }
        let expected_cache = offset
            .checked_mul(self.config.num_key_value_heads)
            .and_then(|count| count.checked_mul(self.config.head_dim))
            .ok_or_else(|| "Vulkan Phi cache size overflow.".to_string())?;
        if self
            .layers
            .first()
            .is_some_and(|layer| layer.keys.len() != expected_cache)
        {
            return Err("Vulkan Phi KV cache position does not match the input offset.".into());
        }
        let token_ids = input
            .to_vec2::<u32>()
            .map_err(|error| format!("Cannot read Vulkan Phi token IDs: {error}"))?;
        let hidden = self.config.hidden_size;
        let mut activations = Vec::with_capacity(sequence * hidden);
        for token_id in &token_ids[0] {
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
                self.config.rope_theta,
            )?;
            layer.keys.extend_from_slice(&keys);
            layer.values.extend_from_slice(&values);
            let attention = cheetah::native::vulkan::attention(
                &query,
                &layer.keys,
                &layer.values,
                sequence,
                offset,
                offset + sequence,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                0,
            )?;
            let attention = layer.dense.forward(&attention, sequence)?;
            let mut feed_forward = layer.fc1.forward(&normalized, sequence)?;
            let ones = vec![1.0; feed_forward.len()];
            feed_forward = match self.config.activation {
                Activation::Silu => cheetah::native::vulkan::silu_mul(&feed_forward, &ones)?,
                Activation::Gelu => cheetah::native::vulkan::gelu_mul(&feed_forward, &ones, false)?,
                Activation::NewGelu => {
                    cheetah::native::vulkan::gelu_mul(&feed_forward, &ones, true)?
                }
            };
            feed_forward = layer.fc2.forward(&feed_forward, sequence)?;
            let residual_attention = cheetah::native::vulkan::add(&residual, &attention)?;
            activations = cheetah::native::vulkan::add(&residual_attention, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized = self
            .final_norm
            .forward(last, 1, hidden, self.config.layer_norm_eps)?;
        let logits = self.lm_head.forward(&normalized, 1)?;
        Tensor::from_vec(logits, (1, 1, self.config.vocab_size), &Device::Cpu)
            .map_err(|error| format!("Cannot create Vulkan Phi logits: {error}"))
    }
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
                num_experts,
                top_k,
                activation,
                ..
            } => FeedForward::MixtureOfExperts {
                router: Linear {
                    name: "router".into(),
                    input_size: 2,
                    output_size: *num_experts,
                    weight: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
                    bias: None,
                },
                experts: (0..*num_experts)
                    .map(|index| Expert {
                        gate: identity_linear(&format!("expert_{index}_gate"), 2),
                        up: identity_linear(&format!("expert_{index}_up"), 2),
                        down: identity_linear(&format!("expert_{index}_down"), 2),
                    })
                    .collect(),
                top_k: *top_k,
                activation: *activation,
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
                keys: Vec::new(),
                values: Vec::new(),
            }],
            final_norm: WeightData::F32(vec![1.0, 1.0].into()),
            lm_head: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
        }
    }

    #[test]
    fn moe_router_selects_top_k_and_renormalizes_weights() {
        let selected = select_experts(&[0.0, 1.0, 2.0, -3.0], 2).unwrap();
        assert_eq!(
            selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [2, 1]
        );
        let total: f32 = selected.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(selected[0].1 > selected[1].1);
    }

    #[test]
    fn moe_router_rejects_invalid_scores_and_k() {
        assert!(select_experts(&[], 1).is_err());
        assert!(select_experts(&[1.0], 0).is_err());
        assert!(select_experts(&[1.0], 2).is_err());
        assert!(select_experts(&[f32::NAN], 1).is_err());
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
            num_experts: 8,
            top_k: 2,
            activation: Activation::Silu,
            router: "moe.router",
            experts_prefix: "moe.experts",
            gate: "gate",
            up: "up",
            down: "down",
        };
        let config = VulkanTransformerConfig::from_raw(&raw, features.clone()).unwrap();
        assert!(matches!(
            config.features.feed_forward,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_experts: 8,
                top_k: 2,
                ..
            }
        ));

        features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
            num_experts: 2,
            top_k: 3,
            activation: Activation::Silu,
            router: "moe.router",
            experts_prefix: "moe.experts",
            gate: "gate",
            up: "up",
            down: "down",
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
            num_experts: 1,
            top_k: 1,
            activation: Activation::Silu,
            router: "router",
            experts_prefix: "experts",
            gate: "gate",
            up: "up",
            down: "down",
        };
        let config = VulkanTransformerConfig::from_raw(&raw, features).unwrap();
        let mut weights = HashMap::new();
        for (name, shape, values) in [
            ("layers.0.router.weight", vec![1, 2], vec![1.0, 0.0]),
            ("layers.0.experts.0.gate.weight", vec![2, 2], vec![1.0; 4]),
            ("layers.0.experts.0.up.weight", vec![2, 2], vec![1.0; 4]),
            ("layers.0.experts.0.down.weight", vec![2, 2], vec![1.0; 4]),
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
            num_experts,
            top_k,
            activation,
            router,
            experts_prefix,
            gate,
            up,
            down,
        } = &config.features.feed_forward
        else {
            unreachable!();
        };
        let feed_forward = load_moe_feed_forward(
            &mut weights,
            "layers.0",
            &config,
            *num_experts,
            *top_k,
            *activation,
            router,
            experts_prefix,
            gate,
            up,
            down,
        )
        .unwrap();
        assert!(weights.is_empty());
        assert!(matches!(
            feed_forward,
            FeedForward::MixtureOfExperts {
                ref router,
                ref experts,
                top_k: 1,
                ..
            } if router.name == "layers.0.router"
                && experts[0].gate.name == "layers.0.experts.0.gate"
                && experts[0].up.name == "layers.0.experts.0.up"
                && experts[0].down.name == "layers.0.experts.0.down"
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
        let device = Device::Cpu;
        let feed_forward_features = [
            TransformerFeedForwardFeatures::Dense,
            TransformerFeedForwardFeatures::MixtureOfExperts {
                num_experts: 2,
                top_k: 1,
                activation: Activation::Silu,
                router: "moe.router",
                experts_prefix: "moe.experts",
                gate: "gate",
                up: "up",
                down: "down",
            },
        ];
        for features in feed_forward_features {
            let mut model = tiny_executor(features);
            let first = Tensor::new(&[[0u32]], &device).unwrap();
            let logits = model.forward(&first, 0).unwrap();
            assert_eq!(logits.dims(), [1, 1, 2]);
            assert!(
                logits
                    .to_vec3::<f32>()
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .all(f32::is_finite)
            );
            let next = Tensor::new(&[[1u32]], &device).unwrap();
            model.forward(&next, 1).unwrap();
            model.clear_cache();
            let repeated = model.forward(&first, 0).unwrap();
            assert_eq!(
                logits.to_vec3::<f32>().unwrap(),
                repeated.to_vec3::<f32>().unwrap()
            );
        }
        cheetah::native::vulkan::shutdown().unwrap();
    }
}
