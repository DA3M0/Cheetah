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
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    sliding_window: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
}

impl VulkanTransformerConfig {
    pub(crate) fn mistral(raw: &Value) -> Result<Self, String> {
        let config = Self::common(raw)?;
        if config.num_local_experts != 0 || config.num_experts_per_tok != 0 {
            return Err("Mistral config must not define MoE experts.".into());
        }
        Ok(config)
    }

    pub(crate) fn mixtral(raw: &Value) -> Result<Self, String> {
        let config = Self::common(raw)?;
        if config.num_local_experts == 0
            || config.num_experts_per_tok == 0
            || config.num_experts_per_tok > config.num_local_experts
        {
            return Err("Mixtral expert counts are inconsistent.".into());
        }
        Ok(config)
    }

    fn common(raw: &Value) -> Result<Self, String> {
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
        let rms_norm_eps = raw
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5) as f32;
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
        let num_local_experts = raw
            .get("num_local_experts")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "num_local_experts does not fit usize.".to_string())?
            .unwrap_or(0);
        let num_experts_per_tok = raw
            .get("num_experts_per_tok")
            .and_then(Value::as_u64)
            .map(usize::try_from)
            .transpose()
            .map_err(|_| "num_experts_per_tok does not fit usize.".to_string())?
            .unwrap_or(0);
        if raw.get("hidden_act").and_then(Value::as_str) != Some("silu") {
            return Err("Vulkan Mistral/Mixtral supports only the SiLU activation.".into());
        }
        if vocab_size == 0
            || hidden_size == 0
            || intermediate_size == 0
            || num_hidden_layers == 0
            || num_attention_heads == 0
            || num_key_value_heads == 0
            || max_position_embeddings == 0
            || hidden_size % num_attention_heads != 0
            || num_attention_heads % num_key_value_heads != 0
            || !rms_norm_eps.is_finite()
            || rms_norm_eps <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err("Mistral/Mixtral model dimensions are inconsistent.".into());
        }
        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            sliding_window,
            num_local_experts,
            num_experts_per_tok,
        })
    }
}

fn required_usize(config: &Value, name: &str) -> Result<usize, String> {
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
                "Vulkan Mistral/Mixtral supports F32, F16, or BF16 weights; tensor '{name}' uses {dtype:?}."
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
        self.weight
            .linear(input, rows, self.input_size, self.output_size)
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
    },
    Sparse {
        router: Linear,
        experts: Vec<Expert>,
        top_k: usize,
    },
}

impl FeedForward {
    fn forward(&self, input: &[f32], rows: usize, hidden_size: usize) -> Result<Vec<f32>, String> {
        match self {
            Self::Dense { gate, up, down } => {
                let gate = gate.forward(input, rows)?;
                let up = up.forward(input, rows)?;
                let activated = cheetah::native::vulkan::silu_mul(&gate, &up)?;
                down.forward(&activated, rows)
            }
            Self::Sparse {
                router,
                experts,
                top_k,
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
                    let activated = cheetah::native::vulkan::silu_mul(&gate, &up)?;
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

fn select_experts(logits: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>, String> {
    if logits.is_empty() || top_k == 0 || top_k > logits.len() {
        return Err("Invalid Mixtral router logits or expert count.".into());
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err("Mixtral router produced a non-finite score.".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probabilities = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let denominator: f32 = probabilities.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("Mixtral router probability normalization failed.".into());
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
        return Err("Mixtral selected expert weights could not be normalized.".into());
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
    pub(crate) fn load_mistral(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
    ) -> Result<Self, String> {
        Self::load(paths, config, false)
    }

    pub(crate) fn load_mixtral(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
    ) -> Result<Self, String> {
        Self::load(paths, config, true)
    }

    fn load(
        paths: &[PathBuf],
        config: &VulkanTransformerConfig,
        sparse: bool,
    ) -> Result<Self, String> {
        let result = (|| {
            cheetah::native::vulkan::initialize()?;
            cheetah::native::vulkan::prepare()?;
            Self::load_initialized(paths, config, sparse)
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
        sparse: bool,
    ) -> Result<Self, String> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_size = config.num_key_value_heads * head_dim;
        let mut weights = load_weights(paths)?;
        let embedding = take_weight(
            &mut weights,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?
        .data;
        let final_norm =
            take_weight(&mut weights, "model.norm.weight", &[config.hidden_size])?.data;
        let lm_head = take_weight(
            &mut weights,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?
        .data;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            let input_norm = take_weight(
                &mut weights,
                &format!("{prefix}.input_layernorm.weight"),
                &[config.hidden_size],
            )?
            .data;
            let post_attention_norm = take_weight(
                &mut weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[config.hidden_size],
            )?
            .data;
            let attention = format!("{prefix}.self_attn");
            let feed_forward = if sparse {
                load_sparse_feed_forward(&mut weights, &prefix, config)?
            } else {
                FeedForward::Dense {
                    gate: take_linear(
                        &mut weights,
                        &format!("{prefix}.mlp.gate_proj"),
                        config.hidden_size,
                        config.intermediate_size,
                    )?,
                    up: take_linear(
                        &mut weights,
                        &format!("{prefix}.mlp.up_proj"),
                        config.hidden_size,
                        config.intermediate_size,
                    )?,
                    down: take_linear(
                        &mut weights,
                        &format!("{prefix}.mlp.down_proj"),
                        config.intermediate_size,
                        config.hidden_size,
                    )?,
                }
            };
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                q_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.q_proj"),
                    config.hidden_size,
                    config.hidden_size,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.k_proj"),
                    config.hidden_size,
                    kv_size,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.v_proj"),
                    config.hidden_size,
                    kv_size,
                )?,
                o_proj: take_linear(
                    &mut weights,
                    &format!("{attention}.o_proj"),
                    config.hidden_size,
                    config.hidden_size,
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
            return Err("Vulkan Mistral/Mixtral requires a non-empty batch of size one.".into());
        }
        let last_position = offset
            .checked_add(sequence)
            .ok_or_else(|| "Vulkan transformer sequence position overflow.".to_string())?;
        if last_position > self.config.max_position_embeddings {
            return Err("Vulkan transformer input exceeds max_position_embeddings.".into());
        }
        let expected_cache = offset
            .checked_mul(self.config.num_key_value_heads)
            .and_then(|count| {
                count.checked_mul(self.config.hidden_size / self.config.num_attention_heads)
            })
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
                activations.push(self.embedding.value(start + index));
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
            (query, keys) = cheetah::native::vulkan::rotary(
                &query,
                &keys,
                sequence,
                offset,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                hidden / self.config.num_attention_heads,
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
                hidden / self.config.num_attention_heads,
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

fn load_sparse_feed_forward(
    weights: &mut HashMap<String, NamedWeight>,
    prefix: &str,
    config: &VulkanTransformerConfig,
) -> Result<FeedForward, String> {
    let moe_prefix = format!("{prefix}.block_sparse_moe");
    let router = take_linear(
        weights,
        &format!("{moe_prefix}.gate"),
        config.hidden_size,
        config.num_local_experts,
    )?;
    let mut experts = Vec::with_capacity(config.num_local_experts);
    for index in 0..config.num_local_experts {
        let expert_prefix = format!("{moe_prefix}.experts.{index}");
        experts.push(Expert {
            gate: take_linear(
                weights,
                &format!("{expert_prefix}.w1"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            up: take_linear(
                weights,
                &format!("{expert_prefix}.w3"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            down: take_linear(
                weights,
                &format!("{expert_prefix}.w2"),
                config.intermediate_size,
                config.hidden_size,
            )?,
        });
    }
    Ok(FeedForward::Sparse {
        router,
        experts,
        top_k: config.num_experts_per_tok,
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

fn take_linear(
    weights: &mut HashMap<String, NamedWeight>,
    prefix: &str,
    input_size: usize,
    output_size: usize,
) -> Result<Linear, String> {
    let weight_name = format!("{prefix}.weight");
    let weight = take_weight(weights, &weight_name, &[output_size, input_size])?;
    Ok(Linear {
        name: prefix.to_owned(),
        input_size,
        output_size,
        weight: weight.data,
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
        }
    }

    fn tiny_executor(sparse: bool) -> VulkanTransformerExecutor {
        let config = VulkanTransformerConfig {
            vocab_size: 2,
            hidden_size: 2,
            intermediate_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            max_position_embeddings: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: 4,
            num_local_experts: usize::from(sparse) * 2,
            num_experts_per_tok: usize::from(sparse),
        };
        let feed_forward = if sparse {
            FeedForward::Sparse {
                router: Linear {
                    name: "router".into(),
                    input_size: 2,
                    output_size: 2,
                    weight: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
                },
                experts: (0..2)
                    .map(|index| Expert {
                        gate: identity_linear(&format!("expert_{index}_gate"), 2),
                        up: identity_linear(&format!("expert_{index}_up"), 2),
                        down: identity_linear(&format!("expert_{index}_down"), 2),
                    })
                    .collect(),
                top_k: 1,
            }
        } else {
            FeedForward::Dense {
                gate: identity_linear("dense_gate", 2),
                up: identity_linear("dense_up", 2),
                down: identity_linear("dense_down", 2),
            }
        };
        VulkanTransformerExecutor {
            config,
            embedding: WeightData::F32(vec![1.0, 0.0, 0.0, 1.0].into()),
            layers: vec![Layer {
                input_norm: WeightData::F32(vec![1.0, 1.0].into()),
                post_attention_norm: WeightData::F32(vec![1.0, 1.0].into()),
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
    fn mixtral_router_selects_top_k_and_renormalizes_weights() {
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
    fn mixtral_router_rejects_invalid_scores_and_k() {
        assert!(select_experts(&[], 1).is_err());
        assert!(select_experts(&[1.0], 0).is_err());
        assert!(select_experts(&[1.0], 2).is_err());
        assert!(select_experts(&[f32::NAN], 1).is_err());
    }

    #[test]
    fn parses_mistral_and_mixtral_runtime_config() {
        let mistral = serde_json::json!({
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "hidden_act": "silu",
            "sliding_window": 4096
        });
        let config = VulkanTransformerConfig::mistral(&mistral).unwrap();
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.sliding_window, 4096);

        let mut mixtral = mistral;
        mixtral["num_local_experts"] = Value::from(8);
        mixtral["num_experts_per_tok"] = Value::from(2);
        assert_eq!(
            VulkanTransformerConfig::mixtral(&mixtral)
                .unwrap()
                .num_local_experts,
            8
        );
        assert!(VulkanTransformerConfig::mistral(&mixtral).is_err());
    }

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn tiny_dense_and_sparse_models_forward_and_reset_cache() {
        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();
        let device = Device::Cpu;
        for sparse in [false, true] {
            let mut model = tiny_executor(sparse);
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
