use crate::color::{self, Color};
use candle_core::{Device, Tensor};
use candle_transformers::models::qwen2;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
                "Vulkan Qwen2 currently supports F32, F16, or BF16 weights; tensor '{name}' uses {dtype:?}."
            )),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) | Self::BF16(values) => values.len(),
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
        if rows == 0 || input_size == 0 || output_size == 0 {
            return Err("Vulkan Qwen2 linear dimensions must be positive.".into());
        }
        let expected_input = rows
            .checked_mul(input_size)
            .ok_or_else(|| "Vulkan Qwen2 input size overflow.".to_string())?;
        if input.len() != expected_input {
            return Err("Vulkan Qwen2 linear input shape is inconsistent.".into());
        }
        const MAX_WEIGHT_TILE_ELEMENTS: usize = 16 * 1024 * 1024;
        const MAX_ROW_TILE: usize = 64;
        let tile_size = output_size
            .min(16_384)
            .min((MAX_WEIGHT_TILE_ELEMENTS / input_size).max(1));
        let output_len = rows
            .checked_mul(output_size)
            .ok_or_else(|| "Vulkan Qwen2 output size overflow.".to_string())?;
        let mut output = vec![0.0; output_len];
        for row_offset in (0..rows).step_by(MAX_ROW_TILE) {
            let tile_rows = (rows - row_offset).min(MAX_ROW_TILE);
            let input_start = row_offset * input_size;
            let input_end = input_start + tile_rows * input_size;
            let input_tile = &input[input_start..input_end];
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
        if input.len() != expected {
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
struct Layer {
    input_norm: WeightData,
    post_attention_norm: WeightData,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct VulkanQwenExecutor {
    config: qwen2::Config,
    embedding: WeightData,
    layers: Vec<Layer>,
    final_norm: WeightData,
    lm_head: WeightData,
}

impl super::models::ModelRuntime for VulkanQwenExecutor {
    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        VulkanQwenExecutor::forward(self, input, offset)
    }

    fn clear_cache(&mut self) {
        VulkanQwenExecutor::clear_cache(self);
    }

    fn device_name(&self) -> &'static str {
        "Vulkan"
    }
}

impl VulkanQwenExecutor {
    pub(crate) fn load(paths: &[PathBuf], config: &qwen2::Config) -> Result<Self, String> {
        let result = (|| {
            cheetah::native::vulkan::initialize()?;
            cheetah::native::vulkan::prepare()?;
            Self::load_initialized(paths, config)
        })();
        if result.is_err() {
            if let Err(shutdown_error) = cheetah::native::vulkan::shutdown() {
                eprintln!(
                    "{}",
                    color::stderr(
                        &format!(
                            "cheetah: Vulkan cleanup after startup failure also failed ({shutdown_error})."
                        ),
                        Color::Red
                    )
                );
            }
        }
        result
    }

    fn load_initialized(paths: &[PathBuf], config: &qwen2::Config) -> Result<Self, String> {
        if config.hidden_size == 0
            || config.intermediate_size == 0
            || config.vocab_size == 0
            || config.num_hidden_layers == 0
            || config.max_position_embeddings == 0
            || config.num_attention_heads == 0
            || config.num_key_value_heads == 0
            || config.hidden_size % config.num_attention_heads != 0
            || config.num_attention_heads % config.num_key_value_heads != 0
        {
            return Err("Qwen2 attention dimensions are inconsistent.".into());
        }
        let mut weights = load_weights(paths)?;
        let embedding = take_weight(
            &mut weights,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let embedding_data = embedding.data.clone();
        let final_norm =
            take_weight(&mut weights, "model.norm.weight", &[config.hidden_size])?.data;
        let lm_head = if let Some(head) = weights.remove("lm_head.weight") {
            validate_shape(
                "lm_head.weight",
                &head.shape,
                &[config.vocab_size, config.hidden_size],
            )?;
            head.data
        } else if config.tie_word_embeddings {
            embedding_data.clone()
        } else {
            return Err("Qwen2 checkpoint is missing lm_head.weight.".into());
        };
        let head_dim = config.hidden_size / config.num_attention_heads;
        let kv_size = config.num_key_value_heads * head_dim;
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
            layers.push(Layer {
                input_norm,
                post_attention_norm,
                q_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.self_attn.q_proj"),
                    config.hidden_size,
                    config.hidden_size,
                    true,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.self_attn.k_proj"),
                    config.hidden_size,
                    kv_size,
                    true,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.self_attn.v_proj"),
                    config.hidden_size,
                    kv_size,
                    true,
                )?,
                o_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.self_attn.o_proj"),
                    config.hidden_size,
                    config.hidden_size,
                    false,
                )?,
                gate_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.mlp.gate_proj"),
                    config.hidden_size,
                    config.intermediate_size,
                    false,
                )?,
                up_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.mlp.up_proj"),
                    config.hidden_size,
                    config.intermediate_size,
                    false,
                )?,
                down_proj: take_linear(
                    &mut weights,
                    &format!("{prefix}.mlp.down_proj"),
                    config.intermediate_size,
                    config.hidden_size,
                    false,
                )?,
                keys: Vec::new(),
                values: Vec::new(),
            });
        }
        Ok(Self {
            config: config.clone(),
            embedding: embedding_data,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub(super) fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.keys.clear();
            layer.values.clear();
        }
    }

    pub(super) fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor, String> {
        let (batch, sequence) = input
            .dims2()
            .map_err(|error| format!("Invalid Vulkan Qwen2 input: {error}"))?;
        if batch != 1 || sequence == 0 {
            return Err("Vulkan Qwen2 currently requires a non-empty batch of size one.".into());
        }
        let last_position = offset
            .checked_add(sequence)
            .ok_or_else(|| "Vulkan Qwen2 sequence position overflow.".to_string())?;
        if last_position > self.config.max_position_embeddings {
            return Err("Vulkan Qwen2 input exceeds max_position_embeddings.".into());
        }
        if self.layers.first().is_some_and(|layer| {
            layer.keys.len()
                != offset
                    * self.config.num_key_value_heads
                    * (self.config.hidden_size / self.config.num_attention_heads)
        }) {
            return Err("Vulkan Qwen2 KV cache position does not match the input offset.".into());
        }
        let tokens = input
            .to_vec2::<u32>()
            .map_err(|error| format!("Cannot read Vulkan Qwen2 token IDs: {error}"))?;
        let hidden = self.config.hidden_size;
        let mut activations = Vec::with_capacity(sequence * hidden);
        for token in &tokens[0] {
            let token = *token as usize;
            if token >= self.config.vocab_size {
                return Err(format!("Token ID {token} exceeds the Qwen2 vocabulary."));
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
                self.config.rms_norm_eps as f32,
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
                self.config.rope_theta as f32,
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
                self.config.rms_norm_eps as f32,
            )?;
            let mut gate = layer.gate_proj.forward(&normalized, sequence)?;
            let up = layer.up_proj.forward(&normalized, sequence)?;
            gate = cheetah::native::vulkan::silu_mul(&gate, &up)?;
            let feed_forward = layer.down_proj.forward(&gate, sequence)?;
            activations = cheetah::native::vulkan::add(&residual, &feed_forward)?;
        }

        let last = &activations[(sequence - 1) * hidden..sequence * hidden];
        let normalized =
            self.final_norm
                .rms_norm(last, 1, hidden, self.config.rms_norm_eps as f32)?;
        let logits = self
            .lm_head
            .linear(&normalized, 1, hidden, self.config.vocab_size)?;
        Tensor::from_vec(logits, (1, 1, self.config.vocab_size), &Device::Cpu)
            .map_err(|error| format!("Cannot create Vulkan Qwen2 logits: {error}"))
    }
}

fn load_weights(paths: &[PathBuf]) -> Result<HashMap<String, NamedWeight>, String> {
    let mut weights = HashMap::new();
    for path in paths {
        load_weight_file(path, &mut weights)?;
    }
    Ok(weights)
}

fn load_weight_file(path: &Path, weights: &mut HashMap<String, NamedWeight>) -> Result<(), String> {
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
    let safetensors = SafeTensors::deserialize(&mapping)
        .map_err(|error| format!("Invalid Vulkan model shard '{}': {error}", path.display()))?;
    for name in safetensors.names() {
        if name.starts_with("__cheetah_padding_") {
            continue;
        }
        let view = safetensors.tensor(name).map_err(|error| {
            format!(
                "Cannot read tensor '{name}' from '{}': {error}",
                path.display()
            )
        })?;
        let shape = view.shape().to_vec();
        let data = WeightData::from_view(name, &view)?;
        if weights
            .insert(name.to_string(), NamedWeight { shape, data })
            .is_some()
        {
            return Err(format!("Duplicate tensor '{name}' across model shards."));
        }
    }
    Ok(())
}

fn take_weight(
    weights: &mut HashMap<String, NamedWeight>,
    name: &str,
    expected_shape: &[usize],
) -> Result<NamedWeight, String> {
    let weight = weights
        .remove(name)
        .ok_or_else(|| format!("Vulkan Qwen2 checkpoint is missing tensor '{name}'."))?;
    validate_shape(name, &weight.shape, expected_shape)?;
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
    require_bias: bool,
) -> Result<Linear, String> {
    let weight_name = format!("{prefix}.weight");
    let weight = take_weight(weights, &weight_name, &[output_size, input_size])?;
    let bias_name = format!("{prefix}.bias");
    let bias = match weights.remove(&bias_name) {
        Some(bias) => {
            validate_shape(&bias_name, &bias.shape, &[output_size])?;
            Some(bias.data)
        }
        None if require_bias => {
            return Err(format!(
                "Vulkan Qwen2 checkpoint is missing tensor '{bias_name}'."
            ));
        }
        None => None,
    };
    Ok(Linear {
        name: prefix.to_string(),
        input_size,
        output_size,
        weight: weight.data,
        bias,
    })
}

#[cfg(test)]
fn apply_rotary_row(
    values: &mut [f32],
    start: usize,
    position: usize,
    head_dim: usize,
    half_dim: usize,
    theta: f32,
) {
    for index in 0..half_dim {
        let frequency = 1.0 / theta.powf((2 * index) as f32 / head_dim as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        let first = values[start + index];
        let second = values[start + half_dim + index];
        values[start + index] = first * cos - second * sin;
        values[start + half_dim + index] = second * cos + first * sin;
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn grouped_query_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    sequence: usize,
    offset: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
) -> Result<Vec<f32>, String> {
    let cache_len = keys.len() / (kv_heads * head_dim);
    if values.len() != keys.len() || cache_len < offset + sequence {
        return Err("Invalid Qwen2 attention cache dimensions.".into());
    }
    let mut output = vec![0.0; sequence * query_heads * head_dim];
    let groups = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    for token_index in 0..sequence {
        let position = offset + token_index;
        let end = position + 1;
        let start = if sliding_window == 0 {
            0
        } else {
            end.saturating_sub(sliding_window)
        };
        let key_count = end - start;
        let mut scores = vec![0.0; key_count];
        for head in 0..query_heads {
            let query_start = (token_index * query_heads + head) * head_dim;
            let kv_head = head / groups;
            for (relative, score) in scores.iter_mut().enumerate() {
                let cache_position = start + relative;
                let key_start = (cache_position * kv_heads + kv_head) * head_dim;
                let dot = (0..head_dim)
                    .map(|dim| query[query_start + dim] * keys[key_start + dim])
                    .sum::<f32>();
                *score = dot * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                denominator += *score;
            }
            let output_start = query_start;
            for dim in 0..head_dim {
                let mut sum = 0.0;
                for (relative, score) in scores.iter().enumerate() {
                    let cache_position = start + relative;
                    let value_start = (cache_position * kv_heads + kv_head) * head_dim;
                    sum += (score / denominator) * values[value_start + dim];
                }
                output[output_start + dim] = sum;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn linear_splits_large_sequences_into_bounded_row_tiles() {
        cheetah::native::vulkan::initialize().unwrap();
        let weight = WeightData::F32(vec![2.0, -1.0].into());
        let output = weight.linear(&vec![3.0; 65], 65, 1, 2).unwrap();
        cheetah::native::vulkan::shutdown().unwrap();
        assert_eq!(output.len(), 130);
        for row in output.chunks_exact(2) {
            assert_eq!(row, &[6.0, -3.0]);
        }
    }

    #[test]
    fn rotary_embedding_matches_split_half_rotation() {
        let mut values = [1.0, 0.0, 0.0, 1.0];
        apply_rotary_row(&mut values, 0, 1, 4, 2, 1.0);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((values[0] - cos).abs() < 1e-6);
        assert!((values[1] + sin).abs() < 1e-6);
        assert!((values[2] - sin).abs() < 1e-6);
        assert!((values[3] - cos).abs() < 1e-6);
    }

    #[test]
    fn grouped_query_attention_uses_causal_kv_positions() {
        let query = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [2.0, 3.0, 5.0, 7.0];
        let output = grouped_query_attention(&query, &keys, &values, 2, 0, 2, 1, 2, 16).unwrap();
        assert_eq!(&output[0..2], &[2.0, 3.0]);
        assert_eq!(&output[2..4], &[2.0, 3.0]);
        let second_weight = 1.0f32 / (1.0 + (-std::f32::consts::FRAC_1_SQRT_2).exp());
        let first_weight = 1.0 - second_weight;
        assert!((output[4] - (first_weight * 2.0 + second_weight * 5.0)).abs() < 1e-6);
        assert!((output[5] - (first_weight * 3.0 + second_weight * 7.0)).abs() < 1e-6);
    }
}
