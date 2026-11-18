//! The phi executor: a different block topology (biased LayerNorms,
//! parallel residual, fused projections, partial rotary) with its own config
//! surface and forward graph. Weight loading and the operator dispatch live
//! in `super::weights` (the precision registry).

use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::color::{self, Color};

use super::required_usize;
use super::transformer::Activation;
use super::weights::{Linear, NamedWeight, WeightData, load_weights, take_linear, take_weight};

#[derive(Clone, Debug)]
#[allow(dead_code)]
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
    #[allow(dead_code)]
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

#[derive(Clone)]
#[allow(dead_code)]
struct PhiNorm {
    weight: WeightData,
    bias: WeightData,
}

impl PhiNorm {
    #[allow(dead_code)]
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

#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) struct VulkanPhiExecutor {
    config: VulkanPhiConfig,
    embedding: WeightData,
    layers: Vec<PhiLayer>,
    final_norm: PhiNorm,
    lm_head: Linear,
}

impl crate::chat::models::ModelRuntime for VulkanPhiExecutor {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        VulkanPhiExecutor::forward(self, tokens, offset)
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
    #[allow(dead_code)]
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
                &BTreeMap::new(),
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
                slot: None,
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
                    &BTreeMap::new(),
                    &format!("{attention}.q_proj"),
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                    true,
                )?,
                k_proj: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.k_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                v_proj: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.v_proj"),
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    true,
                )?,
                dense: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{attention}.dense"),
                    config.num_attention_heads * config.head_dim,
                    config.hidden_size,
                    true,
                )?,
                fc1: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
                    &format!("{prefix}.mlp.fc1"),
                    config.hidden_size,
                    config.intermediate_size,
                    true,
                )?,
                fc2: take_linear(
                    &mut weights,
                    &BTreeMap::new(),
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

    #[allow(dead_code)]
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        let sequence = tokens.len();
        if sequence == 0 {
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
        let hidden = self.config.hidden_size;
        let mut activations = Vec::with_capacity(sequence * hidden);
        for token_id in tokens {
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
        Ok(logits)
    }
}
