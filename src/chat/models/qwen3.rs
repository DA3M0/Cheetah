use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use serde_json::Value;
#[cfg(feature = "vulkan")]
use std::path::PathBuf;
use std::sync::Arc;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["Qwen3ForCausalLM", "qwen3"],
    chat_template: ChatTemplate {
        id: "qwen",
        render: super::templates::qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    loader: load,
    vulkan_loader: vulkan_loader(),
    shaders: &["shared_ops"],
};

#[derive(Clone)]
struct Runtime {
    model: Model,
    device: Device,
}

impl ModelRuntime for Runtime {
    fn forward(
        &mut self,
        input: &Tensor,
        offset: usize,
    ) -> Result<Tensor, String> {
        self.model
            .forward(input, offset)
            .map_err(|error| error.to_string())
    }

    fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn device_name(&self) -> &'static str {
        match &self.device {
            Device::Cpu => "CPU",
            Device::Cuda(_) => "CUDA",
            Device::Metal(_) => "Metal",
        }
    }
}

#[derive(Clone, Deserialize)]
struct Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    hidden_act: Activation,
    #[serde(default)]
    use_sliding_window: bool,
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

impl Config {
    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    fn validate(&self) -> Result<(), String> {
        let head_dim = self.head_dim();
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads.max(1) != 0
            || head_dim == 0
            || head_dim % 2 != 0
            || self.max_position_embeddings == 0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err("Qwen3 model dimensions or normalization settings are inconsistent.".into());
        }
        if self.use_sliding_window {
            return Err("Qwen3 sliding-window attention is not supported.".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
}

impl RmsNorm {
    fn new(size: usize, epsilon: f64, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        Ok(Self {
            inner: candle_nn::rms_norm(size, epsilon, vb)?,
        })
    }
}

impl Module for RmsNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(input)
    }
}

#[derive(Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device, dtype: DType) -> candle_core::Result<Self> {
        let dim = config.head_dim();
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|index| 1.0 / config.rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect::<Vec<_>>();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), device)?;
        let positions =
            Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
                .to_dtype(DType::F32)?
                .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        let frequencies = Tensor::cat(&[&frequencies, &frequencies], candle_core::D::Minus1)?;
        Ok(Self {
            sin: frequencies.sin()?.to_dtype(dtype)?,
            cos: frequencies.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(
        &self,
        query: &Tensor,
        keys: &Tensor,
        offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let sequence = query.dim(2)?;
        let cos = self.cos.narrow(0, offset, sequence)?;
        let sin = self.sin.narrow(0, offset, sequence)?;
        let rotate = |input: &Tensor| -> candle_core::Result<Tensor> {
            let dim = input.dim(candle_core::D::Minus1)?;
            let first = input.narrow(candle_core::D::Minus1, 0, dim / 2)?;
            let second = input.narrow(candle_core::D::Minus1, dim / 2, dim / 2)?;
            let rotated = Tensor::cat(&[&second.neg()?, &first], candle_core::D::Minus1)?;
            let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
            let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
            input.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
        };
        Ok((rotate(query)?, rotate(keys)?))
    }
}

#[derive(Clone)]
struct Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        config: &Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let head_dim = config.head_dim();
        let q_proj = candle_nn::linear_b(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_b(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            config.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rotary,
            cache: None,
        })
    }

    fn forward(
        &mut self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        let query = self.q_proj.forward(input)?;
        let keys = self.k_proj.forward(input)?;
        let values = self.v_proj.forward(input)?;
        let query = query
            .reshape((batch, sequence, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let keys = keys
            .reshape((batch, sequence, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let values = values
            .reshape((batch, sequence, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = self
            .q_norm
            .forward(&query.flatten(0, 2)?)?
            .reshape((batch, self.num_heads, sequence, self.head_dim))?;
        let keys = self
            .k_norm
            .forward(&keys.flatten(0, 2)?)?
            .reshape((batch, self.num_kv_heads, sequence, self.head_dim))?;
        let (query, keys) = self.rotary.apply(&query, &keys, offset)?;
        let (keys, values) = match &self.cache {
            Some((cached_keys, cached_values)) => (
                Tensor::cat(&[cached_keys, &keys], 2)?,
                Tensor::cat(&[cached_values, &values], 2)?,
            ),
            None => (keys, values),
        };
        self.cache = Some((keys.clone(), values.clone()));
        let repeat = self.num_heads / self.num_kv_heads;
        let keys = repeat_kv(keys, repeat)?.contiguous()?;
        let values = repeat_kv(values, repeat)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (query.matmul(&keys.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attention_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        probabilities
            .matmul(&values)?
            .transpose(1, 2)?
            .reshape((batch, sequence, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_cache(&mut self) {
        self.cache = None;
    }
}

fn repeat_kv(input: Tensor, repeat: usize) -> candle_core::Result<Tensor> {
    if repeat == 1 {
        return Ok(input);
    }
    let (batch, heads, sequence, head_dim) = input.dims4()?;
    input
        .unsqueeze(2)?
        .expand((batch, heads, repeat, sequence, head_dim))?
        .reshape((batch, heads * repeat, sequence, head_dim))
}

#[derive(Clone)]
struct DecoderLayer {
    attention: Attention,
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    activation: Activation,
    input_norm: RmsNorm,
    post_attention_norm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        config: &Config,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder<'_>,
    ) -> candle_core::Result<Self> {
        let attention = Attention::new(config, rotary, vb.pp("self_attn"))?;
        let gate_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("mlp").pp("gate_proj"),
        )?;
        let up_proj = candle_nn::linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("mlp").pp("up_proj"),
        )?;
        let down_proj = candle_nn::linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("mlp").pp("down_proj"),
        )?;
        Ok(Self {
            attention,
            gate_proj,
            up_proj,
            down_proj,
            activation: config.hidden_act,
            input_norm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_norm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        offset: usize,
    ) -> candle_core::Result<Tensor> {
        let normalized = self.input_norm.forward(input)?;
        let attended = self.attention.forward(&normalized, attention_mask, offset)?;
        let residual = (input + attended)?;
        let normalized = self.post_attention_norm.forward(&residual)?;
        let gate = self.gate_proj.forward(&normalized)?.apply(&self.activation)?;
        let up = self.up_proj.forward(&normalized)?;
        let feed_forward = (gate * up)?.apply(&self.down_proj)?;
        residual + feed_forward
    }

    fn clear_cache(&mut self) {
        self.attention.clear_cache();
    }
}

#[derive(Clone)]
struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: candle_nn::Linear,
    device: Device,
    dtype: DType,
}

impl Model {
    fn new(config: &Config, vb: VarBuilder<'_>) -> candle_core::Result<Self> {
        let embedding =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(config, vb.device(), vb.dtype())?);
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                DecoderLayer::new(config, rotary.clone(), vb.pp("model.layers").pp(index))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let norm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("model.norm"),
        )?;
        let lm_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            vb.pp("lm_head"),
        )?;
        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (batch, sequence) = input.dims2()?;
        let mut hidden = self.embedding.forward(input)?;
        let mask = if sequence > 1 {
            let values = (0..sequence)
                .flat_map(|query| {
                    (0..sequence + offset).map(move |key| {
                        if key <= query + offset {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect::<Vec<_>>();
            Some(
                Tensor::from_slice(&values, (sequence, sequence + offset), &self.device)?
                    .expand((batch, 1, sequence, sequence + offset))?
                    .to_dtype(self.dtype)?,
            )
        } else {
            None
        };
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, mask.as_ref(), offset)?;
        }
        hidden
            .narrow(1, sequence - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }
}

fn load(
    _config_bytes: &[u8],
    raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    if raw_config
        .get("rope_scaling")
        .is_some_and(|scaling| !scaling.is_null())
    {
        return Err("Qwen3 scaled RoPE is not supported.".into());
    }
    let config: Config = serde_json::from_value(raw_config)
        .map_err(|error| format!("Invalid Qwen3 configuration: {error}"))?;
    config.validate()?;
    let model = Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Qwen3 weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}

const fn vulkan_loader() -> Option<super::VulkanModelLoader> {
    #[cfg(feature = "vulkan")]
    {
        Some(load_vulkan)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
}

#[cfg(feature = "vulkan")]
fn load_vulkan(
    weight_paths: &[PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config = vulkan_config(&raw_config)?;
    Ok(Box::new(
        super::super::vulkan_transformer::VulkanTransformerExecutor::load(weight_paths, &config)?,
    ))
}

#[cfg(feature = "vulkan")]
fn vulkan_config(
    raw_config: &Value,
) -> Result<super::super::vulkan_transformer::VulkanTransformerConfig, String> {
    use super::super::vulkan_transformer::{
        TransformerFeatures, VulkanTransformerConfig, required_usize,
    };

    let hidden_size = required_usize(raw_config, "hidden_size")?;
    let num_attention_heads = required_usize(raw_config, "num_attention_heads")?;
    let head_dim = raw_config
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|_| "head_dim does not fit usize.".to_string())?
        .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
    let mut features = TransformerFeatures::standard("Qwen3");
    features.head_dim_override = Some(head_dim);
    features.qk_norm = true;
    features.attention_bias = raw_config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if raw_config
        .get("use_sliding_window")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || raw_config
            .get("rope_scaling")
            .is_some_and(|scaling| !scaling.is_null())
    {
        return Err("Vulkan Qwen3 does not support sliding windows or scaled RoPE.".into());
    }
    VulkanTransformerConfig::from_raw(raw_config, features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn validates_qwen3_dimensions_and_explicit_head_dim() {
        let config: Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128
        }))
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.head_dim(), 8);
    }

    #[test]
    fn rejects_qwen3_sliding_window_and_invalid_attention_dimensions() {
        let mut config: Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 7,
            "max_position_embeddings": 128
        }))
        .unwrap();
        assert!(config.validate().is_err());
        config.use_sliding_window = true;
        config.head_dim = Some(8);
        assert!(config.validate().is_err());
    }

    #[test]
    #[ignore = "Candle CPU execution initializes Rayon before the CLI thread-pool test"]
    fn candle_model_loads_qwen3_weights_and_resets_its_cache() {
        let device = Device::Cpu;
        let config: Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 16,
            "hidden_act": "silu"
        }))
        .unwrap();
        config.validate().unwrap();

        let mut weights = HashMap::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", (8, 4)),
            ("model.layers.0.self_attn.q_proj.weight", (4, 4)),
            ("model.layers.0.self_attn.k_proj.weight", (4, 4)),
            ("model.layers.0.self_attn.v_proj.weight", (4, 4)),
            ("model.layers.0.self_attn.o_proj.weight", (4, 4)),
            ("model.layers.0.mlp.gate_proj.weight", (8, 4)),
            ("model.layers.0.mlp.up_proj.weight", (8, 4)),
            ("model.layers.0.mlp.down_proj.weight", (4, 8)),
            ("lm_head.weight", (8, 4)),
        ] {
            weights.insert(
                name.to_owned(),
                Tensor::zeros(shape, DType::F32, &device).unwrap(),
            );
        }
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.norm.weight",
        ] {
            weights.insert(
                name.to_owned(),
                Tensor::ones(4, DType::F32, &device).unwrap(),
            );
        }

        let builder = VarBuilder::from_tensors(weights, DType::F32, &device);
        let mut model = Model::new(&config, builder).unwrap();
        let prompt = Tensor::from_vec(vec![1u32, 2], (1, 2), &device).unwrap();
        assert_eq!(model.forward(&prompt, 0).unwrap().dims3().unwrap(), (1, 1, 8));

        let next_token = Tensor::from_vec(vec![3u32], (1, 1), &device).unwrap();
        assert_eq!(
            model.forward(&next_token, 2).unwrap().dims3().unwrap(),
            (1, 1, 8)
        );
        model.clear_kv_cache();
        assert_eq!(
            model.forward(&next_token, 0).unwrap().dims3().unwrap(),
            (1, 1, 8)
        );
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_tests {
    use super::*;

    #[test]
    fn qwen3_vulkan_features_use_head_dim_and_qk_norms() {
        let config = vulkan_config(&serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "attention_bias": false,
            "hidden_act": "silu",
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true
        }))
        .unwrap();
        assert_eq!(config.features.head_dim_override, Some(8));
        assert!(config.features.qk_norm);
        assert!(!config.features.attention_bias);
        assert!(config.features.tie_word_embeddings);
    }

    #[test]
    fn qwen3_vulkan_rejects_scaled_rope() {
        assert!(
            vulkan_config(&serde_json::json!({
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 128,
                "rope_scaling": {"type": "linear", "factor": 2.0}
            }))
            .is_err()
        );
    }
}
