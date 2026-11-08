# Model integration guide

This guide explains how to add a Safetensors decoder architecture to Cheetah's compile-time model registry. It describes Cheetah's integration points, which differ from projects that load arbitrary plugins or infer backend support from a shader file.

## How registration works

At build time, `build.rs` scans `src/chat/models/` for Rust model modules and optional model-specific `.comp` compute shaders. The shared Vulkan operator shaders (`matmul.comp`, `matmul_bf16.comp`, and `shared_ops.comp`) are maintained at the repository root and built independently of model registrations:

- Every `.rs` file in that directory is treated as a model module and must define a `REGISTRATION`. The shared files `mod.rs`, `candle.rs`, `registry.rs`, and `templates.rs` are excluded.
- A model-specific `.comp` file is optional. If present, it is discovered as a shader named after its filename stem and compiled with `glslc` under `--features vulkan`.
- The root-level shared operator shaders are compiled with `glslc` under `--features vulkan`; models using the shared Vulkan operator library do not need their own `.comp` file.
- Generated Rust code declares each model module and combines its registration into the static registry. No central model list needs editing.
- Adding, removing, or changing files triggers a Cargo rebuild.

This is compile-time discovery, not runtime plugin loading. A model-specific `.comp` file is only discovered and compiled; it does not implement a Vulkan model or connect itself to inference. A Vulkan backend must be explicitly provided by the registration's loader and executor.

## Add a Candle model

Create `src/chat/models/<name>.rs`. Use one module per model registration; do not put unrelated helper `.rs` files in this directory, because each non-excluded file is compiled as a model module.

The following is a structural example. Replace the Candle model type, configuration parsing, architecture aliases, template, and stop markers with values appropriate for the architecture:

```rust
use super::{ChatTemplate, ModelRegistration, ModelRuntime};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi;
use serde_json::Value;

pub(super) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: &["ExampleForCausalLM", "example"],
    chat_template: ChatTemplate {
        id: "example",
        render: render_chat,
        header_renderer: None,
    },
    stop_markers: &["<|end|>"],
    loader: load,
    vulkan_loader: None,
    shaders: &[],
};

#[derive(Clone)]
struct Runtime {
    model: phi::Model,
    device: Device,
}

impl ModelRuntime for Runtime {
    fn forward(&mut self, input: &Tensor, _offset: usize) -> Result<Tensor, String> {
        self.model.forward(input).map_err(|error| error.to_string())
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

fn load(
    config_bytes: &[u8],
    _raw_config: Value,
    builder: VarBuilder<'_>,
    device: &Device,
) -> Result<Box<dyn ModelRuntime>, String> {
    let config: phi::Config = serde_json::from_slice(config_bytes)
        .map_err(|error| format!("Invalid Example configuration: {error}"))?;
    let model = phi::Model::new(&config, builder)
        .map_err(|error| format!("Failed to load Example weights: {error}"))?;
    Ok(Box::new(Runtime {
        model,
        device: device.clone(),
    }))
}

fn render_chat(messages: &[crate::chat::ChatMessage], system: Option<&str>) -> String {
    let mut prompt = String::new();
    if let Some(system) = system {
        prompt.push_str("<|system|>\n");
        prompt.push_str(system);
        prompt.push_str("<|end|>\n");
    }
    for message in messages {
        prompt.push_str("<|");
        prompt.push_str(&message.role);
        prompt.push_str("|>\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|end|>\n");
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
```

The example uses Phi's Candle API to show the expected shape; it is not a generic model adapter. Adapt `forward` and `clear_cache` to the actual model and its KV-cache semantics. `ModelRuntime` must be `Send`; the runtime must be `Clone` because generation clones a model instance. The loader must return an owned `Box<dyn ModelRuntime>` and must not retain borrowed data from `VarBuilder`.

The loader receives the original configuration bytes, its parsed JSON value, a Candle `VarBuilder`, and the selected Candle `Device`. Use `config_bytes` when a Candle config type is best deserialized from the source JSON, or `raw_config` when the architecture needs direct JSON access. Return errors with context instead of hiding configuration, weight, or device failures.

### Architecture aliases

List every supported value that may appear in `config.json` under `architectures` or `model_type`. The registry performs an exact string match; aliases must therefore be explicitly listed. Keep aliases unique across registrations.

### Chat template and stop markers

The template's `id` is the stable name written to `cheetah_manifest.json` during `cheetah init`. `render` converts the conversation and optional system prompt into the exact prompt format expected by the tokenizer and model. Keep this compatible with the model's training template.

`header_renderer` is optional and is intended for a second compatible template variant that can be selected using tokenizer/manifest metadata, as with Llama 2 versus Llama 3. Most models should set it to `None`.

`stop_markers` are used as model defaults and are also written into prepared manifests. Choose the actual end-of-turn strings for the model. Cheetah also recognizes stop token IDs from model configuration and accepts per-request stop values through the API.

## Vulkan inference and optional model-specific shaders

Models built on the shared native Vulkan operators need no per-model `.comp` file. Their registration lists `shared_ops` in `shaders` so backend selection can verify that the shared operator pipeline is available. The shared shader sources are at the repository root; they are not discovered from `src/chat/models/`.

Only add `src/chat/models/<name>.comp` when the model requires an additional architecture-specific compute shader. The build discovers it under the stem `<name>` and exposes its compiled path through `super::shader_path("<name>")`. If the executor uses that shader, list the stem in `shaders`.

To opt into Vulkan inference, provide a `vulkan_loader` that initializes and loads a Vulkan executor, returning it as `Box<dyn ModelRuntime>`. List in `shaders` the shared operator pipeline and any optional model-specific shaders that the executor actually uses. Adding a `.comp` file alone does not activate Vulkan inference.

The executor must implement the same `ModelRuntime` interface as the Candle model. Follow the shared Vulkan initialization, resource ownership, tensor validation, and error-handling patterns in `src/chat/vulkan_transformer.rs`; the Candle loader remains necessary for CPU/CUDA/Metal and Vulkan fallback. `vulkan_loader` should be `None` when the model has no Vulkan implementation. Shader presence alone does not activate Vulkan inference.

Qwen2, Qwen3, Gemma, Llama, Mistral, and Mixtral construct `TransformerFeatures` in their registration modules and use the shared transformer executor in `src/chat/vulkan_transformer.rs`. A registration starts with `TransformerFeatures::standard(...)`, adjusts architecture behavior and tensor-layout fields, then builds a `VulkanTransformerConfig` with `VulkanTransformerConfig::from_raw(&raw_config, features)`. Qwen3 uses its explicit `head_dim` and per-head q/k RMS norms. Dense models use the default dense feed-forward feature; MoE models describe expert count, top-k routing, activation, and router/expert tensor names in `TransformerFeedForwardFeatures::MixtureOfExperts`. The executor's single `load(weight_paths, &config)` entry point loads according to that data, without architecture-specific branches or a separate sparse flag. These registrations list the shared `shared_ops` pipeline because they call the common native compute operators; they do not need individual `.comp` files. A new executor may be implemented in its registration module or in a shared chat module. If adding a separate module under `src/chat/`, declare it in `src/chat/mod.rs`; do not place a helper `.rs` next to model registrations unless it also defines a complete model registration.

Build and select Vulkan explicitly:

```sh
cargo build --features vulkan
CHEETAH_DEVICE=vulkan cargo run -- /path/to/model
```

When Vulkan support is not built, or the registered Vulkan loader fails to initialize/load, Cheetah reports the issue and loads the Candle model on CPU. Errors during active inference are returned to the caller.

## Initialization metadata

`cheetah init` reads the model configuration, looks up its architecture in the generated registry, and writes the registered `chat_template.id` and `stop_markers` into the prepared manifest after native weight preparation succeeds. No additional init hook is needed. If the architecture is unregistered, initialization leaves chat metadata as produced by the native initializer.

## Verification checklist

After adding a registration:

```sh
cargo fmt --all
cargo test
cargo check --features vulkan
```

Add focused tests for non-obvious model behavior, such as template rendering, architecture aliases, weight/config loading, cache reset, or Vulkan executor behavior. A Vulkan build needs `glslc` to compile the root-level shared operator shaders (and any optional model-specific shaders); device-dependent tests additionally require a Vulkan-capable system.

Before submitting, check that:

- The registration includes all architecture aliases, a stable template id, a renderer, stop markers, and a Candle loader.
- The runtime can be cloned and its cache is correctly reset between requests.
- The loader uses the expected config format and weight tensor names.
- Shared-operator registrations list `shared_ops`; optional model-specific shader stems match both the filename and the value passed to `shader_path`.
- Vulkan inference has an executor and loader; a model-specific `.comp` file is not required and does not activate Vulkan by itself.
- The new architecture can be selected by `cheetah init`, `cheetah inspect`, and chat using a representative model directory.

## Current registered decoder architectures

| Model | Candle CPU/CUDA/Metal | Vulkan |
| --- | --- | --- |
| Qwen2 | Yes | Yes |
| Qwen3 (dense) | Yes | Yes |
| Gemma | Yes | Yes |
| Llama | Yes | Yes |
| Phi | Yes | No |
| Mistral | Standard 7B configs supported by Candle 0.4.1 | Standard Mistral 7B configs; F32/F16/BF16 weights |
| Mixtral | Configs supported by Candle 0.4.1 | Sparse expert routing; F32/F16/BF16 weights |

Qwen3 support covers dense models with standard full RoPE; sliding-window attention, scaled RoPE, and Qwen3 MoE architectures are not included.
