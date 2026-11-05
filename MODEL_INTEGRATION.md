# Model integration guide

This guide explains how to add a Safetensors decoder architecture to Cheetah's compile-time model registry. It describes Cheetah's integration points, which differ from projects that load arbitrary plugins or infer backend support from a shader file.

## How registration works

At build time, `build.rs` scans `src/chat/models/` for Rust source files and `.comp` compute shaders:

- Every `.rs` file in that directory is treated as a model module and must define a `REGISTRATION`. The shared files `mod.rs`, `candle.rs`, `registry.rs`, and `templates.rs` are excluded.
- Every `.comp` file is discovered as a shader named after its filename stem. Under `--features vulkan`, discovered shaders are compiled with `glslc`.
- Generated Rust code declares each model module and combines its registration into the static registry. No central model list needs editing.
- Adding, removing, or changing files triggers a Cargo rebuild.

This is compile-time discovery, not runtime plugin loading. A `.comp` file is only discovered and compiled; it does not implement a Vulkan model or connect itself to inference. A Vulkan backend must be explicitly provided by the registration's loader and executor.

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

## Optional compute shader and Vulkan inference

For a model that uses a compute shader, add an optional paired file `src/chat/models/<name>.comp`. The build discovers the shader under the stem `<name>` and exposes its compiled path through `super::shader_path("<name>")`.

To opt into Vulkan inference, the model registration must provide both:

1. A `vulkan_loader` that initializes and loads a Vulkan executor, returning it as `Box<dyn ModelRuntime>`.
2. The shader stem(s) used by that executor in `shaders`.

The executor must implement the same `ModelRuntime` interface as the Candle model. Follow the Vulkan initialization, resource ownership, tensor validation, and error-handling patterns in `src/chat/vulkan_qwen.rs`; the Candle loader remains necessary for CPU/CUDA/Metal and Vulkan fallback. `vulkan_loader` should be `None` when the model has no Vulkan implementation. Shader presence alone does not activate Vulkan inference.

Qwen2 demonstrates a feature-gated loader in `src/chat/models/qwen2.rs` and its executor in `src/chat/vulkan_qwen.rs`. Mistral and Mixtral use the shared native operator executor in `src/chat/vulkan_transformer.rs`; Mixtral's sparse expert router and top-k dispatch are model-specific. These implementations list the shared `shared_ops` shader registration because they call its native compute operators; shader names do not describe the model architecture. A new executor may be implemented in its registration module or in a shared chat module. If adding a separate module under `src/chat/`, declare it in `src/chat/mod.rs`; do not place a helper `.rs` next to model registrations unless it also defines a complete model registration.

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

Add focused tests for non-obvious model behavior, such as template rendering, architecture aliases, weight/config loading, cache reset, or Vulkan executor behavior. A Vulkan build needs `glslc` when a model shader is present; device-dependent tests additionally require a Vulkan-capable system.

Before submitting, check that:

- The registration includes all architecture aliases, a stable template id, a renderer, stop markers, and a Candle loader.
- The runtime can be cloned and its cache is correctly reset between requests.
- The loader uses the expected config format and weight tensor names.
- A shader filename stem exactly matches the stem passed to `shader_path` and listed in `shaders`.
- Vulkan inference has an executor and loader; merely adding `.comp` is not sufficient.
- The new architecture can be selected by `cheetah init`, `cheetah inspect`, and chat using a representative model directory.

## Current registered decoder architectures

| Model | Candle CPU/CUDA/Metal | Vulkan |
| --- | --- | --- |
| Qwen2 | Yes | Yes |
| Gemma | Yes | No |
| Llama | Yes | No |
| Phi | Yes | No |
| Mistral | Standard 7B configs supported by Candle 0.4.1 | Standard Mistral 7B configs |
| Mixtral | Configs supported by Candle 0.4.1 | Sparse expert routing; F32/F16/BF16 weights |
