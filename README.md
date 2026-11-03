# Cheetah

Cheetah is a local language-model CLI and OpenAI-compatible gateway using Candle for inference and Safetensors checkpoints for model weights. Supported decoder architectures include Qwen2, Gemma, Llama, and Phi; CPU is the default runtime.

## Build

```sh
cargo build --release
```

### CUDA (NVIDIA)

Build with the CUDA inference backend, then select it when starting Cheetah:

```sh
cargo build --release --features cuda
CHEETAH_DEVICE=cuda ./target/release/cheetah /home/user/models/qwen-0.5b
```

### Metal (macOS)

Build with the Metal inference backend, then select it when starting Cheetah:

```sh
cargo build --release --features metal
CHEETAH_DEVICE=metal ./target/release/cheetah /home/user/models/qwen-0.5b
```

The selected backend must be included at build time. Without `CHEETAH_DEVICE`, Cheetah uses the CPU; `CHEETAH_DEVICE=cpu` explicitly selects it.

The Vulkan feature provides a C compute backend and dense Qwen2 inference. It requires a Vulkan runtime, Vulkan headers/loader, and a C compiler. When `glslc` is available, Cargo compiles and embeds the matmul and Qwen2 compute shaders. The default 800 MiB host-visible/coherent buffer pool can be resized with `CHEETAH_VRAM_POOL_SIZE` (bytes). Select it at runtime with:

```sh
cargo build --release --features vulkan
CHEETAH_DEVICE=vulkan ./target/release/cheetah /home/user/models/qwen-0.5b
```

The current inference path supports dense Qwen2 checkpoints with F32, F16, or BF16 weights. Unsupported architectures, unavailable Vulkan devices, shader/pipeline startup errors, and model-load errors emit a diagnostic and fall back to CPU. Errors during an active Vulkan inference are returned rather than silently switching backends. Other architectures continue to use their existing Candle backends.

The C API, including compute-device initialization and matrix multiplication, is also exposed through `cheetah::native::vulkan`. GPU operator tests are opt-in and require a Vulkan compute device plus `glslc`:

```sh
cargo test --features vulkan
cargo test --features vulkan qwen_operators_match_cpu_reference -- --ignored --nocapture
```

Set `CHEETAH_MATMUL_SPV` or `CHEETAH_QWEN_OPS_SPV` to override the corresponding compiled shader path.

## Static model initialization

`cheetah init` prepares a Hugging Face model directory for static, mmap-friendly loading. It recursively scans Safetensors files, writes per-layer weight shards below `.cheetah/weights/`, and emits `cheetah_manifest.json` at the output root. Shard payloads are aligned to 4 KiB; a missing `config.json` is inferred and generated. Tokenizer files are copied when present. The source directory is not modified.

```sh
cargo run -- init /home/user/models/qwen-0.5b /home/benny/models/qwen-0.5b-prepared
cargo run -- inspect /home/user/models/qwen-0.5b-prepared
```

The output directory is generated model data: initialization replaces its unsharded Safetensors weight files and manages stale generated shards. You can also use `cheetah prepare` as an alias for `cheetah init`.

## Chat

Pass a Hugging Face model directory containing `config.json`, `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` plus its shard files:

```sh
cd /home/benny/models/qwen-0.5b && cheetah
cargo run -- /home/user/models/qwen-0.5b
cargo run -- /home/user/models/qwen-0.5b --max-tokens 128 --temperature 0
```

When Cheetah is launched without arguments, it loads the model from the current directory and starts an interactive chat.

The CLI also accepts a direct `.safetensors` path when its config and tokenizer files are in the same directory. Chat supports multi-turn conversations, `/exit`, `/clear`, `/save <name>`, and named context restore with `--context <name>`. Context archives are stored under the platform's Cheetah application-data directory; override it with `--context-dir <path>`.

Sampling options include `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, and `--seed`. `--threads` controls the CPU thread pool and `--ctx-size` bounds the context window.

## Model inspection

```sh
cargo run -- inspect /home/user/models/qwen-0.5b
cargo run -- inspect /home/user/models/qwen-0.5b/model.safetensors --json --output report.json
cargo run -- inspect /home/user/models/qwen-0.5b-prepared
```

## OpenAI-compatible server

```sh
cargo run -- serve /home/benny/models/qwen-0.5b --host 127.0.0.1 --port 8080
```

The server exposes `POST /v1/chat/completions` (including SSE streaming), non-streaming `POST /v1/completions`, `GET /v1/models`, `GET /v1/models/:id`, and `GET /health`. Unauthenticated non-loopback listeners are rejected; use `--api-key` or `--generate-key` when binding to a non-loopback address.
