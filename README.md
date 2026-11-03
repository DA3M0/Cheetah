# Cheetah

Cheetah is a local language-model CLI and OpenAI-compatible gateway using Candle for inference and Safetensors checkpoints for model weights. Supported decoder architectures include Qwen2, Gemma, Llama, and Phi; CPU is the default runtime.

## Build

```sh
cargo build --release
```

Optional Candle inference backends can be compiled with `--features cuda` or `--features metal`. Set `CHEETAH_DEVICE=cuda` or `CHEETAH_DEVICE=metal` to select a compiled backend at runtime; `CHEETAH_DEVICE=cpu` (the default) selects the CPU.

The separate C Vulkan compute backend is available with `--features vulkan`. It requires a Vulkan SDK/runtime and a C compiler. Its 800 MiB host-visible/coherent memory pool can be resized with `CHEETAH_VRAM_POOL_SIZE` (bytes). The feature exposes the initializer and matrix multiplication through `cheetah::native::vulkan`; Vulkan matrix multiplication also requires a compiled `matmul.spv` shader:

```sh
glslc -fshader-stage=compute matmul.comp -o matmul.spv
cargo test --features vulkan
```

Set `CHEETAH_MATMUL_SPV` to use a shader at another path. The Vulkan compute API is separate from Candle inference device selection.

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
