# Cheetah

**Cheetah** is a local language-model CLI and OpenAI-compatible API gateway, written in Rust and C. It runs Safetensors decoder models with a self-describing prepared-model format and **Ant tiered memory** — real page-level VRAM/RAM/Disk control — across three independent backends.

| | |
|---|---|
| 🧠 **Ant tiered memory** | per-tensor VRAM/RAM/Disk residency acting on real pages — 123 MB steady RSS on a 0.5B model (vs 2.5 GB untiered) |
| 🚀 **native CPU backend** | mmap-backed BF16 weights, AVX2 SIMD kernels — up to 8 tok/s on Qwen2-0.5B |
| 🧩 **compile-time registry** | declarative registration files; adding a model family = two files, zero central edits |
| 🔌 **OpenAI-compatible API** | `/v1/chat/completions` with SSE streaming, tool calls, API keys |
| 🛠️ **no GGUF** | Safetensors in, self-describing prepared directory out (`cheetah init`) |

```
                 ┌─────────────────────────────────────────────┐
  CLI / OpenAI ──▶ Engine (chat/serve) ──▶ ModelRuntime boundary ──▶ ┐
                 └─────────────────────────────────────────────┘      │
                                                                      ▼
        ┌──────────────────┬──────────────────────┬──────────────────────────┐
        │ native (CPU)     │ Vulkan executor      │ Candle (CUDA/Metal only) │
        │ mmap + Ant tiers │ Candle-free executor │ classic bridge           │
        └──────────────────┴──────────────────────┴──────────────────────────┘
```

**Newcomer paths**

- 👤 *I just want to run a model*: follow the [beginner tutorial](#beginner-tutorial) — five steps, ~10 minutes.
- 🧑‍💻 *I want to contribute*: [CONTRIBUTING.md](CONTRIBUTING.md) → [docs/CODEBASE.md](docs/CODEBASE.md) (repository map) → pick a task in §3.
- 🔬 *I want to understand the memory system*: [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md).

**Table of contents** — [Beginner tutorial](#beginner-tutorial) · [Build](#build) · [Chat](#chat) · [Model initialization](#model-initialization) · [Ant memory optimization](#ant-memory-optimization) · [Model inspection](#model-inspection) · [API server](#openai-compatible-api-server) · [Supported models](#supported-models-and-limitations) · [Contributing](#contributing)

Human-readable CLI output uses semantic colors on terminals; colors are disabled for redirected output and when `NO_COLOR` is set. Model text and machine-readable JSON/API responses are never colored.

## Beginner tutorial

This section walks a new user from a clean machine to the first generated response.

**Suggested learning path**

| Time | Goal | Section |
|------|------|---------|
| ~10 min | Build and chat with a real model | this tutorial |
| +15 min | Prepare a model with Ant and watch the tiered-memory banner | [Model initialization](#model-initialization) + [Ant memory optimization](#ant-memory-optimization) |
| +30 min | Serve an OpenAI-compatible endpoint and hit it with curl or the Python client | [API server](#openai-compatible-api-server) |
| First PR | Add a native model family or a kernel tweak | [CONTRIBUTING.md](CONTRIBUTING.md) → [docs/CODEBASE.md](docs/CODEBASE.md) |

### 1. Install prerequisites

You need a Rust toolchain (2024 edition or newer), a C compiler and linker, and Git. On Ubuntu or Debian:

```sh
sudo apt update
sudo apt install -y build-essential git
```

Install Rust via [rustup](https://rustup.rs/) and verify with `rustc --version` and `cargo --version`.

### 2. Build Cheetah

```sh
git clone https://github.com/DA3M0/cheetah.git
cd cheetah
cargo build --release
```

The first build compiles dependencies and may take a while. When it finishes, the executable is `target/release/cheetah`.

### 3. Get a model

Cheetah runs Safetensors decoder models. Point it at a Hugging Face model directory containing `config.json`, `tokenizer.json`, and `model.safetensors` (optionally with `model.safetensors.index.json` for sharded weights), or a single Safetensors file. To fetch one, use any HF client — for example a small Qwen2 model (~1 GB):

```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /tmp/qwen-0.5b
cheetah /tmp/qwen-0.5b
```

(No `huggingface-cli`? Download the files directly from the model page, or point Cheetah at any directory you already have with `config.json`, `tokenizer.json`, and Safetensors weights.)

You can also prepare a model directory for mmap-friendly loading and Ant memory optimization:

```sh
cheetah init /path/to/hf-model /path/to/prepared-model
```

GGUF files are not supported; use a Safetensors conversion of the model instead.

### 4. Run a chat

```sh
cheetah /path/to/model
```

Then type a message at the `> ` prompt and press Enter. Use `/exit` to quit, `/clear` to reset the conversation, `/save <name>` to save it, and `/help` to list chat commands. Start with a small model and a small context for quick feedback:

```sh
cheetah /path/to/small-model --ctx-size 1024 --max-tokens 128
```

### 5. Try the OpenAI-compatible server

```sh
cheetah serve /path/to/model --port 8080
```

Then send chat completion requests to the local endpoint.

See the rest of this README for build options, generation parameters, context management, Ant memory optimization, model inspection, and the full API reference.

## Build

### Beginner-friendly build guide

The default build uses the CPU backend and is the easiest way to get started. Build commands below are run from a terminal in the repository directory.

#### 1. Install the prerequisites

- **Rust and Cargo:** install the latest stable toolchain with [rustup](https://rustup.rs/). This project uses Rust 2024 edition (Rust 1.85 or newer). Check your installation with `rustc --version` and `cargo --version`.
- **Git:** required to download the source code.
- **A native C compiler and linker:** the build compiles a small C component.

On Ubuntu or Debian, install the native build tools with:

```sh
sudo apt update
sudo apt install -y build-essential git
```

On macOS, install Apple's command-line build tools:

```sh
xcode-select --install
```

On Windows, use **WSL2 with Ubuntu** and follow the Linux instructions inside the WSL terminal. Native Windows builds are not currently documented or verified; the C initialization code uses POSIX APIs.

#### 2. Download the source code

```sh
git clone https://github.com/DA3M0/cheetah.git
cd cheetah
```

If you already have the source code, open a terminal in that directory instead.

#### 3. Build the CPU version

```sh
cargo build --release
```

The first build downloads and compiles Rust dependencies and may take a while. When it finishes successfully, the executable is `target/release/cheetah`. Check that it runs:

```sh
./target/release/cheetah help
```

To start chatting, you also need a supported model directory or Safetensors file:

```sh
./target/release/cheetah /path/to/model
```

Without `CHEETAH_DEVICE`, Cheetah uses CPU.

#### Optional: build with a GPU backend

GPU support must be compiled in and selected at runtime. Build only the backend you intend to use; each option has additional platform-specific requirements.

**NVIDIA CUDA (Linux):** install a compatible NVIDIA driver and CUDA Toolkit first. Check that `nvidia-smi` and `nvcc --version` work, then build and run:

```sh
cargo build --release --features cuda
CHEETAH_DEVICE=cuda ./target/release/cheetah /path/to/model
```

**Apple Metal (macOS):** install Xcode command-line tools, then build and run:

```sh
cargo build --release --features metal
CHEETAH_DEVICE=metal ./target/release/cheetah /path/to/model
```

**Vulkan:** install a Vulkan development environment with headers, the loader/runtime, a C compiler, and `glslc`. On Ubuntu or Debian, the development packages are:

```sh
sudo apt install -y build-essential libvulkan-dev glslc
cargo build --release --features vulkan
CHEETAH_DEVICE=vulkan ./target/release/cheetah /path/to/qwen2-model
```

You also need a Vulkan-capable device and its driver. If `glslc` is not on `PATH`, set `GLSLC` to its executable path before building. Vulkan inference is available for Qwen2, Gemma, Llama, Phi, Mistral, and Mixtral when built with `--features vulkan`; support still depends on compatible model configuration and weights. Vulkan builds on platforms other than Linux require a compatible Vulkan SDK and loader and are not covered by the package commands above.

On Windows, GPU builds under WSL2 additionally require the relevant GPU and driver support to be available inside WSL; the Linux commands alone do not install or configure that support.

#### Troubleshooting

- If `cargo` or `rustc` is not found, install Rust with rustup and reopen the terminal so its `PATH` changes take effect.
- If the build reports that a C compiler is missing, install the native build tools listed above.
- If a GPU build fails, check that its platform-specific toolkit, headers, libraries, and runtime/driver are installed. For Vulkan, also verify that `glslc` is available or set `GLSLC`.

### Vulkan

The optional Vulkan feature uses native C compute operators for Qwen2, Gemma, Llama, Phi, Mistral, and Mixtral. See the beginner build guide above for the Vulkan build command and prerequisites. The compute shaders are compiled with `glslc`.

The Vulkan paths support F32, F16, and BF16 weights. Mixtral uses the sparse-expert architecture and its configured top-k routing. Not every configuration is supported (for example, Vulkan Llama does not support RoPE scaling); unavailable devices or model initialization/loading errors report a diagnostic and stop with guidance (there is no silent CPU fallback). An error during active Vulkan inference is returned rather than silently switching devices.

Vulkan operator tests that exercise a real device are ignored by default and require a Vulkan compute device and `glslc`:

```sh
cargo test --features vulkan
cargo test --features vulkan qwen_operators_match_cpu_reference -- --ignored --nocapture
```

`CHEETAH_MATMUL_SPV` and `CHEETAH_QWEN_OPS_SPV` can override the corresponding compiled shader paths.

## Chat

```sh
# Interactive chat from the current model directory
cd /path/to/model && cheetah

# Chat using a model directory or a Safetensors file
cheetah /path/to/model
cheetah chat /path/to/model
cheetah /path/to/model/model.safetensors

# Configure generation
cheetah /path/to/model --threads 8 --max-tokens 128
cheetah /path/to/model --temperature 0.7 --top-k 50 --seed 42
cheetah /path/to/model --system "You are a concise assistant."
```

| Option | Default | Description |
| --- | --- | --- |
| `--threads N` | Physical core count | CPU inference thread count |
| `--max-tokens N` | 256 | Maximum generated tokens per turn |
| `--ctx-size N` | 8192 | Context window size in tokens |
| `--temperature F` | 0 | Sampling temperature; `0` uses deterministic greedy decoding |
| `--top-p F` | 0.9 | Nucleus sampling threshold; used when temperature is greater than zero |
| `--top-k N` | Disabled | Limit sampling to the top K candidates |
| `--repeat-penalty F` | 1 (disabled) | Repetition penalty |
| `--seed N` | Random | Seed for reproducible sampling |
| `--system "..."` | Built-in concise assistant prompt | System prompt, injected at the beginning of the conversation |
| `--context <name>` | None | Restore a saved conversation |
| `--context-dir <path>` | Platform application-data directory | Override context archive storage |

After each reply, interactive chat displays the number of generated tokens and decoding speed in tokens per second. The speed excludes prompt prefill; it is unavailable when no tokens are generated.

At the chat prompt, use `/exit` or `/quit` to leave, `/clear` to reset the conversation, `/save <name>` to save it, and `/help` to list chat commands. `Ctrl-C` interrupts the current generation; `Ctrl-D` exits.

### Context management

Restore a saved conversation by passing the same model and context name:

```sh
cheetah /path/to/model --context work
cheetah context list
cheetah context delete work
```

Contexts preserve the conversation and inference settings. The model path and explicitly supplied settings must be compatible with the saved context. `--context-dir <path>` can be used with chat and context-management commands to choose a different storage directory.

## Model initialization

`cheetah init` prepares a Hugging Face model directory for static, mmap-friendly loading. It recursively scans Safetensors weights, writes aligned per-layer shards below `.cheetah/weights/`, and emits `cheetah_manifest.json`. A missing `config.json` can be inferred; tokenizer files are copied when present. The source directory is not modified. `prepare` is an alias for `init`.

```sh
cheetah init /path/to/hf-model /path/to/prepared-model
cheetah inspect /path/to/prepared-model
```

The destination is generated model data: initialization replaces unsharded Safetensors weight files there and manages stale generated shards.

## Ant memory optimization

Ant is Cheetah's fine-grained model memory optimization for local inference. After `cheetah init`, Safetensors weights are split into atomic units (one per tensor), and each unit is assigned a residency tier:

- **VRAM (tier 0):** hot resident — pages stay faulted in.
- **RAM (tier 1):** resident — pages stay faulted in.
- **Disk (tier 2):** cold — pages dropped with `madvise(DONTNEED)` and streamed back in per step.

On the default CPU path (the native backend), tiers act on **real memory pages**: a disk-tier unit contributes ~0 resident bytes, and the OS page cache absorbs re-reads. Measured on qwen-0.5b with a 16 MB/32 MB budget: steady RSS **123 MB** (vs 2,497 MB for the retired Candle CPU path) with identical output.

Ant CLI overrides: `--ant <bool>`, `--ant-vram <bytes>`, `--ant-ram <bytes>`, `--ant-disk-dir <path>`, `--ant-resident <unit>`, `--ant-eviction-policy <policy>`, `--ant-prefetch-stages <n>`, `--ant-allow-vram-swap <bool>`.

### Interactive Ant preset

`cheetah init <source-dir> <destination-dir> --setup`, `cheetah chat <model-path> --setup`, and `cheetah serve <name-or-path> --setup` run an interactive preset configuration: it shows the model's unit context (count, size, layers), prompts for the VRAM pool, RAM budget, disk directory, eviction policy, prefetch stages, VRAM swap, and pinned resident units (validated against the manifest), and persists the preset to the model directory. `--setup` accepts a bare flag or an explicit boolean; `--ant-setup` is an alias. All three entry points then execute Ant with exactly those settings (CLI overrides still win).

### Backend landscape

| Device | Backend | Notes |
|--------|---------|-------|
| `CHEETAH_DEVICE=cpu` (default) | native mmap backend (Qwen2, Llama, Qwen3) | Ant-tiered, SIMD kernels; unregistered families report guidance |
| `CHEETAH_DEVICE=cuda` / `metal` | Candle | Candle's only remaining role |
| `CHEETAH_DEVICE=vulkan` | independent Vulkan executor (`--features vulkan`) | Candle-free; verified on Lavapipe, Intel Haswell iGPU, NVIDIA GT 730M |

The startup banner prints the registered backends per architecture (`backends: candle, native`). The OpenAI-compatible server accepts the same device/`--ant*` settings as chat; generation requests may additionally pass a `resident_units` override list.

### Developer documentation & contributing

- [CONTRIBUTING.md](CONTRIBUTING.md) — start here: build/test commands, step-by-step contribution workflows, the debugging toolbox, and the error-message cheat sheet.
- [docs/CODEBASE.md](docs/CODEBASE.md) — repository map: every file's role, request data flows, module index, environment surface, test matrix, and the invariants that must not break.
- [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) — Ant memory manager and the native backend: file formats, the parameter-resolution chain, the residency planner, page-level tier control, SIMD kernels, the comparison-test infrastructure, and the add-a-family checklist.
- [docs/REGISTRY.md](docs/REGISTRY.md) — the model registry: declarative registration files, the generated `REGISTRY`/`ALL_ARCHITECTURES`, backend selection, and how to add an architecture, family, or backend.

## Model inspection

```sh
cheetah formats
cheetah inspect /path/to/model
cheetah inspect /path/to/model/model.safetensors --all
cheetah inspect /path/to/model/model.safetensors --json --output report.json
```

`load` is a compatibility alias for `inspect`. Inspection supports Safetensors files and model directories, reports model/tensor metadata, and validates file headers and tensor data ranges. By default, at most 20 tensors are shown; `--all` shows every tensor. `--output` overwrites the specified report file and suppresses terminal output.

## OpenAI-compatible API server

Start a server with a model path, or resolve a model by name under `--model-dir`:

```sh
cheetah serve /path/to/model --host 127.0.0.1 --port 8080

# Resolve "qwen" in ./models and generate an API key
cheetah serve qwen --model-dir ./models --port 8080 --generate-key
```

The service prints its URL and, when enabled, its API key at startup. Press `Ctrl-C` to stop it.

| Option | Default | Description |
| --- | --- | --- |
| `<name-or-path>` | Required | Model name or Safetensors model path |
| `--model <name-or-path>` | Not applicable | Compatibility syntax for supplying the model |
| `--model-dir <directory>` | `.` | Directory searched when a model name is used |
| `--host <address>` | `127.0.0.1` | Listening address |
| `--port <port>` | 8080 | Listening port; `0` selects an available port |
| `--max-concurrent-generations N` | 1 | Maximum simultaneous inference requests |
| `--generate-key` | Disabled | Generate and print an API key |
| `--api-key <KEY>` | None | Set an API key; mutually exclusive with `--generate-key` |

Chat options can also be supplied to `serve` to set generation defaults. Without authentication, the server refuses non-loopback listen addresses. Requests other than `GET /health` require `Authorization: Bearer <key>`.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/chat/completions` | Chat completions, including SSE streaming with `stream: true` |
| `POST` | `/v1/completions` | Non-streaming text completion; the prompt is treated as one user message |
| `GET` | `/v1/models` | List the loaded model |
| `GET` | `/v1/models/:id` | Get the loaded model |
| `GET` | `/health` | Health check; no authentication required |

Example requests:

```sh
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'

curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The standard OpenAI Python client can use the local server by setting `base_url`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

The chat endpoint accepts `model`, `messages`, `stream`, `temperature`, `top_p`, `top_k`, `max_tokens` or `max_completion_tokens`, `stop`, `seed`, `n` (only `1`), `tools`, and `tool_choice`. Supported roles are `system`, `developer` (mapped to system), `user`, `assistant`, and `tool`. Tool calls are returned to the client; Cheetah does not execute external functions. The text-completion endpoint supports `prompt`, `stream` (streaming is not supported), `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, and `echo` (not currently supported).

The server loads one model at a time. Generation requests are queued in arrival order, with concurrency limited by `--max-concurrent-generations`; raising the limit may increase memory use. Hot-loading multiple models, embeddings, audio, and image endpoints are not supported.

## Supported models and limitations

- **Inference:** Safetensors decoder models registered for Qwen2, Llama, Qwen3 (native CPU backend, Ant-tiered), plus Gemma, Mistral, Mixtral, and Phi (Candle on CUDA/Metal; Vulkan executor for all seven with F32, F16, or BF16 weights, subject to model configuration compatibility).
- **Inspection and preparation:** Safetensors model directories and weight files.
- **Not supported:** GGUF inference, ONNX, and PyTorch `.bin`/pickle inference. Cheetah's `formats` command lists its supported inputs.

Model inference requires enough system or GPU memory for the weights, context, and generated output. Actual performance depends on model size, backend, and hardware.

## Contributing

Contributions are welcome — the codebase is structured so the common tasks are mechanical:

- **Add a native model family** (highest value): implement the forward deltas in the shared native runner, write a declarative registration file, extend the synthetic-model comparison test. Step-by-step: [docs/REGISTRY.md](docs/REGISTRY.md) §8 and [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §11. Reference migrations: llama (optional bias, untied lm_head), qwen3 (per-head q/k RMSNorm).
- **Kernels / memory control**: the hot path is the fused BF16 gemv in `src/chat/models/native/kernels.rs` (AVX2 intrinsics with a portable fallback); Ant page control lives in `native/store.rs`. Isolation tests and the candle-comparison oracle guard correctness.
- **Server / CLI work**: `src/serve/` and `src/cli.rs` are self-contained.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) (workflows, debugging toolbox, error-message cheat sheet, review checklist) and the [repository map](docs/CODEBASE.md). Model-specific Vulkan compute shaders (optional `src/chat/models/<name>.comp`) are discovered at build time and validated through the registration's shader list.

## FAQ

**Which backend runs on my machine?**
`cpu` (default) → the native backend for registered families (Qwen2, Llama, Qwen3); `cuda`/`metal` → Candle; `vulkan` → the independent executor. The startup banner prints `backends: ...` for the loaded architecture.

**Why did I get "no native CPU backend registered"?**
That family does not have a native implementation yet and the Candle CPU path is retired. Use `CHEETAH_DEVICE=cuda/metal`, or contribute the family — see §3.1 above.

**Why is the model directory missing `config.json` after `cheetah init`?**
It is not required: init snapshots the model parameters into `cheetah_manifest.json`, and loading falls back to that snapshot (plus registered family defaults) when the file is absent.

**My model loads but outputs look wrong.**
Run the comparison tests (`cargo test native_`) — if the synthetic comparisons pass but your model misbehaves, the issue is likely a config/manifest mismatch rather than the kernels; check the manifest's `*_source` annotations.

**Do I need a GPU?**
No — the native CPU path is the primary backend and needs no GPU. Discrete GPUs currently accelerate less than the native CPU path on this codebase (the Vulkan executor keeps weights in host memory; small-VRAM CUDA is impractical for F32 weights).


## License

Cheetah is distributed under the [Apache-2.0](LICENSE) license.
