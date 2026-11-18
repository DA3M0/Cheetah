# Cheetah

**Cheetah** is a local language-model CLI and OpenAI-compatible API gateway, written in Rust and C. It runs Safetensors decoder models with a self-describing prepared-model format and **Ant tiered memory** — real page-level VRAM/RAM/Disk control — across two independent backends.

| | |
|---|---|
| 🧠 **Ant tiered memory** | per-tensor VRAM/RAM/Disk residency acting on real pages — 123 MB steady RSS on a 0.5B model (vs 2.5 GB untiered) |
| 🚀 **native CPU backend** | mmap-backed BF16 weights, AVX2 SIMD kernels — up to 8 tok/s on Qwen2-0.5B |
| 🧩 **declarative model registry** | one registration file per family — architecture, chat template, stop markers, both backends, family constants; build.rs generates the registry, lookups derive from it |
| 🔌 **OpenAI-compatible API** | `/v1/chat/completions` with SSE streaming, tool calls, API keys |
| 🛠️ **no GGUF** | Safetensors in, self-describing prepared directory out (`cheetah init`) |

```
                 ┌─────────────────────────────────────────────┐
  CLI / OpenAI ──▶ Engine (chat/serve) ──▶ ModelRuntime boundary ──▶ ┐
                 └─────────────────────────────────────────────┘      │
                                                                      ▼
                    ┌──────────────────────┬──────────────────────────┐
                    │ native (CPU)         │ Vulkan executor           │
                    │ mmap + Ant tiers     │ Candle-free, Ant buckets │
                    └──────────────────────┴──────────────────────────┘
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

Cheetah runs Safetensors decoder models. The native CPU backend (the default) requires a **prepared model directory** — run `cheetah init` to convert a Hugging Face model into the mmap-friendly format:

```sh
# Download a small Qwen2 model (~1 GB)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /tmp/qwen-0.5b

# Prepare it for the native backend
cheetah init /tmp/qwen-0.5b /tmp/qwen-0.5b-prepared

# Chat with the prepared model
cheetah /tmp/qwen-0.5b-prepared
```

(No `huggingface-cli`? Download the files directly from the model page.)

The Vulkan backend can also read raw Hugging Face directories directly (set `CHEETAH_DEVICE=vulkan`), but the native backend — which is the default and the fastest CPU path — requires the prepared layout with `.cheetah/weights/` shards and `cheetah_manifest.json`.

GGUF files are not supported; use a Safetensors conversion of the model instead.

### 4. Environment variables

Cheetah reads several environment variables to select the backend and configure memory. These must be **visible to the Cheetah process** — either `export`ed or set inline:

```sh
# Inline (one command only — the variable is gone after the command exits):
CHEETAH_DEVICE=vulkan ./target/release/cheetah /path/to/model

# Export (persists for the entire shell session — all subsequent commands see it):
export CHEETAH_DEVICE=vulkan
./target/release/cheetah /path/to/model
```

> **⚠️ Common pitfall:** writing `CHEETAH_DEVICE=vulkan` on its own line **without `export`** creates a shell variable that child processes cannot see. Cheetah will silently fall back to the default (`cpu`) — no error is printed. Always use `export` or the inline prefix form.

Key variables (full list in [docs/CODEBASE.md](docs/CODEBASE.md) §4):

| Variable | Values | Default |
|----------|--------|---------|
| `CHEETAH_DEVICE` | `cpu` / `vulkan` | `cpu` (native backend) |
| `CHEETAH_QUANT_Q8` | `1` / `0` | `0` (full precision; Vulkan only) |
| `CHEETAH_VRAM_POOL_SIZE` | bytes | `838860800` (800 MiB; Vulkan C substrate pool) |

### 5. Run a chat

```sh
cheetah /path/to/prepared-model
```

Then type a message at the `> ` prompt and press Enter. Use `/exit` to quit, `/clear` to reset the conversation, `/save <name>` to save it, and `/help` to list chat commands. Start with a small model and a small context for quick feedback:

```sh
cheetah /path/to/prepared-model --ctx-size 1024 --max-tokens 128
```

### 6. Try the OpenAI-compatible server

```sh
cheetah serve /path/to/prepared-model --port 8080
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

To start chatting, you also need a prepared model directory:

```sh
./target/release/cheetah init /path/to/hf-model /path/to/prepared-model
./target/release/cheetah /path/to/prepared-model
```

Without `CHEETAH_DEVICE`, Cheetah uses the native CPU backend.

#### Optional: build with the Vulkan backend

The Vulkan backend is the only GPU option. It requires a Vulkan development environment with headers, the loader/runtime, a C compiler, and `glslc`. On Ubuntu or Debian:

```sh
sudo apt install -y build-essential libvulkan-dev glslc
cargo build --release --features vulkan
CHEETAH_DEVICE=vulkan ./target/release/cheetah /path/to/model   # inline = one-shot
```

You also need a Vulkan-capable device and its driver. If `glslc` is not on `PATH`, set `GLSLC` to its executable path before building. Vulkan inference is available for Qwen2, Qwen3, Gemma, Llama, Phi, Mistral, and Mixtral when built with `--features vulkan`; support depends on compatible model configuration and weights. Vulkan builds on platforms other than Linux require a compatible Vulkan SDK and loader.

On Windows, GPU builds under WSL2 additionally require the relevant GPU and driver support to be available inside WSL; the Linux commands alone do not install or configure that support.

#### Troubleshooting

- If `cargo` or `rustc` is not found, install Rust with rustup and reopen the terminal so its `PATH` changes take effect.
- If the build reports that a C compiler is missing, install the native build tools listed above.
- If the Vulkan build fails, check that the Vulkan SDK, headers, libraries, and runtime/driver are installed. Verify that `glslc` is available or set `GLSLC`.

### Vulkan

The optional Vulkan feature uses native C compute operators for all registered architectures (Qwen2, Llama, Qwen3, Gemma, Mistral, Mixtral, Phi). See the beginner build guide above for the Vulkan build command and prerequisites. The compute shaders are compiled with `glslc`.

**Supported weight precisions:** F32, F16, BF16, and **Q8** (row-wise int8 with per-row f32 scales, enabled by `CHEETAH_QUANT_Q8=1`). Q8 roughly halves the weight footprint compared to BF16. Mixtral uses the sparse-expert architecture with per-token expert routing (`num_experts_per_tok`). Not every configuration is supported (for example, Vulkan Llama does not support RoPE scaling); unavailable devices or model initialization/loading errors report a diagnostic and stop with guidance (there is no silent CPU fallback). An error during active Vulkan inference is returned rather than silently switching devices.

**Ant bucketed residency:** VRAM-tier weights are uploaded once into DEVICE_LOCAL memory (real VRAM); disk-tier weights stream through a HOST_COHERENT pool each step. Ant page-in/evict moves weights between the two buckets. On GPUs with small VRAM, oversized uploads automatically fall back to the streaming path per weight.

**Operator library** (`shared_ops.comp`, 10 ops): rms_norm (op 0), elementwise add/sub (ops 1–2), rotary (op 3), attention (op 4), silu_mul (op 5), gelu_mul — exact + tanh (op 6), layer_norm (op 7), embedding_gather (op 8), bias_add (op 9). Dedicated matmul pipelines for f32, bf16, f16, and q8.

Compiled shader paths can be overridden per operator:

| Shader | Override variable | Default |
|--------|-------------------|---------|
| f32 matmul | `CHEETAH_MATMUL_SPV` | embedded at build time |
| bf16 matmul | `CHEETAH_MATMUL_BF16_SPV` | embedded at build time |
| q8 matmul | `CHEETAH_MATMUL_Q8_SPV` | embedded at build time |
| shared ops (10 ops) | `CHEETAH_QWEN_OPS_SPV` | embedded at build time |

Vulkan operator tests that exercise a real device are ignored by default and require a Vulkan compute device and `glslc`:

```sh
cargo test --features vulkan
cargo test --features vulkan qwen_operators_match_cpu_reference -- --ignored --nocapture
```

## Chat

```sh
# Interactive chat with a prepared model directory
cheetah /path/to/prepared-model

# Chat using a Safetensors file (Vulkan backend only)
CHEETAH_DEVICE=vulkan cheetah /path/to/model/model.safetensors

# Configure generation
cheetah /path/to/prepared-model --threads 8 --max-tokens 128
cheetah /path/to/prepared-model --temperature 0.7 --top-k 50 --seed 42
cheetah /path/to/prepared-model --system "You are a concise assistant."
```

| Option | Default | Description |
| --- | --- | --- |
| `--threads N` | Physical core count | CPU inference thread count |
| `--max-tokens N` | 256 | Maximum generated tokens per turn |
| `--ctx-size N` | 8192 | Context window size in tokens |
| `--temperature F` | 0 | Sampling temperature; `0` uses deterministic greedy decoding |
| `--top-p F` | 0.9 | Nucleus sampling threshold; used when temperature is greater than zero |
| `--top-k N` | Disabled | Limit **sampling** to the top K candidates (0 disables; unrelated to MoE expert routing, which is model-configured) |
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
cheetah /path/to/prepared-model --context work
cheetah context list
cheetah context delete work
```

Contexts preserve the conversation and inference settings. The model path and explicitly supplied settings must be compatible with the saved context. `--context-dir <path>` can be used with chat and context-management commands to choose a different storage directory. The default location is platform-specific: `$XDG_DATA_HOME/cheetah/contexts` (or `~/.local/share/cheetah/contexts`) on Linux, `~/Library/Application Support/cheetah/contexts` on macOS, and `%LOCALAPPDATA%\cheetah\contexts` on Windows.

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

Set `export CHEETAH_QUANT_Q8=1` (or inline `CHEETAH_QUANT_Q8=1 cheetah ...`) to quantize linear projections to row-wise int8 on the Vulkan backend (halves the weight footprint; norms, biases and embeddings stay in original precision). The native CPU backend currently loads weights in their original F32/F16/BF16 precision.

On the default CPU path (the native backend), tiers act on **real memory pages**: a disk-tier unit contributes ~0 resident bytes, and the OS page cache absorbs re-reads. Measured on qwen-0.5b with a 16 MB/32 MB budget: steady RSS **123 MB** (vs 2,497 MB for the retired Candle CPU path) with identical output.

Ant CLI overrides: `--ant <bool>`, `--ant-vram <bytes>`, `--ant-ram <bytes>`, `--ant-disk-dir <path>`, `--ant-resident <unit>`, `--ant-eviction-policy <policy>`, `--ant-prefetch-stages <n>`, `--ant-allow-vram-swap <bool>`. Default budgets: VRAM pool 8 GiB, RAM budget 16 GiB.

### Interactive Ant preset

`cheetah init <source-dir> <destination-dir> --setup`, `cheetah chat <model-path> --setup`, and `cheetah serve <name-or-path> --setup` run an interactive preset configuration: it shows the model's unit context (count, size, layers), prompts for the VRAM pool, RAM budget, disk directory, eviction policy, prefetch stages, VRAM swap, and pinned resident units (validated against the manifest), and persists the preset to the model directory. `--setup` accepts a bare flag or an explicit boolean; `--ant-setup` is an alias. All three entry points then execute Ant with exactly those settings (CLI overrides still win).

### Backend landscape

| Device | Backend | Notes |
|--------|---------|-------|
| `CHEETAH_DEVICE=cpu` (default) | native mmap backend (Qwen2, Llama, Qwen3, Mixtral MoE, Mistral, Gemma, Phi) | Ant-tiered, SIMD kernels; requires a prepared model directory (`cheetah init`); architectures without a native backend report guidance |
| `CHEETAH_DEVICE=vulkan` | independent Vulkan executor (`--features vulkan`) | Candle-free; Ant bucketed residency (VRAM/streams); Q8 quantization; verified on Lavapipe, Intel Haswell iGPU, NVIDIA GT 730M |

The startup banner prints the backend capability list per architecture (`backends: native, vulkan`) and the resolved device (`device: native-cpu` / `device: Vulkan`). If the device does not match your expectation, check that `CHEETAH_DEVICE` is exported — an unexported variable silently falls back to the default. The OpenAI-compatible server accepts the same device/`--ant*` settings as chat; generation requests may additionally pass a `resident_units` override list.

### Developer documentation & contributing

- [CONTRIBUTING.md](CONTRIBUTING.md) — start here: build/test commands, step-by-step contribution workflows, the debugging toolbox, and the error-message cheat sheet.
- [docs/CODEBASE.md](docs/CODEBASE.md) — repository map: every file's role, request data flows, module index, environment surface, test matrix, and the invariants that must not break.
- [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) — Ant memory manager and the native backend: file formats, the parameter-resolution chain, the residency planner, page-level tier control, SIMD kernels, the comparison-test infrastructure, and the add-a-family checklist.
- [docs/REGISTRY.md](docs/REGISTRY.md) — the model resolution layer: architecture → chat template / backend capability / family constants, backend selection, and how to add an architecture, family, or backend.

## Model inspection

```sh
cheetah formats
cheetah inspect /path/to/model
cheetah inspect /path/to/model/model.safetensors --all
cheetah inspect /path/to/model/model.safetensors --json --output report.json
```

`load` is a compatibility alias for `inspect`. Inspection supports Safetensors files and model directories, reports model/tensor metadata, and validates file headers and tensor data ranges. By default, at most 20 tensors are shown; `--all` shows every tensor. `--output` overwrites the specified report file and suppresses terminal output.

## OpenAI-compatible API server

Start a server with a prepared model path, or resolve a model by name under `--model-dir`:

```sh
cheetah serve /path/to/prepared-model --host 127.0.0.1 --port 8080

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

- **Inference:** Safetensors decoder models resolved for Qwen2, Llama, Qwen3, Mixtral (native CPU backend, Ant-tiered; Mixtral is the mixture-of-experts with per-token expert routing via `num_experts_per_tok`), plus the dense delta families Mistral (decoding-time sliding window), Gemma (offset norms, sqrt(hidden) embedding scale, GeGLU), and Phi (biased LayerNorms, parallel residual, fused projections, partial rotary). Every family also runs on the independent Vulkan executor with F32, F16, BF16, or Q8 (row-wise int8) weights, subject to model configuration compatibility.
- **Inspection and preparation:** Safetensors model directories and weight files.
- **Not supported:** GGUF inference, ONNX, and PyTorch `.bin`/pickle inference. Cheetah's `formats` command lists its supported inputs.

Model inference requires enough system or GPU memory for the weights, context, and generated output. Actual performance depends on model size, backend, and hardware.

## Contributing

Contributions are welcome — the codebase is structured so the common tasks are mechanical:

- **Add a native model family** (highest value): declare the family's deltas in `DenseFeatures` (offset norms, embedding scale, GeGLU, LayerNorm, parallel residual, fused projections, partial rotary — existing flags cover all published dense deltas), add a declarative registration file in `src/chat/models/registrations/` (build.rs picks it up and regenerates the registry), and extend the native forward-parity tests. Step-by-step: [docs/REGISTRY.md](docs/REGISTRY.md) §4 and [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §11. Reference migrations: llama (optional bias, untied lm_head), qwen3 (per-head q/k RMSNorm), mistral (sliding window), gemma (offset norms + embedding scale + GeGLU), phi (LayerNorm + parallel residual + fused projections).
- **Kernels / memory control**: the hot path is the fused BF16 gemv in `src/chat/models/native/kernels.rs` (AVX2 intrinsics with a portable fallback); Ant page control lives in `native/store.rs`. Isolation tests and the native forward-parity checks guard correctness.
- **Server / CLI work**: `src/serve/` and `src/cli.rs` are self-contained.

Start with [CONTRIBUTING.md](CONTRIBUTING.md) (workflows, debugging toolbox, error-message cheat sheet, review checklist) and the [repository map](docs/CODEBASE.md). Model-specific Vulkan compute shaders (optional `src/chat/models/<name>.comp`) are discovered at build time and exposed through the generated shader registry.

## FAQ

**Which backend runs on my machine?**
`cpu` (default) → the native backend for architecture-resolved families (Qwen2, Llama, Qwen3, Mixtral, Mistral, Gemma, Phi); `vulkan` → the independent executor (requires `--features vulkan` build). The startup banner prints `backends: ...` for the loaded architecture.

**Why did I get "no native CPU backend available"?**
That architecture has no native implementation yet. Use `CHEETAH_DEVICE=vulkan` (independent executor), or contribute the family — see §3.1 above.

**Why is the model directory missing `config.json` after `cheetah init`?**
It is not required: init snapshots the model parameters into `cheetah_manifest.json`, and loading falls back to that snapshot (plus registered family defaults) when the file is absent.

**My model loads but outputs look wrong.**
Run the native backend tests (`cargo test native_`) — if the backend tests pass but your model misbehaves, the issue is likely a config/manifest mismatch rather than the kernels; check the manifest's `*_source` annotations.

**Do I need a GPU?**
No — the native CPU path is the primary backend and needs no GPU. The Vulkan backend can use a GPU but keeps weights in host memory (HOST_COHERENT design); on small models the native CPU path is often faster.

**How do I enable Q8 quantization?**
Set `CHEETAH_QUANT_Q8=1` before starting. Linear projections are quantized to row-wise int8 (roughly half the weight footprint). Norms, biases and embeddings stay in full precision. Vulkan only — the native CPU backend loads weights in their original F32/F16/BF16 precision.

**Why does CHEETAH_DEVICE=VULKAN (uppercase) fail?**
Older builds used case-sensitive device lookup. This was fixed — all casing now works (`vulkan`, `VULKAN`, `Vulkan`). If you still see `Unsupported CHEETAH_DEVICE`, rebuild from the latest code.


## License

Cheetah is distributed under the [Apache-2.0](LICENSE) license.
