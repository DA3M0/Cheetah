# Cheetah

Cheetah is a local language-model CLI and OpenAI-compatible API gateway written in Rust and C. It runs Safetensors decoder models using Candle on CPU, CUDA, or Metal; an optional Vulkan backend supports Qwen2, Gemma, Llama, Phi, Mistral, and Mixtral inference. Cheetah can also prepare model weights for mmap-friendly loading and inspect Safetensors files.

The quickest way to start a chat is to pass a Hugging Face model directory:

```sh
cheetah /path/to/model
```

Human-readable CLI output uses semantic colors when writing to a terminal. Colors are disabled for redirected output and when `NO_COLOR` is set. Generated model text and machine-readable JSON/API responses are not colored.

The directory should contain `config.json`, `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` with its weight shards. You can also launch `cheetah` from inside the model directory.

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

The Vulkan paths support F32, F16, and BF16 weights. Mixtral uses the sparse-expert architecture and its configured top-k routing. Not every configuration is supported (for example, Vulkan Llama does not support RoPE scaling); unavailable devices or model initialization/loading errors report a diagnostic and fall back to Candle on CPU. An error during active Vulkan inference is returned rather than silently switching devices.

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

- **Inference:** Safetensors decoder models registered for Qwen2, Gemma, Llama, Phi, Mistral, and Mixtral architectures. Candle CPU, CUDA, and Metal support depends on the selected architecture and compiled backend. Vulkan inference supports all six registered architectures with F32, F16, or BF16 weights, subject to model configuration compatibility.
- **Inspection and preparation:** Safetensors model directories and weight files.
- **Not supported:** GGUF inference, ONNX, and PyTorch `.bin`/pickle inference. Cheetah's `formats` command lists its supported inputs.

Model inference requires enough system or GPU memory for the weights, context, and generated output. Actual performance depends on model size, backend, and hardware.

## Adding model registrations

The build scans `src/chat/models/*.rs` (excluding shared registry/support modules) and generates the model registry. Add a Rust module with a `REGISTRATION` for architecture aliases, a prompt renderer/template id, stop markers, and a loader implementing `ModelRuntime`; no central registry edit is needed. The runtime trait supports `clone_box` so generation can keep an independent model instance per request.

An optional same-stem `src/chat/models/<name>.comp` is discovered and compiled when building with `--features vulkan`. Its compiled path is available through `super::shader_path("<name>")`; list shader stems in the registration so Vulkan dispatch can validate them. Cargo rebuilds when model or shader sources change. `cheetah init` uses these registrations to add registered template ids and stop markers to the prepared manifest.

See the [model integration guide](MODEL_INTEGRATION.md) for the registration contract, Candle loader example, and optional Vulkan integration details.

## License

Cheetah is distributed under the [Apache-2.0](LICENSE) license.
