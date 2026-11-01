# Cheetah

Cheetah is a local language-model CLI and OpenAI-compatible gateway using Candle for inference and Safetensors checkpoints for model weights. Supported decoder architectures include Qwen2, Gemma, Llama, and Phi; CPU is the default runtime.

## Build

```sh
cargo build --release
```

Optional GPU backends can be compiled with `--features cuda` or `--features metal`. Set `CHEETAH_DEVICE=cuda` or `CHEETAH_DEVICE=metal` to select a compiled backend at runtime; `CHEETAH_DEVICE=cpu` (the default) selects the CPU.

## Chat

Pass a Hugging Face model directory containing `config.json`, `tokenizer.json`, and either `model.safetensors` or `model.safetensors.index.json` plus its shard files:

```sh
cargo run -- /home/benny/models/qwen-0.5b
cargo run -- /home/benny/models/qwen-0.5b --max-tokens 128 --temperature 0
```

The CLI also accepts a direct `.safetensors` path when its config and tokenizer files are in the same directory. Chat supports multi-turn conversations, `/exit`, `/clear`, `/save <name>`, and named context restore with `--context <name>`. Context archives are stored under the platform's Cheetah application-data directory; override it with `--context-dir <path>`.

Sampling options include `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, and `--seed`. `--threads` controls the CPU thread pool and `--ctx-size` bounds the context window.

## Model inspection

```sh
cargo run -- inspect /home/benny/models/qwen-0.5b
cargo run -- inspect /home/benny/models/qwen-0.5b/model.safetensors --json --output report.json
```

## OpenAI-compatible server

```sh
cargo run -- serve /home/benny/models/qwen-0.5b --host 127.0.0.1 --port 8080
```

The server exposes `POST /v1/chat/completions` (including SSE streaming), non-streaming `POST /v1/completions`, `GET /v1/models`, `GET /v1/models/:id`, and `GET /health`. Unauthenticated non-loopback listeners are rejected; use `--api-key` or `--generate-key` when binding to a non-loopback address.
