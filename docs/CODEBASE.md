# Codebase index — developer map

A navigation map of the repository: every source file, what it owns, its key
entry points, the request data flows, the environment surface, and the test
map. Read this first when landing in the codebase; the deep guides
([ANT_MEMORY.md](ANT_MEMORY.md), [REGISTRY.md](REGISTRY.md)) pick up where
this file points.

Line numbers are approximate (they drift); function names are the anchors.

---

## 1. Repository tree

```
cheetah/
├── Cargo.toml                  features: default={} | metal | cuda | vulkan
├── build.rs                    shader/registration discovery + generated registries
├── cheetah_init.c/.h           native model preparation (init) + manifest/units writers
├── cheetah_memory.c/.h         Ant C manager: units index, tier bookkeeping (metadata)
├── cheetah_vulkan.c/.h         Vulkan substrate: device init, VRAM pool, matmul/rope/attention
├── matmul*.comp, shared_ops.comp   shared Vulkan compute shaders
├── src/
│   ├── main.rs                 CLI dispatch (init/chat/serve/inspect/context) + Ant settings resolution
│   ├── cli.rs                  argument parsing for every subcommand (+ Ant options, --setup)
│   ├── ant.rs                  AntSettings/AntUnit/Ant manager, residency planner, FFI wrappers
│   ├── chat/
│   │   ├── mod.rs              chat entry (chat()), model_metadata_for_init
│   │   ├── engine.rs           ★ Engine: load pipeline, generation loop, Ant hooks, sampling
│   │   ├── context.rs          conversation archive/store (persistence)
│   │   └── models/
│   │       ├── mod.rs          ModelRuntime trait + BackendLoader enum + registry include!
│   │       ├── registry.rs     device backends (cpu/cuda/metal/vulkan) + load dispatch + policy
│   │       ├── candle.rs       Candle bridge (VarBuilder, tied-embedding backend)
│   │       ├── qwen2.rs        Candle wrapper: Qwen2 (+ load_vulkan)
│   │       ├── llama.rs        Candle wrapper: Llama (+ Cache)
│   │       ├── gemma/mistral/mixtral/phi.rs   Candle wrappers
│   │       ├── qwen3.rs        Candle wrapper: Qwen3 (self-contained impl, 694 lines)
│   │       ├── templates.rs    chat template renderers
│   │       ├── registrations/  ★ declarative registration files (one per family)
│   │       └── native/         ★ the native mmap backend
│   │           ├── store.rs    MmapWeightStore + madvise tier control
│   │           ├── kernels.rs  gemv (AVX2 + portable), rmsnorm, rope, softmax, silu
│   │           ├── qwen2.rs    decoder runner (Qwen2Native) + Qwen2Config::resolve
│   │           └── mod.rs      NativeCpuRuntime + registry loader + synthetic tests
│   ├── vulkan_transformer.rs   Vulkan model executors (transformer + Phi), Candle-free
│   ├── native.rs               FFI: init/manifest update + Rust wrappers over cheetah_*.c
│   ├── serve/                  OpenAI-compatible API (mod/http/openai/queue)
│   ├── inspect.rs              model inspection (load/inspect command)
│   ├── color.rs, options.rs    CLI coloring, ChatOptions
│   └── context.rs              (see above)
└── docs/                       developer guides (this directory)
```

★ = where most Ant/backend work happens.

---

## 2. Data flows

### 2.1 `cheetah chat <model> [--ant]` (CPU, Qwen2 family)

```
main.rs::run (dispatch)                      src/main.rs
  → run_chat                                 src/main.rs:287
      → resolve_ant_settings                 src/main.rs (defaults → preset → CLI; --setup wizard)
      → chat::chat                           src/chat/mod.rs
          → load_engine_with_progress        src/chat/engine.rs:60
              → Engine::load                 src/chat/engine.rs (§ load pipeline)
                  config.json (or synth from manifest — synth_config_from_manifest:1326)
                  AntSettings resolve → units manifest → plan_residency (src/ant.rs)
                  ant_apply_budget → ant_evicted
                  layer_residency_summary + AntRuntimeState seed
                  backend_loader_for_architecture(arch, "native")   ← registry lookup
                      → native::native_family_loader   src/chat/models/native/mod.rs
                          MmapWeightStore::open        native/store.rs
                          Qwen2Config::resolve         native/qwen2.rs (config→units→family)
                          apply_tiers(plan)            madvise per unit
          → chat_loop                        engine.rs:1500 (REPL)
              → generate_inner               engine.rs:1188
                  apply_ant_request_overrides → prefetch → promote → enforce
                  model.forward(&tokens, offset)     ← backend-agnostic boundary
                      (native: Qwen2Native::forward  native/qwen2.rs)
                  sample_token                   engine.rs:1527
```

### 2.2 `cheetah serve` (OpenAI API)

```
serve::serve            src/serve/mod.rs:17  (auth checks, resolve_model:140)
  → load_engine_with_progress (same pipeline as chat)
  → tiny_http server    http.rs (SSE streaming, queue.rs concurrency tickets)
  → per request: openai.rs (request parsing/validation) → engine.generate_with_chat_params
```

### 2.3 `cheetah init <src> <dst>`

```
run_init_command      src/main.rs:90
  → native::init_model_with_metadata   src/native.rs (C cheetah_init_model + manifest metadata)
      C side: parse safetensors → aligned shards → write_manifest (parameter snapshot)
              → write_units (atomic units) → copy config/tokenizer
  → cli::parse_ant_args (+ --setup wizard)
  → resolve_ant_settings → Ant::plan_residency → page_in per unit
  → prints "ant plan: N unit(s) → X vram, Y ram, Z disk"
```

---

## 3. Module index (key items)

### `src/chat/engine.rs` (~2000 lines) — the inference engine

| Item | Line ~ | Role |
|------|--------|------|
| `load_engine_with_progress` | 60 | spinner-wrapped `Engine::load` |
| `print_model_info` | ~90 | startup banner (backends, ant plan/layers/swap) |
| `Engine` struct | ~211 | model mutex + all Ant fields |
| `AntRuntimeState` | ~250 | Phase-3 bookkeeping (occupancy/evicted/promotion queue) |
| `Engine::load` | ~370 | the load pipeline (see CODEBASE §2.1) |
| `synth_config_from_manifest` | 1326 | config.json-missing recovery from manifest |
| `ant_swap_in_unit` | 711 | budget-checked swap (re-check before each eviction) |
| `prefetch_ant_units` / `promote_ant_units` | ~840/~900 | generation hooks |
| `enforce_ant_budget` | 946 | page-back-in + evict-to-fit |
| `ant_apply_real_tier` | ~640 | drives `AntTierControl` (real pages) |
| `generate_inner` | 1188 | the generation loop (hooks → forward → sample) |
| `sample_token` | 1527 | temperature/top-p/top-k/repeat-penalty sampling |
| `chat_loop` | ~1560 | REPL (slash commands, streaming) |

### `src/chat/models/native/` — the native backend

| Item | File | Role |
|------|------|------|
| `MmapWeightStore` | store.rs | shard mmap + unit index + tier Mutex + `page_in`/`drop_pages`/`drop_range`/`drop_disk_units_of_layer` |
| `gemv` | kernels.rs | row-parallel matrix-vector; AVX2 vs portable dispatch |
| `dot_row_bf16_avx2` | kernels.rs | `cvtepu16_epi32 + slli_epi32(16)` widening + FMA |
| `dot_row_bf16_fused(_portable)` | kernels.rs | fused bit-widening dot (portable fallback) |
| `rmsnorm` / `rope_in_place` / `silu_multiply` / `softmax` | kernels.rs | elementwise/attention helpers |
| `Qwen2Config::resolve` | qwen2.rs | the parameter chain (config → units → family) |
| `Qwen2Native::forward` | qwen2.rs | the decoder forward (see ANT_MEMORY §7.2) |
| `apply_tiers` / `page_in_unit` / `evict_unit` | qwen2.rs | Ant tier application on real pages |
| `native_family_loader` | mod.rs | registry-facing loader (qwen2/llama/qwen3) |
| `prepare_synthetic_decoder` | mod.rs (tests) | synthetic-model generator for comparisons |

### `src/ant.rs` — the planner

| Item | Role |
|------|------|
| `AntSettings` / `AntUnit` | settings schema + unit deserialization |
| `Ant::plan_residency` | layer-ordered greedy tier fill |
| `Ant::layer_residency_summary` | per-layer partial-layer tracking (banner) |
| `ant_swap_fit` | pure swap decision (used by the engine's budget-checked swap) |
| `ant_filter_known_units` | wizard unit-id validation |
| `run_interactive_setup` | the `--setup` wizard |

### `src/chat/models/` — registry & backends

| Item | File | Role |
|------|------|------|
| `ModelRuntime` | mod.rs | backend-agnostic boundary: `forward(&[u32], usize) -> Vec<f32>` |
| `BackendLoader` / `ModelRegistration` / `ModelFamily` | mod.rs | registry schema (see REGISTRY §3) |
| `registration_for_architecture` etc. | mod.rs | lookups |
| `BACKENDS` (device table) / `load_model` / `load_model_for_device` / `retired_cpu_candle_message` | registry.rs | device dispatch + Candle CPU retirement policy |
| `candle::load_model` | candle.rs | VarBuilder bridge + tied-embedding backend (tests' numerical oracle) |
| `VulkanTransformerExecutor` / `VulkanPhiExecutor` | vulkan_transformer.rs | Candle-free Vulkan executors |

### CLI/serve

| Item | File | Role |
|------|------|------|
| `AntCliOptions` / `parse_ant_args` / `parse_chat_args` | cli.rs | all `--ant*`/`--setup` parsing |
| `split_ant_args` | main.rs | value-aware ant-flag extraction for serve |
| `serve::serve` / `resolve_model` | serve/mod.rs | server lifecycle, model resolution by name/dir |
| `http.rs` / `openai.rs` / `queue.rs` | serve/ | routing, request schemas, concurrency tickets |

---

## 4. Environment surface

| Variable | Read by | Effect |
|----------|---------|--------|
| `CHEETAH_DEVICE` | registry.rs, engine.rs | `cpu` (default, native) / `cuda` / `metal` / `vulkan` |
| `CHEETAH_NATIVE` | engine.rs | `0` disables native selection (Candle CPU is retired — this surfaces the guidance error; kept for policy testing) |
| `CHEETAH_VRAM_POOL_SIZE` | cheetah_vulkan.c | Vulkan allocator pool bytes (default 800 MiB) |
| `GLSLC` | build.rs | shader compiler override |
| `NO_COLOR` | color.rs | disable ANSI coloring |

Constant filenames (not env): `cheetah_units.json`, `cheetah_manifest.json`,
`cheetah_ant_settings.json` (see [ANT_MEMORY.md](ANT_MEMORY.md) §3).

---

## 5. Build & test matrix

```sh
cargo build                          # default: native + Candle CPU/metal/cuda paths
cargo build --release                # recommended for any performance measurement
cargo build --release --features vulkan   # + shader compilation (needs glslc) + executors
cargo build --release --features cuda     # Candle CUDA (needs CUDA toolkit)
cargo build --release --features metal    # Candle Metal (macOS)

cargo test                           # default feature tests (135)
cargo test --features vulkan         # + Vulkan executor/shader tests (147)
cargo test native_                   # all native-backend comparison tests
cargo test -- --test-threads=4       # parallel (some tests are timing-sensitive)
```

Tests that need local artifacts (auto-skip when absent):
- `ant_lifecycle_against_real_prepared_model` — `/home/benny/models/qwen-0.5b-prepared`
- `native_logits_match_candle` / `native_layer0_matches_candle_reference` — same model
- `tiny_dense_and_moe_models_forward_and_reset_cache` (vulkan, `#[ignore]`) — a real Vulkan compute device; the serve smoke test is the practical verification

Everything else — including the synthetic decoder comparisons
(`native_llama_matches_candle_on_synthetic_model`,
`native_qwen3_matches_candle_on_synthetic_model`) — is self-contained and
runs anywhere.

---

## 6. Invariants (do not break)

1. **The runtime boundary carries no backend types**:
   `ModelRuntime::forward(&[u32], usize) -> Result<Vec<f32>, String>`.
2. **Registration files contain no runtime code**; implementation files
   contain no `REGISTRATION`.
3. **`Disk`-tier means real released pages** on the native backend — any change
   that keeps disk-tier bytes resident breaks the Ant contract.
4. **Every swap re-checks the budget before it executes** (`ant_swap_in_unit`,
   `ant_swap_fit`).
5. **Bit-math changes require the isolation tests** (`dot_row_bf16_fused` vs
   dequant+dot) AND a native-vs-candle comparison run.
6. **Performance numbers are only valid from a freshly built binary** (see
   ANT_MEMORY §9, lesson 2).
7. **`f32::sin_cos` returns `(sin, cos)`** — the historical rope bug.
8. Registration assertions
   (`backend_registration_is_explicit_per_architecture`) must be updated when
   a family gains/loses a backend.
