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
├── build.rs                    shader discovery + generated shader registry
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
│   │   ├── moe.rs              shared MoE routing (`select_experts`: softmax → top-k → renormalize)
│   │   └── models/
│   │       ├── mod.rs          ModelRuntime trait + BackendLoader enum + ModelRegistration + REGISTRY include
│   │       ├── registrations/  ★ declarative capability per family (build.rs → generated REGISTRY)
│   │       ├── registry.rs     derived lookups (loader/family/backends) + device dispatch
│   │       ├── qwen2.rs        family runtime wiring (native runner + Vulkan executor loader)
│   │       ├── llama.rs        family runtime wiring
│   │       ├── gemma/mistral/mixtral/phi.rs   family runtime wiring
│   │       ├── qwen3.rs        family runtime wiring
│   │       ├── templates.rs    chat template renderers
│   │       └── native/         ★ the native mmap backend
│   │           ├── store.rs    MmapWeightStore + madvise tier control
│   │           ├── kernels.rs  gemv (AVX2 + portable), rmsnorm, layernorm, rope, softmax, silu/gelu
│   │           ├── decoder.rs    decoder runner (DecoderNative) + DecoderConfig::resolve
│   │           └── mod.rs      NativeCpuRuntime + native family loader + native tests
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
                  backend_loader_for_architecture(arch, backend)   ← architecture capability lookup
                      → native::native_family_loader   src/chat/models/native/mod.rs
                          MmapWeightStore::open        native/store.rs
                          DecoderConfig::resolve         native/decoder.rs (config→units→family)
                          apply_tiers(plan)            madvise per unit
          → chat_loop                        engine.rs:1500 (REPL)
              → generate_inner               engine.rs:1188
                  apply_ant_request_overrides → prefetch → promote → enforce
                  model.forward(&tokens, offset)     ← backend-agnostic boundary
                      (native: DecoderNative::forward  native/decoder.rs)
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
| `DecoderConfig::resolve` | decoder.rs | the parameter chain (config → units → family) |
| `DecoderNative::forward` | decoder.rs | the decoder forward (see ANT_MEMORY §7.2) |
| `forward_moe_block` / `forward_dense_mlp` | decoder.rs | sparse block: router → top-k experts mixed by route weight; dense gated MLP (silu / gelu) or phi plain `fc1`/`fc2` otherwise |
| `forward_attention` / `apply_norm_to` / `DenseFeatures` | decoder.rs | family deltas: fused `qkv_proj` + `dense` naming (phi), partial rotary, offset RMSNorm (gemma) vs LayerNorm (phi), output written pre-residual so phi can sum attention + MLP in one update |
| sliding-window attention (`window_start`) | decoder.rs | masks keys older than `sliding_window` (0 = full attention), same rule as `shared_ops.comp` |
| `apply_tiers` / `page_in_unit` / `evict_unit` | decoder.rs | Ant tier application on real pages |
| `native_family_loader` | mod.rs | native family loader (llama / mixtral / qwen2 / qwen3 / mistral / gemma / phi) |

### `src/ant.rs` — the planner

| Item | Role |
|------|------|
| `AntSettings` / `AntUnit` | settings schema + unit deserialization |
| `Ant::plan_residency` | layer-ordered greedy tier fill |
| `Ant::layer_residency_summary` | per-layer partial-layer tracking (banner) |
| `ant_swap_fit` | pure swap decision (used by the engine's budget-checked swap) |
| `ant_filter_known_units` | wizard unit-id validation |
| `run_interactive_setup` | the `--setup` wizard |

### `src/chat/models/` — runtime & backend resolution

| Item | File | Role |
|------|------|------|
| `ModelRuntime` / `AntTierControl` | mod.rs | backend-agnostic boundary: `forward(&[u32], usize) -> Vec<f32>` + Ant tier control |
| `BackendLoader` / `ModelFamily` / `ModelRegistration` | mod.rs | backend enum + family constants + declarative registration entry |
| `registrations/<family>.rs` | registrations/ | declarative capability (architectures / template / stops / backends / family); build.rs → generated `REGISTRY` |
| `backend_loader_for_architecture` / `vulkan_loader_for_architecture` / `registered_backends` / `family_for_architecture` / `chat_template_for_architecture` / `stop_markers_for_architecture` | mod.rs / registry.rs | lookups derived from the generated `REGISTRY` |
| `load_model` / `load_model_for_device` | registry.rs | device/backend dispatch + Candle CPU retirement policy |
| `VulkanTransformerExecutor` / `VulkanPhiExecutor` | vulkan_transformer.rs | Candle-free Vulkan executors (Ant GPU-bucket residency) |

### CLI/serve

| Item | File | Role |
|------|------|------|
| `AntCliOptions` / `parse_ant_args` / `parse_chat_args` | cli.rs | all `--ant*`/`--setup` parsing |
| `split_ant_args` | main.rs | value-aware ant-flag extraction for serve |
| `serve::serve` / `resolve_model` | serve/mod.rs | server lifecycle, model resolution by name/dir |
| `http.rs` / `openai.rs` / `queue.rs` | serve/ | routing, request schemas, concurrency tickets |

---

## 4. Environment surface

> **All variables below must be visible to the Cheetah process.** Use `export VAR=value` (persistent for the shell session) or `VAR=value ./target/release/cheetah ...` (inline, one command). A bare `VAR=value` without `export` creates a shell variable that child processes cannot see — Cheetah will silently use the default.

### 4.1 Device & backend selection

| Variable | Read by | Values | Effect |
|----------|---------|--------|--------|
| `CHEETAH_DEVICE` | registry.rs, engine.rs | `cpu` / `cuda` / `metal` / `vulkan` | backend selection. `cpu` (default) → native mmap backend for architecture-resolved families (Qwen2, Llama, Qwen3); `cuda`/`metal` → Candle; `vulkan` → independent executor. |
| `CHEETAH_NATIVE` | engine.rs | `0` / `1` | `0` skips native selection (surfaces the guidance error for architectures without a native backend on CPU). Kept for policy testing. |
| `CHEETAH_QUANT_Q8` | vulkan_transformer.rs (take_linear) | `1` / `true` | Quantizes all Linear projections to row-wise int8 (Q8) at load time. Halves weight footprint; norms/biases/embeddings stay in original precision. Vulkan only. |

### 4.2 Vulkan substrate

| Variable | Read by | Values | Effect |
|----------|---------|--------|--------|
| `CHEETAH_VRAM_POOL_SIZE` | cheetah_vulkan.c | bytes (default 800 MiB) | HOST_COHERENT allocator pool size; also filters devices (heap must ≥ pool) |
| `CHEETAH_MATMUL_SPV` | cheetah_vulkan.c | .spv path | overrides the f32 matmul shader |
| `CHEETAH_MATMUL_BF16_SPV` | cheetah_vulkan.c | .spv path | overrides the bf16 matmul shader |
| `CHEETAH_MATMUL_Q8_SPV` | cheetah_vulkan.c | .spv path | overrides the q8 matmul shader |
| `CHEETAH_QWEN_OPS_SPV` | cheetah_vulkan.c | .spv path | overrides the shared_ops shader (rmsnorm/rope/attention/silu/gelu/elementwise) |

### 4.3 Build-time

| Variable | Read by | Effect |
|----------|---------|--------|
| `GLSLC` | build.rs | shader compiler path (default: `glslc` on PATH) |

### 4.4 Testing

| Variable | Read by | Effect |
|----------|---------|--------|
| `CHEETAH_TEST_VULKAN_Q8` | vulkan_transformer.rs (tests) | set `1` to run the q8 resident-matmul forward test (needs a Vulkan device) |
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
cargo test --features vulkan         # + Vulkan executor/shader tests (151 passed, 5 device-gated ignored)
cargo test native_                   # all native-backend tests
cargo test -- --test-threads=4       # parallel (some tests are timing-sensitive)
```

Tests that need local artifacts (auto-skip when absent):
- `ant_lifecycle_against_real_prepared_model` — `/home/benny/models/qwen-0.5b-prepared`

Everything else — including the native kernel / forward tests — is
self-contained and runs anywhere.

---

## 6. Invariants (do not break)

1. **The runtime boundary carries no backend types**:
   `ModelRuntime::forward(&[u32], usize) -> Result<Vec<f32>, String>`.
2. **Architecture resolution contains no runtime code**; the declarative
   registration files in `src/chat/models/registrations/` are the single
   source of truth, build.rs generates `REGISTRY` from them, and every lookup
   (backends, chat metadata, family constants) derives from that table.
3. **`Disk`-tier means real released pages** on the native backend — any change
   that keeps disk-tier bytes resident breaks the Ant contract.
4. **Every swap re-checks the budget before it executes** (`ant_swap_in_unit`,
   `ant_swap_fit`).
5. **Bit-math changes require the isolation tests** (`dot_row_bf16_fused` vs
   dequant+dot) AND a native forward-parity run.
6. **Performance numbers are only valid from a freshly built binary** (see
   ANT_MEMORY §9, lesson 2).
7. **`f32::sin_cos` returns `(sin, cos)`** — the historical rope bug.
8. Backend capability assertions must be updated when
   a family gains/loses a backend.

## Architecture support table

| Architecture | Backends      | Type         | Notes                                          |
|--------------|---------------|--------------|------------------------------------------------|
| llama        | native+vulkan | Dense        | GQA, untied `lm_head`                          |
| qwen2        | native+vulkan | Dense        | attention bias, tied embeddings                |
| qwen3        | native+vulkan | Dense        | per-head q/k norm, GQA, tied embeddings        |
| mixtral      | native+vulkan | MoE (num_experts_per_tok) | router + route-weighted expert mix, per-expert tiers |
| mistral      | native+vulkan | Dense        | decoding-time sliding-window attention         |
| gemma        | native+vulkan | Dense        | `(1 + weight)` norm multiply, sqrt(hidden) embedding scale, GeGLU, tied embeddings |
| phi          | native+vulkan | Dense        | biased LayerNorms, parallel residual, fused `qkv_proj`/`dense` + `fc1`/`fc2`, partial rotary (`VulkanPhiExecutor` on Vulkan) |

`native+vulkan` means both `backend_loader_for_architecture(arch, "native")` and
the Vulkan load path (`vulkan_backends()` + `vulkan_loader_for_architecture`)
resolve; the two Vulkan tables are asserted to agree by
`vulkan_capability_table_and_loader_table_agree`.

Sparse (sliding-window) attention and MoE routing are backend-agnostic: both
runners resolve `use_sliding_window` / `sliding_window` the same way (`false`
or `0` → attend to the whole cache) and route experts with
`chat::moe::select_experts` (softmax → top-k → renormalize).
