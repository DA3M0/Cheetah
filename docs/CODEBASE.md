# Codebase index — developer map

A navigation map of the repository: every source file, what it owns, its key
entry points, the request data flows, the environment surface, and the test
map. Read this first when landing in the codebase; the deep guides
([ANT_MEMORY.md](ANT_MEMORY.md), [REGISTRY.md](REGISTRY.md)) pick up where
this file points, and the conventions + per-function reference live in
[CODING_STANDARDS.md](CODING_STANDARDS.md).

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
│   │   ├── mod.rs              chat entry (chat()), model_metadata_for_init, generated REGISTRY include
│   │   ├── engine.rs           ★ Engine: load pipeline, generation loop, Ant hooks, MCP tool loop, sampling
│   │   ├── context.rs          conversation archive/store (persistence)
│   │   ├── moe.rs              shared MoE routing (`select_experts`: softmax → top-k → renormalize)
│   │   ├── toolcalls.rs        shared tool-call shapes + parsing (engine ↔ serve ↔ MCP)
│   │   ├── registry.rs         ★ ModelForward trait + BackendLoader + ModelRegistration + derived lookups
│   │   ├── templates.rs        chat template renderers
│   │   ├── registrations/      ★ one file per family: declarative capability (build.rs → generated REGISTRY)
│   │   │                       ★ AND the family's ModelForward assembly (layer stacking) + its loaders
│   │   ├── native/             ★ the native mmap backend's shared infra (family-free)
│   │   │   ├── ops.rs          ★ OpsToolbox: math kernels (gemv/rmsnorm/rope/...) + store/KV-cache/
│   │   │   │                   quant/expert-residency/Ant plumbing — no family branches
│   │   │   ├── store.rs        MmapWeightStore + madvise tier control
│   │   │   ├── kernels.rs      compatibility alias re-exporting ops (precision registry path)
│   │   │   ├── decoder.rs      DecoderConfig::resolve + DenseFeatures/MoEExpertLayout + native tests
│   │   │   └── mod.rs          module glue
│   │   ├── precision.rs        ★ weight-precision registry: shared types, lookups, typed byte views
│   │   ├── precision/          ★ one file per format: operators + the format's declarative registration
│   │   │                       (f32/f16/bf16/q4/q8 → native+vulkan ops; build.rs → generated REGISTRY)
│   │   └── vulkan/             ★ the Vulkan backend (shared executor machinery)
│   │       ├── transformer.rs  generic transformer executor (llama-shaped families) + Ant GPU-bucket residency
│   │       └── weights.rs      shared weight layer + precision-registry dispatch
│   ├── mcp.rs                  ★ MCP client: server configs, tool registry, stdio transport (src/mcp/client.rs)
│   ├── native.rs               FFI: init/manifest update + Rust wrappers over cheetah_*.c
│   ├── serve/                  OpenAI-compatible API (mod/http/openai/queue; axum)
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
                      → <family>_native_loader         src/chat/registrations/<family>.rs
                          OpsToolbox::open             native/ops.rs
                              MmapWeightStore::open      native/store.rs
                              DecoderConfig::resolve     native/decoder.rs (config→units→family)
                              apply_tiers(plan)          madvise per unit
                          <Family>Native::new          the family's ModelForward (layer stacking)
          → chat_loop                        engine.rs (REPL; with --mcp-*: agentic tool loop)
              → generate_inner               engine.rs:1188
                  apply_ant_request_overrides → prefetch → promote → enforce
                  model.forward(&tokens, offset)     ← backend-agnostic boundary (Box<dyn ModelForward>)
                      (native: <Family>Native in registrations/<family>.rs,
                       composing the native/ops.rs OpsToolbox)
                  sample_token                   engine.rs:1527
```

### 2.2 `cheetah serve` (OpenAI API)

```
serve::serve            src/serve/mod.rs:17  (auth checks, resolve_model:140)
  → load_engine_with_progress (same pipeline as chat)
  → axum server         http.rs (SSE streaming, queue.rs concurrency tickets)
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

### `src/chat/native/` — the native backend's shared infra (family-free)

| Item | File | Role |
|------|------|------|
| `OpsToolbox` | ops.rs | shared inference state + operators: weight store access, KV caches, quantized linear dispatch (`linear`), GQA attention (`attention`, sliding-window aware), norms (`rmsnorm`/`rmsnorm_offset`/`layernorm`/`qk_norm`), `add_bias`, `embedding_row_into`, `rope_heads`, `lm_head_into` (chunked Ant sweep), expert residency, `apply_tiers`/`page_in_unit`/`evict_unit` |
| `MmapWeightStore` | store.rs | shard mmap + unit index + tier Mutex + `page_in`/`drop_pages`/`drop_range`/`drop_disk_units_of_layer` |
| `gemv` | ops.rs | row-parallel matrix-vector; AVX2 vs portable dispatch (names frozen; `kernels.rs` re-exports the module for the precision registry) |
| `dot_row_bf16_avx2` | ops.rs | `cvtepu16_epi32 + slli_epi32(16)` widening + FMA |
| `dot_row_bf16_fused(_portable)` | ops.rs | fused bit-widening dot (portable fallback) |
| `rmsnorm` / `rope_in_place` / `rope_heads` / `silu_multiply` / `softmax` | ops.rs | elementwise/attention helpers |
| `DecoderConfig::resolve` | decoder.rs | the parameter chain (config → units → family) |
| `DenseFeatures` / `MoEExpertLayout` | decoder.rs | declarative delta/layout tags the resolution probes read |
| native forward tests | decoder.rs | reference-math + residency-parity suites driving the registration types |

### `src/chat/registrations/` — one family, one file (capability + assembly)

| Item | File | Role |
|------|------|------|
| `<Family>Native` (e.g. `LlamaNative`, `MixtralNative`, `PhiNative`) | registrations/<family>.rs | the family's `ModelForward` implementation: the complete layer stacking (Attention → Norm → FFN → residual) composed from OpsToolbox ops; `forward_moe_block` lives in mixtral.rs, phi's parallel residual in phi.rs, gemma's GeGLU in gemma.rs |
| `<family>_native_loader` | registrations/<family>.rs | `OpsToolbox::open` → family struct → `Box<dyn ModelForward>` |
| `<family>_vulkan_loader` / `<family>_vulkan_config` | registrations/<family>.rs | `#[cfg(feature = "vulkan")]`: the family's own executor entry (no central `model_type` dispatch) |
| family Vulkan feature tests | registrations/<family>.rs | `#[cfg(all(test, feature = "vulkan"))]` modules moved out of the former central loader |

### `src/ant.rs` — the planner

| Item | Role |
|------|------|
| `AntSettings` / `AntUnit` | settings schema + unit deserialization |
| `Ant::plan_residency` | layer-ordered greedy tier fill |
| `Ant::layer_residency_summary` | per-layer partial-layer tracking (banner) |
| `ant_swap_fit` | pure swap decision (used by the engine's budget-checked swap) |
| `ant_filter_known_units` | wizard unit-id validation |
| `run_interactive_setup` | the `--setup` wizard |

### `src/mcp.rs` + `src/mcp/client.rs` — MCP client

| Item | File | Role |
|------|------|------|
| `McpServerConfig` / `parse_server_spec` / `server_configs` / `registry_from_flags` | mcp.rs | `--mcp-server name=command args...` + `--mcp-config <file>` parsing; registry construction (spawn → initialize → tools/list); `None` when nothing configured |
| `McpRegistry` | mcp.rs | Connected servers + flattened tool list; `tool_definitions()` renders OpenAI-style definitions, `call()` routes `tools/call` to the owning server (per-client mutex, `Arc`-shared across request workers); `Drop` kills children |
| `McpClient` / `LineTransport` / `ChildTransport` | mcp/client.rs | JSON-RPC 2.0 over newline-delimited stdio: `initialize` handshake, `tools/list` (paginated), `tools/call`; id-correlated requests with a 30s deadline; reader thread pumps stdout; server→client requests answered with a protocol error |

### `src/chat/` — runtime boundary & backend resolution

| Item | File | Role |
|------|------|------|
| `ModelForward` / `AntTierControl` | registry.rs | backend-agnostic boundary: `forward(&[u32], usize) -> Result<Vec<f32>, String>` + Ant tier control; the registry's loader table returns `Box<dyn ModelForward>` |
| `BackendLoader` / `ModelFamily` / `ModelRegistration` | registry.rs | backend enum + family constants + declarative registration entry |
| generated `REGISTRY` include | mod.rs | build.rs → `OUT_DIR/model_registry.rs`; the `<family>_registration` modules are children of `chat`, so registration files reach the boundary types via `super::` |
| `backend_loader_for_architecture` / `vulkan_loader_for_architecture` / `registered_backends` / `family_for_architecture` / `chat_template_for_architecture` / `stop_markers_for_architecture` | registry.rs | lookups derived from the generated `REGISTRY` |
| `load_model` / `load_model_for_device` | registry.rs | device/backend dispatch + Candle CPU retirement policy |
| `VulkanTransformerExecutor` / `VulkanPhiExecutor` | vulkan/transformer.rs; registrations/phi.rs | Candle-free Vulkan executors (Ant GPU-bucket residency); the phi executor lives in the phi registration file — one family, one file on both backends |
| `WeightData` / `Linear` / `take_linear` / resident upload | vulkan/weights.rs | shared weight layer; streaming/resident matmuls dispatch through the precision registry |

### `src/chat/precision.rs` + `src/chat/precision/` — weight-precision operator registry

| Item | File | Role |
|------|------|------|
| `PrecisionRegistration` / `PrecisionKind` / `PrecisionBackend` | precision.rs | shared registration types; the per-format entries live in `precision/{f32,f16,bf16,q4,q8}.rs`, each file carrying its operators AND its declarative registration; build.rs scans `src/chat/precision/` → generated `REGISTRY` |
| `VulkanOps` | precision.rs | per-format Vulkan operator table: load-time `quantize`, streaming matmul, resident upload / resident matmul, shared-ops format code, shader-override env |
| `NativeOps` | precision.rs | per-format native operator table: `quantize` (quantized kinds) + `gemv` computed in the format's own precision domain |
| `PrecisionBuffer` | precision.rs | quantized buffer abstraction (q8 layout: f32 scale head + packed int8 rows) |
| `selected_format` / `selected_vulkan_ops` / `selected_native_ops` | precision.rs | selector-env driven lookups (`CHEETAH_QUANT_Q8` selects q8) |
| `f32_view` / `u16_view` | precision.rs | typed views over the registry's uniform byte payloads (no copy; vulkan-only) |
| `gemv_q8` / `gemv_q4` / `dequant_values` | native/ops.rs (re-exported by `native/kernels.rs`) | int8/int4 ×scale per-row gemv (same math as `matmul_q8.comp` / `matmul_q4.comp`) + f32 dequant feeding the quantize operators |
| `OpsToolbox::linear` | native/ops.rs | native linear dispatch: quantize-at-first-use + registered gemv; original precisions dispatch by dtype format id |

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
| `CHEETAH_QUANT_Q8` | precision registry (`chat::precision::selected_format`) → both backends | `1` / `true` | Selects the registered `q8` format: Linear projections quantize to row-wise int8 at load time through the format's registered operators — Vulkan (streaming + resident matmul) and native (quantize-at-first-use gemv, source pages released). Halves the weight footprint; norms/biases/embeddings stay in original precision. |
| `CHEETAH_QUANT_Q4` | precision registry (`chat::precision::selected_format`) → both backends | `1` / `true` | Selects the registered `q4` format: Linear projections quantize to row-wise symmetric int4 (two codes per byte, scale = max\|w\|/7) — a quarter of the BF16 footprint. Same dual-backend operator table as q8 (`matmul_q4.comp` pipeline on Vulkan, `gemv_q4` on native). |

### 4.2 Vulkan substrate

| Variable | Read by | Values | Effect |
|----------|---------|--------|--------|
| `CHEETAH_VRAM_POOL_SIZE` | cheetah_vulkan.c | bytes (default 800 MiB) | HOST_COHERENT allocator pool size; also filters devices (heap must ≥ pool) |
| `CHEETAH_MATMUL_SPV` | cheetah_vulkan.c | .spv path | overrides the f32 matmul shader |
| `CHEETAH_MATMUL_BF16_SPV` | cheetah_vulkan.c | .spv path | overrides the bf16 matmul shader |
| `CHEETAH_MATMUL_Q8_SPV` | cheetah_vulkan.c | .spv path | overrides the q8 matmul shader |
| `CHEETAH_MATMUL_Q4_SPV` | cheetah_vulkan.c | .spv path | overrides the q4 matmul shader |
| `CHEETAH_QWEN_OPS_SPV` | cheetah_vulkan.c | .spv path | overrides the shared_ops shader (rmsnorm/rope/attention/silu/gelu/elementwise) |

### 4.3 Build-time

| Variable | Read by | Effect |
|----------|---------|--------|
| `GLSLC` | build.rs | shader compiler path (default: `glslc` on PATH) |

### 4.4 Testing

| Variable | Read by | Effect |
|----------|---------|--------|
| `CHEETAH_TEST_VULKAN_Q8` | vulkan/transformer.rs (tests) | set `1` to run the q8 resident-matmul forward test (needs a Vulkan device) |
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

cargo test                           # default feature tests (161)
cargo test --features vulkan         # + Vulkan executor/shader tests (180 passed, 3 device-gated ignored)
cargo test native_                   # all native-backend tests
cargo test -- --test-threads=4       # parallel (some tests are timing-sensitive)
```

Tests that need local artifacts (auto-skip when absent):
- `ant_lifecycle_against_real_prepared_model` — `/home/benny/models/qwen-0.5b-prepared`

Everything else — including the native kernel / forward tests — is
self-contained and runs anywhere.

---

## 6. Invariants (do not break)

1. **The runtime boundary carries no backend or family types**:
   `ModelForward::forward(&[u32], usize) -> Result<Vec<f32>, String>`;
   the registry's loader table returns `Box<dyn ModelForward>`.
2. **Registration files are the single source of truth** for architecture
   capability AND family forward structure: the declarative entries in
   `src/chat/registrations/` feed the generated `REGISTRY` (every
   lookup derives from it), and each family's `ModelForward` lives in its own
   file. The main program (`native/ops.rs`, executors, engine) contains **no
   family branches** — control flow vs. operators split; the low-level math
   stays statically dispatched free functions with frozen names
   (`gemv`, `rmsnorm`, `rope_in_place`, ...; `native/kernels.rs` re-exports
   them for the precision registry).
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
`vulkan_capability_table_and_loader_table_agree`, and every registered
Vulkan loader is exercised by `vulkan_loaders_reach_family_feature_selection`.

Each row's native forward is the registration file's own `ModelForward`
assembly (`src/chat/registrations/<family>.rs`) driving the shared
`OpsToolbox`; the `DenseFeatures` column of the family constants remains as
the resolution metadata the parameter fallback chain probes.

Sparse (sliding-window) attention and MoE routing are backend-agnostic: both
runners resolve `use_sliding_window` / `sliding_window` the same way (`false`
or `0` → attend to the whole cache) and route experts with
`chat::moe::select_experts` (softmax → top-k → renormalize).
