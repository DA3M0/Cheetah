# Ant memory manager — developer guide

This is the developer documentation for Ant, Cheetah's model memory manager,
and for the native CPU inference backend that implements it. It covers the
file formats, the parameter-resolution chain, the residency planner, the
native backend architecture, the SIMD kernels, the testing infrastructure, and
the step-by-step process for adding a new model family.

For the registration mechanism (how a model family declares itself and its
backends), read [REGISTRY.md](REGISTRY.md) alongside this document.

---

## 1. Design goals and context

Cheetah decided early on **not to support GGUF**. Ant (VRAM / RAM / Disk
tiers) is therefore the project's only memory-management direction, and the
native CPU backend is its execution vehicle.

The core idea: after `cheetah init`, a model is split into **atomic units**
(one per tensor). Every unit is assigned a **residency tier**:

| Tier | Meaning | Real memory behavior (native backend) |
|------|---------|----------------------------------------|
| `VRAM` (0) | hot resident | pages stay faulted in across generation steps |
| `RAM` (1) | resident | pages stay faulted in across generation steps |
| `Disk` (2) | cold | pages are dropped with `madvise(MADV_DONTNEED)` and stream back in per step |

On the native backend, tiers act on **real memory pages**, not bookkeeping.
That is the central design point: a `Disk`-tier unit contributes ~0 resident
bytes; its bytes re-fault from the OS page cache on the next access.

History: the first implementation phase ran tiers on the Candle backend, where
they were bookkeeping-only (measured RSS was identical with Ant on or off —
see §13 for the full measurement history). The native backend exists because
Candle's owned-tensor design cannot keep weights mmap-backed.

---

## 2. Terminology

- **Atomic unit** — one Safetensors tensor in a prepared model, identified by
  its tensor name (`unit_id`, e.g. `model.layers.0.self_attn.q_proj.weight`).
- **Prepared model directory** — output of `cheetah init`: aligned shards
  under `.cheetah/weights/`, plus `cheetah_units.json`, `cheetah_manifest.json`,
  a copied `config.json` and `tokenizer.json`.
- **Residency plan** — map `unit_id → tier` produced by the planner.
- **Family** — a model family (llama, qwen2, …) with its own fallback
  inference constants (`ModelFamily`) and native backend support.
- **Runner** — the native forward implementation
  (`Qwen2Native` in `src/chat/models/native/qwen2.rs`); despite the historical
  name it serves the whole decoder family (llama / qwen2 / qwen3 shape).

---

## 3. Prepared-directory file formats

### 3.1 `cheetah_units.json` (written by `cheetah init`, C side)

A JSON array, one entry per tensor. Schema:

```json
{
  "unit_id": "model.layers.0.self_attn.q_proj.weight",
  "name": "model.layers.0.self_attn.q_proj.weight",
  "dtype": "BF16",
  "shape": [896, 896],
  "bytes": 1605632,
  "source_file": "layer_0.safetensors",
  "source_offset": 28000256,
  "shard": "layer_0.safetensors",
  "layer_id": 0
}
```

Notes:

- **`source_offset` is relative to the shard's payload section**, not the file
  start. Absolute byte position = `8 + header_length + source_offset`
  (Safetensors layout: 8-byte LE header length, JSON header, payload).
- **`layer_id` semantics**: tensors whose name matches `layers.N` get `N`;
  everything else (embeddings, final norm, head) shares `layer_id 0` with the
  first decoder layer. The decoder layer count is `max(layer_id) + 1`.
- `dtype` is the *stored* dtype of the prepared shards (typically `BF16`).

### 3.2 `cheetah_manifest.json` (written by `cheetah init`)

The manifest makes a prepared directory **self-describing**. The `model`
object is the authoritative parameter snapshot:

```json
{
  "format": "cheetah-vulkan-safetensors-v1",
  "alignment": 4096,
  "model": {
    "model_type": "qwen2",
    "num_hidden_layers": 24,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": true
  },
  "chat_template": "qwen",
  "stop_markers": ["<|im_end|>", "<|endoftext|>"],
  "shards": [ { "group": "layer_0.safetensors", "file": "...", "size": 29843456 } ],
  "tensors": [ ... ]
}
```

Every heuristic-derived field carries a `_source` annotation:

- `model_type_source`: `"source config.json"` or `"tensor-name heuristic"`.
- `num_attention_heads_source`: `"source config.json"`,
  `"inferred from q_norm and q_proj tensor shapes"` (exact), or
  `"estimated from hidden_size; verify"` (heuristic — the config path takes
  precedence when available).
- `max_position_embeddings_source`: `"source config.json"` or
  `"default (4096)"`.
- `rope_theta_source` / `rms_norm_eps_source` /
  `tie_word_embeddings_source`: `"source config.json"` (present when the
  source config carried the field).

Source layout in `cheetah_init.c`: `json_number_field_from_file` /
`json_bool_field_from_file` snapshot the fields out of the source
`config.json` during init; `finalize_inference` applies config values over the
shape heuristics (heuristic `head_dim`: `hidden >= 2048 → 128`, else 64).

### 3.3 `cheetah_ant_settings.json` (user-managed)

Written by the interactive wizard (`--setup`) or by init defaults. Schema:

```json
{
  "enabled": true,
  "vram_pool_bytes": 16000000,
  "ram_budget_bytes": 32000000,
  "disk_dir": null,
  "resident_units": [],
  "eviction_policy": "lru",
  "prefetch_stages": 2,
  "allow_vram_swap": true
}
```

- `resident_units` are **pinned** units (VRAM tier in the planner).
- `prefetch_stages` = units paged back in per generation step.
- `allow_vram_swap` = permit evict-to-fit during VRAM-targeted swap-ins.
- `eviction_policy` (`lru|clock|clock2`) is metadata today.

Resolution order (see `resolve_ant_settings` in `src/main.rs`):
**defaults → preset file (when `enabled`) → CLI overrides**. `--setup` /
`--ant-setup` runs the interactive wizard first and persists its preset; all
three entry points (init / chat / serve) then execute with exactly those
settings.

---

## 4. Parameter-resolution chain

The native backend must keep working when the model directory's
`config.json` is missing or incomplete ("directory completeness"). Parameters
resolve field-by-field through a three-level fallback:

```
config.json field  →  units-manifest derivation  →  registered family default
```

1. **config.json** (authoritative when present). `Engine::load` reads it; if
   the file is missing, `synth_config_from_manifest` rebuilds a minimal config
   from the manifest snapshot, merging the family's `extra_config_json`
   (Candle-struct fields such as `hidden_act`, `sliding_window`) and family
   fallback values for any field the snapshot does not carry (backwards
   compatible with manifests written before the snapshot was extended).
2. **units-manifest derivation** (`Qwen2Config::resolve`):
   - `hidden_size`, `vocab_size` ← `model.embed_tokens.weight` shape
     (`[vocab, hidden]`),
   - `num_hidden_layers` ← `max(layer_id) + 1`,
   - `intermediate_size` ← `mlp.gate_proj.weight` out-features,
   - `q_out`/`k_out` ← `self_attn.q_proj.weight` / `k_proj.weight`
     out-features.
3. **family defaults** (`ModelFamily`, see [REGISTRY.md](REGISTRY.md) §5):
   `rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, and the **size
   profile table**.

The one thing that cannot be derived from shapes is the
**q_heads / kv_heads / head_dim split** (hidden 4096 could be 32×128 or
28×128, and hidden alone is ambiguous inside a family — llama2-7B and
llama3-8B share hidden 4096 but differ in vocab and kv heads). The family
`profiles: &[(hidden_size, vocab_size, q_heads, kv_heads, head_dim)]` table
declares it for published sizes; when neither config nor profile resolves it,
the backend errors with a clear message instead of guessing.

---

## 5. Residency planning

`Ant::plan_residency` (in `src/ant.rs`) is a pure function:
`(units, settings) → BTreeMap<unit_id, tier>`.

Algorithm:

1. Units whose `unit_id` appears in `settings.resident_units` are **pinned**
   and sorted first (they get VRAM unconditionally).
2. Remaining units sort by `(layer_id, unit_id)` — **numeric layer order**, so
   the VRAM→RAM boundary falls inside one layer deterministically. (The first
   implementation sorted by `unit_id` string, which scattered `layers.10`
   before `layers.2`; layer-ordered fill makes partial layers meaningful.)
3. Greedy fill per unit:
   - `bytes > vram_pool` → never fits VRAM,
   - else if `vram_used + bytes <= vram_pool` → `vram`,
   - else if `ram_used + bytes <= ram_budget` → `ram`,
   - else → `disk`.

Because units are per-tensor and some are tiny (layernorms ~1.8 KB, biases
~256 B), a layer whose budget sliver still admits its small units becomes a
**partial layer** (some units VRAM, rest RAM/Disk). Partial layers are
explicitly tracked: `Ant::layer_residency_summary` returns per-layer
VRAM/RAM/Disk unit+byte counts with states `full_vram` / `partial` /
`full_ram` / `full_disk`, and the startup banner prints the breakdown
(`ant layers: 0 full-vram, 24 partial, ...`).

`ant_apply_budget` runs after planning at load time: when the total resident
bytes exceed the budget (and the device is an accelerator), units are evicted
lowest-priority-first (RAM tier before VRAM, higher layers first; pinned
units never move).

---

## 6. Load pipeline (`Engine::load`)

The full sequence, in order:

1. Read `config.json` (or synthesize from the manifest — §4).
2. Parse `ModelConfig` (model_type → architecture registration lookup,
   `max_position_embeddings`, eos ids).
3. Read tokenizer, resolve chat template / stop markers.
4. Resolve Ant settings (§3.3).
5. Open the units manifest and plan residency (`plan_residency`).
6. Apply the budget (`ant_apply_budget`) → `ant_evicted` list.
7. Derive the layer summary (§5) and seed the runtime bookkeeping (§7.4).
8. **Native selection**: on CPU devices, look up
   `backend_loader_for_architecture(architecture, "native")` (see
   [REGISTRY.md](REGISTRY.md) §7). When registered, the native loader
   receives `(model_dir, config_bytes, raw_config, plan, family)` and builds
   the runtime with tiers applied to real pages.
9. Otherwise fall through to the device backends: CUDA/Metal → Candle;
   Vulkan → the independent executor. The Candle CPU path is retired
   (an explicit guidance error, not a silent fallback).
10. Seed `AntRuntimeState` from the plan (§7.4) and print the startup banner
    (`ant plan`, `ant layers`, `ant swap`, `backends`).

---

## 7. Native backend architecture

Files (all under `src/chat/models/native/`):

| File | Role |
|------|------|
| `store.rs` | `MmapWeightStore`: shard mapping + unit index + tier table + madvise control |
| `kernels.rs` | compute kernels: gemv, rmsnorm, rope, softmax, silu, dequant |
| `qwen2.rs` | the decoder runner (`Qwen2Native` — historical name; serves llama / qwen2 / qwen3 shape) + `Qwen2Config` resolution |
| `mod.rs` | `NativeCpuRuntime` (ModelRuntime + AntTierControl bridge), registry-facing loader, synthetic-model tests |

### 7.1 `MmapWeightStore`

- Opens every shard referenced by the units manifest (one `memmap2::Mmap`
  each), parses the Safetensors header (8-byte LE length + JSON) and records
  `data_start = 8 + header_len` per shard.
- Unit → absolute byte range: `data_start + source_offset`, length `bytes`.
- `slice(unit_id) -> (&[u8], WeightDType, &[usize])` — zero-copy view;
  `unit_shape` / `contains` / `max_layer_id` support parameter derivation.
- **Tier state** lives in a `Mutex<HashMap<String, Tier>>` (the store is
  shared as `Arc` across runtime clones; interior mutability required).
- **Page control**:
  - `page_in(unit)` — touches one byte per page so RSS reflects the tier
    immediately;
  - `drop_pages(unit)` / `drop_range(unit, start, len)` —
    `madvise(MADV_DONTNEED)`; the mapping stays valid and re-faults on the
    next access;
  - `drop_disk_units_of_layer(layer)` — called after every layer's step so a
    generation holds only the planned resident bytes (plus one transient
    layer);
  - `run_lm_head` sweeps the embedding in 16k-row chunks and drops each chunk
    (the 272 MB embedding is the largest transient).

### 7.2 Forward pipeline (`Qwen2Native::forward`)

Per position (looped over the token sequence):

```
hidden = embed_row(token)                       // gather hidden bf16 → f32
for each layer:
    normed = rmsnorm(hidden, input_layernorm)   // f32 activations
    q = gemv(q_proj, normed) [+ bias]           // optional bias (llama: none)
    k = gemv(k_proj, normed) [+ bias]
    v = gemv(v_proj, normed) [+ bias]
    [q_norm per head] [k_norm per head]         // qwen3-style, optional, pre-rope
    rope(q, pos), rope(k, pos)                  // rotate-half, matches candle
    kv cache append (post-rope keys)
    attention: GQA (q_heads/kv_heads groups), causal via cache length,
               4-accumulator softmax-weighted value sum
    hidden += o_proj(attn_out)
    normed = rmsnorm(hidden, post_attention_layernorm)
    hidden += down(silu(gate(normed)) ⊙ up(normed))
    drop_disk_units_of_layer(layer)             // Ant: release cold pages
hidden = rmsnorm(hidden, model.norm)
logits = lm_head(hidden)                        // tied → embed rows; untied → lm_head.weight
drop_disk_units_of_layer(0)                     // embed/norm bucket
```

- **KV cache**: per layer `Vec<f32>` flattened as
  `[position * kv_heads * head_dim + index]`, f32.
- **Optional units** are resolved by `store.contains` — attention biases and
  q/k norms are family-optional; tied-vs-untied embeddings likewise. This is
  what lets one runner serve llama (no bias, untied), qwen2 (bias, tied) and
  qwen3 (no bias, q/k norms, tie varies).
- **`Qwen2Config::resolve`** implements the parameter chain (§4).
- `forward` returns the final-position logits only (`Vec<f32>`), matching the
  backend-agnostic `ModelRuntime::forward(&[u32], usize)` boundary.

### 7.3 Compute kernels

- `gemv(weight_bytes, dtype, rows, cols, x, y)` — rows ≤ `GEMV_CHUNK_ROWS`
  (128) run sequentially (rayon fork/join would exceed the compute); larger
  matrices split into 128-row rayon chunks.
- `dot_row_bf16_fused` — the hot path. Two implementations behind one call
  site:
  - **AVX2+FMA intrinsics** (x86_64, `is_x86_feature_detected!` guarded,
    `#[target_feature(enable = "avx2,fma")]`): one 256-bit load = 16 bf16
    lanes; each 128-bit half is zero-extended (`cvtepu16_epi32`) and shifted
    (`slli_epi32(16)`) — the u32 lane becomes `bf16 << 16`, which bit-casts to
    the f32 value with lanes preserved — then two `_mm256_fmadd_ps` per
    16-element chunk; horizontal reduction once per row (store + sum).
  - **Portable fused loop** (`as_chunks::<8>` row × `as_chunks::<4>` x, four
    accumulators, pure bit arithmetic). On AVX2 hardware LLVM compiles this to
    equivalent vector code (measured parity — see §10), so the portable path
    is the correctness reference and the non-AVX2 fallback in one.
- `rmsnorm`, `rope_in_place` (rotate-half; **`f32::sin_cos` returns
  `(sin, cos)` — historically swapped, guarded by a comment and tests**),
  `silu_multiply`, `softmax` (stable, in place).

### 7.4 Engine-side runtime accounting (`AntRuntimeState`)

Phase-3 bookkeeping that the generation hooks keep up to date (in `Engine`,
behind a `Mutex`):

- `vram_used_bytes` / `ram_used_bytes` — live occupancy;
- `evicted` — units evicted at runtime (start = load-time evictions);
- `runtime_resident` — disk-tier units paged in by overrides;
- `promotion_queue` — RAM-tier resident units in layer order, next in line
  for VRAM promotion;
- counters (`swaps_in` / `swaps_out` / `swap_skips`).

Every generation: `apply_ant_request_overrides` → `prefetch_ant_units` →
`promote_ant_units` → `enforce_ant_budget`. Each swap routes through
`ant_swap_in_unit`, which **re-checks the budget before executing** (and again
after each eviction), evicts residents only when `--ant-allow-vram-swap`
permits it, and updates the engine accounting plus the real pages via
`AntTierControl` (`page_in_unit` / `evict_unit`).

---

## 8. C layer (`cheetah_memory.c`)

The C manager predates the native backend and today fills two roles:

1. **Manifest/units parsing** (`cheetah_ant_load_model`,
   `load_units_from_json`) and metadata queries (`unit_tier`, `unit_list`).
2. **Bookkeeping target** for the engine hooks: `page_in` / `evict_unit` /
   `unit_pin` flip tier integers (the native backend performs the real page
   operations through `AntTierControl`).

It also owns the prepared-directory check (`model_is_initialized` —
`.cheetah/weights` presence) and the not-initialized warning. Known semantic
quirks: `unit_pin(true)` sets tier RAM (not VRAM), and the unit index append
must copy `tier`/`source_offset` (fixed in ant patch-2 — the index previously
contained uninitialized tiers from realloc memory).

The Vulkan substrate (`cheetah_vulkan.c`) is a separate C component (device
init, VRAM allocator, matmul/rope/attention shaders) consumed by the
independent Vulkan executor — see [REGISTRY.md](REGISTRY.md) §8 for the
feature build. `cheetah_vulkan.c` logs the selected device at startup
(`cheetah-vulkan: device '...' (API ..., discrete=N)`); the executor requires
a HOST_VISIBLE|HOST_COHERENT heap, so weights stay in host RAM and the GPU
runs compute shaders only (measured: GT 730M 0.3 tok/s vs 8 tok/s native CPU
— see §10).

---

## 9. Testing infrastructure

- **Synthetic decoder generator** (`native/mod.rs::prepare_synthetic_decoder`):
  builds a tiny model in-process (Rust `safetensors::serialize` — 2 layers,
  hidden 64, GQA 4/2, head_dim 16, vocab 128, deterministic pseudo-random
  weights), runs `cheetah init` over it, and compares **native vs candle**
  logits. Parameters: `tie` (tied/untied lm_head) and `qk_norm` (qwen3 per-head
  norms). Current harnesses: llama (no bias, untied) and qwen3 (q/k norms,
  explicit head_dim).
- **Real-model comparisons** (`native/qwen2.rs` tests): qwen-0.5b prepared
  model, native vs the internal Candle loader (numerical oracle) — full-width
  logits and a stage-wise layer-0 capture (`DebugCapture`: normed / q_raw /
  q_rope / scores / attn_out) that pinpoints which stage diverges.
- **Kernel isolation tests** (`kernels.rs`): `dot_row_bf16_fused` vs
  dequant+dot on exact bf16 integers (8 and 896 lanes — the full-width test
  exercises AVX2 tail handling). These caught the acc[1]/acc[3] word-mapping
  bug immediately.
- The lifecycle test (`ant_lifecycle_against_real_prepared_model`) skips
  itself when the local prepared model is absent.

Lessons worth keeping:

1. **Bit-math paths need isolation tests** — the AVX2/fused acc mapping bug
   was invisible in end-to-end numbers until the exact-integer test.
2. **Never measure performance from a stale binary** — two measurement rounds
   ran against failed builds; always confirm zero compile errors first.
3. **Comparison against a second implementation is the oracle** — the
   sin/cos swap bug was caught by the candle-reference test, not by review.

---

## 10. Performance history (qwen-0.5b, CPU, 16M/32M tiered unless noted)

| Iteration | tok/s | Change |
|-----------|-------|--------|
| Candle baseline (F32 materialized) | 3.5–4.0 | retired CPU path; 2.5 GB RSS |
| Native, scalar `from_le_bytes` gemv | 0.40 | first correct version |
| + dequant scratch + 4-acc f32 dot | 1.46 | bulk dequant, vectorizable |
| + 8-lane accumulator (`as_chunks::<8>`) | 2.9 | fixed-width inner loop vectorizes |
| + fused bf16 (bit widening in accumulator) | 3.7 | parity with retired Candle path |
| + AVX2 explicit intrinsics | 3.0–3.7 | parity (LLVM already auto-vectorized the portable path) |
| Machine idle, latest build | **7.9–8.4** | same binary, unloaded host |

Memory (16M/32M tiered): steady RSS **123 MB**, inference peak **152 MB**
(lm_head chunked with per-chunk page release), vs Candle 2,497/3,704 MB.

Ant tiering overhead vs full-resident: ~10 % (page-cache re-faults; per-layer
`DONTNEED` keeps transients to one layer plus the chunked lm_head sweep).

Vulkan executor devices (HOST_COHERENT design — weights in host RAM, shaders
on GPU): GT 730M 0.3 tok/s, Lavapipe and Haswell iGPU verified functionally.

---

## 11. Adding a new family — checklist

1. **Implement the forward differences** in the native runner (only the
   deltas: e.g. gemma's `(1 + weight)` norm multiply, mistral's sliding
   window, mixtral's router/experts). Reuse everything else.
2. **Extend `Qwen2Config::resolve`** if the family carries new config fields.
3. **Write the registration file** `src/chat/models/registrations/<family>.rs`
   — see [REGISTRY.md](REGISTRY.md) §8: `backends` gains the native loader,
   `family` gains constants + profiles (`(hidden, vocab, q, kv, head_dim)` for
   every published size; hidden alone is ambiguous within a family).
4. **Extend the synthetic generator** (`prepare_synthetic_decoder`) for the
   family's structural deltas and add the comparison test.
5. **Run the comparison** — native vs candle logits must agree to <1e-3 before
   the family is announced.
6. Update the registration assertion test
   (`backend_registration_is_explicit_per_architecture`) and this document.

---

## 12. Known limitations

- llama3 rope-scaling is not implemented (llama2-style theta only).
- Untied `lm_head` is supported; tied/untied is driven by
  `tie_word_embeddings`.
- Batched prefill (one weight pass over all prompt tokens) is pending —
  prefill currently walks positions sequentially, re-reading resident pages
  per position.
- MoE routing-aware offload (per-expert residency) is the remaining item from
  the original full-offload design.
- The C-side `eviction_policy` (`lru|clock|clock2`) is metadata only; the
  native backend's effective policy is tier-driven.

---

## 13. Measurement history — why the native backend exists

The first phase ran tiers on the Candle CPU backend. Measured (2026-11-13,
qwen-0.5b, serve RSS): Ant off, the 16M/32M preset, and forced near-full
budgets all produced identical RSS (~2,565 MB), and all-disk budgets failed
with `cannot find tensor`. Three structural causes, all rooted in Candle's
owned-tensor design:

1. **Sliver effect**: tiny units from every layer fit into budget slivers, so
   every shard keeps a resident unit and file-level exclusion never fires.
2. **The builder materializes every tensor**: construction requests each
   tensor by name from all mapped shards; excluding a required shard breaks
   the build.
3. **DType::F32 conversion dominates**: BF16 (~988 MB) shards upconvert to
   ~2 GB regardless of tiering.

A custom `SimpleBackend` could not fix this (Candle tensors always own their
bytes; `from_raw_buffer` copies). The native backend keeps weights mmap-backed
instead — that is the entire reason it exists.
