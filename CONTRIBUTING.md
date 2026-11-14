# Contributing to Cheetah

This guide gets a new contributor from clone to first contribution. The deep
technical guides are [docs/CODEBASE.md](docs/CODEBASE.md) (repository map),
[docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) (memory manager + native backend),
and [docs/REGISTRY.md](docs/REGISTRY.md) (model registry).

---

## 1. Quick start

```sh
# Build (native backend + Candle bridges; no GPU needed)
cargo build --release

# Run the full test suite (self-contained; no model downloads)
cargo test -- --test-threads=4

# Try it end to end (skip the wizard prompts with piped defaults)
printf "16000000\n32000000\n\nlru\n2\ntrue\n\n" | \
  ./target/release/cheetah init <hf-model-dir> /tmp/my-model --setup
printf "hi\n/quit\n" | ./target/release/cheetah chat /tmp/my-model --ant --max-tokens 24
```

Feature builds:

```sh
cargo build --release --features vulkan   # needs glslc (Vulkan shader compilation)
cargo build --release --features cuda     # needs a CUDA toolkit
cargo build --release --features metal    # macOS only
```

Device selection is `CHEETAH_DEVICE` (`cpu` default → native; `cuda`/`metal` →
Candle; `vulkan` → the independent executor). See
[docs/CODEBASE.md](docs/CODEBASE.md) §4 for the full environment surface.

---

## 2. Where to look for what

| Task / question | Document |
|-----------------|----------|
| "Where does X live in the code?" | [docs/CODEBASE.md](docs/CODEBASE.md) (§1 tree, §2 flows, §3 module index) |
| "How do Ant tiers actually free memory?" | [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §7 |
| "How do model parameters resolve without a complete directory?" | [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §4 |
| "How do I register a model / add a backend?" | [docs/REGISTRY.md](docs/REGISTRY.md) §8 |
| "Which backend runs on this machine and why?" | [docs/REGISTRY.md](docs/REGISTRY.md) §5 |
| "The logits look wrong / numerics diverge" | [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §9 (comparison harness + isolation tests) |
| "How do I benchmark honestly?" | [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §10 + lesson 2 |

---

## 3. Common contributions, step by step

### 3.1 Add a native model family (the highest-value contribution)

Full checklist: [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §11 and
[docs/REGISTRY.md](docs/REGISTRY.md) §8.2. Short form:

1. Implement the forward deltas in the shared native runner
   (`src/chat/models/native/qwen2.rs`) — optional units are detected via
   `store.contains`, so families with missing pieces (biases, q/k norms,
   separate lm_head) need no fork.
2. Add the registration file `src/chat/models/registrations/<family>.rs`:
   `backends` gains `("native", super::native::native_family_loader)`,
   `family` gains constants + the published-size profile table.
3. Extend `prepare_synthetic_decoder` (native/mod.rs tests) for the structural
   deltas and add a native-vs-candle comparison test.
4. Update `backend_registration_is_explicit_per_architecture`.
5. Acceptance: comparison logits agree to <1e-3 on the synthetic model, and a
   real prepared model (when available) matches the Candle output.

Reference migrations to copy from: llama (optional bias, untied lm_head) and
qwen3 (per-head q/k RMSNorm).

### 3.2 Touch the kernels or memory control

- Every bit-math change needs the kernel isolation tests to pass
  (`dot_row_bf16_fused_matches_dequant_dot`, the 896-lane full-width test) AND
  a native-vs-candle comparison run — see the acc-mapping bug history in
  [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md) §9.
- Anything that changes what stays resident must keep the Ant contract:
  disk-tier pages are dropped every step (invariant 3 in
  [docs/CODEBASE.md](docs/CODEBASE.md) §6).

### 3.3 Server / CLI work

`serve/` and `cli.rs` are self-contained; the Ant options flow through
`parse_ant_args` (chat) and `split_ant_args` (serve). Registration-facing CLI
changes belong in `parse_ant_args` so init/chat/serve keep one syntax.

---

## 4. Debugging toolbox

### 4.1 Startup diagnostics

The banner and logs answer most "which path am I on" questions:

```
device:        native-cpu          ← backend actually selected
backends:      candle, native      ← what the registry declares for this architecture
ant layers:    0 full-vram, 24 partial, 0 ram, 0 disk
ant swap:      10 queued for VRAM promotion, 0 evicted at load
cheetah-vulkan: device '...' (API ..., discrete=N)   ← vulkan builds only
```

### 4.2 Environment knobs

| Variable | Use when |
|----------|----------|
| `CHEETAH_DEVICE=cpu\|cuda\|metal\|vulkan` | force a device path |
| `CHEETAH_NATIVE=0` | policy testing: skip native selection (Candle CPU is retired — expect the guidance error) |
| `CHEETAH_VRAM_POOL_SIZE=<bytes>` | Vulkan allocator pool size |
| `GLSLC=<path>` | nonstandard shader compiler location |

### 4.3 Known error messages → causes

| Error | Cause | Fix |
|-------|-------|-----|
| `Architecture 'X' has no native CPU backend registered...` | the family has no native implementation yet | use `CHEETAH_DEVICE=cuda/metal`, or contribute the family (§3.1) |
| `cannot determine the attention head layout for hidden size N` | neither config.json nor the family profile covers this size | provide config.json, or add the profile row to the family registration |
| `Invalid Qwen2 configuration: missing field ...` (synth path) | family `extra_config_json` lacks a Candle-struct field | extend the family's `extra_config_json` |
| `Native backend: unknown unit '...'` | the runner requested a unit the family does not carry | make the unit optional (bias/qk-norm pattern) or fix the name |
| `neither tied embeddings nor a separate 'lm_head.weight' tensor` | the config's `tie_word_embeddings` contradicts the units manifest | fix the config or the manifest (re-run `cheetah init`) |
| `no Vulkan device has a compute queue and a sufficiently large ... heap` | no ICD satisfies the pool requirement | install a driver/ICD, or lower `CHEETAH_VRAM_POOL_SIZE` |
| `Failed to read .../config.json ... manifest snapshot is missing '...'` | directory was never `cheetah init`-ed | run `cheetah init` |

---

## 5. Conventions

- **Comments**: implementation comments are English; user-facing strings and
  some section comments are bilingual (EN + 中文) — follow the file's existing
  style.
- **Errors are sentences**, not codes: `"Native backend: cannot determine ..."`,
  and retirement/guidance messages always name the alternative
  (`use CHEETAH_DEVICE=cuda or metal ...`).
- **Registrations are `const`**: everything in a registration file must be
  const-constructible (fn pointers, `&'static` tables).
- **Tests**: any behavioral change ships with a test; numerics changes ship
  with a comparison test; registry changes update the assertion test.
- **Performance claims** require a fresh release build and a like-for-like
  baseline (same token count, same stop behavior) — see the stale-binary and
  9-vs-24-token measurement traps in [docs/ANT_MEMORY.md](docs/ANT_MEMORY.md).
- **Commit style**: short imperative subjects, historically `ant patch-N` /
  `update create` for the memory-manager line; group related work into one
  patch with the tests included.

---

## 6. Contribution review checklist

Before opening a change, verify:

- [ ] `cargo build` + `cargo test` clean (default features)
- [ ] `cargo test --features vulkan` clean if `vulkan_transformer.rs`,
      `registry.rs`, or shader-adjacent code changed
- [ ] New native paths: comparison test added and passing (<1e-3)
- [ ] Ant-touching changes: tier invariants hold
      ([docs/CODEBASE.md](docs/CODEBASE.md) §6)
- [ ] Docs updated: REGISTRY.md §4 table (registration state), ANT_MEMORY.md
      §10 (performance history), CODEBASE.md §1/§3 (new files/items)
- [ ] Banner-visible behavior changes are reflected in `print_model_info`
      tests or output examples
