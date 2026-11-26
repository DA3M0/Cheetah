# Model resolution — developer guide

This is the developer documentation for Cheetah's model resolution layer: how
a model architecture declared in a model's own `config.json` resolves to a chat
template, stop markers, backend capability, and family constants, and how the
engine picks a backend.

The resolution layer is **declarative registration files**. Every file in
`src/chat/registrations/` declares one family's capability —
architectures, chat template, stop markers, backend loaders, family
constants — and hosts the family's runtime forward assemblies (each file
implements `ModelForward` by composing the backend toolboxes: the native
`OpsToolbox` or a Vulkan executor). build.rs collects the declarative
constants into the generated `OUT_DIR/model_registry.rs` (`REGISTRY` /
`ALL_ARCHITECTURES`). All runtime lookups derive from that table: adding a
family is adding a file, with no table edits and no core-code branches
anywhere.

For the memory manager and the native backend that implement the Ant tiers,
read [ANT_MEMORY.md](ANT_MEMORY.md) alongside this document.

The same declarative pattern backs two companion registries:
`src/chat/precision/` (weight formats) and `src/chat/ops/` (the generic
operator catalog, §4.5). Both are discovered by `build.rs` and expose generated
`REGISTRY` tables; adding a format or an operator is adding a file.

---

## 1. Overview — resolution layers

```text
src/chat/registrations/<family>.rs   declarative capability + the family's
        │                            ModelForward assemblies (native + vulkan)
        │  build.rs: discover_registrations + generate_registry
        ▼
OUT_DIR/model_registry.rs          generated REGISTRY / ALL_ARCHITECTURES
        │  include!(...) in src/chat/mod.rs (the <family>_registration
        │  modules are children of chat; they reach the boundary types via super::)
        ▼
src/chat/registry.rs               ModelForward / BackendLoader / ModelRegistration types
                                   + lookups: registration / loader / family / backend
                                   + chat template + stop markers derived from REGISTRY
        │
        ▼
Engine::load / model_metadata_for_init     src/chat/engine.rs, src/chat/mod.rs
```

Resolution order at load time:

1. Read `config.json` from the (prepared or source) model directory; take the
   first entry of `architectures`, falling back to `model_type`.
2. `registration_for_architecture(arch)` finds the declarative registration
   serving that name (normalized: lowercase, `ForCausalLM` suffix stripped —
   `Qwen2ForCausalLM` and `qwen2` resolve identically).
3. Chat metadata derives from the same registration:
   `chat_template_for_architecture(arch)` and
   `stop_markers_for_architecture(arch)` (in `src/chat/registry.rs`).
4. Backend capability derives from the registration's `backends` table:
   - `backend_loader_for_architecture(arch, backend) -> Option<BackendLoader>`
   - `registered_backends(arch)` → the declared backend names in order, e.g.
     `["native", "vulkan"]`
   - `family_for_architecture(arch) -> Option<ModelFamily>` — the family
     constants the native backend's parameter fallback chain reads
   - the Vulkan load path (`load_vulkan_or_fallback`) dispatches through
     `vulkan_loader_for_architecture(arch)`, which reads the same table

Every published family (llama / mixtral / qwen2 / qwen3 / mistral / gemma /
phi) serves `["native", "vulkan"]`; mixtral carries the mixture-of-experts
feed-forward, and mistral / gemma / phi are dense delta families (sliding
window; offset norms + embedding scale + GeGLU; LayerNorm + parallel
residual + fused projections respectively) — expressed as each family's own
forward assembly in its registration file, with `DenseFeatures` remaining as
the resolution metadata the config fallback chain probes.

## 2. The registration file (`src/chat/registrations/<family>.rs`)

A registration file has two halves. The declarative half declares one
`ModelRegistration` constant:

```rust
pub(crate) const ARCHITECTURES: &[&str] = &["LlamaForCausalLM", "llama"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate { id: "llama", render: llama2, header_renderer: Some(llama3) },
    stop_markers: &["<|eot_id|>", "</s>"],
    backends: model_backends(),
    family: Some(llama_family()),
};
```

- **`architectures`** — the names this registration serves; the generated
  `ALL_ARCHITECTURES` index collects them at build time. Names must be unique
  across all registration files (asserted by a test — a duplicate would
  silently shadow one family's template/backends).
- **`chat_template`** — the prompt renderer (`ChatTemplate { id, render,
  header_renderer }`); renderers live in `src/chat/templates.rs`.
- **`stop_markers`** — generation stop token names for the family.
- **`backends`** — `&[(&'static str, BackendLoader)]` in declared order; the
  order is what `registered_backends` reports and what the startup banner
  prints. Both loader entries point into this same file: the family's
  `<family>_native_loader` (opens the mmap store, resolves the decoder
  config, installs the Ant plan, builds the family's `ModelForward` struct)
  and, behind `#[cfg(feature = "vulkan")]`, `<family>_vulkan_loader` (builds
  the executor around the family's declared feature set). There is no
  central `model_type` dispatch left — each registration owns its executor
  entry.
- **`family`** — `Option<ModelFamily>`: fallback inference constants
  (`rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, size
  `profiles: (hidden, vocab, q_heads, kv_heads, head_dim)`,
  `extra_config_json`, `max_position_embeddings`, the sparse-MoE
  `moe_layout`, and the dense `features: DenseFeatures`). Every registration
  that claims the native backend must carry a family (asserted by a test).

The runtime half of the file implements `ModelForward` (+ `AntTierControl`
where the backend owns real pages): the native struct (e.g. `LlamaNative`)
composes the `OpsToolbox` ops in the family's order — see
[ANT_MEMORY.md](ANT_MEMORY.md) §7.2 for the pipeline and
`src/chat/native/ops.rs` for the operator surface — and the
`#[cfg(all(test, feature = "vulkan"))]` module carries the family's Vulkan
feature tests.

The generated `model_registry.rs` (do not edit) declares one
`<family>_registration` module per registration file and assembles:

- `REGISTRY: &[&ModelRegistration]` — the lookup table
- `ALL_ARCHITECTURES: &[&[&str]]` — every architecture name, in registry order

## 3. Runtime lookups (`src/chat/registry.rs` + `mod.rs`)

All of these derive from `REGISTRY` — none of them hardcode family data:

- `registration_for_architecture(arch) -> Result<&'static ModelRegistration, String>`
- `backend_loader_for_architecture(arch, backend) -> Option<BackendLoader>`
- `registered_backends(arch) -> Vec<&'static str>`
- `family_for_architecture(arch) -> Option<ModelFamily>`
- `vulkan_loader_for_architecture(arch) -> Option<BackendLoader>` (feature
  `vulkan`) — reads the same backend table
- `vulkan_backends()` — test-support view (vulkan-gated) listing the
  architectures that declare a Vulkan loader; the native side asserts
  capability through `backend_loader_for_architecture` directly
- `chat_template_for_architecture(arch)` / `stop_markers_for_architecture(arch)`
  (in `mod.rs`) — derived from the registration

The only static table left in `registry.rs` is the **device selection** table
(`BACKENDS`: `cpu` → native, `vulkan` → Vulkan executor), keyed by
`CHEETAH_DEVICE` — that is device dispatch, not model capability.

## 4. How to add things

### 4.1 A new architecture (the common case)

1. **Add `src/chat/registrations/<name>.rs`** with the architecture
   names, chat template, stop markers, backend loaders, family constants,
   and the family's `ModelForward` implementation — copy the closest
   existing file. The forward composes the existing `OpsToolbox` ops in the
   family's order; shared math operators live family-free in
   `native/ops.rs`, so a new shape is new *composition*, not new core code
   ([ANT_MEMORY.md](ANT_MEMORY.md) §11).
2. Extend `DecoderConfig::resolve` only when the family carries config
   fields no existing probe reads.
3. Rebuild: build.rs picks the file up automatically (the directory is the
   contract — no exclusion list, no table edits). The registry tests assert
   name uniqueness, native→family presence, and Vulkan loader agreement.

### 4.2 A new native family

1. Add the registration file with the family constants + profile table
   (`(hidden, vocab, q_heads, kv_heads, head_dim)` for every published size —
   hidden alone is ambiguous within a family) and the family's `ModelForward`
   (the complete native transformer assembly; see §2 above and
   [ANT_MEMORY.md](ANT_MEMORY.md) §11 checklist).
2. Declare the family's deltas in `DenseFeatures` (native/decoder.rs) only
   when the config resolution needs new probes.
3. Add the forward-parity tests (reference math + residency parity, same
   pattern as the gemma / phi suites in native/decoder.rs).

### 4.3 A new backend kind

1. Add a `BackendLoader` variant with the loader signature natural to the
   backend's data path.
2. Reference it from the registration files' `model_backends()` tables.
3. Add backend-name assertions to the registry tests.

### 4.4 A Vulkan model executor

The generic executor machinery lives in `src/chat/vulkan/` —
`transformer.rs` owns the generic compute graph (llama-shaped families,
parameterized by each registration's feature set) and `weights.rs` is the
shared weight layer whose streaming/resident matmuls dispatch through the
precision registry (`crate::chat::precision`). There is no central loader:
each registration file owns its `<family>_vulkan_loader`, building the
generic transformer executor around the family's feature set — and a family
with a different block topology hosts its own executor in its registration
file (phi's executor lives in `registrations/phi.rs`), keeping
one-family-one-file true on both backends. Keep the executors Candle-free (the shared shader library lives in
`*.comp` at the repo root; shaders are discovered from `src/chat/registrations/` at
build time).

### 4.5 A new operator (the `xxx_ops` catalog)

The generic operator catalog lives in `src/chat/ops/`; it carries non-matmul
transformer operators (activations, elementwise, norms, position/embedding,
stateful recurrent steps) on **f32 buffers**. Large linear projections stay in
the precision registry — callers dequantize small norm weights/biases to f32
before dispatching a catalog op.

1. **Add `src/chat/ops/<family>.rs`** declaring `REGISTRATION: OpRegistration`
   with one `OpEntry` per operator id. Each entry lists both backends:
   - `OpBackend::Native(NativeOp::…)` — a plain fn pointer in one of the
     signature families (`Unary`, `InPlaceBinop`, `Binary`, `Norm`, `NormL2`,
     `Softmax`, `Rope`, `Embedding`, `Conv1dStep`, `GatedDeltaStep`).
   - `OpBackend::Vulkan(VulkanOp { … })` — either a shared `shared_ops` op code
     (`0..17`, with `kind` + `sub_op`) or a custom shader.
2. **Custom shader ops** (op code `>= CUSTOM_OP_BASE`, i.e. `100`) set
   `custom_shader: Some(CustomShader { name, env })` and add
   `src/chat/ops/shaders/<name>.comp`. The shader uses the `shared_ops` ABI
   unchanged (5 storage buffers, 8 uint push constants, `local_size_x = 64`,
   entry `main`); `env` names its runtime SPIR-V override variable (e.g.
   `CHEETAH_MISH_OPS_SPV`). A shader compile failure is a build warning, not an
   error — set the override variable to recover without rebuilding.
3. **Rebuild.** `build.rs` discovers every `.rs` file in `src/chat/ops/` (the
   directory is the contract), generates `OUT_DIR/ops_registry.rs`, compiles
   custom shaders into `OUT_DIR/custom_op_shaders.rs`, and the runtime derives
   `native_op_for` / `vulkan_op_for` from the generated table. No core
   dispatch edits.
4. **Tests.** `cargo test ops::` runs the native unit tests plus registry
   invariants (unique ids, both backends, op-kind agreement, custom-shader
   metadata). `cargo test --features vulkan ops::` adds the native↔Vulkan
   parity suite for every operator; it skips cleanly when no Vulkan device is
   present.

## 5. Design decisions worth remembering

1. **Declarative registration files.** The registration directory is the
   single source of truth for what the binary serves; build.rs generates the
   registry, and the runtime derives every lookup from it. The alternative —
   hardcoded capability tables plus per-family functions in `registry.rs` —
   was tried (RC1..RC3) and reverted: it made "add a family" a four-table edit
   across two files, and capability drift (a table claiming a loader that no
   implementation backs) became possible again. The generated-table design
   makes drift a build/test failure instead.
2. **Vocab-keyed profiles.** Hidden size alone is ambiguous within a family
   (llama2-7B vs llama3-8B); the profile key is `(hidden, vocab)`.
3. **Backend-agnostic runtime boundary.** `ModelForward::forward` keeps
   the engine core free of backend AND family types; the registry's loader
   table returns `Box<dyn ModelForward>`, and both the native mmap backend
   and the Vulkan executor implement `AntTierControl` on top of that
   boundary.
4. **Control flow vs. operators split.** The registration files own the
   per-family forward assemblies (residual topology, projection naming,
   activation); the main program owns family-free operators and state
   (`native/ops.rs` OpsToolbox on the CPU side, the generic executors on the
   Vulkan side). Low-level math is statically dispatched free functions with
   frozen names — the only `dyn` is the engine boundary.
5. **Candle CPU path retirement.** The Candle CPU backend is retired (explicit
   guidance error) — CPU belongs to native, Candle remains the CUDA/Metal option
   via `CHEETAH_DEVICE=cuda|metal`.
6. **Vulkan resident bucket.** Vulkan executor devices keep weights in host RAM
   with DEVICE_LOCAL bucketing for VRAM-tier units; oversized uploads fall back
   to the host-coherent streaming path per weight (see
   [ANT_MEMORY.md](ANT_MEMORY.md) §12).
