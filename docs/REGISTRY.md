# Model resolution — developer guide

This is the developer documentation for Cheetah's model resolution layer: how
a model architecture declared in a model's own `config.json` resolves to a chat
template, stop markers, backend capability, and family constants, and how the
engine picks a backend.

The resolution layer is **declarative registration files**. Every file in
`src/chat/models/registrations/` declares one family's capability —
architectures, chat template, stop markers, backend loaders, family
constants — and build.rs collects them into the generated
`OUT_DIR/model_registry.rs` (`REGISTRY` / `ALL_ARCHITECTURES`). All runtime
lookups derive from that table: adding a family is adding a file, with no
table edits anywhere.

For the memory manager and the native backend that implement the Ant tiers,
read [ANT_MEMORY.md](ANT_MEMORY.md) alongside this document.

---

## 1. Overview — resolution layers

```text
src/chat/models/registrations/<family>.rs   declarative capability (no runtime code)
        │  build.rs: discover_registrations + generate_model_registry
        ▼
OUT_DIR/model_registry.rs          generated REGISTRY / ALL_ARCHITECTURES
        │  include!(...) in mod.rs
        ▼
src/chat/models/registry.rs        lookups: registration / loader / family / backend
src/chat/models/mod.rs             chat template + stop markers derived from REGISTRY
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
   `stop_markers_for_architecture(arch)` (in `src/chat/models/mod.rs`).
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
feed-forward, and mistral / gemma / phi are dense delta families driven by
`DenseFeatures` (sliding window; offset norms + embedding scale + GeGLU;
LayerNorm + parallel residual + fused projections respectively).

## 2. The registration file (`src/chat/models/registrations/<family>.rs`)

A registration file intentionally contains **no runtime code** — it declares
one `ModelRegistration` constant:

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
  header_renderer }`); renderers live in `src/chat/models/templates.rs`.
- **`stop_markers`** — generation stop token names for the family.
- **`backends`** — `&[(&'static str, BackendLoader)]` in declared order; the
  order is what `registered_backends` reports and what the startup banner
  prints. `BackendLoader::Native(super::native::native_family_loader)` for the
  native mmap backend; `BackendLoader::Vulkan(super::vulkan_executors::load)`
  behind `#[cfg(feature = "vulkan")]` — one generic loader, executor features
  derived from the config's `model_type`.
- **`family`** — `Option<ModelFamily>`: fallback inference constants
  (`rope_theta`, `rms_norm_eps`, `tie_word_embeddings`, size
  `profiles: (hidden, vocab, q_heads, kv_heads, head_dim)`,
  `extra_config_json`, `max_position_embeddings`, the sparse-MoE
  `moe_layout`, and the dense `features: DenseFeatures`). Every registration
  that claims the native backend must carry a family (asserted by a test).

The generated `model_registry.rs` (do not edit) declares one
`<family>_registration` module per file and assembles:

- `REGISTRY: &[&ModelRegistration]` — the lookup table
- `ALL_ARCHITECTURES: &[&[&str]]` — every architecture name, in registry order

## 3. Runtime lookups (`src/chat/models/registry.rs` + `mod.rs`)

All of these derive from `REGISTRY` — none of them hardcode family data:

- `registration_for_architecture(arch) -> Result<&'static ModelRegistration, String>`
- `backend_loader_for_architecture(arch, backend) -> Option<BackendLoader>`
- `registered_backends(arch) -> Vec<&'static str>`
- `family_for_architecture(arch) -> Option<ModelFamily>`
- `vulkan_loader_for_architecture(arch) -> Option<BackendLoader>` (feature
  `vulkan`) — reads the same backend table
- `native_backends()` / `vulkan_backends()` — test-support views listing the
  architectures that declare each loader
- `chat_template_for_architecture(arch)` / `stop_markers_for_architecture(arch)`
  (in `mod.rs`) — derived from the registration

The only static table left in `registry.rs` is the **device selection** table
(`BACKENDS`: `cpu` → native, `vulkan` → Vulkan executor), keyed by
`CHEETAH_DEVICE` — that is device dispatch, not model capability.

## 4. How to add things

### 4.1 A new architecture (the common case)

1. Implement the runtime — usually nothing: the native runner needs no new
   code when `DenseFeatures` already covers the deltas
   ([ANT_MEMORY.md](ANT_MEMORY.md) §11), and the Vulkan side needs no new
   code when `vulkan_executors.rs` has a `model_type` arm with the right
   executor features. Both sides only change when the family introduces a
   structure no existing arm/flag covers.
2. **Add `src/chat/models/registrations/<name>.rs`** with the architecture
   names, chat template, stop markers, backend loaders, and family constants
   — copy the closest existing file.
3. Rebuild: build.rs picks the file up automatically (the directory is the
   contract — no exclusion list, no table edits). The registry tests assert
   name uniqueness, native→family presence, and Vulkan loader agreement.

### 4.2 A new native family

1. Declare the family's deltas in `DenseFeatures` (native/decoder.rs) if they
   are new; existing flags cover all published dense deltas
   ([ANT_MEMORY.md](ANT_MEMORY.md) §11 checklist).
2. Add the registration file with the family constants + profile table
   (`(hidden, vocab, q_heads, kv_heads, head_dim)` for every published size —
   hidden alone is ambiguous within a family).
3. Extend the native runner / forward tests for structural deltas and add the
   forward-parity tests.

### 4.3 A new backend kind

1. Add a `BackendLoader` variant with the loader signature natural to the
   backend's data path.
2. Reference it from the registration files' `model_backends()` tables.
3. Add backend-name assertions to the registry tests.

### 4.4 A Vulkan model executor

The executor implementation (`vulkan_transformer.rs`) owns the actual compute
graph and the Ant GPU-bucket residency; every registration points its
`vulkan` backend name at the one generic loader (`super::vulkan_executors::load`),
which selects executor features by the config's `model_type` and dispatches to
`VulkanTransformerExecutor` (transformer + Phi variants). Keep the executors
Candle-free (the shared shader library lives in `*.comp` at the repo root;
shaders are discovered from `src/chat/models/` at build time).

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
3. **Backend-agnostic runtime boundary.** `forward(&[u32]) -> Vec<f32>` keeps
   the engine core free of backend types; both the native mmap backend and the
   Vulkan executor implement `AntTierControl` on top of that boundary.
4. **Candle CPU path retirement.** The Candle CPU backend is retired (explicit
   guidance error) — CPU belongs to native, Candle remains the CUDA/Metal option
   via `CHEETAH_DEVICE=cuda|metal`.
5. **Vulkan resident bucket.** Vulkan executor devices keep weights in host RAM
   with DEVICE_LOCAL bucketing for VRAM-tier units; oversized uploads fall back
   to the host-coherent streaming path per weight (see
   [ANT_MEMORY.md](ANT_MEMORY.md) §12).
