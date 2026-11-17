# Model resolution — developer guide

This is the developer documentation for Cheetah's model resolution layer: how
a model architecture declared in a model's own `config.json` resolves to a chat
template, stop markers, backend capability, and family constants, and how the
engine picks a backend.

There is no compile-time model registration anymore. The former declarative
registration files (`src/chat/models/registrations/<family>.rs`) and the
generated `OUT_DIR/model_registry.rs` (`REGISTRY` / `ALL_ARCHITECTURES`) were
retired; architecture names are now the single key the whole stack resolves
against.

For the memory manager and the native backend that implement the Ant tiers,
read [ANT_MEMORY.md](ANT_MEMORY.md) alongside this document.

---

## 1. Overview — resolution layers

```text
src/chat/models/<family>.rs        runtime implementations (native/*, vulkan_transformer.rs)
        │
        ▼
src/chat/models/registry.rs       backend capability tables + family constants + loaders
        │
        ▼
src/chat/models/mod.rs             chat template + stop marker resolution
        │
        ▼
Engine::load / model_metadata_for_init     src/chat/engine.rs, src/chat/mod.rs
```

Resolution order at load time:

1. Read `config.json` from the (prepared or source) model directory; take the
   first entry of `architectures`, falling back to `model_type`.
2. Resolve chat metadata via `chat_template_for_architecture(arch)` and stop
   markers via `stop_markers_for_architecture(arch)` (in `src/chat/models/mod.rs`).
3. Resolve backend capability via the tables in `src/chat/models/registry.rs`:
   - native-capable architectures: `llama`, `mixtral`, `qwen2`, `qwen3` → native
     mmap backend (mixtral carries the mixture-of-experts feed-forward; the
     native forward implements the router + top-k experts)
   - Vulkan-capable architectures: `vulkan_backends()` — llama / qwen2 / qwen3
     / gemma / mistral / mixtral / phi. gemma, mistral and phi are
     Vulkan-only (no native backend); mixtral carries MoE routing on both
     backends.
   - The Vulkan load path (`load_vulkan_or_fallback`) guards on
     `architecture_has_backend(arch, "vulkan")` and then dispatches through
     `vulkan_loader_for_architecture(arch) -> Option<BackendLoader>` — the
     capability table and the loader table are asserted to agree by
     `vulkan_capability_table_and_loader_table_agree`.
   - `backend_loader_for_architecture(arch, backend) -> Option<BackendLoader>`
   - `registered_backends(arch)` → e.g. `["native", "vulkan"]` / `["vulkan"]`
4. Resolve family fallback constants via `family_for_architecture(arch)` for
   native families that need them (llama / mixtral / qwen2 / qwen3).

Architectures are normalized (lowercase + `ForCausalLM` suffix stripped) so
`Qwen2ForCausalLM` and `qwen2` resolve identically.

## 2. Backend capability tables (`src/chat/models/registry.rs`)

- `native_backends()` — architectures served by the native mmap + Ant tiering backend
  (`llama`, `mixtral`, `qwen2`, `qwen3`; mixtral carries the MoE feed-forward).
- `vulkan_backends()` — architectures served by the Vulkan compute executor.
- `vulkan_loader_for_architecture(architecture) -> Option<BackendLoader>` — the
  Vulkan executor loader for one architecture (the dispatch half of the
  Vulkan capability table; both halves are asserted to agree).
- `registered_backends(architecture) -> Vec<&'static str>` — backend names served
  for one architecture.
- `backend_loader_for_architecture(architecture, backend) -> Option<BackendLoader>`.
- `family_for_architecture(architecture) -> Option<ModelFamily>` — fallback
  inference constants for native families.

## 3. Chat metadata (`src/chat/models/mod.rs`)

- `ChatTemplate { id, render, header_renderer }` + `chat_template_for_architecture`.
- `stop_markers_for_architecture`.

## 4. How to add things

### 4.1 A new architecture

1. Add the runtime implementation (`src/chat/models/<name>.rs` native runner or
   the Vulkan executor wiring in `src/chat/vulkan_transformer.rs`).
2. Register backend capability: add to the native/vulkan backend tables, add a
   `vulkan_loader_for_architecture` arm when it has a Vulkan executor (the
   `vulkan_capability_table_and_loader_table_agree` test keeps the two Vulkan
   tables in step), and — for native families — add the family constants +
   profile table.
3. Add the chat template and stop markers to the resolution tables.
4. Add backend capability tests in `src/chat/models/registry.rs` tests.

### 4.2 A new native family

1. Implement the forward deltas in the native runner
   ([ANT_MEMORY.md](ANT_MEMORY.md) §11 checklist).
2. Add the family to the native backend capability table and to
   `family_for_architecture`. For mixture-of-experts families (e.g. mixtral)
   this means teaching the native forward the router + top-k experts block.
3. Extend the native runner / forward tests for the structural deltas and
   add the forward tests.
4. Flip the backend capability test to include the family.

### 4.3 A new backend kind

1. Add a `BackendLoader` variant with the loader signature natural to the
   backend's data path.
2. Add selection logic in `Engine::load` (or `registry.rs`).
3. Add backend-name assertions to the registry tests.

### 4.4 A Vulkan model executor

Vulkan executors are not generated from registrations — the executor
implementation (`vulkan_transformer.rs`) owns the actual compute graph and the
Ant GPU-bucket residency. Follow `VulkanTransformerExecutor` (transformer +
Phi variants) and keep the executor Candle-free (the shared shader library
lives in `*.comp` at the repo root; the shader registry maps names to compiled
artifacts under the feature).

## 5. Design decisions worth remembering

1. **Architecture-name resolution instead of registration files.** The former
   design scanned a `registrations/` directory and generated a registry from
   declarations; the current design keys everything off the architecture name
   that the model's own `config.json` carries, resolved through small capability
   tables in `src/chat/models/registry.rs`.
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
