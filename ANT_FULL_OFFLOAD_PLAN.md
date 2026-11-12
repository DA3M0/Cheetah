# Ant Full Offload Plan

## Status

- **Phase 1 + Phase 4 executed** in `ant patch-1` (atomic-unit full offload: residency
  planning, budget enforcement, disk-tier exclusion, prefetch, page-in).
- **Phase 2 + Phase 3 executed** in `ant patch-2`: layer-ordered residency planning with
  explicit partial-layer tracking, and budget-checked swap-in with runtime bookkeeping
  (VRAM/RAM occupancy, runtime eviction set, promotion queue).
- Remaining future work: MoE routing-aware offload (per-expert residency) and true
  per-unit device placement inside the Candle backend (single device per load).
- Context: Cheetah decided not to support GGUF; Ant memory optimization (VRAM / RAM / Disk tiers) is the active memory-management direction.

## Goal

Implement **full offload** for the Ant memory manager: move model weight residency toward a complete offload pipeline where resident memory holds only what is needed and cold data rests on disk.

## Planned phases

### Phase 1: Full offload — done (`ant patch-1`)

- Enable complete offloading across the Ant tiers (VRAM, RAM, Disk) for prepared Safetensors models.
- Resident set is no longer implicitly "all weights"; it becomes an explicit offload result.

### Phase 2: Partial layers enter VRAM — done (`ant patch-2`)

- Layer-ordered residency planning (`plan_residency` fills tiers layer by layer in
  numeric layer order) so the VRAM boundary falls inside one layer deterministically.
- That boundary layer becomes an explicitly tracked **partial layer**:
  `Ant::layer_residency_summary` reports per-layer VRAM/RAM/Disk unit and byte counts,
  with `full_vram` / `partial` / `full_ram` / `full_disk` states.
- The remainder of a partial layer stays in RAM or on disk, and the startup banner
  reports the layer breakdown.

### Phase 3: In-memory portion about to swap into VRAM — done (`ant patch-2`)

- The engine keeps runtime bookkeeping (`AntRuntimeState`): VRAM/RAM occupancy, the
  runtime eviction set, disk-tier units paged in at runtime, and a **promotion queue**
  of RAM-tier resident units ordered by layer — the in-memory portion next in line
  for promotion into VRAM.
- Every swap goes through the budget-checked `ant_swap_in_unit`: the VRAM pool is
  re-checked before each swap (and after each eviction), residents are evicted to make
  room only when `--ant-allow-vram-swap` permits it, and swaps that cannot fit are
  skipped and counted.
- Promotion (`promote_ant_units`) only swaps a queued unit in when it fits without
  evicting anyone, walking the queue in layer order.
- Overrides, prefetch and budget enforcement all route through the same swap path.

### Phase 4: Cold layers go to disk — done (`ant patch-1`, refined in `ant patch-2`)

- After offload, evict / mark remaining layers as cold (disk tier) so they are not held resident.
- Page cold units back in on demand (including via the configured prefetch stages), now with per-swap budget re-checks and runtime eviction accounting.

## MoE considerations (future)

- Apply the same offload strategy to Mixture-of-Experts (MoE) models (for example Mixtral):
  - Residency granularity per expert rather than only per layer.
  - Routing-aware offloading: keep experts that are likely needed resident and move cold experts to disk.

## Acceptance criteria

- Resident memory stays within configured budgets (VRAM pool and RAM budget).
- Disk-tier units are excluded from loading and paged in on demand.
- Generation-time budget enforcement and prefetch continue to work with the full-offload pipeline.

## Out of scope (current V1 known limitation)

- True per-unit device placement inside the Candle backend (single device per load currently).

## Measured status (2026-11-13, qwen-0.5b prepared, CPU backend, serve RSS)

Ant V1 does **not yet reduce real memory usage**. Measured process RSS is identical
with Ant off, with the 16M/32M wizard preset, and with forced near-full budgets:

| Configuration | RSS after load |
|---|---|
| Ant disabled (`--ant false`) | 2,564,028 kB |
| Ant preset 16M VRAM / 32M RAM (151 units planned to disk) | 2,564,872 kB |
| Ant 300M VRAM / 35M RAM (near-full, 148 units to disk) | 2,564,624 kB |
| Ant 0 / 0 (all units disk → every shard excluded) | load fails: `cannot find tensor model.embed_tokens.weight` |

Three structural reasons, all rooted in the Candle loader:

1. **Sliver effect:** units are per-tensor (layernorms are ~1.8 KB, biases ~256 B),
   so after the big units fill a budget, every remaining layer's tiny units still fit
   into the leftover slivers. Every shard keeps at least one resident unit, so
   `ant_resident_weight_paths` never excludes a file → nothing is ever skipped.
2. **The builder materializes every tensor anyway:** `VarBuilder::from_mmaped_safetensors`
   is per-file, but model construction requests each tensor by name from every mapped
   shard; excluding a required tensor's shard breaks the build (0/0 case above).
   Tier state is invisible to the loader.
3. **DType::F32 conversion dominates:** prepared shards are f16 (~988 MB) but the
   loader converts every tensor to F32, so the materialized model is ~2 GB regardless
   of tiering.

Generation-time page-in / eviction / promotion is likewise bookkeeping-only (tier
metadata flips in the C manager; no bytes move). Making Ant actually save memory
requires loader-side work, in order of realism:

- Keep tensors **mmap-backed** (never copy into owned buffers) and implement the RAM
  tier as page-cache control (`madvise(DONTNEED)` for cold tiers, prefetch as
  `madvise(WILLNEED)`), so the OS owns tiering.
- Or a custom `SimpleBackend` that materializes only resident tensors and pages the
  rest in on demand — which also requires the model forward pass to tolerate absent
  tensors (offload-to-RAM-per-step), i.e. the true per-unit placement work.
