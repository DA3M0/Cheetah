# Ant Full Offload Plan

## Status

- **Not executed.** This document describes a future plan; no offload implementation has been carried out yet.
- Context: Cheetah decided not to support GGUF; Ant memory optimization (VRAM / RAM / Disk tiers) is the active memory-management direction.

## Goal

Implement **full offload** for the Ant memory manager: move model weight residency toward a complete offload pipeline where resident memory holds only what is needed and cold data rests on disk.

## Planned phases

### Phase 1: Full offload

- Enable complete offloading across the Ant tiers (VRAM, RAM, Disk) for prepared Safetensors models.
- Resident set is no longer implicitly "all weights"; it becomes an explicit offload result.

### Phase 2: Partial layers enter VRAM

- Implement partial-layer residency granularity (finer than whole atomic units) so only part of a layer's weights reside in VRAM.
- Keep the remainder of the layer in RAM or on disk.

### Phase 3: In-memory portion about to swap into VRAM

- Track the in-memory (resident) layers that are next in line for promotion / swapping into VRAM.
- Enforce the configured VRAM pool while swapping: only swap in as memory becomes available, and re-check the budget before each swap.

### Phase 4: Cold layers go to disk

- After offload, evict / mark remaining layers as cold (disk tier) so they are not held resident.
- Page cold units back in on demand (including via the configured prefetch stages).

## MoE considerations

- Apply the same offload strategy to Mixture-of-Experts (MoE) models (for example Mixtral):
  - Residency granularity per expert rather than only per layer.
  - Routing-aware offloading: keep experts that are likely needed resident and move cold experts to disk.

## Acceptance criteria

- Resident memory stays within configured budgets (VRAM pool and RAM budget).
- Disk-tier units are excluded from loading and paged in on demand.
- Generation-time budget enforcement and prefetch continue to work with the full-offload pipeline.

## Out of scope (current V1 known limitation)

- True per-unit device placement inside the Candle backend (single device per load currently).
