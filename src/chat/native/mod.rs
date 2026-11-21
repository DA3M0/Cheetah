//! Native CPU inference backend. Weights stay mmap-backed from
//! the prepared Ant shards and every Ant tier acts on real pages via madvise,
//! so the resident set only holds what the plan keeps resident.
//!
//! Module layout (the main-program side of the decoder split — the per-model
//! forward assemblies live in the registration files):
//!
//! - [`ops`] — the OpsToolbox: math kernels (`gemv`, `rmsnorm`, `rope`,
//!   ...) plus the shared inference state (store access, KV caches,
//!   quantized linear dispatch, routing-aware expert residency, Ant tier
//!   plumbing). No family branches.
//! - [`decoder`] — `DecoderConfig` resolution (config.json → units manifest
//!   → family constants) plus the declarative feature/layout tags and the
//!   native forward test suite.
//! - [`store`] — the mmap weight store and its madvise tier control.
//! - [`kernels`] — compatibility alias re-exporting [`ops`].

pub(super) mod decoder;
pub(crate) mod ops;
// kernels and store are reached by the precision operator registry
// (src/chat/precision/<format>.rs wires the registered gemv operators).
pub(crate) mod kernels;
pub(crate) mod store;
