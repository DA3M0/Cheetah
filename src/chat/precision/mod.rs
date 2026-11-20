//! Weight-precision operators under the declarative registration mode.
//!
//! Every file in `src/chat/precision/registrations/` declares one weight
//! precision — original storage precisions (f32 / f16 / bf16) and quantized
//! formats (q8) — with its id, the environment selector that enables it
//! (quantized formats only), and the per-backend operator table (load-time
//! quantize, gemv/matmul streaming and resident, resident upload). build.rs
//! collects them into the generated `OUT_DIR/precision_registry.rs`
//! (`REGISTRY`); both backends derive every dispatch from that table: adding
//! a precision is adding a file, no table edits.
//!
//! Compute contract: every operator computes in the weight's own precision
//! domain — f32 rows dot in f32, bf16 rows run the fused bf16 kernel, f16
//! rows decode IEEE half per row, and a quantized format computes from its
//! packed codes and scales. No operator converts the weight into another
//! storage width (there is no upcast-to-FP16 step anywhere).
//!
//! How dispatch flows through the registry:
//!
//! ```text
//! checkpoint dtype (safetensors metadata) ──► format id ("f32"/"f16"/"bf16")
//! selector env (CHEETAH_QUANT_Q4/Q8)      ──► selected_format()      [quantized only]
//!                 │
//!                 ▼
//! registration_for_format ──► PrecisionRegistration.backends
//!                 │                             ├─ "vulkan" → VulkanOps  (vulkan/weights.rs)
//!                 │                             └─ "native" → NativeOps  (native decoder)
//!                 ▼
//! per-matmul operator call (fn pointers — zero dynamic dispatch per tile
//! beyond the one table lookup the caller caches)
//! ```
//!
//! Format codes reported to the C substrate (`cheetah_vulkan.c`): the
//! shared_ops encoding is f32 = 0, f16 = 2, bf16 = 3 (decoded by
//! `shared_ops.comp::load_weight`); quantized formats continue the sequence
//! with q8 = 4 and q4 = 5, served by their dedicated matmul pipelines
//! (`matmul_q8.comp` / `matmul_q4.comp`).
//!
//! What lives where: `registrations/<format>.rs` declares (no runtime
//! code); `<format>.rs` implements the operators; `bytes.rs` restores typed
//! views over the uniform byte payloads; the executors
//! (`src/chat/vulkan/weights.rs`, `src/chat/models/native/decoder.rs`)
//! consume the tables.

use std::sync::Arc;

mod bf16;
#[cfg(feature = "vulkan")]
mod bytes;
mod f16;
mod f32;
mod q4;
mod q8;

/// Which compute domain a format's operators run in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrecisionKind {
    /// Original storage precision: the weight bytes are f32 / f16 / bf16
    /// values and every operator reads them directly in that precision.
    Original,
    /// Quantized representation: the weight is packed codes plus per-row
    /// scales, produced by the registered load-time `quantize` operator, and
    /// every operator computes from that representation.
    Quantized,
}

/// A quantized weight buffer: the GPU/CPU-facing byte layout plus the shape
/// the matmul operators need. One implementation per registered quantized
/// format (see `src/chat/precision/<format>.rs`); the executors hold these
/// behind `WeightData::Quantized` / the decoder's quantized-unit table and
/// dispatch through the format's registered operator table.
///
/// Layout contract (identical on both backends, and mirrored by the C
/// substrate and the shaders): the buffer is a head of `rows` f32 row
/// scales (raw little-endian bits, one word per row) followed by the packed
/// payloads, row-major, each row padded so its byte stride is a multiple of
/// four — GPU shaders read whole u32 words per row, so the stride formula
/// lives in the format implementation (`row_bytes`) and must stay in sync
/// with the shader and the C entry points. q8 pads `cols` int8 codes to a
/// 4-byte stride; q4 packs two int4 codes per byte and pads to the same
/// stride rule.
///
/// `Send + Sync` because the Vulkan executor clones weight data across
/// threads (rayon tiles) and the native decoder shares buffers through
/// `Arc` inside `QuantState`.
#[allow(dead_code)] // buffer diagnostics (format_id/dequant_value) serve the
// Vulkan executor and the tests; the CPU gemv reads rows/cols/buffer only.
pub(crate) trait PrecisionBuffer: Send + Sync {
    /// The registered format id this buffer was produced by (e.g. "q8").
    fn format_id(&self) -> &'static str;
    /// Matmul output features (`n`): one quantized row per output channel.
    fn rows(&self) -> usize;
    /// Matmul reduction width (`k`): elements per quantized row.
    fn cols(&self) -> usize;
    /// The complete buffer (scales head + packed payloads).
    fn buffer(&self) -> &[u8];
    /// Padded per-row byte stride of the packed payloads.
    fn row_bytes(&self) -> usize;
    /// Dequantized element value (tests and diagnostics).
    fn dequant_value(&self, row: usize, col: usize) -> f32;
    /// Total logical elements (`rows * cols`).
    fn element_count(&self) -> usize {
        self.rows()
            .checked_mul(self.cols())
            .expect("quantized buffer element count overflow")
    }
}

/// The Vulkan operator set one registered precision serves: everything the
/// executor needs to quantize (quantized formats), upload, and run matmuls
/// in that precision.
///
/// Two matmul paths share one buffer layout:
/// - **streaming** (`matmul_streaming`): the weight bytes stage through the
///   HOST_COHERENT pool on every call — the default path and the fallback
///   when a resident upload is refused (e.g. the buffer would exceed the
///   device's storage-buffer range).
/// - **resident** (`upload` + `matmul_resident`): VRAM-tier units upload
///   once into the DEVICE_LOCAL bucket (or a persistent host-coherent
///   allocation when device-local memory is unavailable) and stay bound;
///   only activations and outputs pass through the pool per call. The
///   executor releases the handle via `weight_release` when Ant evicts the
///   unit.
///
/// All function pointers are plain `fn` — the tables are `Copy` consts in
/// `&'static` registrations, so a dispatch costs one table lookup, not a
/// vtable call.
#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
pub(crate) struct VulkanOps {
    /// Quantize a full-precision weight matrix (row-major, `rows * cols`
    /// values) into this format at load time. `None` for original
    /// precisions — their weights are never converted.
    pub(crate) quantize:
        Option<fn(&[f32], rows: usize, cols: usize) -> Result<Arc<dyn PrecisionBuffer>, String>>,
    /// Host-streaming matmul: `weight` is this format's raw weight bytes for
    /// the tile being multiplied (typed views are restored inside the
    /// format's implementation without copying).
    pub(crate) matmul_streaming:
        fn(input: &[f32], weight: &[u8], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String>,
    /// Upload a weight buffer into the Ant resident bucket; `row_bytes` is
    /// the padded per-row stride quantized formats validate against (0 for
    /// original precisions). `None` keeps the per-step streaming path.
    pub(crate) upload: Option<fn(weight: &[u8], row_bytes: usize) -> Option<u64>>,
    /// Resident matmul: the weight stays bound on the GPU behind `handle`
    /// (from `upload`); only activations and outputs pass through the pool.
    pub(crate) matmul_resident: Option<
        fn(handle: u64, input: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String>,
    >,
    /// shared_ops weight-format code reported to the C substrate
    /// (f32 = 0, f16 = 2, bf16 = 3, q8 = 4).
    pub(crate) weight_format: u32,
    /// Runtime shader-path override variable for this format's matmul
    /// pipeline — declared registration metadata, consumed by the C
    /// substrate through `getenv` (cheetah_vulkan.c) and asserted by the
    /// registry tests. The f16 matmul shares the bf16 pipeline — the shader
    /// decodes both 16-bit layouts, selected by a push constant.
    #[allow(dead_code)]
    pub(crate) shader_env: &'static str,
}

/// The native CPU operator set one registered precision serves.
///
/// The native decoder (`src/chat/models/native/decoder.rs`) holds the
/// selected format's `NativeOps` in its `QuantState`: original precisions
/// dispatch straight to `gemv` with the mmap'd unit bytes; a quantized
/// format quantizes a unit at first use, releases the mmap pages (the
/// memory win), and computes from the owned buffer afterwards. Norms,
/// biases and the embedding never route through here — they keep their
/// stored precision by design.
#[derive(Clone, Copy)]
pub(crate) struct NativeOps {
    /// Quantize a full-precision weight matrix into this format at load time
    /// (`None` for original precisions).
    pub(crate) quantize:
        Option<fn(&[f32], rows: usize, cols: usize) -> Result<Arc<dyn PrecisionBuffer>, String>>,
    /// `y = W·x` over this format's raw weight bytes: one dot product per
    /// output row, computed in the format's own precision domain (bf16 rows
    /// use the fused bf16 kernel, f16 rows decode half per row, quantized
    /// rows unpack codes × scale).
    pub(crate) gemv: fn(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]),
}

/// One inference backend implementation for a weight precision. The enum
/// mirrors the model registry's `BackendLoader` shape — a new backend kind
/// (cuda/metal) adds a variant plus per-format entries in the registration
/// files; the lookups below dispatch on it.
#[derive(Clone, Copy)]
pub(crate) enum PrecisionBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanOps),
    Native(NativeOps),
}

/// One declarative weight-precision registration: what the format is
/// (id/kind/label), how a quantized format is enabled at runtime, and which
/// backends serve it. Entries live in
/// `src/chat/precision/registrations/<format>.rs` and carry no runtime code;
/// build.rs assembles them into `REGISTRY` (generated below).
#[derive(Clone, Copy)]
pub(crate) struct PrecisionRegistration {
    /// Format id (lowercase, e.g. "q8", "bf16"); also
    /// `PrecisionBuffer::format_id` for quantized buffers.
    pub(crate) id: &'static str,
    /// Human-readable name for diagnostics and docs.
    #[allow(dead_code)]
    pub(crate) label: &'static str,
    /// Original storage precision or quantized representation (tests and
    /// diagnostics assert the operator tables match the kind).
    #[allow(dead_code)]
    pub(crate) kind: PrecisionKind,
    /// Environment variable that enables a quantized format ("1" or "true",
    /// case-insensitive — read by `selected_format` at load time). Original
    /// precisions are always available and carry `None`.
    pub(crate) selector_env: Option<&'static str>,
    /// Backend operator tables keyed by backend name, in declared order.
    pub(crate) backends: &'static [(&'static str, PrecisionBackend)],
}

impl PrecisionRegistration {
    /// Backend operator table by name, case-insensitive.
    pub(crate) fn loader_for(&self, backend: &str) -> Option<PrecisionBackend> {
        self.backends
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(backend))
            .map(|(_, loader)| *loader)
    }

    /// Registered backend names, in declared order (registry tests).
    #[allow(dead_code)]
    pub(crate) fn backend_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.backends.iter().map(|(name, _)| *name)
    }
}

// Generated by build.rs from src/chat/precision/registrations/: the
// `<format>_registration` modules plus the `REGISTRY` table.
include!(concat!(env!("OUT_DIR"), "/precision_registry.rs"));

/// The registration serving a format id, or a listing error. The error
/// carries the registered id list so a typo in a selector or a test names
/// the valid formats; the lookup is case-insensitive because the ids
/// surface in prose and environment values as well as code.
pub(crate) fn registration_for_format(
    format: &str,
) -> Result<&'static PrecisionRegistration, String> {
    REGISTRY
        .iter()
        .copied()
        .find(|registration| registration.id.eq_ignore_ascii_case(format))
        .ok_or_else(|| {
            let registered: Vec<&str> =
                REGISTRY.iter().map(|registration| registration.id).collect();
            format!(
                "Weight precision '{format}' is not registered; registered precisions: {registered:?}."
            )
        })
}

/// Display label for a format id (diagnostic messages).
pub(crate) fn format_label(format: &str) -> &'static str {
    registration_for_format(format)
        .map(|registration| registration.label)
        .unwrap_or("Quantized")
}

/// Vulkan operator table for a format id, from the registration's backend
/// table. A registration without a Vulkan entry (or the wrong loader kind)
/// resolves to `None` — a registration-file bug surfaces as a dispatch error.
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_ops_for_format(format: &str) -> Option<VulkanOps> {
    registration_for_format(format)
        .ok()
        .and_then(|registration| match registration.loader_for("vulkan") {
            Some(PrecisionBackend::Vulkan(ops)) => Some(ops),
            _ => None,
        })
}

/// Vulkan operators for an existing quantized buffer (dispatch by the
/// buffer's registered format id).
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_ops_for_buffer(buffer: &dyn PrecisionBuffer) -> Option<VulkanOps> {
    vulkan_ops_for_format(buffer.format_id())
}

/// Native CPU operator table for a format id, from the registration's
/// backend table.
pub(crate) fn native_ops_for_format(format: &str) -> Option<NativeOps> {
    registration_for_format(format)
        .ok()
        .and_then(|registration| match registration.loader_for("native") {
            Some(PrecisionBackend::Native(ops)) => Some(ops),
            _ => None,
        })
}

/// Selector semantics shared by every registration: "1" or "true",
/// case-insensitive; anything else leaves the format off.
fn selector_enabled(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

/// The quantization format enabled for this run through its registration's
/// selector environment variable, if any. Both backends quantize linear
/// projections with this format at load time.
///
/// First match in registry order wins; the uniqueness test below guarantees
/// selector variables are distinct, so at most one format can be selected
/// no matter how many variables are exported. When nothing matches (or the
/// value is not "1"/"true") every unit runs in its stored precision —
/// quantization is strictly opt-in.
pub(crate) fn selected_format() -> Option<&'static PrecisionRegistration> {
    REGISTRY.iter().copied().find(|registration| {
        registration.selector_env.is_some_and(|variable| {
            std::env::var(variable)
                .map(|value| selector_enabled(&value))
                .unwrap_or(false)
        })
    })
}

/// The selected quantization format's Vulkan operator table, if the format is
/// registered for Vulkan.
#[cfg(feature = "vulkan")]
pub(crate) fn selected_vulkan_ops() -> Option<VulkanOps> {
    selected_format().and_then(|registration| match registration.loader_for("vulkan") {
        Some(PrecisionBackend::Vulkan(ops)) => Some(ops),
        _ => None,
    })
}

/// The selected quantization format's native CPU operator table, if the
/// format is registered for the native backend. `None` (no selector set, or
/// the format carries no native entry) keeps the source precision.
pub(crate) fn selected_native_ops() -> Option<NativeOps> {
    selected_format().and_then(|registration| match registration.loader_for("native") {
        Some(PrecisionBackend::Native(ops)) => Some(ops),
        _ => None,
    })
}

#[cfg(test)]
pub(crate) fn registered_format_ids() -> Vec<&'static str> {
    REGISTRY
        .iter()
        .map(|registration| registration.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registered precisions: three original storage precisions and the
    /// quantized formats (q8 row-wise int8, q4 row-wise int4).
    #[test]
    fn registry_contains_original_and_quantized_formats() {
        assert_eq!(registered_format_ids(), ["bf16", "f16", "f32", "q4", "q8"]);
        for id in ["bf16", "f16", "f32"] {
            let registration = registration_for_format(id).unwrap();
            assert_eq!(registration.kind, PrecisionKind::Original);
            assert_eq!(registration.selector_env, None);
        }
        for (id, selector) in [("q4", "CHEETAH_QUANT_Q4"), ("q8", "CHEETAH_QUANT_Q8")] {
            let registration = registration_for_format(id).unwrap();
            assert_eq!(registration.kind, PrecisionKind::Quantized);
            assert_eq!(registration.selector_env, Some(selector));
            assert_eq!(format_label(id), registration.label);
        }
    }

    /// Format ids and selector variables must be globally unique — a
    /// duplicate id would silently shadow one format's operators, and a
    /// duplicate selector would make one environment variable enable two
    /// formats at once.
    #[test]
    fn registrations_have_unique_ids_and_selectors() {
        let mut ids = std::collections::HashSet::new();
        let mut selectors = std::collections::HashSet::new();
        for registration in REGISTRY {
            assert!(
                ids.insert(registration.id),
                "duplicate format id '{}'",
                registration.id
            );
            if let Some(selector) = registration.selector_env {
                assert!(
                    selectors.insert(selector),
                    "duplicate selector env '{selector}'"
                );
            }
        }
    }

    /// Every registration must carry at least one backend with the full
    /// operator table (a missing entry is a registration-file bug, not a
    /// dispatch gap): Vulkan entries carry a streaming matmul and the
    /// shared-ops format code; CPU entries carry a gemv. Quantized formats
    /// additionally quantize; original precisions never do.
    #[test]
    fn every_registration_carries_backends() {
        for registration in REGISTRY {
            let names: Vec<_> = registration.backend_names().collect();
            assert!(
                !names.is_empty(),
                "{} registers no backends",
                registration.id
            );
            for name in names {
                if name == "vulkan" && !cfg!(feature = "vulkan") {
                    continue;
                }
                match registration.loader_for(name) {
                    #[cfg(feature = "vulkan")]
                    Some(PrecisionBackend::Vulkan(ops)) => {
                        let expected_format = match registration.id {
                            "f32" => 0,
                            "f16" => 2,
                            "bf16" => 3,
                            "q8" => 4,
                            "q4" => 5,
                            other => panic!("unexpected format {other}"),
                        };
                        assert_eq!(ops.weight_format, expected_format);
                        let expected_shader = match registration.id {
                            "f32" => "CHEETAH_MATMUL_SPV",
                            // f16 shares the bf16 pipeline (the shader decodes
                            // both 16-bit layouts via a push constant).
                            "f16" | "bf16" => "CHEETAH_MATMUL_BF16_SPV",
                            "q8" => "CHEETAH_MATMUL_Q8_SPV",
                            "q4" => "CHEETAH_MATMUL_Q4_SPV",
                            other => panic!("unexpected format {other}"),
                        };
                        assert_eq!(ops.shader_env, expected_shader);
                        assert_eq!(
                            ops.quantize.is_some(),
                            registration.kind == PrecisionKind::Quantized,
                            "{}: quantize operator presence must match the kind",
                            registration.id
                        );
                    }
                    Some(PrecisionBackend::Native(ops)) => {
                        assert_eq!(
                            ops.quantize.is_some(),
                            registration.kind == PrecisionKind::Quantized,
                            "{}: quantize operator presence must match the kind",
                            registration.id
                        );
                    }
                    None => panic!("{}: backend '{name}' lookup failed", registration.id),
                }
            }
        }
    }

    /// Original precisions never carry a selector, and only quantized
    /// formats do — the selector is what makes a format opt-in.
    #[test]
    fn selectors_belong_to_quantized_formats_only() {
        for registration in REGISTRY {
            assert_eq!(
                registration.selector_env.is_some(),
                registration.kind == PrecisionKind::Quantized,
                "{}: selector presence must match the kind",
                registration.id
            );
        }
    }

    /// Cross-backend coverage invariant: every registered precision serves
    /// BOTH backends (native always; vulkan when compiled in) — a format is
    /// not considered done until it runs everywhere.
    #[test]
    fn every_format_is_registered_for_both_backends() {
        for registration in REGISTRY {
            let names: Vec<_> = registration.backend_names().collect();
            assert!(
                names.contains(&"native"),
                "{}: no native backend registered",
                registration.id
            );
            if cfg!(feature = "vulkan") {
                assert!(
                    names.contains(&"vulkan"),
                    "{}: no vulkan backend registered",
                    registration.id
                );
            }
        }
    }

    /// Lookups normalize case and reject unknown formats with the registered
    /// listing in the message.
    #[test]
    fn format_lookup_normalizes_and_reports_unknown() {
        assert!(registration_for_format("Q8").is_ok());
        assert!(registration_for_format("BF16").is_ok());
        let error = match registration_for_format("q3") {
            Err(error) => error,
            Ok(_) => panic!("q3 must not resolve to a registration"),
        };
        assert!(
            error.contains("registered precisions: [\"bf16\", \"f16\", \"f32\", \"q4\", \"q8\"]"),
            "{error}"
        );
    }

    /// Selector semantics: "1"/"true" in any casing enable; other values
    /// (including "0" and "yes") do not.
    #[test]
    fn selector_accepts_only_one_and_true() {
        for value in ["1", "true", "TRUE", "True"] {
            assert!(selector_enabled(value), "{value} must enable");
        }
        for value in ["0", "false", "yes", "", "on"] {
            assert!(!selector_enabled(value), "{value} must not enable");
        }
    }
}
