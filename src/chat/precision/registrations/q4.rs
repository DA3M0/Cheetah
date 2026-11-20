//! Declarative registration for the `q4` quantization format.
//!
//! This file is DATA ONLY: build.rs scans `src/chat/precision/registrations/`
//! and splices every file into the generated `OUT_DIR/precision_registry.rs`
//! as a `<name>_registration` module, assembling the `REGISTRY` table the
//! runtime lookups (`registration_for_format`, `selected_format`, the
//! per-backend operator lookups) derive from. Adding a weight precision is
//! adding one file here plus its implementation module — no table edits
//! anywhere, matching the model-family registration mode.
//!
//! Registration contract enforced by the registry tests:
//! - format ids and selector variables are globally unique;
//! - every format serves BOTH backends (native always, vulkan when compiled
//!   in) — a half-registered format fails the build's tests;
//! - `quantize` presence must match `PrecisionKind` (original precisions
//!   never convert, quantized formats always do);
//! - original precisions carry `selector_env: None`; only quantized formats
//!   are opt-in.
//!
//! The operator implementations live in `super::q4`.

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "q4",
    label: "Q4 (row-wise int4)",
    kind: PrecisionKind::Quantized,
    // `CHEETAH_QUANT_Q4=1` quantizes linear projections to row-wise int4 at
    // load time (a quarter of the BF16 footprint, half of q8). Norms, biases
    // and the embedding stay in original precision.
    selector_env: Some("CHEETAH_QUANT_Q4"),
    backends: precision_backends(),
};

/// Backends registered for this format: both the native CPU operators
/// (load-time quantize + int4×scale gemv) and, when compiled in, the Vulkan
/// executor operators (quantize at load, resident upload, streaming and
/// resident matmul).
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: Some(super::q4::quantize),
                    gemv: super::q4::gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: Some(super::q4::quantize),
                    matmul_streaming: super::q4::matmul_streaming,
                    upload: Some(super::q4::upload),
                    matmul_resident: Some(super::q4::matmul_resident),
                    weight_format: 5,
                    shader_env: "CHEETAH_MATMUL_Q4_SPV",
                }),
            ),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[(
            "native",
            PrecisionBackend::Native(super::NativeOps {
                quantize: Some(super::q4::quantize),
                gemv: super::q4::gemv,
            }),
        )]
    }
}
