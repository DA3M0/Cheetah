//! Declarative registration for the `f32` original precision.
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
//! The operator implementations live in `super::f32`.

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "f32",
    label: "F32 (original precision)",
    kind: PrecisionKind::Original,
    // Original precisions are always available; no selector.
    selector_env: None,
    backends: precision_backends(),
};

/// Backends registered for this precision: the native CPU gemv (f32 dot) and,
/// when compiled in, the Vulkan executor's f32 matmul pipeline. Original
/// precisions never quantize and carry no resident upload (weights stream
/// per step from their typed payload).
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: None,
                    gemv: super::f32::gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: None,
                    matmul_streaming: super::f32::matmul_streaming,
                    upload: None,
                    matmul_resident: None,
                    weight_format: 0,
                    shader_env: "CHEETAH_MATMUL_SPV",
                }),
            ),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[(
            "native",
            PrecisionBackend::Native(super::NativeOps {
                quantize: None,
                gemv: super::f32::gemv,
            }),
        )]
    }
}
