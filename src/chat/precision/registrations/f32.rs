//! Declarative registration for the `f32` original precision.
//!
//! build.rs collects every file in `src/chat/precision/registrations/` into
//! the generated `REGISTRY`; the operator implementations live in
//! `super::f32`. This file intentionally contains no runtime code.

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
