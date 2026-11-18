//! Declarative registration for the `f16` original precision.
//!
//! build.rs collects every file in `src/chat/precision/registrations/` into
//! the generated `REGISTRY`; the operator implementations live in
//! `super::f16`. This file intentionally contains no runtime code.

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "f16",
    label: "F16 (original precision)",
    kind: PrecisionKind::Original,
    // Original precisions are always available; no selector.
    selector_env: None,
    backends: precision_backends(),
};

/// Backends registered for this precision: the native CPU gemv (half decoded
/// per row) and, when compiled in, the Vulkan executor's f16 matmul (the
/// shared 16-bit pipeline in fp16 decode mode). Original precisions never
/// quantize and carry no resident upload.
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: None,
                    gemv: super::f16::gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: None,
                    matmul_streaming: super::f16::matmul_streaming,
                    upload: None,
                    matmul_resident: None,
                    weight_format: 2,
                    shader_env: "CHEETAH_MATMUL_BF16_SPV",
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
                gemv: super::f16::gemv,
            }),
        )]
    }
}
