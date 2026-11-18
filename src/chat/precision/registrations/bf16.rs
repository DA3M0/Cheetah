//! Declarative registration for the `bf16` original precision.
//!
//! build.rs collects every file in `src/chat/precision/registrations/` into
//! the generated `REGISTRY`; the operator implementations live in
//! `super::bf16`. This file intentionally contains no runtime code.

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "bf16",
    label: "BF16 (original precision)",
    kind: PrecisionKind::Original,
    // Original precisions are always available; no selector.
    selector_env: None,
    backends: precision_backends(),
};

/// Backends registered for this precision: the native CPU fused bf16 gemv
/// and, when compiled in, the Vulkan executor's bf16 matmul pipeline
/// (streaming plus the bf16 resident upload / resident matmul pair). Original
/// precisions never quantize.
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: None,
                    gemv: super::bf16::gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: None,
                    matmul_streaming: super::bf16::matmul_streaming,
                    upload: Some(super::bf16::upload),
                    matmul_resident: Some(super::bf16::matmul_resident),
                    weight_format: 3,
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
                gemv: super::bf16::gemv,
            }),
        )]
    }
}
