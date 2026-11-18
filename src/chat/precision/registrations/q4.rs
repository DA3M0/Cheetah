//! Declarative registration for the `q4` quantization format.
//!
//! build.rs collects every file in `src/chat/precision/registrations/` into
//! the generated `REGISTRY`; the operator implementations live in
//! `super::q4`. This file intentionally contains no runtime code.

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
