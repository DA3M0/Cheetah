//! Declarative registration for the `q8` quantization format.
//!
//! build.rs collects every file in `src/chat/precision/registrations/` into
//! the generated `REGISTRY`; the operator implementations live in
//! `super::q8`. This file intentionally contains no runtime code.

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "q8",
    label: "Q8 (row-wise int8)",
    kind: PrecisionKind::Quantized,
    // `CHEETAH_QUANT_Q8=1` quantizes linear projections to row-wise int8 at
    // load time (halves the weight footprint for both the streaming and the
    // resident path). Norms, biases and the embedding stay in original
    // precision.
    selector_env: Some("CHEETAH_QUANT_Q8"),
    backends: precision_backends(),
};

/// Backends registered for this format: both the native CPU operators
/// (load-time quantize + int8×scale gemv) and, when compiled in, the Vulkan
/// executor operators (quantize at load, resident upload, streaming and
/// resident matmul).
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: Some(super::q8::quantize),
                    gemv: super::q8::gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: Some(super::q8::quantize),
                    matmul_streaming: super::q8::matmul_streaming,
                    upload: Some(super::q8::upload),
                    matmul_resident: Some(super::q8::matmul_resident),
                    weight_format: 4,
                    shader_env: "CHEETAH_MATMUL_Q8_SPV",
                }),
            ),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[(
            "native",
            PrecisionBackend::Native(super::NativeOps {
                quantize: Some(super::q8::quantize),
                gemv: super::q8::gemv,
            }),
        )]
    }
}
