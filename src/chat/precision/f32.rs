//! F32 operators: the original storage precision. Every operator reads the
//! stored f32 weights directly — the streaming matmul runs the dedicated f32
//! pipeline and the CPU gemv dots f32 rows in f32. No conversion step.
//!
//! Registered facts (asserted by the registry tests): shared-ops format
//! code 0, shader override `CHEETAH_MATMUL_SPV`, no resident upload (an
//! f32 weight is the widest layout, so VRAM-tier units stream per step
//! like every other non-uploadable format) and no quantize (an original
//! precision never converts).

#[cfg(feature = "vulkan")]
use super::f32_view;

/// Host-streaming f32 matmul (dedicated f32 pipeline; weights stay f32).
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_streaming(
    input: &[f32],
    weight: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_f32_transposed(input, f32_view(weight)?, m, k, n)
}

/// `y = W·x` over stored f32 rows (f32 multiply-accumulate).
pub(crate) fn gemv(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    use crate::chat::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::F32, rows, cols, x, y);
}

// ---------------------------------------------------------------------------
// Declarative registration — this format's entry in the generated REGISTRY
// (build.rs scans src/chat/precision/ and splices every file there as a
// `<name>` module; adding a weight precision is adding one such file).
// ---------------------------------------------------------------------------

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
                    gemv: gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: None,
                    matmul_streaming: matmul_streaming,
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
                gemv: gemv,
            }),
        )]
    }
}
