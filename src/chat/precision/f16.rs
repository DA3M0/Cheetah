//! F16 operators: the original storage precision. Every operator reads the
//! stored IEEE half weights directly — the streaming matmul runs the 16-bit
//! pipeline in fp16 decode mode and the CPU gemv decodes half per row. No
//! conversion into another storage width.
//!
//! Pipeline sharing: the C substrate keeps ONE 16-bit matmul pipeline
//! (`g_matmul_bf16_pipeline`); a push constant selects the decode rule
//! (bf16 bit-shift vs `unpackHalf2x16`). That is why this registration
//! reports `CHEETAH_MATMUL_BF16_SPV` as its shader override and format
//! code 2 (the shared_ops encoding; bf16 uses 3). On the native side the
//! gemv decodes IEEE half per row into f32 scratch and accumulates in f32 —
//! the standard mixed-precision dot, not a storage conversion.

#[cfg(feature = "vulkan")]
use super::u16_view;

/// Host-streaming f16 matmul: the fp16 decode flag selects IEEE half on the
/// shared 16-bit pipeline; weights stay f16.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_streaming(
    input: &[f32],
    weight: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_fp16_transposed(input, u16_view(weight)?, m, k, n)
}

/// `y = W·x` over stored f16 rows (half decoded per row, f32 accumulate).
pub(crate) fn gemv(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    use crate::chat::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::F16, rows, cols, x, y);
}

// ---------------------------------------------------------------------------
// Declarative registration — this format's entry in the generated REGISTRY
// (build.rs scans src/chat/precision/ and splices every file there as a
// `<name>` module; adding a weight precision is adding one such file).
// ---------------------------------------------------------------------------

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
                gemv: gemv,
            }),
        )]
    }
}
