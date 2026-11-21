//! BF16 operators: the original storage precision. Every operator reads the
//! stored bf16 weights directly — the streaming matmul runs the dedicated
//! bf16 pipeline, the resident path uploads and multiplies the bf16 bits, and
//! the CPU gemv uses the fused bf16 kernel. No conversion into another
//! storage width.
//!
//! BF16 is the most complete original precision in the registry: it is the
//! only one with a resident upload (the C substrate stores u16 payloads,
//! so `cheetah_vulkan_weight_upload_bf16` serves it directly) and the only
//! one with a fused CPU kernel that reads the bits without a dequant
//! scratch (`dot_row_bf16_fused` — the `bf16 << 16` widening is exact, and
//! the FMA accumulation matches the dequant+dot reference bit for bit).
//! Registered facts: shared-ops format code 3, shader override
//! `CHEETAH_MATMUL_BF16_SPV`.

#[cfg(feature = "vulkan")]
use super::u16_view;

/// Host-streaming bf16 matmul (dedicated bf16 pipeline; weights stay bf16).
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_streaming(
    input: &[f32],
    weight: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_bf16_transposed(input, u16_view(weight)?, m, k, n)
}

/// Upload the bf16 bits into the Ant resident bucket; `None` keeps the
/// per-step streaming path (e.g. an oversized upload). `row_bytes` does not
/// apply to original precisions.
#[cfg(feature = "vulkan")]
pub(crate) fn upload(weight: &[u8], _row_bytes: usize) -> Option<u64> {
    cheetah::native::vulkan::weight_upload_bf16(u16_view(weight).ok()?)
}

/// Ant resident bf16 matmul: the weight stays bound on the GPU behind
/// `handle` (from `upload`); only activations and outputs pass through the
/// pool.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_resident(
    handle: u64,
    input: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_bf16_transposed_resident(handle, input, m, k, n)
}

/// `y = W·x` over stored bf16 rows (fused bf16 kernel, f32 accumulate).
pub(crate) fn gemv(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    use crate::chat::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::Bf16, rows, cols, x, y);
}

// ---------------------------------------------------------------------------
// Declarative registration — this format's entry in the generated REGISTRY
// (build.rs scans src/chat/precision/ and splices every file there as a
// `<name>` module; adding a weight precision is adding one such file).
// ---------------------------------------------------------------------------

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
                    gemv: gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: None,
                    matmul_streaming: matmul_streaming,
                    upload: Some(upload),
                    matmul_resident: Some(matmul_resident),
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
                gemv: gemv,
            }),
        )]
    }
}
