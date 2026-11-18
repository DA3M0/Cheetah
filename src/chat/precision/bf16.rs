//! BF16 operators: the original storage precision. Every operator reads the
//! stored bf16 weights directly — the streaming matmul runs the dedicated
//! bf16 pipeline, the resident path uploads and multiplies the bf16 bits, and
//! the CPU gemv uses the fused bf16 kernel. No conversion into another
//! storage width.

#[cfg(feature = "vulkan")]
use super::bytes::u16_view;

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
    use crate::chat::models::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::Bf16, rows, cols, x, y);
}
