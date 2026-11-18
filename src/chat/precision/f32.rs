//! F32 operators: the original storage precision. Every operator reads the
//! stored f32 weights directly — the streaming matmul runs the dedicated f32
//! pipeline and the CPU gemv dots f32 rows in f32. No conversion step.

#[cfg(feature = "vulkan")]
use super::bytes::f32_view;

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
    use crate::chat::models::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::F32, rows, cols, x, y);
}
