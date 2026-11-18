//! F16 operators: the original storage precision. Every operator reads the
//! stored IEEE half weights directly — the streaming matmul runs the 16-bit
//! pipeline in fp16 decode mode and the CPU gemv decodes half per row. No
//! conversion into another storage width.

#[cfg(feature = "vulkan")]
use super::bytes::u16_view;

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
    use crate::chat::models::native::{kernels, store::WeightDType};
    kernels::gemv(weight, WeightDType::F16, rows, cols, x, y);
}
