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
