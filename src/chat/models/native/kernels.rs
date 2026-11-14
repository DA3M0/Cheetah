//! Compute kernels for the native CPU backend. Weight bytes are read directly
//! from the mmap (BF16/F16 upconverted on the fly, F32 as-is) so no weight is
//! ever copied into an owned buffer — resident-set growth comes only from the
//! pages the kernel actually touches.

use rayon::prelude::*;

/// Rows per parallel chunk in the gemv kernel.
const GEMV_CHUNK_ROWS: usize = 128;

#[inline]
pub fn bf16_f32(bits: u16) -> f32 {
    // BF16 is the upper half of an F32 — widening is a 16-bit shift.
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f16_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[inline]
fn widen_bf16(lo: u8, hi: u8) -> f32 {
    bf16_f32(u16::from_le_bytes([lo, hi]))
}

/// Interpret a weight byte slice as little-endian scalar values.
#[inline]
pub fn scalar_at(bytes: &[u8], index: usize, dtype: super::store::WeightDType) -> f32 {
    use super::store::WeightDType;
    match dtype {
        WeightDType::Bf16 => bf16_f32(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])),
        WeightDType::F16 => f16_f32(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])),
        WeightDType::F32 => f32::from_le_bytes([
            bytes[4 * index],
            bytes[4 * index + 1],
            bytes[4 * index + 2],
            bytes[4 * index + 3],
        ]),
    }
}

/// Dequantize one BF16 weight row into `out` (f32). Two rows land per u32
/// load; the widening is pure bit arithmetic, which auto-vectorizes strongly
/// (SSE/AVX bitcast conversion).
fn dequant_bf16_row(row_bytes: &[u8], out: &mut [f32]) {
    let (pairs, tail) = row_bytes.as_chunks::<4>();
    for (index, chunk) in pairs.iter().enumerate() {
        let word = u32::from_le_bytes(*chunk);
        out[2 * index] = f32::from_bits((word & 0xFFFF) << 16);
        out[2 * index + 1] = f32::from_bits(word & 0xFFFF_0000);
    }
    let tail_u16: &[u8] = tail;
    for (index, chunk) in tail_u16.chunks_exact(2).enumerate() {
        out[2 * pairs.len() + index] = bf16_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

/// Dequantize one weight row into `out` (f32).
#[inline]
fn dequant_row(row_bytes: &[u8], dtype: super::store::WeightDType, out: &mut [f32]) {
    use super::store::WeightDType;
    match dtype {
        WeightDType::Bf16 => dequant_bf16_row(row_bytes, out),
        WeightDType::F16 => {
            for (chunk, value) in row_bytes.chunks_exact(2).zip(out.iter_mut()) {
                *value = f16_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        WeightDType::F32 => {
            for (chunk, value) in row_bytes.chunks_exact(4).zip(out.iter_mut()) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
    }
}

/// Dot product of two f32 slices with an 8-lane accumulator array — the
/// fixed-width inner loop lowers to wide vector FMA sequences.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks::<8>();
    for (a8, b8) in a_chunks.iter().zip(b_chunks.iter()) {
        for lane in 0..8 {
            acc[lane] += a8[lane] * b8[lane];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (a_value, b_value) in a_tail.iter().zip(b_tail.iter()) {
        sum += a_value * b_value;
    }
    sum
}

/// Dot product of one BF16 row against x with the widening fused into the
/// accumulator: each 8-byte chunk holds 4 bf16 lanes feeding 4 accumulators,
/// so no intermediate buffer exists and the inner loop stays in fixed-width
/// vector form. On AVX2+FMA hardware this dispatches to an explicit
/// intrinsics path (unpack-widen + FMA); otherwise the portable fused loop.
#[inline]
fn dot_row_bf16_fused(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above; the
            // intrinsics read exactly row/len bytes and x/len floats.
            unsafe { return dot_row_bf16_avx2(row, x) }
        }
    }
    dot_row_bf16_fused_portable(row, x)
}

/// Portable fused bf16 dot: each 8-byte chunk holds 4 bf16 lanes feeding 4
/// accumulators, so no intermediate buffer exists and the inner loop stays in
/// fixed-width vector form.
#[inline]
fn dot_row_bf16_fused_portable(row: &[u8], x: &[f32]) -> f32 {
    let (row_chunks, row_tail) = row.as_chunks::<8>();
    let (x_chunks, x_tail) = x.as_chunks::<4>();
    let mut acc = [0f32; 4];
    for (chunk8, x4) in row_chunks.iter().zip(x_chunks.iter()) {
        let w0 = u32::from_le_bytes([chunk8[0], chunk8[1], chunk8[2], chunk8[3]]);
        let w1 = u32::from_le_bytes([chunk8[4], chunk8[5], chunk8[6], chunk8[7]]);
        acc[0] += f32::from_bits((w0 & 0xFFFF) << 16) * x4[0];
        acc[1] += f32::from_bits(w0 & 0xFFFF_0000) * x4[1];
        acc[2] += f32::from_bits((w1 & 0xFFFF) << 16) * x4[2];
        acc[3] += f32::from_bits(w1 & 0xFFFF_0000) * x4[3];
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for (index, x_value) in x_tail.iter().enumerate() {
        let byte = row_chunks.len() * 8 + index * 2;
        sum += widen_bf16(row[byte], row[byte + 1]) * x_value;
    }
    let _ = row_tail;
    sum
}

/// AVX2+FMA bf16 dot: one 256-bit load carries 16 bf16 lanes; each 128-bit
/// half is zero-extended to u32 and shifted left 16 (exactly `bf16 << 16`,
/// which bit-casts to the f32 value), then two FMA accumulate against x.
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support (runtime detection).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_bf16_avx2(row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let (full, tail) = row.as_chunks::<32>(); // 32 bytes = 16 bf16 per chunk
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for (index, chunk) in full.iter().enumerate() {
        let w = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let lo = _mm256_castsi256_si128(w); // bf16[0..8]
        let hi = _mm256_extracti128_si256::<1>(w); // bf16[8..16]
        // Zero-extend u16 → u32 then shift left 16: the u32 lane becomes
        // `bf16 << 16`, bit-casting to the exact f32 value, lanes preserved.
        let f_lo = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(lo),
            16,
        ));
        let f_hi = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_cvtepu16_epi32(hi),
            16,
        ));
        let base = index * 16;
        let x0 = _mm256_loadu_ps(x.as_ptr().add(base));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(base + 8));
        acc0 = _mm256_fmadd_ps(f_lo, x0, acc0);
        acc1 = _mm256_fmadd_ps(f_hi, x1, acc1);
    }
    // Horizontal reduction of the two accumulators (once per row — scalar
    // store + sum is negligible next to the FMA loop).
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
    let mut lanes1 = [0f32; 8];
    _mm256_storeu_ps(lanes1.as_mut_ptr(), acc1);
    let total: f32 = lanes.iter().sum::<f32>() + lanes1.iter().sum::<f32>();
    let mut tail_total = 0f32;
    for (index, byte_pair) in tail.chunks_exact(2).enumerate() {
        let x_index = full.len() * 16 + index;
        tail_total += widen_bf16(byte_pair[0], byte_pair[1]) * x[x_index];
    }
    total + tail_total
}

pub fn gemv(
    weight: &[u8],
    dtype: super::store::WeightDType,
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
) {
    assert_eq!(x.len(), cols, "gemv input width mismatch");
    assert_eq!(y.len(), rows, "gemv output width mismatch");
    let element_size = dtype.element_size();
    let row_stride = cols * element_size;
    let dot_row = |row_bytes: &[u8]| -> f32 {
        match dtype {
            super::store::WeightDType::Bf16 => dot_row_bf16_fused(row_bytes, x),
            _ => {
                let mut scratch = vec![0f32; cols];
                dequant_row(row_bytes, dtype, &mut scratch);
                dot_f32(&scratch, x)
            }
        }
    };
    if rows <= GEMV_CHUNK_ROWS {
        for (row, y_value) in y.iter_mut().enumerate() {
            let row_bytes = &weight[row * row_stride..(row + 1) * row_stride];
            *y_value = dot_row(row_bytes);
        }
        return;
    }
    y.par_chunks_mut(GEMV_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk, out)| {
            let row_start = chunk * GEMV_CHUNK_ROWS;
            for (index_in_chunk, y_value) in out.iter_mut().enumerate() {
                let row = row_start + index_in_chunk;
                let row_bytes = &weight[row * row_stride..(row + 1) * row_stride];
                *y_value = dot_row(row_bytes);
            }
        });
}

/// RMSNorm: x / sqrt(mean(x²) + eps) · weight, computed in f32 in place.
pub fn rmsnorm(x: &mut [f32], weight: &[u8], dtype: super::store::WeightDType, eps: f32) {
    let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_square + eps).sqrt();
    for (value, index) in x.iter_mut().zip(0..) {
        *value *= inv * scalar_at(weight, index, dtype);
    }
}

/// Rotary embedding (GPT-NeoX rotate-half style, matching candle's rope):
/// for each head of dim d, split [x1 | x2] and pair angle i with both halves.
pub fn rope_in_place(values: &mut [f32], heads: usize, head_dim: usize, position: usize, theta: f32) {
    let half = head_dim / 2;
    for head in 0..heads {
        let base = head * head_dim;
        for i in 0..half {
            let freq = theta.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = position as f32 * freq;
            // NOTE: f32::sin_cos returns (sin, cos) — do not swap these.
            let (sin, cos) = angle.sin_cos();
            let first = base + i;
            let second = base + half + i;
            let (a, b) = (values[first], values[second]);
            values[first] = a * cos - b * sin;
            values[second] = b * cos + a * sin;
        }
    }
}

/// SwiGLU activation: silu(gate) ⊙ up, written into `gate`.
pub fn silu_multiply(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g *= u / (1.0 + (-*g).exp());
    }
}

/// Numerically stable in-place softmax over a masked score vector.
pub fn softmax(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    let inv = 1.0 / sum;
    for score in scores.iter_mut() {
        *score *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::store::WeightDType;

    #[test]
    fn bf16_widening_matches_bit_shift() {
        assert_eq!(bf16_f32(0x3F80), 1.0);
        assert_eq!(bf16_f32(0xC000), -2.0);
        assert_eq!(bf16_f32(0x0000), 0.0);
    }

    #[test]
    fn gemv_multiplies_rows_in_parallel() {
        // W = [[1, 2, 3], [4, 5, 6]] (F32), x = [1, 0, -1]
        let mut weight = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            weight.extend_from_slice(&value.to_le_bytes());
        }
        let x = vec![1.0f32, 0.0, -1.0];
        let mut y = vec![0f32; 2];
        gemv(&weight, WeightDType::F32, 2, 3, &x, &mut y);
        assert_eq!(y, vec![-2.0, -2.0]);

        // Same matrix as BF16.
        let mut weight_bf16 = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            let bits = (value.to_bits() >> 16) as u16;
            weight_bf16.extend_from_slice(&bits.to_le_bytes());
        }
        gemv(&weight_bf16, WeightDType::Bf16, 2, 3, &x, &mut y);
        assert_eq!(y, vec![-2.0, -2.0]);
    }

    #[test]
    fn rmsnorm_matches_reference_computation() {
        let x = [3.0f32, 4.0];
        let weight_bits = (1.0f32.to_bits() >> 16) as u16; // weight of ones
        let weight_bytes = weight_bits.to_le_bytes().repeat(2);
        let mut y = x;
        rmsnorm(&mut y, &weight_bytes, WeightDType::Bf16, 1e-6);
        let expected = 1.0 / (12.5f32 + 1e-6).sqrt();
        assert!((y[0] - 3.0 * expected).abs() < 1e-5);
        assert!((y[1] - 4.0 * expected).abs() < 1e-5);
    }

    #[test]
    fn rope_preserves_magnitude_and_matches_reference() {
        // Two-dimensional head (half = 1): i = 0 gives freq = theta^0 = 1, so
        // position 1 rotates by exactly 1 radian. Rotation must preserve length.
        let mut v = vec![3.0f32, 4.0];
        rope_in_place(&mut v, 1, 2, 1, 1_000_000.0);
        assert!((v[0] * v[0] + v[1] * v[1] - 25.0).abs() < 1e-4);
        let angle = 1.0f32;
        let cos = angle.cos();
        let sin = angle.sin();
        assert!((v[0] - (3.0 * cos - 4.0 * sin)).abs() < 1e-5);
        assert!((v[1] - (4.0 * cos + 3.0 * sin)).abs() < 1e-5);
    }

    #[test]
    fn silu_multiply_matches_reference() {
        let mut gate = vec![1.0f32, -2.0];
        silu_multiply(&mut gate, &[2.0f32, 3.0]);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert!((gate[0] - silu(1.0) * 2.0).abs() < 1e-6);
        assert!((gate[1] - silu(-2.0) * 3.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_is_shift_invariant() {
        let mut scores = vec![1.0f32, 2.0, 3.0];
        softmax(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[2] > scores[1] && scores[1] > scores[0]);
        let mut shifted = vec![1001.0f32, 1002.0, 1003.0];
        softmax(&mut shifted);
        assert!((scores[0] - shifted[0]).abs() < 1e-5);
    }

    /// Replicate candle's qwen2 rope (duplicate-frequency rotate-half) with
    /// plain tensor math and compare against the native kernel at positions > 0.
    #[test]
    fn rope_matches_candle_rotate_half_at_positive_positions() {
        let head_dim = 64usize;
        let theta = 1_000_000.0f32;
        let half = head_dim / 2;
        let mut input = Vec::with_capacity(head_dim);
        for i in 0..head_dim {
            input.push(((i * 37 % 11) as f32 / 7.0) - 0.6);
        }
        for position in 1..=8usize {
            let mut native = input.clone();
            rope_in_place(&mut native, 1, head_dim, position, theta);
            // Candle reference: inv_freq[i] = 1 / theta^(2i/dim) for i in 0..half;
            // cos/sin rows duplicate each frequency across both halves.
            let mut expected = vec![0f32; head_dim];
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = position as f32 * freq;
                let cos = angle.cos();
                let sin = angle.sin();
                expected[i] = input[i] * cos - input[i + half] * sin;
                expected[i + half] = input[i + half] * cos + input[i] * sin;
            }
            for index in 0..head_dim {
                assert!(
                    (native[index] - expected[index]).abs() < 1e-4,
                    "position {position} dim {index}: native {} vs expected {}",
                    native[index],
                    expected[index]
                );
            }
        }
    }
}

#[cfg(test)]
mod dot_tests {
    use super::*;
    use super::super::store::WeightDType;

    #[test]
    fn dot_row_bf16_fused_matches_dequant_dot() {
        // bf16 values of exact small integers (0.0..255.0 are exactly
        // representable) so both paths must agree to the bit.
        let cols = 8usize;
        let mut row_bytes = Vec::with_capacity(cols * 2);
        for index in 0..cols {
            let bits = (((index % 251) as f32).to_bits() >> 16) as u16;
            row_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let x: Vec<f32> = (0..cols).map(|i| (i % 17) as f32 / 8.0 - 1.0).collect();

        let fused = dot_row_bf16_fused(&row_bytes, &x);
        let mut scratch = vec![0f32; cols];
        dequant_bf16_row(&row_bytes, &mut scratch);
        let naive = dot_f32(&scratch, &x);

        eprintln!("scratch[0..8]={:?}\nx[0..8]={:?}", &scratch, &x);
        eprintln!("fused={fused:.6} naive={naive:.6}");
        assert!(
            (fused - naive).abs() < 1e-2,
            "fused ({fused}) diverges from dequant+dot ({naive})"
        );
    }

    /// Full-width dot (896 lanes like a real hidden size) exercising the
    /// AVX2 path with tail handling against the dequant+dot reference.
    #[test]
    fn dot_row_bf16_fused_full_width_matches_reference() {
        let cols = 896usize;
        let mut row_bytes = Vec::with_capacity(cols * 2);
        for index in 0..cols {
            let value = ((index % 251) as f32) * 0.5 - 60.0; // inexact bf16 rounding on purpose
            let bits = (value.to_bits() >> 16) as u16;
            row_bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let x: Vec<f32> = (0..cols)
            .map(|index| ((index % 31) as f32) * 0.25 - 4.0)
            .collect();

        let fused = dot_row_bf16_fused(&row_bytes, &x);
        let mut scratch = vec![0f32; cols];
        dequant_bf16_row(&row_bytes, &mut scratch);
        let reference = dot_f32(&scratch, &x);

        // Both paths dequantize the same bits; the sum order differs, so the
        // tolerance is a few ULPs of the magnitude rather than exact.
        eprintln!("full-width: fused={fused:.4} reference={reference:.4}");
        assert!(
            (fused - reference).abs() / reference.abs().max(1.0) < 1e-5,
            "full-width fused ({fused}) diverges from reference ({reference})"
        );
    }
}
