//! The native-backend OpsToolbox: the low-level math operators plus the
//! shared inference plumbing the per-model forward assemblies drive.
//!
//! This module is the main-program side of the decoder split. It owns
//! **operators and state, not model structure**: there is no family branch
//! anywhere here. Each model family's layer stacking (residual topology,
//! projection naming, norm kind, activation) lives in its registration file
//! (`src/chat/registrations/<family>.rs`), which composes the toolbox
//! ops in the order its architecture prescribes.
//!
//! ## Math kernels
//!
//! `gemv`, `rmsnorm`, `rope_in_place`, `softmax`, and friends are free
//! functions with statically dispatched internals (AVX2 vs portable paths are
//! chosen by runtime feature detection inside the kernel, never through trait
//! objects). Names are frozen: the precision operator registry
//! (`src/chat/precision`) wires `gemv_q8` / `gemv_q4` / `dequant_values`
//! directly, and `kernels` re-exports this module for compatibility.
//!
//! ## The toolbox
//!
//! [`OpsToolbox`] carries the per-generation state that every model shares:
//! the mmap-backed weight store, the per-layer KV caches, the quantized
//! linear buffers, the routing-aware expert residency, and the Ant tier
//! plumbing. Its methods are parameterized by explicit arguments (unit ids,
//! shapes, positions), so a registration file picks the units and the norm
//! flavor while the toolbox executes the math and the memory contract.
//!
//! Borrow discipline: the toolbox owns the store and the caches, while the
//! activation buffers live in [`Scratch`] next to the toolbox in the family
//! struct. That disjoint ownership is what lets a family forward call
//! `ctx.linear(...)` with `scratch` slices as arguments.

use rayon::prelude::*;

use super::decoder::DecoderConfig;
use super::store::{MmapWeightStore, Tier, WeightDType};
use crate::chat::precision::{self, PrecisionBuffer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Math kernels (moved verbatim from `kernels`; names are frozen).
// ---------------------------------------------------------------------------

/// Rows per parallel chunk in the gemv kernel.
const GEMV_CHUNK_ROWS: usize = 128;

/// Widen one BF16 bit pattern to its exact f32 value.
///
/// BF16 shares the f32 exponent and stores a truncated mantissa, so the
/// widening is a left shift into a u32's high half — pure bit arithmetic,
/// no rounding, which is what makes the fused dot kernels exact.
#[inline]
pub fn bf16_f32(bits: u16) -> f32 {
    // BF16 is the upper half of an F32 — widening is a 16-bit shift.
    f32::from_bits((bits as u32) << 16)
}

/// Widen an IEEE f16 bit pattern to its f32 value.
///
/// Unlike bf16, the 10-bit mantissa and biased 5-bit exponent make this
/// more than a shift (denormals, exponent rebias), so the conversion goes
/// through the `half` crate.
#[inline]
pub fn f16_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Widen one little-endian BF16 byte pair (`lo`, `hi`) to its f32 value —
/// the byte-pair convenience form of [`bf16_f32`] used by the dot-kernel
/// tails.
#[inline]
fn widen_bf16(lo: u8, hi: u8) -> f32 {
    bf16_f32(u16::from_le_bytes([lo, hi]))
}

/// Interpret a weight byte slice as little-endian scalar values.
#[inline]
pub fn scalar_at(bytes: &[u8], index: usize, dtype: WeightDType) -> f32 {
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
    for (index, chunk) in tail_u16.as_chunks::<2>().0.iter().enumerate() {
        out[2 * pairs.len() + index] = bf16_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
}

/// Dequantize one weight row into `out` (f32).
#[inline]
fn dequant_row(row_bytes: &[u8], dtype: WeightDType, out: &mut [f32]) {
    match dtype {
        WeightDType::Bf16 => dequant_bf16_row(row_bytes, out),
        WeightDType::F16 => {
            for (chunk, value) in row_bytes.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
                *value = f16_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        WeightDType::F32 => {
            for (chunk, value) in row_bytes.as_chunks::<4>().0.iter().zip(out.iter_mut()) {
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
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
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
    unsafe {
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
            let f_lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
            let f_hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16));
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
        for (index, byte_pair) in tail.as_chunks::<2>().0.iter().enumerate() {
            let x_index = full.len() * 16 + index;
            tail_total += widen_bf16(byte_pair[0], byte_pair[1]) * x[x_index];
        }
        total + tail_total
    }
}

/// Row-parallel matrix-vector `y = W · x` over stored-precision weight bytes.
///
/// BF16 rows take the fused widening kernel (AVX2+FMA when the CPU supports
/// it, the portable fused loop otherwise); every other dtype dequantizes one
/// row at a time and dots in f32. Up to `GEMV_CHUNK_ROWS` (128) rows run
/// sequentially (a rayon fork/join would cost more than the compute), larger
/// matrices split into 128-row rayon chunks.
pub fn gemv(weight: &[u8], dtype: WeightDType, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols, "gemv input width mismatch");
    assert_eq!(y.len(), rows, "gemv output width mismatch");
    let element_size = dtype.element_size();
    let row_stride = cols * element_size;
    let dot_row = |row_bytes: &[u8]| -> f32 {
        match dtype {
            WeightDType::Bf16 => dot_row_bf16_fused(row_bytes, x),
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

/// Q8 matvec over the registered quantized buffer layout (`q8` in
/// `crate::chat::precision`): a head of `rows` f32 row scales followed by
/// row-major int8 payloads, each row padded to a multiple of four bytes. The
/// dot runs in the quantized domain — codes are unpacked as int8 and
/// multiplied by their row scale, f32 accumulate; the weights are never
/// bulk-dequantized into another storage width (the same math the
/// `matmul_q8.comp` shader performs).
pub fn gemv_q8(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols, "gemv_q8 input width mismatch");
    assert_eq!(y.len(), rows, "gemv_q8 output width mismatch");
    let row_bytes = (cols + 3) / 4 * 4;
    let head_bytes = rows * 4;
    let dot_row = |row: usize| -> f32 {
        let scale = f32::from_le_bytes([
            weight[row * 4],
            weight[row * 4 + 1],
            weight[row * 4 + 2],
            weight[row * 4 + 3],
        ]);
        let codes = &weight[head_bytes + row * row_bytes..head_bytes + row * row_bytes + cols];
        let mut sum = 0f32;
        for (column, code) in codes.iter().enumerate() {
            sum += scale * (*code as i8 as f32) * x[column];
        }
        sum
    };
    if rows <= GEMV_CHUNK_ROWS {
        for (row, y_value) in y.iter_mut().enumerate() {
            *y_value = dot_row(row);
        }
        return;
    }
    y.par_chunks_mut(GEMV_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk, out)| {
            let row_start = chunk * GEMV_CHUNK_ROWS;
            for (index_in_chunk, y_value) in out.iter_mut().enumerate() {
                *y_value = dot_row(row_start + index_in_chunk);
            }
        });
}

/// Q4 matvec over the registered quantized buffer layout (`q4` in
/// `crate::chat::precision`): a head of `rows` f32 row scales followed by
/// row-major int4 payloads, two codes per byte (low nibble = even column),
/// each row padded to a multiple of four bytes. The dot runs in the
/// quantized domain — codes are unpacked as int4 and multiplied by their row
/// scale, f32 accumulate; the weights are never bulk-dequantized into another
/// storage width (the same math the `matmul_q4.comp` shader performs).
pub fn gemv_q4(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols, "gemv_q4 input width mismatch");
    assert_eq!(y.len(), rows, "gemv_q4 output width mismatch");
    let row_bytes = (cols + 1).div_ceil(2).div_ceil(4) * 4;
    let head_bytes = rows * 4;
    let dot_row = |row: usize| -> f32 {
        let scale = f32::from_le_bytes([
            weight[row * 4],
            weight[row * 4 + 1],
            weight[row * 4 + 2],
            weight[row * 4 + 3],
        ]);
        let row_start = head_bytes + row * row_bytes;
        let mut sum = 0f32;
        for (column, value) in x.iter().enumerate() {
            let byte = weight[row_start + column / 2];
            let nibble = if column % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            let code = if nibble >= 8 {
                nibble as i8 - 16
            } else {
                nibble as i8
            };
            sum += scale * (code as f32) * value;
        }
        sum
    };
    if rows <= GEMV_CHUNK_ROWS {
        for (row, y_value) in y.iter_mut().enumerate() {
            *y_value = dot_row(row);
        }
        return;
    }
    y.par_chunks_mut(GEMV_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk, out)| {
            let row_start = chunk * GEMV_CHUNK_ROWS;
            for (index_in_chunk, y_value) in out.iter_mut().enumerate() {
                *y_value = dot_row(row_start + index_in_chunk);
            }
        });
}

/// Dequantize a contiguous weight array (row-major) to f32 values — the
/// load-time quantize path feeds the registered quantize operators with this.
pub fn dequant_values(bytes: &[u8], dtype: WeightDType, count: usize) -> Vec<f32> {
    debug_assert_eq!(
        bytes.len(),
        count * dtype.element_size(),
        "dequant_values length mismatch"
    );
    (0..count)
        .map(|index| scalar_at(bytes, index, dtype))
        .collect()
}

/// RMSNorm: x / sqrt(mean(x²) + eps) · weight, computed in f32 in place.
pub fn rmsnorm(x: &mut [f32], weight: &[u8], dtype: WeightDType, eps: f32) {
    let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_square + eps).sqrt();
    for (value, index) in x.iter_mut().zip(0..) {
        *value *= inv * scalar_at(weight, index, dtype);
    }
}

/// gemma-flavored RMSNorm: the affine weight is stored as `weight` but applied
/// as `(1 + weight)` — `x / sqrt(mean(x²) + eps) · (1 + weight)`.
pub fn rmsnorm_offset(x: &mut [f32], weight: &[u8], dtype: WeightDType, eps: f32) {
    let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_square + eps).sqrt();
    for (value, index) in x.iter_mut().zip(0..) {
        *value *= inv * (1.0 + scalar_at(weight, index, dtype));
    }
}

/// LayerNorm with affine parameters (phi): `(x - mean) / sqrt(var + eps)`
/// scaled by the weight and shifted by the bias, both stored-dtype reads. An
/// empty bias slice skips the shift.
pub fn layernorm(
    x: &mut [f32],
    weight: &[u8],
    weight_dtype: WeightDType,
    bias: &[u8],
    bias_dtype: WeightDType,
    eps: f32,
) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance = x
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / x.len() as f32;
    let inv = 1.0 / (variance + eps).sqrt();
    for (value, index) in x.iter_mut().zip(0..) {
        let mut normalized = (*value - mean) * inv * scalar_at(weight, index, weight_dtype);
        if !bias.is_empty() {
            normalized += scalar_at(bias, index, bias_dtype);
        }
        *value = normalized;
    }
}

/// Rotary embedding (GPT-NeoX rotate-half style):
/// for each head of dim d, split [x1 | x2] and pair angle i with both halves.
pub fn rope_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) {
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

/// Rotary embedding across heads over the leading `rotary_dim` dims of each
/// head: `head_dim` for the full-rotation families, a partial share for
/// phi-style partial rotary (remaining dims pass through untouched). The
/// per-head loop is part of the operator, so a registration file rotates a
/// whole projection with one call.
pub fn rope_heads(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) {
    for head in 0..heads {
        let start = head * head_dim;
        rope_in_place(
            &mut values[start..start + rotary_dim],
            1,
            rotary_dim,
            position,
            theta,
        );
    }
}

/// SwiGLU activation: silu(gate) ⊙ up, written into `gate`.
pub fn silu_multiply(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g *= u / (1.0 + (-*g).exp());
    }
}

/// GELU with the tanh approximation (`gelu_new` / `gelu_pytorch_tanh`),
/// written in place:
/// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 x³)))`.
pub fn gelu(values: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    for value in values.iter_mut() {
        let x = *value;
        let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
        *value = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// GeGLU activation (gemma): gelu(gate) ⊙ up, written into `gate`.
pub fn gelu_multiply(gate: &mut [f32], up: &[f32]) {
    gelu(gate);
    for (g, &u) in gate.iter_mut().zip(up.iter()) {
        *g *= u;
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

// ---------------------------------------------------------------------------
// Activation scratch + quantization state (shared by every family's forward).
// ---------------------------------------------------------------------------

/// Reusable activation buffers for one forward pass.
///
/// Buffers are allocated once with their maximum capacity when the family
/// runtime is constructed and then cleared/resized on every use: `Resize`
/// keeps the capacity, so the hot path does no allocator work. Most values
/// are overwritten wholesale, but `hidden` is the persistent residual stream
/// across layers and positions.
///
/// The buffers live in the family runtime struct next to its
/// [`OpsToolbox`]: toolbox methods take `&mut [f32]` arguments, and the
/// disjoint ownership (store/caches in the toolbox, activations outside it)
/// is what keeps the borrow checker happy on the hot path.
#[derive(Clone)]
pub(crate) struct Scratch {
    /// Residual stream for the token currently being decoded; length
    /// `hidden_size`. Seeded from the embedding row and updated by each block.
    pub(crate) hidden: Vec<f32>,
    /// Output of the most recent norm (input norm, post-attention norm, or
    /// phi's shared input norm). Attention and MLP both read this.
    pub(crate) normed: Vec<f32>,
    /// Query states after projection, optional q_norm, and rope; flattened as
    /// `[q_heads * head_dim]`. For phi's fused path it temporarily holds the
    /// stacked q/k/v rows before they are split.
    pub(crate) q: Vec<f32>,
    /// Key states after projection, optional k_norm, and rope, flattened as
    /// `[kv_heads * head_dim]`. Appended to the layer's KV cache.
    pub(crate) k: Vec<f32>,
    /// Value states after projection, flattened as `[kv_heads * head_dim]`.
    pub(crate) v: Vec<f32>,
    /// Attention output before the output projection, flattened as
    /// `[q_heads * head_dim]`. This is the softmax-weighted value mix.
    pub(crate) attn_out: Vec<f32>,
    /// Attention output after `o_proj`/`dense` (plus optional bias), length
    /// `hidden_size`. The caller adds it to the residual stream.
    pub(crate) attn_projected: Vec<f32>,
    /// Gate projection / fc1 output, length `intermediate_size`. Also holds
    /// the activated intermediate (`act(gate) * up`) that feeds the down
    /// projection.
    pub(crate) gate: Vec<f32>,
    /// Up projection output, length `intermediate_size`, consumed by the
    /// gated activation.
    pub(crate) up: Vec<f32>,
    /// Feed-forward output of the current block, length `hidden_size`. Also
    /// the per-expert scratch buffer inside the MoE loop.
    pub(crate) mlp_out: Vec<f32>,
    /// MoE accumulator: each selected expert's scaled output lands here
    /// before swapping into `mlp_out`.
    pub(crate) mlp_out_weighted: Vec<f32>,
    /// Router logits for the sparse block, length `num_local_experts`; turned
    /// into a softmax top-k selection by `select_experts`.
    pub(crate) router: Vec<f32>,
    /// Final vocabulary logits, length `vocab_size`. `forward` moves this out
    /// with `mem::take`, leaving an empty vector that is resized next time.
    pub(crate) logits: Vec<f32>,
}

impl Scratch {
    /// Preallocate every buffer at its maximum capacity for this config.
    pub(crate) fn new(cfg: &DecoderConfig) -> Self {
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        Self {
            router: Vec::new(),
            hidden: Vec::with_capacity(cfg.hidden_size),
            normed: Vec::with_capacity(cfg.hidden_size),
            q: Vec::with_capacity(cfg.q_heads * cfg.head_dim),
            k: Vec::with_capacity(kv_dim),
            v: Vec::with_capacity(kv_dim),
            attn_out: Vec::with_capacity(cfg.q_heads * cfg.head_dim),
            attn_projected: Vec::with_capacity(cfg.hidden_size),
            gate: Vec::with_capacity(cfg.intermediate_size),
            up: Vec::with_capacity(cfg.intermediate_size),
            mlp_out: Vec::with_capacity(cfg.hidden_size),
            mlp_out_weighted: Vec::with_capacity(cfg.hidden_size),
            logits: Vec::with_capacity(cfg.vocab_size),
        }
    }
}

/// The selected quantization format's native operators (from the precision
/// registry) plus the quantized unit buffers it has produced. `None`-free
/// design: without a selected format both operators stay `None` and every
/// unit keeps its stored precision.
///
/// The two function pointers are the format's own entry points: `quantize`
/// converts an f32 value vector into the format's packed buffer, and `gemv`
/// computes a matvec over that packed representation. They travel together
/// because a format that can compute cannot be useful without the ability to
/// produce its buffer in the first place.
#[derive(Clone)]
pub(crate) struct QuantState {
    /// Registered quantizer: (values, rows, cols) -> packed buffer. `None`
    /// when no quantization format is selected.
    pub(crate) quantize:
        Option<fn(&[f32], usize, usize) -> Result<Arc<dyn PrecisionBuffer>, String>>,
    /// Registered quantized matvec: (packed bytes, rows, cols, x, y). `None`
    /// when no quantization format is selected.
    pub(crate) gemv: Option<fn(&[u8], usize, usize, &[f32], &mut [f32])>,
    /// Quantized linear units, keyed by unit id: first use quantizes the
    /// source unit through the registered operator and drops its source
    /// pages; later steps compute from the quantized buffer.
    pub(crate) buffers: HashMap<String, Arc<dyn PrecisionBuffer>>,
}

/// Stage-wise debug capture of layer 0's attention at the last input
/// position. Test instrumentation only: the softmax scores of query head 0
/// (used by the sliding-window test) and the pre-projection attention mix.
#[allow(dead_code)]
pub(crate) struct DebugCapture {
    /// Softmax scores of query head 0 over all cached keys.
    pub(crate) scores: Vec<f32>,
    /// Attention output before the output projection, [q_heads * head_dim].
    pub(crate) attn_out: Vec<f32>,
}

// ---------------------------------------------------------------------------
// The OpsToolbox.
// ---------------------------------------------------------------------------

/// Shared native inference state and the operators that act on it.
///
/// One toolbox backs one model runtime. It owns the weight store, the KV
/// caches, the quantized linear buffers, the routing-aware expert residency,
/// and the debug capture; every method is a statically dispatched call with
/// explicit unit ids and shapes. Model structure (which units run in which
/// order, residual topology, norm flavor) belongs to the registration file
/// that drives the toolbox — nothing here branches on a family name.
pub(crate) struct OpsToolbox {
    /// Fully resolved hyper-parameters (config → units manifest → family
    /// constants). The family runtime keeps its own copy for composition;
    /// this one backs the toolbox-internal ops.
    pub(crate) cfg: DecoderConfig,
    /// Shared handle to the mapped weight store. Clones share the same
    /// mapping, so cloning a runtime does not duplicate weight pages.
    store: Arc<MmapWeightStore>,
    /// Per-layer key cache, flattened as
    /// [position * kv_heads * head_dim + head * head_dim + dim].
    kv_k: Vec<Vec<f32>>,
    /// Per-layer value cache with the same layout as `kv_k`.
    kv_v: Vec<Vec<f32>>,
    /// Softmax scores for one query head, length = number of visible cached
    /// keys. Attention-internal scratch.
    scores: Vec<f32>,
    /// Selected quantization: linear units quantize at first use and compute
    /// from the quantized buffer.
    pub(crate) quant: QuantState,
    /// RC2-B: routing-aware per-expert residency, keyed by
    /// (layer_index, expert_index) -> resident. Kept in sync with the MoE
    /// router's top-k selections while a generation walks the layers.
    pub(crate) expert_residency: HashMap<(usize, usize), bool>,
    /// Test hook: capture layer-0 attention scores/attn-out when armed.
    debug_capture_armed: bool,
    /// Filled by `attention` when the hook is armed, then taken by the test
    /// helper. `None` during normal generation.
    debug_capture_buf: Option<DebugCapture>,
}

impl Clone for OpsToolbox {
    /// Clone semantics for the toolbox: the weight store is shared through the
    /// Arc, while the KV caches, quantized-buffer table, expert-residency map
    /// and debug state start fresh — the engine clones the runtime per
    /// generation and clears the cache anyway.
    fn clone(&self) -> Self {
        // Weights are shared through the Arc'd store; a clone starts from a
        // fresh KV cache and fresh quantized buffers (the engine clears the
        // cache before every generation anyway) — the same semantics the
        // decoder clone always had.
        Self::with_store(self.cfg.clone(), Arc::clone(&self.store))
    }
}

impl OpsToolbox {
    /// Build a toolbox around an already-resolved config and store (test and
    /// fixture path; the loader path uses [`OpsToolbox::open`]).
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        Self {
            kv_k: vec![Vec::new(); cfg.num_layers],
            kv_v: vec![Vec::new(); cfg.num_layers],
            scores: Vec::with_capacity(4096),
            cfg,
            store,
            quant: QuantState::selected(),
            expert_residency: HashMap::new(),
            debug_capture_armed: false,
            debug_capture_buf: None,
        }
    }

    /// Load-path constructor: open the mmap store, resolve the configuration
    /// through the parameter fallback chain (config.json → units manifest →
    /// family constants), and install the Ant residency plan.
    pub(crate) fn open(
        model_dir: &Path,
        raw_config: &Value,
        plan: &BTreeMap<String, String>,
        family: crate::chat::registry::ModelFamily,
    ) -> Result<Self, String> {
        let store = Arc::new(MmapWeightStore::open(model_dir)?);
        let cfg = DecoderConfig::resolve(Some(raw_config), &store, Some(family))?;
        let mut toolbox = Self::with_store(cfg, Arc::clone(&store));
        toolbox.apply_tiers(plan);
        Ok(toolbox)
    }

    /// Reject an empty token batch and ids beyond the vocabulary.
    pub(crate) fn validate_tokens(&self, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("Native backend: empty token input.".into());
        }
        for &token in tokens {
            if (token as usize) >= self.cfg.vocab_size {
                return Err(format!(
                    "Native backend: token id {token} exceeds the vocabulary size {}.",
                    self.cfg.vocab_size
                ));
            }
        }
        Ok(())
    }

    /// Copy one embedding row into `out` (replacing its contents). The row is
    /// read in the unit's stored precision; any scaling (gemma's
    /// `sqrt(hidden_size)`) is the caller's model logic.
    pub(crate) fn embedding_row_into(
        &self,
        token: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let (bytes, dtype, shape) = self
            .store
            .slice("model.embed_tokens.weight")
            .map_err(|error| format!("Native backend: {error}"))?;
        if shape.len() != 2 || shape[1] != self.cfg.hidden_size {
            return Err("Native backend: unexpected embedding shape.".into());
        }
        let base = token * self.cfg.hidden_size;
        out.clear();
        out.extend((0..self.cfg.hidden_size).map(|index| scalar_at(bytes, base + index, dtype)));
        Ok(())
    }

    /// RMSNorm against the `{unit}.weight` bytes, in place on `target`.
    pub(crate) fn rmsnorm(&self, unit: &str, target: &mut [f32]) -> Result<(), String> {
        let (weight, dtype, _) = self.store.slice(&format!("{unit}.weight"))?;
        rmsnorm(target, weight, dtype, self.cfg.rms_eps);
        Ok(())
    }

    /// gemma-flavored RMSNorm `x/rms · (1 + weight)` against `{unit}.weight`.
    pub(crate) fn rmsnorm_offset(&self, unit: &str, target: &mut [f32]) -> Result<(), String> {
        let (weight, dtype, _) = self.store.slice(&format!("{unit}.weight"))?;
        rmsnorm_offset(target, weight, dtype, self.cfg.rms_eps);
        Ok(())
    }

    /// LayerNorm against `{unit}.weight` (+ `{unit}.bias` when present), in
    /// place on `target`.
    pub(crate) fn layernorm(&self, unit: &str, target: &mut [f32]) -> Result<(), String> {
        let (weight, dtype, _) = self.store.slice(&format!("{unit}.weight"))?;
        let (bias, bias_dtype, _) =
            self.store
                .slice(&format!("{unit}.bias"))
                .unwrap_or((&[], dtype, &[]));
        layernorm(target, weight, dtype, bias, bias_dtype, self.cfg.rms_eps);
        Ok(())
    }

    /// Per-head RMSNorm on q/k states (qwen3): each `head_dim` slice of
    /// `values` is normalized against `{unit}.weight`. A no-op when the unit
    /// is absent, so a family can call it unconditionally.
    pub(crate) fn qk_norm(
        &self,
        values: &mut [f32],
        heads: usize,
        head_dim: usize,
        unit: &str,
    ) -> Result<(), String> {
        let unit = format!("{unit}.weight");
        if !self.store.contains(&unit) {
            return Ok(());
        }
        let (weight, dtype, _) = self.store.slice(&unit)?;
        for head in 0..heads {
            let start = head * head_dim;
            rmsnorm(
                &mut values[start..start + head_dim],
                weight,
                dtype,
                self.cfg.rms_eps,
            );
        }
        Ok(())
    }

    /// Add a `{unit}.bias` tensor's values to `values`; a no-op when the
    /// unit is absent (families without attention/MLP biases).
    pub(crate) fn add_bias(&self, values: &mut [f32], unit: &str) {
        let Ok((bias_bytes, dtype, _)) = self.store.slice(&format!("{unit}.bias")) else {
            return;
        };
        if bias_bytes.is_empty() {
            return;
        }
        for (value, index) in values.iter_mut().zip(0..) {
            *value += scalar_at(bias_bytes, index, dtype);
        }
    }

    /// Compute one linear layer `y = W · x` for the named weight unit.
    ///
    /// Three paths, checked in this order:
    /// 1. the unit was already quantized: run the registered quantized gemv
    ///    over the cached buffer;
    /// 2. a quantization format is selected: dequantize the source bytes to
    ///    f32 once, quantize them, drop the source pages, then gemv;
    /// 3. no format: look up the native operator table for the unit's stored
    ///    dtype and run its gemv directly over the mapped bytes.
    ///
    /// `rows`/`cols` are the logical shape of W (row-major) and must match
    /// both the buffer and the destination length (`y.len() == rows`).
    pub(crate) fn linear(
        &mut self,
        unit: &str,
        rows: usize,
        cols: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        // Fast path: this unit has already been quantized at least once. The
        // buffer owns its shape, so the caller's `rows`/`cols` are not used to
        // compute — they only describe the expected logical shape.
        if let Some(buffer) = self.quant.buffers.get(unit) {
            let gemv = self
                .quant
                .gemv
                .expect("quantized unit present without a registered gemv operator");
            (gemv)(buffer.buffer(), buffer.rows(), buffer.cols(), x, y);
            return Ok(());
        }
        if let (Some(quantize), Some(gemv)) = (self.quant.quantize, self.quant.gemv) {
            // Lazy load-time quantization: dequantize the source unit to f32
            // values once, run the registered quantize operator, and release
            // the source pages — the unit computes from the quantized buffer
            // (roughly half the footprint) from then on. Norms, biases and
            // the embedding never reach this path.
            let (bytes, dtype, _) = self.store.slice(unit)?;
            let count = rows
                .checked_mul(cols)
                .ok_or_else(|| format!("Linear unit '{unit}' size overflow."))?;
            if bytes.len() != count * dtype.element_size() {
                return Err(format!(
                    "Linear unit '{unit}' does not match its {rows}x{cols} shape."
                ));
            }
            let values = dequant_values(bytes, dtype, count);
            let buffer = quantize(&values, rows, cols)?;
            self.quant.buffers.insert(unit.to_owned(), buffer);
            // The source bytes are never read again once quantized.
            self.store.drop_pages(unit);
            let buffer = self
                .quant
                .buffers
                .get(unit)
                .expect("quantized buffer was just inserted");
            (gemv)(buffer.buffer(), buffer.rows(), buffer.cols(), x, y);
            return Ok(());
        }
        let (bytes, dtype, _) = self.store.slice(unit)?;
        let format_id = dtype.format_id();
        let ops = precision::native_ops_for_format(format_id).ok_or_else(|| {
            format!(
                "Weight precision '{format_id}' ({}) has no native operator table for unit '{unit}'.",
                precision::format_label(format_id)
            )
        })?;
        (ops.gemv)(bytes, rows, cols, x, y);
        Ok(())
    }

    /// Append one position's K/V rows to a layer's cache.
    pub(crate) fn cache_append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.kv_k[layer].extend_from_slice(k);
        self.kv_v[layer].extend_from_slice(v);
    }

    /// Grouped-query attention over the cached keys for `position`.
    ///
    /// Every `q_heads / kv_heads` query heads of `q` (post-rope, flattened
    /// `[q_heads * head_dim]`) read one KV head; scores use the
    /// `1/sqrt(head_dim)` scale and the resolved `sliding_window` (0 = the
    /// whole cache; keys older than the window are masked to -inf before the
    /// softmax, and the window start is clamped to the newest key so an
    /// all-masked row is impossible). The softmax-weighted value mix lands in
    /// `out` (pre-zeroed by the caller). This is the same math the
    /// `shared_ops.comp` attention shader runs on the GPU side.
    pub(crate) fn attention(&mut self, layer: usize, position: usize, q: &[f32], out: &mut [f32]) {
        let kv_dim = self.cfg.kv_heads * self.cfg.head_dim;
        let cache_len = self.kv_k[layer].len() / kv_dim;
        let head_group = self.cfg.q_heads / self.cfg.kv_heads;
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
        // Sparse (sliding-window) attention: the query at `position` only
        // sees keys from `end - window` onward (`end` is one past this
        // position), matching the Vulkan attention shader.
        // window == 0 attends to the whole cache.
        let window_start = if self.cfg.sliding_window == 0 {
            0
        } else {
            (position + 1).saturating_sub(self.cfg.sliding_window)
        };
        // Never start past the cache: an all-masked softmax would be NaN.
        let window_start = window_start.min(cache_len.saturating_sub(1));
        let mut head0_scores: Option<Vec<f32>> = None;
        for head in 0..self.cfg.q_heads {
            let kv_head = head / head_group;
            let q_base = head * self.cfg.head_dim;
            let kv_base = kv_head * self.cfg.head_dim;
            self.scores.clear();
            self.scores.resize(cache_len, f32::NEG_INFINITY);
            for pos in window_start..cache_len {
                let k_base = pos * kv_dim + kv_base;
                let score: f32 = self.kv_k[layer][k_base..k_base + self.cfg.head_dim]
                    .iter()
                    .zip(&q[q_base..q_base + self.cfg.head_dim])
                    .map(|(k, q)| k * q)
                    .sum();
                self.scores[pos] = score * scale;
            }
            softmax(&mut self.scores);
            if head == 0 && self.debug_capture_armed {
                head0_scores = Some(self.scores.clone());
            }
            for pos in window_start..cache_len {
                let weight = self.scores[pos];
                if weight == 0.0 {
                    continue;
                }
                let v_base = pos * kv_dim + kv_base;
                for (index, out_value) in out[q_base..q_base + self.cfg.head_dim]
                    .iter_mut()
                    .enumerate()
                {
                    *out_value += weight * self.kv_v[layer][v_base + index];
                }
            }
        }
        if head0_scores.is_some() {
            self.debug_capture_armed = false;
            self.debug_capture_buf = Some(DebugCapture {
                scores: head0_scores.expect("armed capture recorded head-0 scores"),
                attn_out: out.to_vec(),
            });
        }
    }

    /// RC2-B: routing-aware per-expert residency management. Evict the
    /// experts the router moved away from, page in the newly selected ones.
    /// State keyed by (layer, expert) persists across positions.
    pub(crate) fn update_expert_residency(&mut self, layer: usize, picked: &[usize]) {
        for expert in 0..self.cfg.num_local_experts {
            let key = (layer, expert);
            let was_resident = self.expert_residency.get(&key).copied().unwrap_or(false);
            if was_resident && !picked.contains(&expert) {
                self.set_expert_resident(layer, expert, false);
                self.expert_residency.insert(key, false);
            }
        }
        for &expert in picked {
            let key = (layer, expert);
            if !self.expert_residency.get(&key).copied().unwrap_or(false) {
                self.set_expert_resident(layer, expert, true);
                self.expert_residency.insert(key, true);
            }
        }
    }

    /// RC2-B: apply one expert's residency transition. The router-selected
    /// expert's gate/up/down units move together between the resident and disk
    /// tiers; unit ids the store does not know are ignored.
    fn set_expert_resident(&self, layer: usize, expert: usize, resident: bool) {
        let prefix = format!("model.layers.{layer}.");
        let layout = &self.cfg.moe_layout;
        let gate_unit = layout.expert_unit(&prefix, expert, layout.gate);
        let up_unit = layout.expert_unit(&prefix, expert, layout.up);
        let down_unit = layout.expert_unit(&prefix, expert, layout.down);
        let tier = if resident { Tier::Resident } else { Tier::Disk };
        self.store.set_tier(&gate_unit, tier);
        self.store.set_tier(&up_unit, tier);
        self.store.set_tier(&down_unit, tier);
    }

    /// Vocabulary logits over the residual stream: a chunked row sweep over
    /// the head unit that releases each chunk's pages right away when the
    /// unit sits on the disk tier, so the lm_head pass does not transiently
    /// hold the whole (largest) tensor resident.
    ///
    /// The head unit choice (tied embedding vs separate `lm_head.weight`) is
    /// the caller's model logic; the sweep and the precision dispatch are the
    /// toolbox's. `logits` is cleared and resized to `vocab_size`.
    pub(crate) fn lm_head_into(
        &self,
        head_unit: &str,
        hidden: &[f32],
        logits: &mut Vec<f32>,
    ) -> Result<(), String> {
        if !self.store.contains(head_unit) {
            return Err(format!(
                "Native backend: neither tied embeddings nor a separate '{head_unit}' tensor are present."
            ));
        }
        let (bytes, dtype, shape) = self
            .store
            .slice(head_unit)
            .map_err(|error| format!("Native backend: {error}"))?;
        if shape.len() != 2 || shape[1] != self.cfg.hidden_size {
            return Err("Native backend: unexpected lm_head shape.".into());
        }
        logits.clear();
        logits.resize(self.cfg.vocab_size, 0.0);
        // lm_head keeps its stored precision (the registered quantize formats
        // cover the layer linear projections, matching the Vulkan executor's
        // take_linear coverage) — the chunked sweep below is already the
        // memory-optimal path for the largest tensor.
        let format_id = dtype.format_id();
        let ops = precision::native_ops_for_format(format_id).ok_or_else(|| {
            format!(
                "Weight precision '{format_id}' has no native operator table for '{head_unit}'."
            )
        })?;
        let chunk_rows = 16384usize;
        let row_stride = self.cfg.hidden_size * dtype.element_size();
        let mut start_row = 0usize;
        while start_row < self.cfg.vocab_size {
            let end_row = (start_row + chunk_rows).min(self.cfg.vocab_size);
            let byte_start = start_row * row_stride;
            let byte_len = (end_row - start_row) * row_stride;
            (ops.gemv)(
                &bytes[byte_start..byte_start + byte_len],
                end_row - start_row,
                self.cfg.hidden_size,
                hidden,
                &mut logits[start_row..end_row],
            );
            if self.store.tier(head_unit) == Tier::Disk {
                self.store.drop_range(head_unit, byte_start, byte_len);
            }
            start_row = end_row;
        }
        Ok(())
    }

    /// Cached positions in one layer's KV cache (test/diagnostics view:
    /// `kv_k[layer].len() / kv_dim`).
    #[cfg(test)]
    pub(crate) fn kv_len(&self, layer: usize) -> usize {
        let kv_dim = self.cfg.kv_heads * self.cfg.head_dim;
        self.kv_k[layer].len() / kv_dim.max(1)
    }

    /// Clear every layer's KV cache.
    pub(crate) fn clear_kv(&mut self) {
        for cache in self.kv_k.iter_mut().chain(self.kv_v.iter_mut()) {
            cache.clear();
        }
    }

    /// Apply the Ant residency plan to the mapped pages: resident tiers are
    /// faulted in, disk tiers are dropped.
    pub(crate) fn apply_tiers(&mut self, plan: &BTreeMap<String, String>) {
        for unit_id in self.store.unit_ids().map(str::to_owned).collect::<Vec<_>>() {
            let tier = match plan.get(&unit_id).map(String::as_str) {
                Some("disk") => Tier::Disk,
                Some(_) | None => Tier::Resident,
            };
            self.store.set_tier(&unit_id, tier);
        }
    }

    /// Ant page-in on the native backend: fault a unit's pages back in.
    /// Returns whether the units manifest knows the unit.
    pub(crate) fn page_in_unit(&mut self, unit_id: &str) -> bool {
        if self.store.unit_ids().any(|id| id == unit_id) {
            self.store.set_tier(unit_id, Tier::Resident);
            true
        } else {
            false
        }
    }

    /// Ant evict on the native backend: drop a unit's pages to the disk tier.
    /// Returns whether the units manifest knows the unit.
    pub(crate) fn evict_unit(&mut self, unit_id: &str) -> bool {
        if self.store.unit_ids().any(|id| id == unit_id) {
            self.store.set_tier(unit_id, Tier::Disk);
            true
        } else {
            false
        }
    }

    /// Drop every disk-tier unit of one layer after the layer's step
    /// completed, so the transient footprint stays near one layer, not the
    /// whole model. Dropped pages re-fault from the OS page cache.
    pub(crate) fn drop_disk_units_of_layer(&self, layer_id: i32) {
        self.store.drop_disk_units_of_layer(layer_id);
    }

    /// Test helper: arm the layer-0 attention capture for the next
    /// `attention` call on layer 0.
    #[cfg(test)]
    pub(crate) fn arm_debug_capture(&mut self) {
        self.debug_capture_armed = true;
    }

    /// Test helper: take the armed capture (None when it never fired).
    #[cfg(test)]
    pub(crate) fn take_debug_capture(&mut self) -> Option<DebugCapture> {
        self.debug_capture_buf.take()
    }
}

/// Lazy per-unit quantization and the matvec dispatch used by every linear
/// projection.
impl QuantState {
    /// The selected quantization format's native operator table (precision
    /// registry), or an inert state when no format is selected / the format
    /// does not serve the native backend.
    fn selected() -> Self {
        // `selected_native_ops` is `Option`: no selection (or a format that
        // has no native kernels) yields `None`, which leaves both operator
        // slots empty and makes `OpsToolbox::linear` use stored-precision
        // paths.
        let ops = precision::selected_native_ops();
        Self {
            quantize: ops.and_then(|ops| ops.quantize),
            gemv: ops.map(|ops| ops.gemv),
            buffers: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// gemv_q8: the dot runs in the quantized domain — per-row scale × int8
    /// codes accumulated in f32 — and ignores the 4-byte row padding, matching
    /// the `matmul_q8.comp` shader's math bit for bit.
    #[test]
    fn gemv_q8_matches_rowwise_scale_code_reference() {
        let rows = 3usize;
        let cols = 6usize; // not a multiple of four: exercises the padded tail
        let row_bytes = (cols + 3) / 4 * 4;
        let x: Vec<f32> = (0..cols).map(|index| 0.5f32 - index as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = vec![
            vec![1.0, -2.0, 0.0, 3.5, -0.25, 2.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // zero row: scale falls back to 1.0
            vec![-7.0, 7.0, -1.0, 1.0, -0.5, 0.5],
        ];
        let mut buffer = vec![0u8; rows * 4 + rows * row_bytes];
        let mut expected = vec![0f32; rows];
        for row in 0..rows {
            let max_abs = weights[row]
                .iter()
                .fold(0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            buffer[row * 4..row * 4 + 4].copy_from_slice(&scale.to_le_bytes());
            let mut sum = 0f32;
            for (col, value) in weights[row].iter().enumerate() {
                let code = (value / scale).round().clamp(-127.0, 127.0) as i8;
                buffer[rows * 4 + row * row_bytes + col] = code as u8;
                sum += scale * (code as f32) * x[col];
            }
            expected[row] = sum;
        }
        let mut y = vec![0f32; rows];
        gemv_q8(&buffer, rows, cols, &x, &mut y);
        for (row, (got, want)) in y.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-5, "row {row}: {got} vs {want}");
        }
    }

    /// gemv_q4: the dot runs in the quantized domain — per-row scale × int4
    /// codes (two per byte, low nibble = even column) accumulated in f32 —
    /// matching the `matmul_q4.comp` shader's math bit for bit.
    #[test]
    fn gemv_q4_matches_rowwise_scale_code_reference() {
        let rows = 3usize;
        let cols = 9usize; // odd width: exercises both nibble parities + padding
        let row_bytes = (cols + 1).div_ceil(2).div_ceil(4) * 4;
        let x: Vec<f32> = (0..cols).map(|index| 0.5f32 - index as f32 * 0.1).collect();
        let weights: Vec<Vec<f32>> = vec![
            vec![1.0, -2.0, 0.0, 3.5, -0.25, 2.0, -1.5, 0.75, 4.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![-7.0, 7.0, -1.0, 1.0, -0.5, 0.5, -3.0, 3.0, 0.0],
        ];
        let mut buffer = vec![0u8; rows * 4 + rows * row_bytes];
        let mut expected = vec![0f32; rows];
        for row in 0..rows {
            let max_abs = weights[row]
                .iter()
                .fold(0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            buffer[row * 4..row * 4 + 4].copy_from_slice(&scale.to_le_bytes());
            let mut sum = 0f32;
            for (col, value) in weights[row].iter().enumerate() {
                let code = (value / scale).round().clamp(-7.0, 7.0) as i8;
                let index = rows * 4 + row * row_bytes + col / 2;
                buffer[index] |= ((code as u8) & 0x0F) << ((col % 2) * 4);
                sum += scale * (code as f32) * x[col];
            }
            expected[row] = sum;
        }
        let mut y = vec![0f32; rows];
        gemv_q4(&buffer, rows, cols, &x, &mut y);
        for (row, (got, want)) in y.iter().zip(&expected).enumerate() {
            assert!((got - want).abs() < 1e-5, "row {row}: {got} vs {want}");
        }
    }

    /// The registered quantize operator and the native q4 gemv agree: feeding
    /// quantized weights back through gemv_q4 reconstructs the dequantized
    /// matvec (the two operators share one buffer layout by contract).
    #[test]
    fn gemv_q4_consumes_the_registered_quantize_output() {
        let rows = 5usize;
        let cols = 9usize;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.3)
            .collect();
        let quantize = crate::chat::precision::native_ops_for_format("q4")
            .and_then(|ops| ops.quantize)
            .expect("q4 must register a native quantize operator");
        let buffer = quantize(&values, rows, cols).unwrap();
        let x: Vec<f32> = (0..cols)
            .map(|index| (index as f32 * 0.7 - 2.0).sin())
            .collect();
        let mut y = vec![0f32; rows];
        gemv_q4(buffer.buffer(), buffer.rows(), buffer.cols(), &x, &mut y);
        for row in 0..rows {
            let want: f32 = (0..cols)
                .map(|col| buffer.dequant_value(row, col) * x[col])
                .sum();
            assert!(
                (y[row] - want).abs() < 1e-4,
                "row {row}: {} vs {want}",
                y[row]
            );
        }
    }

    /// The registered quantize operator and the native q8 gemv agree: feeding
    /// quantized weights back through gemv_q8 reconstructs the dequantized
    /// matvec (the two operators share one buffer layout by contract).
    #[test]
    fn gemv_q8_consumes_the_registered_quantize_output() {
        let rows = 5usize;
        let cols = 9usize;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 23) as f32 - 11.0) * 0.3)
            .collect();
        let quantize = crate::chat::precision::native_ops_for_format("q8")
            .and_then(|ops| ops.quantize)
            .expect("q8 must register a native quantize operator");
        let buffer = quantize(&values, rows, cols).unwrap();
        let x: Vec<f32> = (0..cols)
            .map(|index| (index as f32 * 0.7 - 2.0).sin())
            .collect();
        let mut y = vec![0f32; rows];
        gemv_q8(buffer.buffer(), buffer.rows(), buffer.cols(), &x, &mut y);
        for row in 0..rows {
            let want: f32 = (0..cols)
                .map(|col| buffer.dequant_value(row, col) * x[col])
                .sum();
            assert!(
                (y[row] - want).abs() < 1e-4,
                "row {row}: {} vs {want}",
                y[row]
            );
        }
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

    /// rope_heads drives the per-head loop of the kernel: full rotary spans
    /// every head, a partial span leaves the trailing dims untouched.
    #[test]
    fn rope_heads_rotates_leading_dims_of_every_head() {
        // Two heads of dim 4 with a partial span of 2: only dims {0,1} and
        // {4,5} rotate; {2,3} and {6,7} must pass through untouched.
        let mut values = vec![1.0f32, 0.0, 9.0, 8.0, 0.0, 1.0, 7.0, 6.0];
        rope_heads(&mut values, 2, 4, 2, 1, 10_000.0);
        // head 0: [1, 0] rotated by 1 radian (freq = theta^0 = 1).
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((values[0] - (1.0 * cos - 0.0 * sin)).abs() < 1e-5);
        assert!((values[1] - (0.0 * cos + 1.0 * sin)).abs() < 1e-5);
        // head 1: [0, 1] rotated by the same angle.
        assert!((values[4] - (0.0 * cos - 1.0 * sin)).abs() < 1e-5);
        assert!((values[5] - (1.0 * cos + 0.0 * sin)).abs() < 1e-5);
        // Untouched trailing dims.
        assert_eq!(&values[2..4], &[9.0, 8.0]);
        assert_eq!(&values[6..8], &[7.0, 6.0]);
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

    /// Replicate the qwen2 rope (duplicate-frequency rotate-half) with
    /// plain tensor math and compare against the native kernel at positions > 0.
    #[test]
    fn rope_matches_reference_rotate_half() {
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
            // Expected: inv_freq[i] = 1 / theta^(2i/dim) for i in 0..half;
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

    /// The AVX2-vs-dequant+dot bit-math isolation: fused widening must track
    /// the dequantize-then-dot reference.
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
