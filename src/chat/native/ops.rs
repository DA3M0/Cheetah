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
use crate::chat::registry::{KvSnapshot, QuantizedKvLayer, QuantizedKvSnapshot};
use serde_json::Value;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;

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
/// row at a time into a per-chunk scratch buffer and dots in f32. Up to
/// `GEMV_CHUNK_ROWS` (128) rows run sequentially (a rayon fork/join would
/// cost more than the compute), larger matrices split into 128-row rayon
/// chunks. The scratch buffer is allocated once per chunk, never per row —
/// the old per-row allocation dominated f16/f32 decode.
pub fn gemv(weight: &[u8], dtype: WeightDType, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), cols, "gemv input width mismatch");
    assert_eq!(y.len(), rows, "gemv output width mismatch");
    let element_size = dtype.element_size();
    let row_stride = cols * element_size;
    if matches!(dtype, WeightDType::Bf16) {
        let dot_row = |row_bytes: &[u8]| -> f32 { dot_row_bf16_fused(row_bytes, x) };
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
        return;
    }
    // f16/f32: one dequant scratch per sequential call or per rayon chunk.
    let dot_row = |row_bytes: &[u8], scratch: &mut [f32]| -> f32 {
        dequant_row(row_bytes, dtype, scratch);
        dot_f32(scratch, x)
    };
    if rows <= GEMV_CHUNK_ROWS {
        let mut scratch = vec![0f32; cols];
        for (row, y_value) in y.iter_mut().enumerate() {
            let row_bytes = &weight[row * row_stride..(row + 1) * row_stride];
            *y_value = dot_row(row_bytes, &mut scratch);
        }
        return;
    }
    y.par_chunks_mut(GEMV_CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk, out)| {
            let row_start = chunk * GEMV_CHUNK_ROWS;
            let mut scratch = vec![0f32; cols];
            for (index_in_chunk, y_value) in out.iter_mut().enumerate() {
                let row = row_start + index_in_chunk;
                let row_bytes = &weight[row * row_stride..(row + 1) * row_stride];
                *y_value = dot_row(row_bytes, &mut scratch);
            }
        });
}

/// Rows per GEMM row-block: sized so a block's weight bytes (~115 KB at
/// Qwen-scale widths) share the L2 with the active token group.
const GEMM_BLOCK_ROWS: usize = 64;
/// Tokens per GEMM token-group: the group's x rows (~14 KB at Qwen-scale
/// widths) stay L1-resident while the row block streams past them.
const GEMM_TOKEN_GROUP: usize = 4;
/// Cache length above which attention fans its heads out to rayon: below
/// this the fork/join costs more than the per-head math.
const ATTENTION_PARALLEL_MIN_CACHE: usize = 384;

/// Row-parallel matrix-matrix `Y = X · Wᵀ` over stored-precision weight bytes:
/// `x` is `[n, cols]` row-major and `y` is `[n, rows]` row-major.
///
/// BF16 (the prepared-model dtype) is where chunked prefill gets its win: the
/// weight bytes stream from RAM once per batch instead of once per token.
/// Blocking is two-level — rayon owns `[GEMM_BLOCK_ROWS, n]` row-block
/// slices of a transposed scratch, and inside each block the tokens are
/// walked in `GEMM_TOKEN_GROUP`-sized groups through a multi-token AVX2
/// kernel, so the weight block lives in L2 while each 4-token x group lives
/// in L1 (at Qwen widths the full stacked x is far bigger than a mobile
/// L2 — a naive row-major loop re-reads it from L3 once per weight row and
/// gives every cache win back). Every other dtype (and `n == 1`) runs the
/// existing per-token [`gemv`] unchanged, so non-bf16 paths lose nothing.
///
/// Bit-exactness with the per-token path: every output element is a dot of
/// the same weight row against the same input row with the same accumulation
/// order per element — the batching changes only the loop order, never the
/// math.
pub fn gemm(
    weight: &[u8],
    dtype: WeightDType,
    rows: usize,
    cols: usize,
    x: &[f32],
    n: usize,
    y: &mut [f32],
) {
    assert_eq!(x.len(), n * cols, "gemm input width mismatch");
    assert_eq!(y.len(), n * rows, "gemm output width mismatch");
    if n == 1 || !matches!(dtype, WeightDType::Bf16) {
        for index in 0..n {
            gemv(
                weight,
                dtype,
                rows,
                cols,
                &x[index * cols..(index + 1) * cols],
                &mut y[index * rows..(index + 1) * rows],
            );
        }
        return;
    }
    // Transposed scratch `y_t[row][token]`: each row-block owns one
    // contiguous slice, so the parallel chunks write disjoint memory while
    // sharing the read-only x.
    let mut y_t = vec![0f32; rows * n];
    let block_rows = GEMM_BLOCK_ROWS.min(rows);
    if rows <= block_rows {
        gemm_row_block(weight, cols, x, n, 0, rows, &mut y_t);
    } else {
        y_t.par_chunks_mut(block_rows * n)
            .enumerate()
            .for_each(|(chunk, out)| {
                let row_start = chunk * block_rows;
                let rows_in_block = out.len() / n;
                gemm_row_block(weight, cols, x, n, row_start, rows_in_block, out);
            });
    }
    // Transpose the scratch `[rows][n]` into the caller's `[n][rows]` layout.
    for (row, out_row) in y_t.chunks_exact(n).enumerate() {
        for (token, value) in out_row.iter().enumerate() {
            y[token * rows + row] = *value;
        }
    }
}

/// Dots weight rows `row_start..row_start + block_rows` against every input
/// row, writing `y_t[row][token]` into `out` (`block_rows` rows of `n`
/// tokens each). Token groups outer, weight rows inner: the group's x rows
/// stay L1-hot across the whole row block, and the row block stays L2-hot
/// across all token groups.
fn gemm_row_block(
    weight: &[u8],
    cols: usize,
    x: &[f32],
    n: usize,
    row_start: usize,
    block_rows: usize,
    out: &mut [f32],
) {
    let row_stride = cols * WeightDType::Bf16.element_size();
    for token_start in (0..n).step_by(GEMM_TOKEN_GROUP) {
        let token_count = GEMM_TOKEN_GROUP.min(n - token_start);
        for index_in_block in 0..block_rows {
            let row = row_start + index_in_block;
            let row_bytes = &weight[row * row_stride..(row + 1) * row_stride];
            let out_row = &mut out[index_in_block * n..(index_in_block + 1) * n];
            dot_row_bf16_multi_fused(
                row_bytes,
                x,
                token_start,
                token_count,
                &mut out_row[token_start..token_start + token_count],
            );
        }
    }
}

/// Dots one bf16 weight row against `token_count` consecutive input rows at
/// once, writing the `token_count` results into `out`. The weight bytes and
/// their bf16→f32 widening are amortized across the token group; each x row
/// is `token_stride` floats wide (the caller passes the full stacked x).
/// AVX2+FMA path: per 16-weight chunk the widened halves feed 4 tokens × 2
/// FMA accumulators; the per-dot horizontal reductions shrink by the same
/// group factor. Falls back to per-token [`dot_row_bf16_fused`] without AVX2.
fn dot_row_bf16_multi_fused(
    row: &[u8],
    x: &[f32],
    token_start: usize,
    token_count: usize,
    out: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above.
            unsafe {
                return dot_row_bf16_avx2_multi(row, x, token_start, token_count, out);
            }
        }
    }
    let cols = row.len() / WeightDType::Bf16.element_size();
    for (index, slot) in out.iter_mut().enumerate() {
        let token = token_start + index;
        *slot = dot_row_bf16_fused(row, &x[token * cols..(token + 1) * cols]);
    }
}

/// Multi-token bf16 row dot over AVX2+FMA: one weight row against
/// `token_count` (≤ 4) input rows of `cols = row.len()/2` floats each,
/// starting at x row `token_start`. The four token accumulators are held in
/// *explicitly unrolled* ymm registers — a runtime-indexed accumulator array
/// spills to the stack on LLVM, serializing every FMA through memory — so
/// the weight load and its bf16→f32 widening are amortized across the group
/// while each FMA chain stays in registers. Each x row is `cols` floats wide
/// (the caller passes the full stacked x).
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support (runtime detection).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_bf16_avx2_multi(
    row: &[u8],
    x: &[f32],
    token_start: usize,
    token_count: usize,
    out: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;

        let cols = row.len() / 2;
        let (full, tail) = row.as_chunks::<32>(); // 32 bytes = 16 bf16 per chunk
        // One accumulator pair per token lane, unrolled: 8 ymm registers.
        let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        // x row base pointers for the (up to) four lanes; inactive lanes
        // reuse lane 0 — the extra FMAs land in accumulators that are never
        // reduced into `out`.
        let base0 = x.as_ptr().add(token_start * cols);
        let base1 = if token_count > 1 {
            x.as_ptr().add((token_start + 1) * cols)
        } else {
            base0
        };
        let base2 = if token_count > 2 {
            x.as_ptr().add((token_start + 2) * cols)
        } else {
            base0
        };
        let base3 = if token_count > 3 {
            x.as_ptr().add((token_start + 3) * cols)
        } else {
            base0
        };
        for (chunk_index, chunk) in full.iter().enumerate() {
            let w = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let lo = _mm256_castsi256_si128(w);
            let hi = _mm256_extracti128_si256::<1>(w);
            // Zero-extend u16 → u32 then shift left 16: the u32 lane becomes
            // `bf16 << 16`, bit-casting to the exact f32 value.
            let f_lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
            let f_hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16));
            let base = chunk_index * 16;
            a00 = _mm256_fmadd_ps(f_lo, _mm256_loadu_ps(base0.add(base)), a00);
            a01 = _mm256_fmadd_ps(f_hi, _mm256_loadu_ps(base0.add(base + 8)), a01);
            a10 = _mm256_fmadd_ps(f_lo, _mm256_loadu_ps(base1.add(base)), a10);
            a11 = _mm256_fmadd_ps(f_hi, _mm256_loadu_ps(base1.add(base + 8)), a11);
            a20 = _mm256_fmadd_ps(f_lo, _mm256_loadu_ps(base2.add(base)), a20);
            a21 = _mm256_fmadd_ps(f_hi, _mm256_loadu_ps(base2.add(base + 8)), a21);
            a30 = _mm256_fmadd_ps(f_lo, _mm256_loadu_ps(base3.add(base)), a30);
            a31 = _mm256_fmadd_ps(f_hi, _mm256_loadu_ps(base3.add(base + 8)), a31);
        }
        let full_count = full.len() * 16;
        let tails = tail.as_chunks::<2>().0;
        let mut results = [0f32; GEMM_TOKEN_GROUP];
        let accs = [[a00, a01], [a10, a11], [a20, a21], [a30, a31]];
        // Only the active lanes reduce: inactive lanes' accumulators hold
        // recomputed lane-0 values and their x rows may not exist.
        for (token_in_group, slot) in results[..token_count].iter_mut().enumerate() {
            let x_row = &x
                [(token_start + token_in_group) * cols..(token_start + token_in_group + 1) * cols];
            let mut tail_total = 0f32;
            for (index, byte_pair) in tails.iter().enumerate() {
                let x_index = full_count + index;
                tail_total += widen_bf16(byte_pair[0], byte_pair[1]) * x_row[x_index];
            }
            *slot = horizontal_sum(accs[token_in_group][0])
                + horizontal_sum(accs[token_in_group][1])
                + tail_total;
        }
        out[..token_count].copy_from_slice(&results[..token_count]);
    }
}

/// Sums the 8 f32 lanes of one accumulator (once per weight row per token —
/// negligible next to the FMA loop).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut storage = std::mem::MaybeUninit::<[f32; 8]>::uninit();
        _mm256_storeu_ps(storage.as_mut_ptr() as *mut f32, value);
        storage.assume_init().iter().sum()
    }
}

/// Rows per chunked-prefill GEMM batch. Overridable with
/// `CHEETAH_PREFILL_CHUNK` for experiments; 256 keeps the stacked inputs
/// (roughly a megabyte at Qwen-scale widths) cache-friendly while amortizing
/// the weight traffic across the whole chunk.
pub(crate) fn prefill_chunk_size() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("CHEETAH_PREFILL_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(PREFILL_CHUNK_ROWS)
    })
}

/// Default rows per chunked prefill batch (see [`prefill_chunk_size`]).
const PREFILL_CHUNK_ROWS: usize = 256;

/// Dot product of one int8 code row against f32 activations. Codes are
/// sign-extended to f32 and accumulated with FMA when AVX2 is available; the
/// portable path is the same math without explicit intrinsics.
#[inline]
fn dot_i8_f32(codes: &[i8], x: &[f32]) -> f32 {
    debug_assert_eq!(codes.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above; the
            // slices have equal length, so every load stays in bounds.
            unsafe { return dot_i8_f32_avx2(codes, x) }
        }
    }
    codes
        .iter()
        .zip(x.iter())
        .map(|(code, value)| *code as f32 * *value)
        .sum()
}

/// AVX2+FMA int8 dot: eight codes widen to one ymm of f32 lanes per step, so
/// the weight traffic is a quarter of the f32 path and the conversion stays
/// off the scalar pipeline.
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support (runtime detection).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_i8_f32_avx2(codes: &[i8], x: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let mut acc = _mm256_setzero_ps();
        let mut index = 0usize;
        while index + 8 <= codes.len() {
            let raw = _mm_loadl_epi64(codes.as_ptr().add(index) as *const __m128i);
            let widened = _mm256_cvtepi8_epi32(raw);
            let floats = _mm256_cvtepi32_ps(widened);
            acc = _mm256_fmadd_ps(floats, _mm256_loadu_ps(x.as_ptr().add(index)), acc);
            index += 8;
        }
        let mut sum = horizontal_sum(acc);
        while index < codes.len() {
            sum += codes[index] as f32 * x[index];
            index += 1;
        }
        sum
    }
}

/// `out += weight * codes` over an int8 row — the attention value-mix inner
/// loop. AVX2 widens eight codes at a time; the portable fallback is scalar.
#[inline]
fn accumulate_i8_f32(codes: &[i8], weight: f32, out: &mut [f32]) {
    debug_assert_eq!(codes.len(), out.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above; the
            // slices have equal length, so every load/store stays in bounds.
            unsafe { return accumulate_i8_f32_avx2(codes, weight, out) }
        }
    }
    for (out_value, code) in out.iter_mut().zip(codes.iter()) {
        *out_value += weight * *code as f32;
    }
}

/// AVX2+FMA weighted int8 accumulation (see [`accumulate_i8_f32`]).
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support (runtime detection).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_i8_f32_avx2(codes: &[i8], weight: f32, out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;

        let weight_lanes = _mm256_set1_ps(weight);
        let mut index = 0usize;
        while index + 8 <= codes.len() {
            let raw = _mm_loadl_epi64(codes.as_ptr().add(index) as *const __m128i);
            let widened = _mm256_cvtepi8_epi32(raw);
            let floats = _mm256_cvtepi32_ps(widened);
            let current = _mm256_loadu_ps(out.as_ptr().add(index));
            let updated = _mm256_fmadd_ps(floats, weight_lanes, current);
            _mm256_storeu_ps(out.as_mut_ptr().add(index), updated);
            index += 8;
        }
        while index < codes.len() {
            out[index] += weight * codes[index] as f32;
            index += 1;
        }
    }
}

/// Dot product of one packed int4 row (two codes per byte, low nibble = even
/// column) against f32 activations, in the codes' signed domain.
#[inline]
fn dot_q4_codes(row: &[u8], cols: usize, x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by the runtime feature detection above; the
            // packed row carries at least ceil(cols/2) bytes.
            unsafe { return dot_q4_codes_avx2(row, cols, x) }
        }
    }
    let mut sum = 0f32;
    for (column, value) in x.iter().enumerate().take(cols) {
        let byte = row[column / 2];
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
        sum += code as f32 * *value;
    }
    sum
}

/// AVX2+FMA packed-int4 dot: 32 bytes expand to 64 nibbles per step (four
/// 128-bit lane groups of 16 codes each, widened eight at a time).
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support (runtime detection) and a row
/// slice covering `ceil(cols / 2)` packed bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_codes_avx2(row: &[u8], cols: usize, x: &[f32]) -> f32 {
    unsafe {
        use std::arch::x86_64::*;

        let mask = _mm256_set1_epi8(0x0F);
        let sign = _mm256_set1_epi8(8);
        let full_vecs = (cols / 2) / 32;
        let mut acc = _mm256_setzero_ps();
        let mut column = 0usize;
        for vector in 0..full_vecs {
            let packed = _mm256_loadu_si256(row.as_ptr().add(vector * 32) as *const __m256i);
            let low = _mm256_and_si256(packed, mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            // unpacklo/hi interleave the two nibble streams back into column
            // order; `(nibble ^ 8) - 8` sign-extends 0..15 to -8..7.
            let even = _mm256_sub_epi8(
                _mm256_xor_si256(_mm256_unpacklo_epi8(low, high), sign),
                sign,
            );
            let odd = _mm256_sub_epi8(
                _mm256_xor_si256(_mm256_unpackhi_epi8(low, high), sign),
                sign,
            );
            let even_low = _mm256_castsi256_si128(even);
            let even_high = _mm256_extracti128_si256::<1>(even);
            let odd_low = _mm256_castsi256_si128(odd);
            let odd_high = _mm256_extracti128_si256::<1>(odd);
            // Columns: even lane 0 = 0..15, odd lane 0 = 16..31,
            // even lane 1 = 32..47, odd lane 1 = 48..63.
            acc = fma_i8_lane(acc, even_low, x.as_ptr().add(column));
            acc = fma_i8_lane(
                acc,
                _mm_srli_si128::<8>(even_low),
                x.as_ptr().add(column + 8),
            );
            acc = fma_i8_lane(acc, odd_low, x.as_ptr().add(column + 16));
            acc = fma_i8_lane(
                acc,
                _mm_srli_si128::<8>(odd_low),
                x.as_ptr().add(column + 24),
            );
            acc = fma_i8_lane(acc, even_high, x.as_ptr().add(column + 32));
            acc = fma_i8_lane(
                acc,
                _mm_srli_si128::<8>(even_high),
                x.as_ptr().add(column + 40),
            );
            acc = fma_i8_lane(acc, odd_high, x.as_ptr().add(column + 48));
            acc = fma_i8_lane(
                acc,
                _mm_srli_si128::<8>(odd_high),
                x.as_ptr().add(column + 56),
            );
            column += 64;
        }
        let mut sum = horizontal_sum(acc);
        while column < cols {
            let byte = row[column / 2];
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
            sum += code as f32 * x[column];
            column += 1;
        }
        sum
    }
}

/// Eight signed int8 lanes from `codes` widened to f32 and FMA'd into `acc`.
///
/// # Safety
/// Callers must guarantee AVX2 + FMA support and that `x` points at eight
/// readable f32 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_i8_lane(
    acc: std::arch::x86_64::__m256,
    codes: std::arch::x86_64::__m128i,
    x: *const f32,
) -> std::arch::x86_64::__m256 {
    unsafe {
        use std::arch::x86_64::*;
        let floats = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
        _mm256_fmadd_ps(floats, _mm256_loadu_ps(x), acc)
    }
}

/// Q8 matvec over the registered quantized buffer layout (`q8` in
/// `crate::chat::precision`): a head of `rows` f32 row scales followed by
/// row-major int8 payloads, each row padded to a multiple of four bytes. The
/// dot runs in the quantized domain — codes are unpacked as int8 and
/// multiplied by their row scale, f32 accumulate; the weights are never
/// bulk-dequantized into another storage width (the same math the
/// `matmul_q8.comp` shader performs). The row dot is AVX2+FMA when available.
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
        // SAFETY: int8 codes and f32 values have the same signed numeric
        // domain; the slice is exactly `cols` bytes long.
        let codes = unsafe { std::slice::from_raw_parts(codes.as_ptr().cast::<i8>(), cols) };
        scale * dot_i8_f32(codes, x)
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
/// storage width (the same math the `matmul_q4.comp` shader performs). The
/// row dot is AVX2+FMA when available.
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
        scale * dot_q4_codes(&weight[row_start..row_start + row_bytes], cols, x)
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
/// Kept as the single-head entry point (the test suite and any external
/// caller use it); [`rope_heads`] drives the multi-head path with one shared
/// angle table.
#[allow(dead_code)]
pub fn rope_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
) {
    rope_span_in_place(values, heads, head_dim, head_dim, position, theta);
}

/// Rotary embedding across heads over the leading `rotary_dim` dims of each
/// head: `head_dim` for the full-rotation families, a partial share for
/// phi-style partial rotary (remaining dims pass through untouched). The
/// per-head loop is part of the operator, so a registration file rotates a
/// whole projection with one call.
///
/// The sin/cos table is computed once per call and reused by every head:
/// a decode step used to run `powf` + `sin_cos` per head per dim (thousands
/// of transcendental calls per token), while the angles depend only on the
/// position and the frequency index.
pub fn rope_heads(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) {
    rope_span_in_place(values, heads, head_dim, rotary_dim, position, theta);
}

/// Shared rotate-half kernel over `heads` heads of `head_dim`, rotating the
/// leading `rotary_dim` lanes. One angle table per call, one rotation per
/// head — the math is unchanged from the per-head kernel.
fn rope_span_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: usize,
    theta: f32,
) {
    let half = rotary_dim / 2;
    let mut table = Vec::with_capacity(half);
    for index in 0..half {
        let freq = theta.powf(-2.0 * index as f32 / rotary_dim as f32);
        let angle = position as f32 * freq;
        table.push(angle.sin_cos());
    }
    for head in 0..heads {
        let base = head * head_dim;
        for (index, (sin, cos)) in table.iter().copied().enumerate() {
            let first = base + index;
            let second = base + half + index;
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

/// SiLU in place (`x · sigmoid(x)`), used by the Qwen3.5 gated linear
/// attention's convolution and gated norm paths.
pub fn silu(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value /= 1.0 + (-*value).exp();
    }
}

/// Numerically stable `softplus(x) = ln(1 + eˣ)`: the linear branch keeps
/// large positive inputs from overflowing the exponential.
pub fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// L2-normalize each `head_dim` slice in place:
/// `x / sqrt(sum(x²) + eps)`, matching the FLA gated-delta-rule kernel.
pub fn l2norm_heads(values: &mut [f32], heads: usize, head_dim: usize) {
    const EPS: f32 = 1e-6;
    for head in 0..heads {
        let start = head * head_dim;
        let norm: f32 = values[start..start + head_dim]
            .iter()
            .map(|value| value * value)
            .sum();
        let inv = (norm + EPS).sqrt().recip();
        for value in &mut values[start..start + head_dim] {
            *value *= inv;
        }
    }
}

/// Gated RMSNorm used by Qwen3.5's linear-attention output:
/// per-head `x / rms(x) · weight · silu(gate)`. `weight` is one head wide
/// and shared by every head; `gate` is `heads * head_dim` long.
///
/// Intentionally unused until the Qwen3.5 family registers: its siblings
/// (`causal_conv1d_step`, `gated_delta_step`) are already wired through the
/// operator catalog; this kernel joins when that family lands.
#[allow(dead_code)]
pub fn rmsnorm_gated(
    values: &mut [f32],
    gate: &[f32],
    weight: &[f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for head in 0..heads {
        let start = head * head_dim;
        let slice = &mut values[start..start + head_dim];
        let mean_square: f32 =
            slice.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
        let inv = (mean_square + eps).sqrt().recip();
        for (index, value) in slice.iter_mut().enumerate() {
            let g = gate[start + index];
            *value *= inv * weight[index] * (g / (1.0 + (-g).exp()));
        }
    }
}

/// One causal depthwise-conv step for Qwen3.5's linear attention.
///
/// `state` is `channels × (kernel - 1)` in chronological order (oldest
/// first) and is shifted in place; `weight` is `channels × kernel` in the
/// safetensors `[channels, 1, kernel]` order. The produced row is
/// `sum_k weight[channel, k] · x[t - (kernel - 1) + k]`, i.e. the exact
/// convolution `causal_conv1d_update` computes before the caller's
/// activation.
pub fn causal_conv1d_step(
    input: &[f32],
    weight: &[f32],
    kernel: usize,
    state: &mut [f32],
    output: &mut [f32],
) {
    let history = kernel - 1;
    let channels = input.len();
    for channel in 0..channels {
        let weight_row = &weight[channel * kernel..(channel + 1) * kernel];
        let state_row = &mut state[channel * history..(channel + 1) * history];
        let mut value = weight_row[kernel - 1] * input[channel];
        for index in 0..history {
            value += weight_row[index] * state_row[index];
        }
        output[channel] = value;
        if history > 0 {
            state_row.rotate_left(1);
            state_row[history - 1] = input[channel];
        }
    }
}

/// One token of Qwen3.5's gated delta rule (linear attention).
///
/// `query`/`key` are `num_k_heads × head_k_dim` (already L2-normalized, with
/// the query scaled by `head_k_dim^-0.5`), `value` is
/// `num_v_heads × head_v_dim`, and `state` is
/// `num_v_heads × head_k_dim × head_v_dim` in `[head, key, value]` order.
/// When the value-head count exceeds the key-head count each key head serves
/// `num_v_heads / num_k_heads` consecutive value heads, matching HF's
/// `repeat_interleave` grouping.
pub fn gated_delta_step(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
) {
    let group = num_v_heads / num_k_heads.max(1);
    for v_head in 0..num_v_heads {
        let k_head = v_head / group.max(1);
        let k_base = k_head.min(num_k_heads.saturating_sub(1)) * head_k_dim;
        let v_base = v_head * head_v_dim;
        let state_base = v_head * head_k_dim * head_v_dim;
        let decay = gate[v_head].exp();
        let beta = beta[v_head];
        // Decay the recurrent state, then project the current key through it.
        let state_slice = &mut state[state_base..state_base + head_k_dim * head_v_dim];
        let mut memory = vec![0f32; head_v_dim];
        for key_index in 0..head_k_dim {
            let k = key[k_base + key_index];
            let row = &mut state_slice[key_index * head_v_dim..(key_index + 1) * head_v_dim];
            for (index, entry) in row.iter_mut().enumerate() {
                *entry *= decay;
                memory[index] += *entry * k;
            }
        }
        // Delta rule update: write only the part the old state missed.
        let mut delta = vec![0f32; head_v_dim];
        for index in 0..head_v_dim {
            delta[index] = (value[v_base + index] - memory[index]) * beta;
        }
        for key_index in 0..head_k_dim {
            let k = key[k_base + key_index];
            let row = &mut state_slice[key_index * head_v_dim..(key_index + 1) * head_v_dim];
            for index in 0..head_v_dim {
                row[index] += k * delta[index];
            }
        }
        // Read the updated state with the query.
        let q_base = k_base;
        for index in 0..head_v_dim {
            let mut sum = 0f32;
            for key_index in 0..head_k_dim {
                sum += state_slice[key_index * head_v_dim + index] * query[q_base + key_index];
            }
            output[v_base + index] = sum;
        }
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
// Reductions over one axis of a row-major `[rows, width]` buffer.
// `axis == 0` reduces each row over `width` (one output per row — the
// logits / per-row shape); `axis == 1` reduces each column over `rows` (one
// output per column — the mean-pooling shape). `out.len()` must match the
// reduced shape.
// ---------------------------------------------------------------------------

/// Sum over one axis; NaN inputs propagate.
pub fn reduce_sum(x: &[f32], rows: usize, width: usize, axis: u32, out: &mut [f32]) {
    match axis {
        0 => {
            assert_eq!(out.len(), rows, "reduce_sum axis 0 needs one output per row");
            for (row, value) in out.iter_mut().enumerate() {
                *value = x[row * width..(row + 1) * width].iter().sum();
            }
        }
        1 => {
            assert_eq!(out.len(), width, "reduce_sum axis 1 needs one output per column");
            for value in out.iter_mut() {
                *value = 0.0;
            }
            for row in 0..rows {
                for (column, value) in out.iter_mut().enumerate() {
                    *value += x[row * width + column];
                }
            }
        }
        other => panic!("reduce_sum: unsupported axis {other} (0 = over width, 1 = over rows)"),
    }
}

/// Mean over one axis (`reduce_sum` scaled by the reduced extent; an empty
/// extent yields NaN, matching torch.mean over a zero-length dim).
pub fn reduce_mean(x: &[f32], rows: usize, width: usize, axis: u32, out: &mut [f32]) {
    let extent = match axis {
        0 => width,
        1 => rows,
        other => panic!("reduce_mean: unsupported axis {other} (0 = over width, 1 = over rows)"),
    };
    reduce_sum(x, rows, width, axis, out);
    for value in out.iter_mut() {
        *value /= extent as f32;
    }
}

/// Maximum over one axis; NaN inputs propagate (NaN dominates, matching
/// torch.max).
pub fn reduce_max(x: &[f32], rows: usize, width: usize, axis: u32, out: &mut [f32]) {
    match axis {
        0 => {
            assert_eq!(out.len(), rows, "reduce_max axis 0 needs one output per row");
            for (row, value) in out.iter_mut().enumerate() {
                let mut best = f32::NEG_INFINITY;
                let mut saw_nan = false;
                for &entry in &x[row * width..(row + 1) * width] {
                    if entry.is_nan() {
                        saw_nan = true;
                        break;
                    }
                    if entry > best {
                        best = entry;
                    }
                }
                *value = if saw_nan { f32::NAN } else { best };
            }
        }
        1 => {
            assert_eq!(out.len(), width, "reduce_max axis 1 needs one output per column");
            for value in out.iter_mut() {
                *value = f32::NEG_INFINITY;
            }
            for row in 0..rows {
                for (column, value) in out.iter_mut().enumerate() {
                    let entry = x[row * width + column];
                    if entry.is_nan() {
                        *value = f32::NAN;
                    } else if !value.is_nan() && entry > *value {
                        *value = entry;
                    }
                }
            }
        }
        other => panic!("reduce_max: unsupported axis {other} (0 = over width, 1 = over rows)"),
    }
}

/// Argmax over one axis, written as the f32 index value. Replicates the
/// host-side greedy scan (`sample_token`'s `max_by(f32::total_cmp)`)
/// element-for-element: `total_cmp` ordering, equal maxima keep the later
/// index, and NaN ranks above every number.
pub fn reduce_argmax(x: &[f32], rows: usize, width: usize, axis: u32, out: &mut [f32]) {
    let (count, inner, stride, base_step) = match axis {
        0 => (rows, width, 1usize, width),
        1 => (width, rows, width, 1usize),
        other => panic!("reduce_argmax: unsupported axis {other} (0 = over width, 1 = over rows)"),
    };
    assert_eq!(
        out.len(),
        count,
        "reduce_argmax needs one output per reduced row/column"
    );
    for index in 0..count {
        let base = index * base_step;
        let mut best = f32::NEG_INFINITY;
        let mut best_index = 0usize;
        for step in 0..inner {
            let value = x[base + step * stride];
            if value.total_cmp(&best) != std::cmp::Ordering::Less {
                best = value;
                best_index = step;
            }
        }
        out[index] = best_index as f32;
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

/// Chunked-prefill scratch: the same intermediates as [`Scratch`], stacked
/// `[chunk_rows, width]` row-major so the batched projections run as GEMMs.
/// The final position's residual row lands back in the per-token
/// [`Scratch::hidden`] for the LM head.
///
/// Cloning a runtime keeps the buffers (like [`Scratch`], they are pure
/// scratch — contents never leak into results).
#[derive(Clone)]
pub(crate) struct ChunkScratch {
    /// Residual stream rows, `[rows, hidden_size]`.
    pub(crate) hidden: Vec<f32>,
    /// Most recent norm output per row, `[rows, hidden_size]`.
    pub(crate) normed: Vec<f32>,
    /// Query rows after projection, optional norms, and rope, `[rows, q_dim]`.
    pub(crate) q: Vec<f32>,
    /// Key rows after projection, optional norms, and rope, `[rows, kv_dim]`.
    pub(crate) k: Vec<f32>,
    /// Value rows after projection, `[rows, kv_dim]`.
    pub(crate) v: Vec<f32>,
    /// Attention output rows before the output projection, `[rows, q_dim]`.
    pub(crate) attn_out: Vec<f32>,
    /// Attention output rows after `o_proj`/`dense`, `[rows, hidden_size]`.
    pub(crate) attn_projected: Vec<f32>,
    /// Gate projection rows / activated intermediate, `[rows, intermediate]`.
    pub(crate) gate: Vec<f32>,
    /// Up projection rows, `[rows, intermediate]`.
    pub(crate) up: Vec<f32>,
    /// Feed-forward output rows, `[rows, hidden_size]`.
    pub(crate) mlp_out: Vec<f32>,
    /// Rows the buffers were last sized for.
    capacity_rows: usize,
}

impl ChunkScratch {
    /// Empty chunk scratch; buffers grow on the first
    /// [`ChunkScratch::resize_for`].
    pub(crate) fn new() -> Self {
        Self {
            hidden: Vec::new(),
            normed: Vec::new(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            attn_out: Vec::new(),
            attn_projected: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            mlp_out: Vec::new(),
            capacity_rows: 0,
        }
    }

    /// Sizes every buffer for `rows` chunk rows (a no-op when the capacity
    /// already covers them; grows when `CHEETAH_PREFILL_CHUNK` yields a
    /// larger final chunk than previous calls saw).
    pub(crate) fn resize_for(&mut self, cfg: &DecoderConfig, rows: usize) {
        if rows <= self.capacity_rows {
            return;
        }
        let kv_dim = cfg.kv_heads * cfg.head_dim;
        self.hidden = vec![0.0; rows * cfg.hidden_size];
        self.normed = vec![0.0; rows * cfg.hidden_size];
        self.q = vec![0.0; rows * cfg.q_heads * cfg.head_dim];
        self.k = vec![0.0; rows * kv_dim];
        self.v = vec![0.0; rows * kv_dim];
        self.attn_out = vec![0.0; rows * cfg.q_heads * cfg.head_dim];
        self.attn_projected = vec![0.0; rows * cfg.hidden_size];
        self.gate = vec![0.0; rows * cfg.intermediate_size];
        self.up = vec![0.0; rows * cfg.intermediate_size];
        self.mlp_out = vec![0.0; rows * cfg.hidden_size];
        self.capacity_rows = rows;
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
// KV cache storage (f32 or opt-in Q8).
// ---------------------------------------------------------------------------

/// Whether the KV cache stores row-wise int8 codes instead of f32 values.
/// Selected once per process through `CHEETAH_KV_Q8=1` (or the
/// `CHEETAH_KV_QUANT_Q8` alias); unset/`0` keeps full f32 precision.
fn kv_q8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        ["CHEETAH_KV_Q8", "CHEETAH_KV_QUANT_Q8"]
            .iter()
            .any(|variable| {
                std::env::var(variable)
                    .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
            })
    })
}

/// Per-layer KV storage. The layout is identical to the old f32 caches:
/// `[position * kv_heads * head_dim + head * head_dim + dim]`; Q8 adds one
/// f32 scale per `(position, head)`.
enum KvStorage {
    /// Full-precision values (default).
    F32(Vec<f32>),
    /// Signed int8 codes plus per-(position, head) scales.
    Q8 { codes: Vec<i8>, scales: Vec<f32> },
}

/// One layer's key or value cache.
pub(crate) struct KvCache {
    storage: KvStorage,
    /// Cached positions (rows).
    len: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
}

impl KvCache {
    /// Empty cache in the requested storage precision.
    fn new(kv_heads: usize, head_dim: usize, q8: bool) -> Self {
        let storage = if q8 {
            KvStorage::Q8 {
                codes: Vec::new(),
                scales: Vec::new(),
            }
        } else {
            KvStorage::F32(Vec::new())
        };
        Self {
            storage,
            len: 0,
            kv_heads,
            head_dim,
            kv_dim: kv_heads * head_dim,
        }
    }

    fn is_q8(&self) -> bool {
        matches!(self.storage, KvStorage::Q8 { .. })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn clear(&mut self) {
        match &mut self.storage {
            KvStorage::F32(values) => values.clear(),
            KvStorage::Q8 { codes, scales } => {
                codes.clear();
                scales.clear();
            }
        }
        self.len = 0;
    }

    /// Reserve capacity for `positions` total cached rows (never shrinks).
    fn reserve(&mut self, positions: usize) {
        match &mut self.storage {
            KvStorage::F32(values) => {
                values.reserve(
                    positions
                        .saturating_mul(self.kv_dim)
                        .saturating_sub(values.len()),
                );
            }
            KvStorage::Q8 { codes, scales } => {
                codes.reserve(
                    positions
                        .saturating_mul(self.kv_dim)
                        .saturating_sub(codes.len()),
                );
                scales.reserve(
                    positions
                        .saturating_mul(self.kv_heads)
                        .saturating_sub(scales.len()),
                );
            }
        }
    }

    /// Append one position's row, quantizing per head when Q8 is selected.
    fn append(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.kv_dim);
        match &mut self.storage {
            KvStorage::F32(values) => values.extend_from_slice(row),
            KvStorage::Q8 { codes, scales } => {
                quantize_kv_row(codes, scales, row, self.kv_heads, self.head_dim)
            }
        }
        self.len += 1;
    }

    /// Restore `positions` f32 rows (converting to Q8 when this cache is Q8).
    fn restore_f32(&mut self, values: &[f32], positions: usize) {
        let kv_heads = self.kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = self.kv_dim;
        let keep = positions.min(values.len() / kv_dim.max(1));
        match &mut self.storage {
            KvStorage::F32(out) => {
                out.clear();
                out.extend_from_slice(&values[..keep * kv_dim]);
            }
            KvStorage::Q8 { codes, scales } => {
                codes.clear();
                scales.clear();
                for position in 0..keep {
                    let row = &values[position * kv_dim..(position + 1) * kv_dim];
                    quantize_kv_row(codes, scales, row, kv_heads, head_dim);
                }
            }
        }
        self.len = keep;
    }

    /// Restore `positions` rows from a quantized snapshot layer, dequantizing
    /// when this cache runs in f32 mode.
    fn restore_q8(&mut self, layer: &QuantizedKvLayer, positions: usize) {
        let kv_heads = self.kv_heads;
        let head_dim = self.head_dim;
        let kv_dim = self.kv_dim;
        let keep = positions
            .min(layer.codes.len() / kv_dim.max(1))
            .min(layer.scales.len() / kv_heads.max(1));
        match &mut self.storage {
            KvStorage::Q8 { codes, scales } => {
                codes.clear();
                codes.extend_from_slice(&layer.codes[..keep * kv_dim]);
                scales.clear();
                scales.extend_from_slice(&layer.scales[..keep * kv_heads]);
            }
            KvStorage::F32(out) => {
                out.clear();
                out.reserve(keep * kv_dim);
                for position in 0..keep {
                    for head in 0..kv_heads {
                        let scale = layer.scales[position * kv_heads + head];
                        let base = position * kv_dim + head * head_dim;
                        out.extend(
                            layer.codes[base..base + head_dim]
                                .iter()
                                .map(|code| *code as f32 * scale),
                        );
                    }
                }
            }
        }
        self.len = keep;
    }

    /// Full-precision rows when this cache stores f32.
    fn f32_values(&self) -> Option<&[f32]> {
        match &self.storage {
            KvStorage::F32(values) => Some(values),
            KvStorage::Q8 { .. } => None,
        }
    }

    /// Quantized rows when this cache stores Q8.
    fn q8_layer(&self) -> Option<QuantizedKvLayer> {
        match &self.storage {
            KvStorage::Q8 { codes, scales } => Some(QuantizedKvLayer {
                codes: codes.clone(),
                scales: scales.clone(),
            }),
            KvStorage::F32(_) => None,
        }
    }

    /// One dequantized value (tests/diagnostics).
    #[cfg(test)]
    fn value_at(&self, position: usize, dim: usize) -> f32 {
        match &self.storage {
            KvStorage::F32(values) => values[position * self.kv_dim + dim],
            KvStorage::Q8 { codes, scales } => {
                let head = dim / self.head_dim;
                codes[position * self.kv_dim + dim] as f32 * scales[position * self.kv_heads + head]
            }
        }
    }
}

/// Quantize one KV row head-by-head: symmetric int8 with `scale =
/// max|value| / 127` (zero rows use scale 1.0, codes all zero).
fn quantize_kv_row(
    codes: &mut Vec<i8>,
    scales: &mut Vec<f32>,
    row: &[f32],
    kv_heads: usize,
    head_dim: usize,
) {
    for head in 0..kv_heads {
        let values = &row[head * head_dim..(head + 1) * head_dim];
        let max_abs = values.iter().fold(0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        codes.extend(
            values
                .iter()
                .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8),
        );
    }
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
    /// Per-layer key cache. f32 by default; `CHEETAH_KV_Q8=1` stores int8
    /// codes plus one scale per `(position, head)`, cutting KV memory ~4x.
    kv_k: Vec<KvCache>,
    /// Per-layer value cache with the same storage mode as `kv_k`.
    kv_v: Vec<KvCache>,
    /// Softmax scores for one query head, length = number of visible cached
    /// keys. Reused across heads/positions so decode attention does not
    /// allocate a score vector per head.
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
    /// fixture path; the loader path uses [`OpsToolbox::open`]). The KV cache
    /// precision follows the `CHEETAH_KV_Q8` selector.
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        Self::with_store_kv_q8(cfg, store, kv_q8_enabled())
    }

    /// [`OpsToolbox::with_store`] with an explicit KV precision — the test
    /// hook that lets Q8 and f32 caches be compared in one process.
    pub(crate) fn with_store_kv_q8(
        cfg: DecoderConfig,
        store: Arc<MmapWeightStore>,
        kv_q8: bool,
    ) -> Self {
        let kv_heads = cfg.kv_heads;
        let head_dim = cfg.head_dim;
        let num_layers = cfg.num_layers;
        Self {
            kv_k: (0..num_layers)
                .map(|_| KvCache::new(kv_heads, head_dim, kv_q8))
                .collect(),
            kv_v: (0..num_layers)
                .map(|_| KvCache::new(kv_heads, head_dim, kv_q8))
                .collect(),
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
        self.embedding_row_into_unit("model.embed_tokens.weight", token, out)
    }

    /// [`Self::embedding_row_into`] for a family whose embedding tensor is not
    /// named `model.embed_tokens.weight` (Qwen3.5 wraps its text model as
    /// `model.language_model.embed_tokens.weight`).
    pub(crate) fn embedding_row_into_unit(
        &self,
        unit: &str,
        token: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let (bytes, dtype, shape) = self
            .store
            .slice(unit)
            .map_err(|error| format!("Native backend: {error}"))?;
        if shape.len() != 2 || shape[1] != self.cfg.hidden_size {
            return Err(format!(
                "Native backend: unexpected embedding shape for '{unit}'."
            ));
        }
        let base = token * self.cfg.hidden_size;
        out.clear();
        out.extend((0..self.cfg.hidden_size).map(|index| scalar_at(bytes, base + index, dtype)));
        Ok(())
    }

    /// Dequantize one unit to f32 values (small parameter tensors such as a
    /// gated-delta `A_log`, convolution taps, or a head norm).
    ///
    /// Intentionally unused until the Qwen3.5 family registers: its forward
    /// will read the gated-delta parameters through this accessor, the same
    /// way dense families read norm weights through `rmsnorm`.
    #[allow(dead_code)]
    pub(crate) fn unit_values(&self, unit: &str) -> Result<Vec<f32>, String> {
        let (bytes, dtype, shape) = self
            .store
            .slice(unit)
            .map_err(|error| format!("Native backend: {error}"))?;
        let count = shape.iter().product::<usize>();
        Ok((0..count)
            .map(|index| scalar_at(bytes, index, dtype))
            .collect())
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

    /// Batched linear: `y[n, rows] = x[n, cols] · Wᵀ` against `unit` — the
    /// chunked-prefill counterpart of [`OpsToolbox::linear`]. Stored-precision
    /// bf16 runs the batched [`gemm`]; quantized units (registered buffer or
    /// lazy first-use quantization) keep their packed-domain per-row matvec,
    /// run once per input row, so their math is unchanged.
    pub(crate) fn linear_batch(
        &mut self,
        unit: &str,
        rows: usize,
        cols: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        let n = x.len() / cols;
        // Fast path: this unit has already been quantized at least once.
        if let Some(buffer) = self.quant.buffers.get(unit) {
            let gemv = self
                .quant
                .gemv
                .expect("quantized unit present without a registered gemv operator");
            let bytes = buffer.buffer();
            for index in 0..n {
                (gemv)(
                    bytes,
                    buffer.rows(),
                    buffer.cols(),
                    &x[index * cols..(index + 1) * cols],
                    &mut y[index * rows..(index + 1) * rows],
                );
            }
            return Ok(());
        }
        if let (Some(quantize), Some(gemv)) = (self.quant.quantize, self.quant.gemv) {
            // Lazy load-time quantization — the same first-use conversion
            // `linear` performs, then one matvec per input row.
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
            self.store.drop_pages(unit);
            let buffer = self
                .quant
                .buffers
                .get(unit)
                .expect("quantized buffer was just inserted");
            let bytes = buffer.buffer();
            for index in 0..n {
                (gemv)(
                    bytes,
                    buffer.rows(),
                    buffer.cols(),
                    &x[index * cols..(index + 1) * cols],
                    &mut y[index * rows..(index + 1) * rows],
                );
            }
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
        if matches!(dtype, WeightDType::Bf16) {
            gemm(bytes, dtype, rows, cols, x, n, y);
        } else {
            for index in 0..n {
                (ops.gemv)(
                    bytes,
                    rows,
                    cols,
                    &x[index * cols..(index + 1) * cols],
                    &mut y[index * rows..(index + 1) * rows],
                );
            }
        }
        Ok(())
    }

    /// Append one position's K/V rows to a layer's cache.
    pub(crate) fn cache_append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.kv_k[layer].append(k);
        self.kv_v[layer].append(v);
    }

    /// Pre-allocates every layer's KV cache for `positions` upcoming rows, so
    /// a long prefill grows into reserved capacity instead of repeatedly
    /// doubling (each doubling memcpys the whole cache). Never shrinks or
    /// moves existing contents; a no-op when the capacity is already there.
    pub(crate) fn reserve_kv(&mut self, positions: usize) {
        for (keys, values) in self.kv_k.iter_mut().zip(self.kv_v.iter_mut()) {
            keys.reserve(positions);
            values.reserve(positions);
        }
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
    ///
    /// The cache may be f32 or Q8 (`CHEETAH_KV_Q8`); both paths produce the
    /// same shape of result, with Q8 codes expanded on the fly so no
    /// dequantized copy of the cache exists. The decode path reuses one score
    /// buffer across heads instead of allocating per head.
    pub(crate) fn attention(&mut self, layer: usize, position: usize, q: &[f32], out: &mut [f32]) {
        let kv_dim = self.cfg.kv_heads * self.cfg.head_dim;
        let cache_len = self.kv_k[layer].len();
        let head_group = self.cfg.q_heads / self.cfg.kv_heads;
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
        // Sparse (sliding-window) attention: the query at `position` only
        // sees keys from `end - window` onward (`end` is one past this
        // position), matching the Vulkan attention shader. The window
        // resolves per layer (hybrid pattern / explicit layer_types).
        // window == 0 attends to the whole cache.
        let window = self.cfg.window_for_layer(layer);
        let window_start = if window == 0 {
            0
        } else {
            (position + 1).saturating_sub(window)
        };
        // Never start past the cache: an all-masked softmax would be NaN.
        let window_start = window_start.min(cache_len.saturating_sub(1));
        let head_dim = self.cfg.head_dim;
        let kv_heads = self.cfg.kv_heads;
        let q_heads = self.cfg.q_heads;
        let keys = &self.kv_k[layer];
        let values = &self.kv_v[layer];
        // Heads are independent: scores → softmax → value mix is a pure
        // function of the cache + q. At long caches this loop dominates
        // prefill, so heads fan out to rayon (short caches and single-head
        // configs run sequentially — the fork/join would cost more than the
        // math). The math per head is unchanged, so results stay bit-exact.
        // The debug capture hook forces the sequential path so it can keep
        // head 0's score vector.
        let parallel =
            q_heads > 1 && cache_len >= ATTENTION_PARALLEL_MIN_CACHE && !self.debug_capture_armed;
        if parallel {
            let head_results: Vec<(Vec<f32>, Vec<f32>)> = (0..q_heads)
                .into_par_iter()
                .map(|head| {
                    let mut scores = vec![f32::NEG_INFINITY; cache_len];
                    let mut head_out = vec![0f32; head_dim];
                    Self::attention_head(
                        keys,
                        values,
                        q,
                        cache_len,
                        window_start,
                        kv_dim,
                        kv_heads,
                        head_group,
                        head_dim,
                        scale,
                        head,
                        &mut scores,
                        &mut head_out,
                    );
                    (scores, head_out)
                })
                .collect();
            for (head, (_, head_out)) in head_results.iter().enumerate() {
                out[head * head_dim..(head + 1) * head_dim].copy_from_slice(head_out);
            }
        } else {
            let mut scores = std::mem::take(&mut self.scores);
            scores.clear();
            scores.resize(cache_len, f32::NEG_INFINITY);
            let mut head0_scores = None;
            for head in 0..q_heads {
                // Every head rewrites the visible window; reset the mask
                // before the next head so stale softmax values cannot leak.
                scores.fill(f32::NEG_INFINITY);
                let head_out = &mut out[head * head_dim..(head + 1) * head_dim];
                Self::attention_head(
                    keys,
                    values,
                    q,
                    cache_len,
                    window_start,
                    kv_dim,
                    kv_heads,
                    head_group,
                    head_dim,
                    scale,
                    head,
                    &mut scores,
                    head_out,
                );
                if self.debug_capture_armed && head == 0 {
                    head0_scores = Some(scores.clone());
                }
            }
            self.scores = scores;
            if self.debug_capture_armed {
                self.debug_capture_armed = false;
                self.debug_capture_buf = Some(DebugCapture {
                    scores: head0_scores.expect("armed capture always records head 0"),
                    attn_out: out.to_vec(),
                });
            }
        }
    }

    /// One attention head: scaled scores over the visible cache window,
    /// softmax, weighted value mix. Side-effect-free over the cache — the
    /// per-head fan-out inside [`OpsToolbox::attention`] relies on that.
    #[allow(clippy::too_many_arguments)]
    fn attention_head(
        keys: &KvCache,
        values: &KvCache,
        q: &[f32],
        cache_len: usize,
        window_start: usize,
        kv_dim: usize,
        kv_heads: usize,
        head_group: usize,
        head_dim: usize,
        scale: f32,
        head: usize,
        scores: &mut [f32],
        out_head: &mut [f32],
    ) {
        match (&keys.storage, &values.storage) {
            (KvStorage::F32(keys), KvStorage::F32(values)) => Self::attention_head_f32(
                keys,
                values,
                q,
                cache_len,
                window_start,
                kv_dim,
                head_group,
                head_dim,
                scale,
                head,
                scores,
                out_head,
            ),
            (
                KvStorage::Q8 {
                    codes: key_codes,
                    scales: key_scales,
                },
                KvStorage::Q8 {
                    codes: value_codes,
                    scales: value_scales,
                },
            ) => Self::attention_head_q8(
                key_codes,
                key_scales,
                value_codes,
                value_scales,
                q,
                cache_len,
                window_start,
                kv_dim,
                kv_heads,
                head_group,
                head_dim,
                scale,
                head,
                scores,
                out_head,
            ),
            _ => unreachable!("key and value caches always share one storage mode"),
        }
    }

    /// f32 attention head (the default precision).
    #[allow(clippy::too_many_arguments)]
    fn attention_head_f32(
        keys: &[f32],
        values: &[f32],
        q: &[f32],
        cache_len: usize,
        window_start: usize,
        kv_dim: usize,
        head_group: usize,
        head_dim: usize,
        scale: f32,
        head: usize,
        scores: &mut [f32],
        out_head: &mut [f32],
    ) {
        let kv_head = head / head_group;
        let q_base = head * head_dim;
        let kv_base = kv_head * head_dim;
        for pos in window_start..cache_len {
            let k_base = pos * kv_dim + kv_base;
            let score: f32 = keys[k_base..k_base + head_dim]
                .iter()
                .zip(&q[q_base..q_base + head_dim])
                .map(|(k, q)| k * q)
                .sum();
            scores[pos] = score * scale;
        }
        softmax(scores);
        // Weighted value mix: per position, one contiguous `head_dim` slice
        // of the V cache. Slices (not per-element indexing) keep this
        // auto-vectorizable — a bounds-checked `kv_v[base+index]` per element
        // ran this loop ~10x slower at long contexts.
        out_head.fill(0.0);
        for pos in window_start..cache_len {
            let weight = scores[pos];
            if weight == 0.0 {
                continue;
            }
            let v_base = pos * kv_dim + kv_base;
            let v_row = &values[v_base..v_base + head_dim];
            for (out_value, v) in out_head.iter_mut().zip(v_row) {
                *out_value += weight * v;
            }
        }
    }

    /// Q8 attention head: key scores run as int8×f32 dots (per-head scale
    /// applied after the dot), and the value mix expands int8 codes directly
    /// into the f32 accumulator — the cache is never dequantized as a whole.
    #[allow(clippy::too_many_arguments)]
    fn attention_head_q8(
        key_codes: &[i8],
        key_scales: &[f32],
        value_codes: &[i8],
        value_scales: &[f32],
        q: &[f32],
        cache_len: usize,
        window_start: usize,
        kv_dim: usize,
        kv_heads: usize,
        head_group: usize,
        head_dim: usize,
        scale: f32,
        head: usize,
        scores: &mut [f32],
        out_head: &mut [f32],
    ) {
        let kv_head = head / head_group;
        let q_base = head * head_dim;
        let kv_base = kv_head * head_dim;
        for pos in window_start..cache_len {
            let code_base = pos * kv_dim + kv_base;
            let raw = dot_i8_f32(
                &key_codes[code_base..code_base + head_dim],
                &q[q_base..q_base + head_dim],
            );
            let key_scale = key_scales[pos * kv_heads + kv_head];
            scores[pos] = raw * key_scale * scale;
        }
        softmax(scores);
        out_head.fill(0.0);
        for pos in window_start..cache_len {
            let weight = scores[pos] * value_scales[pos * kv_heads + kv_head];
            if weight == 0.0 {
                continue;
            }
            let code_base = pos * kv_dim + kv_base;
            accumulate_i8_f32(
                &value_codes[code_base..code_base + head_dim],
                weight,
                out_head,
            );
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

    /// Cached positions in one layer's KV cache (test/diagnostics view).
    #[cfg(test)]
    pub(crate) fn kv_len(&self, layer: usize) -> usize {
        self.kv_k[layer].len()
    }

    /// Captures the current KV state (all layers, all stored positions) for
    /// cross-generation prefix reuse. A Q8 cache snapshots its codes and
    /// scales directly, so the prefix cache does not inflate back to f32.
    pub(crate) fn kv_snapshot(&self) -> KvSnapshot {
        let positions = self.kv_k.first().map(KvCache::len).unwrap_or(0);
        if self.kv_k.first().is_some_and(KvCache::is_q8) {
            let q8 = QuantizedKvSnapshot {
                kv_k: self
                    .kv_k
                    .iter()
                    .map(|cache| {
                        cache
                            .q8_layer()
                            .expect("a Q8 toolbox always snapshots Q8 layers")
                    })
                    .collect(),
                kv_v: self
                    .kv_v
                    .iter()
                    .map(|cache| {
                        cache
                            .q8_layer()
                            .expect("a Q8 toolbox always snapshots Q8 layers")
                    })
                    .collect(),
            };
            KvSnapshot {
                positions,
                kv_k: Vec::new(),
                kv_v: Vec::new(),
                q8: Some(q8),
            }
        } else {
            KvSnapshot {
                positions,
                kv_k: self
                    .kv_k
                    .iter()
                    .map(|cache| {
                        cache
                            .f32_values()
                            .expect("an f32 toolbox always snapshots f32 layers")
                            .to_vec()
                    })
                    .collect(),
                kv_v: self
                    .kv_v
                    .iter()
                    .map(|cache| {
                        cache
                            .f32_values()
                            .expect("an f32 toolbox always snapshots f32 layers")
                            .to_vec()
                    })
                    .collect(),
                q8: None,
            }
        }
    }

    /// Restores the first `positions` positions of `snapshot` into this
    /// toolbox's (freshly cleared) KV caches, truncating anything beyond.
    /// The result must be bit-equivalent to having prefilled those tokens
    /// here — the decoder KV-continuation tests pin that property. Snapshots
    /// and caches may use different precisions; the restore converts when
    /// they do (Q8 snapshots into an f32 cache, or vice versa).
    pub(crate) fn restore_kv_prefix(&mut self, snapshot: &KvSnapshot, positions: usize) {
        let keep = positions.min(snapshot.positions);
        for cache in self.kv_k.iter_mut().chain(self.kv_v.iter_mut()) {
            cache.clear();
        }
        if let Some(q8) = &snapshot.q8 {
            for (cache, layer) in self.kv_k.iter_mut().zip(q8.kv_k.iter()) {
                cache.restore_q8(layer, keep);
            }
            for (cache, layer) in self.kv_v.iter_mut().zip(q8.kv_v.iter()) {
                cache.restore_q8(layer, keep);
            }
        } else {
            for (cache, layer) in self.kv_k.iter_mut().zip(snapshot.kv_k.iter()) {
                cache.restore_f32(layer, keep);
            }
            for (cache, layer) in self.kv_v.iter_mut().zip(snapshot.kv_v.iter()) {
                cache.restore_f32(layer, keep);
            }
        }
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

    /// gemm vs per-row gemv: every output element of the batched kernel must
    /// match a per-token `gemv` call bit for bit — the chunked prefill path
    /// depends on this equivalence (same dot, different loop order only).
    #[test]
    fn gemm_matches_per_row_gemv_bit_exact() {
        // Deterministic pseudo-random weights/inputs; bf16-rounded.
        let rows = GEMV_CHUNK_ROWS * 2 + 3; // exercises the parallel chunk path
        let cols = 8usize;
        let n = 7usize;
        let weight: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 37 % 23) as f32 - 11.0) / 8.0)
            .collect();
        let mut weight_bf16 = Vec::with_capacity(rows * cols * 2);
        for value in &weight {
            let bits = (value.to_bits() >> 16) as u16;
            weight_bf16.extend_from_slice(&bits.to_le_bytes());
        }
        let x: Vec<f32> = (0..n * cols)
            .map(|index| ((index * 11 % 17) as f32 - 8.0) / 4.0)
            .collect();

        let mut y_gemm = vec![0f32; n * rows];
        gemm(
            &weight_bf16,
            WeightDType::Bf16,
            rows,
            cols,
            &x,
            n,
            &mut y_gemm,
        );
        for token in 0..n {
            let mut y_gemv = vec![0f32; rows];
            gemv(
                &weight_bf16,
                WeightDType::Bf16,
                rows,
                cols,
                &x[token * cols..(token + 1) * cols],
                &mut y_gemv,
            );
            assert_eq!(
                &y_gemm[token * rows..(token + 1) * rows],
                &y_gemv[..],
                "token {token}: gemm must equal per-token gemv bit for bit"
            );
        }

        // Non-bf16 dtypes fall back to per-row gemv — must match trivially.
        let weight_f32: Vec<u8> = weight
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut y_f32 = vec![0f32; n * rows];
        gemm(&weight_f32, WeightDType::F32, rows, cols, &x, n, &mut y_f32);
        for token in 0..n {
            let mut y_gemv = vec![0f32; rows];
            gemv(
                &weight_f32,
                WeightDType::F32,
                rows,
                cols,
                &x[token * cols..(token + 1) * cols],
                &mut y_gemv,
            );
            assert_eq!(&y_f32[token * rows..(token + 1) * rows], &y_gemv[..]);
        }
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

    /// Q8 KV cache round trip: per-head int8 quantization stays within half a
    /// code step, and snapshot layers restore bit-for-bit into another Q8
    /// cache (or convert losslessly into f32).
    #[test]
    fn kv_cache_q8_round_trip_and_conversion() {
        let kv_heads = 2usize;
        let head_dim = 4usize;
        let kv_dim = kv_heads * head_dim;
        let rows: Vec<Vec<f32>> = (0..5)
            .map(|position| {
                (0..kv_dim)
                    .map(|dim| ((position * 7 + dim * 3) % 19) as f32 * 0.1 - 0.9)
                    .collect()
            })
            .collect();

        let mut f32_cache = KvCache::new(kv_heads, head_dim, false);
        let mut q8_cache = KvCache::new(kv_heads, head_dim, true);
        for row in &rows {
            f32_cache.append(row);
            q8_cache.append(row);
        }
        assert_eq!(q8_cache.len(), rows.len());
        assert!(q8_cache.is_q8());
        for position in 0..rows.len() {
            let max_abs = rows[position]
                .iter()
                .fold(0f32, |acc, value| acc.max(value.abs()));
            let half_code = (max_abs / 127.0) / 2.0;
            for dim in 0..kv_dim {
                let restored = q8_cache.value_at(position, dim);
                let original = f32_cache.value_at(position, dim);
                assert!(
                    (restored - original).abs() <= half_code + 1e-6,
                    "position {position} dim {dim}: {restored} vs {original}, half-code {half_code}"
                );
            }
        }

        let layer = q8_cache.q8_layer().expect("Q8 cache must snapshot Q8");
        let mut restored_q8 = KvCache::new(kv_heads, head_dim, true);
        restored_q8.restore_q8(&layer, 3);
        assert_eq!(restored_q8.len(), 3);
        for position in 0..3 {
            for dim in 0..kv_dim {
                assert_eq!(
                    restored_q8.value_at(position, dim),
                    q8_cache.value_at(position, dim)
                );
            }
        }

        let mut restored_f32 = KvCache::new(kv_heads, head_dim, false);
        restored_f32.restore_q8(&layer, rows.len());
        for position in 0..rows.len() {
            for dim in 0..kv_dim {
                assert!(
                    (restored_f32.value_at(position, dim) - q8_cache.value_at(position, dim)).abs()
                        < 1e-6
                );
            }
        }

        let values = f32_cache.f32_values().unwrap().to_vec();
        let mut q8_from_f32 = KvCache::new(kv_heads, head_dim, true);
        q8_from_f32.restore_f32(&values, rows.len());
        assert_eq!(q8_from_f32.len(), rows.len());
        assert_eq!(
            q8_from_f32.value_at(4, 7),
            q8_cache.value_at(4, 7),
            "restoring the same rows must produce the same codes/scales"
        );

        q8_from_f32.clear();
        assert_eq!(q8_from_f32.len(), 0);
        assert!(q8_from_f32.q8_layer().unwrap().codes.is_empty());
    }

    /// Q8 attention matches the f32 path within quantization error: the same
    /// K/V rows fed to both caches (one f32, one Q8) produce head outputs
    /// that differ only by the int8 rounding of the keys/values. Window
    /// masking and a cache length with an AVX2 tail are exercised.
    #[test]
    fn attention_q8_matches_f32_within_quantization_error() {
        let kv_heads = 2usize;
        let head_dim = 8usize;
        let q_heads = 4usize;
        let head_group = q_heads / kv_heads;
        let kv_dim = kv_heads * head_dim;
        let cache_len = 305usize; // 8-lane remainder 1: exercises the tails
        let window_start = 123usize;

        let keys: Vec<Vec<f32>> = (0..cache_len)
            .map(|position| {
                (0..kv_dim)
                    .map(|dim| ((position * 13 + dim * 7) % 37) as f32 / 18.0 - 1.0)
                    .collect()
            })
            .collect();
        let values: Vec<Vec<f32>> = (0..cache_len)
            .map(|position| {
                (0..kv_dim)
                    .map(|dim| ((position * 5 + dim * 11) % 31) as f32 / 15.0 - 1.0)
                    .collect()
            })
            .collect();

        let mut f32_keys = KvCache::new(kv_heads, head_dim, false);
        let mut f32_values = KvCache::new(kv_heads, head_dim, false);
        let mut q8_keys = KvCache::new(kv_heads, head_dim, true);
        let mut q8_values = KvCache::new(kv_heads, head_dim, true);
        for position in 0..cache_len {
            f32_keys.append(&keys[position]);
            f32_values.append(&values[position]);
            q8_keys.append(&keys[position]);
            q8_values.append(&values[position]);
        }

        let q: Vec<f32> = (0..q_heads * head_dim)
            .map(|index| ((index * 3) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let f32_keys = f32_keys.f32_values().unwrap();
        let f32_values = f32_values.f32_values().unwrap();
        let KvStorage::Q8 {
            codes: key_codes,
            scales: key_scales,
        } = &q8_keys.storage
        else {
            panic!("Q8 cache must store codes");
        };
        let KvStorage::Q8 {
            codes: value_codes,
            scales: value_scales,
        } = &q8_values.storage
        else {
            panic!("Q8 cache must store codes");
        };

        let mut max_abs_error = 0f32;
        for head in 0..q_heads {
            let mut f32_scores = vec![f32::NEG_INFINITY; cache_len];
            let mut f32_out = vec![0f32; head_dim];
            OpsToolbox::attention_head_f32(
                f32_keys,
                f32_values,
                &q,
                cache_len,
                window_start,
                kv_dim,
                head_group,
                head_dim,
                scale,
                head,
                &mut f32_scores,
                &mut f32_out,
            );

            let mut q8_scores = vec![f32::NEG_INFINITY; cache_len];
            let mut q8_out = vec![0f32; head_dim];
            OpsToolbox::attention_head_q8(
                key_codes,
                key_scales,
                value_codes,
                value_scales,
                &q,
                cache_len,
                window_start,
                kv_dim,
                kv_heads,
                head_group,
                head_dim,
                scale,
                head,
                &mut q8_scores,
                &mut q8_out,
            );

            for dim in 0..head_dim {
                max_abs_error = max_abs_error.max((f32_out[dim] - q8_out[dim]).abs());
            }
        }
        assert!(
            max_abs_error < 3e-2,
            "Q8 attention diverged from f32 by {max_abs_error}"
        );
    }

    /// Full-width Q8/Q4 matvecs exercise the AVX2 widening kernels against the
    /// scalar dequantized reference (the small existing tests only hit the
    /// tail paths).
    #[test]
    fn quantized_gemv_full_width_matches_dequant_reference() {
        let rows = 17usize;
        let cols = 896usize;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index * 37 % 251) as f32 - 125.0) / 64.0)
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|index| ((index * 13 % 29) as f32 - 14.0) / 16.0)
            .collect();

        for (format, gemv) in [
            ("q8", gemv_q8 as fn(&[u8], usize, usize, &[f32], &mut [f32])),
            ("q4", gemv_q4),
        ] {
            let quantize = crate::chat::precision::native_ops_for_format(format)
                .and_then(|ops| ops.quantize)
                .expect("quantized formats register a native quantizer");
            let buffer = quantize(&values, rows, cols).unwrap();
            let mut y = vec![0f32; rows];
            gemv(buffer.buffer(), rows, cols, &x, &mut y);
            for row in 0..rows {
                let want: f32 = (0..cols)
                    .map(|col| buffer.dequant_value(row, col) * x[col])
                    .sum();
                assert!(
                    (y[row] - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "{format} row {row}: {} vs {want}",
                    y[row]
                );
            }
        }
    }
}
