//! Q4 row-wise int4 quantization operators (the registered `q4` format).
//!
//! Buffer layout (shared by the streaming and resident paths, the CPU gemv,
//! and the `matmul_q4.comp` shader): a head of `rows` f32 row scales (raw
//! bits) followed by row-major int4 payloads, two codes per byte (low nibble
//! = even column), each row padded to a multiple of four bytes so the shader
//! unpacks eight lanes per u32 word.
//!
//! ```text
//! offset  0                head: rows × 4 B — f32 scale per OUTPUT row
//! offset  rows*4           payloads: rows rows of `row_bytes` bytes each
//!
//! one payload byte (column pair):  [high nibble]  [low nibble]
//!                                   col 2b+1       col 2b
//!                                   (code >> 4)    (code & 0xF)
//!
//! one u32 word (shader view):      8 columns, nibble n at bit n*4;
//!                                  word index = column / 8 within the row
//! ```
//!
//! Bit-level correspondence (the contract between packer, CPU gemv and
//! shader): column `c` lives in byte `c / 2` at nibble `c % 2`, i.e. bit
//! offset `(c % 8) * 4` inside u32 word `c / 8` of the row. Both decoders
//! sign-extend the 4-bit lane (nibble ≥ 8 → nibble − 16) so the ±7 code
//! range survives as int4.
//!
//! Numerics: per-row symmetric quantization with `scale = max|w| / 7` and
//! codes clamped to ±7 — the same scheme as q8 scaled down to the 4-bit
//! range, so one code step is `scale` and the error is bounded by
//! ±scale/2 (±max|w|/14). A zero row keeps `scale = 1.0`. The quantized
//! footprint is a quarter of BF16 and half of q8.
//!
//! Compute contract: operators work from the quantized representation —
//! codes are unpacked as int4 and multiplied by the row scale; the weights
//! are never bulk-dequantized into another storage width. Symmetric range:
//! scale = max|w| / 7 per row, codes clamped to ±7.

use super::PrecisionBuffer;
use std::sync::Arc;

/// Padded per-row byte stride: `(cols + 1) / 2` payload bytes rounded up to
/// a multiple of four (the shader reads whole u32 words per row).
///
/// Two-stage rounding: `cols` codes need `(cols + 1) / 2` bytes (two codes
/// per byte), and the shader indexes the row as u32 words —
/// `q.values[head + row * (row_bytes / 4) + column / 8]` — so the stride
/// must be a multiple of four. The C entry point computes the same formula
/// from `k`; a unit test pins the Rust/C agreement via the buffer length
/// the FFI validates.
pub(crate) fn row_stride(cols: usize) -> usize {
    (cols + 1).div_ceil(2).div_ceil(4) * 4
}

/// Decode one int4 code from `byte` (`half` selects the low/high nibble).
///
/// Sign extension: the packer stored `code & 0xF`, so codes −8..−1 come back
/// as nibbles 8..15 and map back by subtracting 16. `half == 0` reads the
/// low nibble (even column), `half == 1` the high nibble (odd column).
fn decode_code(byte: u8, half: usize) -> i8 {
    let nibble = if half == 0 { byte & 0x0F } else { byte >> 4 };
    if nibble >= 8 {
        nibble as i8 - 16
    } else {
        nibble as i8
    }
}

pub(crate) struct Q4Matrix {
    rows: usize,
    cols: usize,
    buffer: Arc<[u8]>,
}

impl PrecisionBuffer for Q4Matrix {
    fn format_id(&self) -> &'static str {
        "q4"
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    fn row_bytes(&self) -> usize {
        row_stride(self.cols)
    }

    fn dequant_value(&self, row: usize, col: usize) -> f32 {
        let scale = f32::from_le_bytes([
            self.buffer[row * 4],
            self.buffer[row * 4 + 1],
            self.buffer[row * 4 + 2],
            self.buffer[row * 4 + 3],
        ]);
        let byte = self.buffer[self.rows * 4 + row * self.row_bytes() + col / 2];
        scale * decode_code(byte, col % 2) as f32
    }
}

/// Row-wise symmetric int4 quantization: scale = max|w| / 7 per row.
///
/// Mirrors `q8::quantize` step for step, scaled to the 4-bit range:
/// 1. scan the row for `max|w|`;
/// 2. `scale = max|w| / 7` (or 1.0 for an all-zero row);
/// 3. write the scale's f32 bits into the head at `row * 4`;
/// 4. round to a code: `(w / scale).round()` clamped to ±7.
///
/// Payload writes are read-modify-write (`|=`) because each byte holds two
/// independent column codes — the low nibble for the even column is written
/// first, then the high nibble ORs in without disturbing it. The padded
/// tail bytes stay zero (never decoded: the gemv stops at `cols`, the
/// shader guards `column < n`).
pub(crate) fn quantize(
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Arc<dyn PrecisionBuffer>, String> {
    let expected = rows
        .checked_mul(cols)
        .ok_or("Q4 quantization size overflow")?;
    if values.len() != expected {
        return Err(format!(
            "Q4 quantization expected {expected} elements, found {}.",
            values.len()
        ));
    }
    let row_bytes = row_stride(cols);
    let mut buffer = vec![0u8; rows * 4 + rows * row_bytes];
    for row in 0..rows {
        let start = row * cols;
        let max_abs = values[start..start + cols]
            .iter()
            .fold(0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let scale_base = row * 4;
        buffer[scale_base..scale_base + 4].copy_from_slice(&scale.to_le_bytes());
        for (col, value) in values[start..start + cols].iter().enumerate() {
            let quantized = (value / scale).round().clamp(-7.0, 7.0) as i8;
            let index = rows * 4 + row * row_bytes + col / 2;
            buffer[index] |= ((quantized as u8) & 0x0F) << ((col % 2) * 4);
        }
    }
    Ok(Arc::new(Q4Matrix {
        rows,
        cols,
        buffer: buffer.into(),
    }))
}

/// Upload the quantized buffer into the Ant resident bucket; `None` keeps the
/// per-step streaming path (e.g. an oversized upload). `row_bytes` is the
/// padded per-row stride the resident matmul validates against.
///
/// Same contract as `q8::upload`; the C side records format code q4 = 5 in
/// the resident record and the resident matmul dispatches the q4 pipeline
/// through the format table.
#[cfg(feature = "vulkan")]
pub(crate) fn upload(buffer: &[u8], row_bytes: usize) -> Option<u64> {
    cheetah::native::vulkan::weight_upload_q4(buffer, row_bytes as u32)
}

/// Host-streaming q4 matmul: the whole quantized buffer (scale head + packed
/// rows) stages in one pass — a quarter of the BF16 footprint and half of q8.
///
/// `k` is the reduction width (input features), `n` the output features;
/// the expected buffer length is `n * 4` scale bytes plus `n * row_bytes(k)`
/// payload bytes — validated here and inside the C entry point. The row
/// stride is derived from `k` on both sides (`row_stride`), so a Rust/C
/// drift fails loudly at the length check instead of corrupting reads.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_streaming(
    input: &[f32],
    buffer: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let row_bytes = row_stride(k);
    let expected = n * 4 + n * row_bytes;
    if buffer.len() != expected {
        return Err(format!(
            "Vulkan q4 matmul expected {expected} weight bytes, received {}.",
            buffer.len()
        ));
    }
    cheetah::native::vulkan::matmul_q4_transposed(input, buffer, n, k, m, k, n)
}

/// Ant resident q4 matmul: the quantized weight buffer stays bound on the
/// GPU behind `handle` (from `upload`); only activations and outputs pass
/// through the pool.
///
/// The C side resolves the record's format code (q4 = 5) through
/// `g_quant_formats`, lazily creates the q4 pipeline on first use and
/// dispatches the same shader as the streaming path — the weight buffer is
/// descriptor 1 of the resident record instead of a per-call pool staging.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_resident(
    handle: u64,
    input: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_q4_transposed_resident(handle, input, m, k, n)
}

/// `y = W·x` over the quantized buffer: int4 codes × per-row scale, f32
/// accumulate — the same math the q4 shader performs.
///
/// Thin forward to `kernels::gemv_q4` (one symbol per backend, mirroring
/// q8::gemv) so the decoder's quantized path and the registration table
/// stay decoupled from the kernel module layout.
pub(crate) fn gemv(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    crate::chat::native::kernels::gemv_q4(weight, rows, cols, x, y);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quantized buffer length must match the FFI contract in
    /// `cheetah::native::vulkan::matmul_q4_transposed` (n f32 scales plus n
    /// padded int4 rows) — the two formulas drifting apart would fail every
    /// q4 matmul at the length check.
    #[test]
    fn buffer_layout_matches_ffi_contract() {
        for (rows, cols) in [(3usize, 5usize), (8, 64), (128, 151936 / 128)] {
            let values: Vec<f32> = (0..rows * cols)
                .map(|index| (index % 17) as f32 - 8.0)
                .collect();
            let buffer = quantize(&values, rows, cols).unwrap();
            assert_eq!(buffer.buffer().len(), rows * 4 + rows * row_stride(cols));
            assert_eq!(buffer.row_bytes(), row_stride(cols));
            assert_eq!(buffer.rows(), rows);
            assert_eq!(buffer.cols(), cols);
            assert_eq!(buffer.element_count(), rows * cols);
            assert_eq!(buffer.format_id(), "q4");
        }
    }

    /// Quantize/dequantize round-trip: each value lands within half a code
    /// step (scale/2) of its row, and the max-abs row drives the scale
    /// (max|w| / 7 for the symmetric ±7 int4 range).
    #[test]
    fn quantize_round_trip_within_one_step() {
        let rows = 2usize;
        let cols = 9usize;
        let mut values = vec![0.25f32; rows * cols];
        values[cols] = -40.0; // row 1 max abs
        let buffer = quantize(&values, rows, cols).unwrap();
        let scale = |row: usize| {
            let base = row * 4;
            f32::from_le_bytes([
                buffer.buffer()[base],
                buffer.buffer()[base + 1],
                buffer.buffer()[base + 2],
                buffer.buffer()[base + 3],
            ])
        };
        assert!((scale(0) - 0.25 / 7.0).abs() < 1e-9);
        assert!((scale(1) - 40.0 / 7.0).abs() < 1e-9);
        for (index, value) in values.iter().enumerate() {
            let row = index / cols;
            let col = index % cols;
            let restored = buffer.dequant_value(row, col);
            // One code = scale; rounding lands within half a code.
            let half_code = scale(row) / 2.0;
            assert!(
                (restored - value).abs() <= half_code + 1e-6,
                "value[{index}] {value} restored {restored} half-code {half_code}"
            );
        }
    }

    /// Nibble packing survives the round trip for both parities: every odd
    /// column reads the high nibble, every even column the low nibble.
    #[test]
    fn nibble_packing_matches_column_parity() {
        let rows = 1usize;
        let cols = 8usize;
        let values: Vec<f32> = (0..cols).map(|index| (index as f32 + 1.0) * 0.5).collect();
        let buffer = quantize(&values, rows, cols).unwrap();
        for col in 0..cols {
            let restored = buffer.dequant_value(0, col);
            let expected_code =
                ((values[col] / buffer_dequant_scale(&buffer)).round() as i8).clamp(-7, 7);
            let expected = expected_code as f32 * buffer_dequant_scale(&buffer);
            assert!(
                (restored - expected).abs() < 1e-6,
                "col {col}: restored {restored} expected {expected}"
            );
        }
    }

    /// Scale helper for the parity test (reads the head of a one-row buffer).
    fn buffer_dequant_scale(buffer: &Arc<dyn PrecisionBuffer>) -> f32 {
        f32::from_le_bytes([
            buffer.buffer()[0],
            buffer.buffer()[1],
            buffer.buffer()[2],
            buffer.buffer()[3],
        ])
    }

    /// An element count mismatch is rejected before any buffer is built.
    #[test]
    fn quantize_rejects_size_mismatch() {
        let error = match quantize(&[0.0; 10], 3, 5) {
            Err(error) => error,
            Ok(_) => panic!("a size mismatch must be rejected"),
        };
        assert!(error.contains("expected 15 elements"), "{error}");
    }
}

// ---------------------------------------------------------------------------
// Declarative registration — this format's entry in the generated REGISTRY
// (build.rs scans src/chat/precision/ and splices every file there as a
// `<name>` module; adding a weight precision is adding one such file).
// ---------------------------------------------------------------------------

use super::{PrecisionBackend, PrecisionKind, PrecisionRegistration};

pub(crate) const REGISTRATION: PrecisionRegistration = PrecisionRegistration {
    id: "q4",
    label: "Q4 (row-wise int4)",
    kind: PrecisionKind::Quantized,
    // `CHEETAH_QUANT_Q4=1` quantizes linear projections to row-wise int4 at
    // load time (a quarter of the BF16 footprint, half of q8). Norms, biases
    // and the embedding stay in original precision.
    selector_env: Some("CHEETAH_QUANT_Q4"),
    backends: precision_backends(),
};

/// Backends registered for this format: both the native CPU operators
/// (load-time quantize + int4×scale gemv) and, when compiled in, the Vulkan
/// executor operators (quantize at load, resident upload, streaming and
/// resident matmul).
const fn precision_backends() -> &'static [(&'static str, PrecisionBackend)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            (
                "native",
                PrecisionBackend::Native(super::NativeOps {
                    quantize: Some(quantize),
                    gemv: gemv,
                }),
            ),
            (
                "vulkan",
                PrecisionBackend::Vulkan(super::VulkanOps {
                    quantize: Some(quantize),
                    matmul_streaming: matmul_streaming,
                    upload: Some(upload),
                    matmul_resident: Some(matmul_resident),
                    weight_format: 5,
                    shader_env: "CHEETAH_MATMUL_Q4_SPV",
                }),
            ),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[(
            "native",
            PrecisionBackend::Native(super::NativeOps {
                quantize: Some(quantize),
                gemv: gemv,
            }),
        )]
    }
}
