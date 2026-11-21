//! Q8 row-wise int8 quantization operators (the registered `q8` format).
//!
//! Buffer layout (shared by the streaming and resident paths, the CPU gemv,
//! and the `matmul_q8.comp` shader): a head of `rows` f32 row scales (raw
//! bits) followed by the row-major int8 payloads, each row padded to a
//! multiple of four bytes so packed words decode four lanes at a time.
//!
//! ```text
//! offset  0                head: rows × 4 B — f32 scale per OUTPUT row
//!                          (little-endian f32 bits, one word per row;
//!                          the shader reads word `row` directly)
//! offset  rows*4           payloads: rows rows of `row_bytes` bytes each,
//!                          row_bytes = (cols + 3) / 4 * 4 — one i8 per
//!                          weight, the tail padded with zero bytes so the
//!                          shader's u32 word indexing stays in bounds
//! ```
//!
//! Numerics: per-row symmetric quantization — `scale = max|w| / 127`, codes
//! clamped to ±127, dequant `scale * code`. The error is bounded by half a
//! code (±scale/2 = ±max|w|/254). A zero row keeps `scale = 1.0` (codes all
//! zero, dequant exactly 0) instead of dividing by zero.
//!
//! Compute contract: operators work from the quantized representation —
//! codes are unpacked as int8 and multiplied by the row scale; the weights
//! are never bulk-dequantized into another storage width.

use super::PrecisionBuffer;
use std::sync::Arc;

pub(crate) struct Q8Matrix {
    rows: usize,
    cols: usize,
    buffer: Arc<[u8]>,
}

impl PrecisionBuffer for Q8Matrix {
    fn format_id(&self) -> &'static str {
        "q8"
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
        (self.cols + 3) / 4 * 4
    }

    fn dequant_value(&self, row: usize, col: usize) -> f32 {
        let scale = f32::from_le_bytes([
            self.buffer[row * 4],
            self.buffer[row * 4 + 1],
            self.buffer[row * 4 + 2],
            self.buffer[row * 4 + 3],
        ]);
        let raw = self.buffer[self.rows * 4 + row * self.row_bytes() + col] as i8 as f32;
        scale * raw
    }
}

/// Row-wise symmetric int8 quantization: scale = max|w| / 127 per row.
///
/// The input is the row-major weight matrix in f32 — the executor flattens
/// whatever storage precision the checkpoint carries through
/// `WeightData::value` before calling (the quantize step itself is the only
/// place precision is intentionally lost). Steps per row:
/// 1. scan the row for `max|w|`;
/// 2. `scale = max|w| / 127` (or 1.0 for an all-zero row) — one code step
///    equals `scale`, so the quantized range covers exactly ±max|w|;
/// 3. write the scale's f32 bits into the head at `row * 4`;
/// 4. round every value to a code: `(w / scale).round()` clamped to ±127,
///    stored as the raw i8 bit pattern in the padded payload row.
///
/// The padding bytes stay zero from the `vec!` initialization — the shader
/// reads whole u32 words and the gemv reads only `cols` codes, so the tail
/// is never interpreted.
pub(crate) fn quantize(
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Arc<dyn PrecisionBuffer>, String> {
    let expected = rows
        .checked_mul(cols)
        .ok_or("Q8 quantization size overflow")?;
    if values.len() != expected {
        return Err(format!(
            "Q8 quantization expected {expected} elements, found {}.",
            values.len()
        ));
    }
    let row_bytes = (cols + 3) / 4 * 4;
    let mut buffer = vec![0u8; rows * 4 + rows * row_bytes];
    for row in 0..rows {
        let start = row * cols;
        let max_abs = values[start..start + cols]
            .iter()
            .fold(0f32, |acc, value| acc.max(value.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let scale_base = row * 4;
        buffer[scale_base..scale_base + 4].copy_from_slice(&scale.to_le_bytes());
        for (col, value) in values[start..start + cols].iter().enumerate() {
            let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
            buffer[rows * 4 + row * row_bytes + col] = quantized as u8;
        }
    }
    Ok(Arc::new(Q8Matrix {
        rows,
        cols,
        buffer: buffer.into(),
    }))
}

/// Upload the quantized buffer into the Ant resident bucket; `None` keeps the
/// per-step streaming path (e.g. an oversized upload). `row_bytes` is the
/// padded per-row stride the resident matmul validates against.
///
/// The C side (`cheetah_vulkan_weight_upload_q8`) copies the buffer once
/// into device-local memory (or a persistent host-coherent allocation when
/// the device has no host-visible device-local heap) and records the format
/// code (q8 = 4) plus the stride in the resident record; the resident
/// matmul later binds the record's buffer directly as descriptor 1. A
/// `None` here is silent by contract — the executor logs the fallback and
/// the per-step streaming path takes over.
#[cfg(feature = "vulkan")]
pub(crate) fn upload(buffer: &[u8], row_bytes: usize) -> Option<u64> {
    cheetah::native::vulkan::weight_upload_q8(buffer, row_bytes as u32)
}

/// Host-streaming q8 matmul: the whole quantized buffer (scale head + packed
/// rows) stages in one pass — roughly half the traffic of the BF16 streaming
/// path — so no column tiling is needed.
///
/// `k` is the reduction width (the input feature count) and `n` the output
/// feature count; the buffer must be exactly the q8 layout for `n × k`
/// (checked here AND inside the C entry point — the two length formulas
/// drifting apart is guarded by a unit test). The weight bytes arrive
/// row-padded, so the C kernel computes the stride itself from `k`.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_streaming(
    input: &[f32],
    buffer: &[u8],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let row_bytes = (k + 3) / 4 * 4;
    let expected = n * 4 + n * row_bytes;
    if buffer.len() != expected {
        return Err(format!(
            "Vulkan q8 matmul expected {expected} weight bytes, received {}.",
            buffer.len()
        ));
    }
    cheetah::native::vulkan::matmul_q8_transposed(input, buffer, n, k, m, k, n)
}

/// Ant resident q8 matmul: the quantized weight buffer stays bound on the
/// GPU behind `handle` (from `upload`); only activations and outputs pass
/// through the pool.
///
/// The C side looks the resident record up by handle, binds its buffer as
/// descriptor 1 and dispatches the same q8 shader — identical math to the
/// streaming path, minus the per-call weight staging. `m × k` activations
/// and `m × n` outputs still flow through the pool because the executor
/// tiles `m` (MAX_ROW_TILE) to bound the pool usage.
#[cfg(feature = "vulkan")]
pub(crate) fn matmul_resident(
    handle: u64,
    input: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    cheetah::native::vulkan::matmul_q8_transposed_resident(handle, input, m, k, n)
}

/// `y = W·x` over the quantized buffer: int8 codes × per-row scale, f32
/// accumulate — the same math the q8 shader performs.
///
/// Thin forward to `kernels::gemv_q8`; kept as a named operator so the q8
/// registration wires one symbol per backend and the decoder's quantized
/// path needs no knowledge of the kernel module layout.
pub(crate) fn gemv(weight: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    crate::chat::native::kernels::gemv_q8(weight, rows, cols, x, y);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quantized buffer length must match the FFI contract in
    /// `cheetah::native::vulkan::matmul_q8_transposed` (n f32 scales plus n
    /// padded int8 rows) — the two formulas drifting apart would fail every
    /// q8 matmul at the length check.
    #[test]
    fn buffer_layout_matches_ffi_contract() {
        for (rows, cols) in [(3usize, 5usize), (8, 64), (128, 151936 / 128)] {
            let values: Vec<f32> = (0..rows * cols)
                .map(|index| (index % 17) as f32 - 8.0)
                .collect();
            let buffer = quantize(&values, rows, cols).unwrap();
            let row_bytes = (cols + 3) / 4 * 4;
            assert_eq!(buffer.buffer().len(), rows * 4 + rows * row_bytes);
            assert_eq!(buffer.rows(), rows);
            assert_eq!(buffer.cols(), cols);
            assert_eq!(buffer.element_count(), rows * cols);
            assert_eq!(buffer.row_bytes(), row_bytes);
        }
    }

    /// Quantize/dequantize round-trip: each value lands within half a code
    /// step (scale/2) of its row, and the max-abs row drives the scale.
    #[test]
    fn quantize_round_trip_within_one_step() {
        let rows = 2usize;
        let cols = 7usize;
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
        assert!((scale(0) - 0.25 / 127.0).abs() < 1e-9);
        assert!((scale(1) - 40.0 / 127.0).abs() < 1e-9);
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
    id: "q8",
    label: "Q8 (row-wise int8)",
    kind: PrecisionKind::Quantized,
    // `CHEETAH_QUANT_Q8=1` quantizes linear projections to row-wise int8 at
    // load time (halves the weight footprint for both the streaming and the
    // resident path). Norms, biases and the embedding stay in original
    // precision.
    selector_env: Some("CHEETAH_QUANT_Q8"),
    backends: precision_backends(),
};

/// Backends registered for this format: both the native CPU operators
/// (load-time quantize + int8×scale gemv) and, when compiled in, the Vulkan
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
                    weight_format: 4,
                    shader_env: "CHEETAH_MATMUL_Q8_SPV",
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
