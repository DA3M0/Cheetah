//! Vulkan weight layer shared by both executors: typed payloads
//! (`WeightData` — an original storage precision or a registered quantized
//! format's buffer), the `Linear` projection, and the loader plumbing
//! (`load_weights` / `take_weight` / `take_linear`, resident upload). Every
//! operator dispatch goes through the precision registry (`chat::precision`)
//! by the payload's registered format id.

use crate::chat::precision::{self, PrecisionBuffer};
use crate::color::{self, Color};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

/// One weight payload: an original storage precision (f32 / f16 / bf16) or a
/// registered quantization format's buffer. Every operator dispatch goes
/// through the precision registry (`chat::precision`) by the payload's
/// registered format id.
#[derive(Clone)]
pub(crate) enum WeightData {
    F32(Arc<[f32]>),
    F16(Arc<[u16]>),
    BF16(Arc<[u16]>),
    /// A registered quantization format's buffer (see `chat::precision`);
    /// every operation on it dispatches through the format's registered
    /// operator table.
    Quantized(Arc<dyn PrecisionBuffer>),
}

impl WeightData {
    /// Converts a Safetensors tensor view into the typed payload (F32 / F16 /
    /// BF16); any other dtype is rejected loudly — the Vulkan shaders serve
    /// those three storage widths only.
    fn from_view(name: &str, view: &TensorView<'_>) -> Result<Self, String> {
        let bytes = view.data();
        match view.dtype() {
            Dtype::F32 => {
                if bytes.len() % 4 != 0 {
                    return Err(format!("Tensor '{name}' has an invalid F32 byte length."));
                }
                Ok(Self::F32(
                    bytes
                        .chunks_exact(4)
                        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                        .collect::<Vec<_>>()
                        .into(),
                ))
            }
            Dtype::F16 | Dtype::BF16 => {
                if bytes.len() % 2 != 0 {
                    return Err(format!(
                        "Tensor '{name}' has an invalid 16-bit byte length."
                    ));
                }
                let values: Arc<[u16]> = bytes
                    .chunks_exact(2)
                    .map(|value| u16::from_le_bytes([value[0], value[1]]))
                    .collect::<Vec<_>>()
                    .into();
                if view.dtype() == Dtype::F16 {
                    Ok(Self::F16(values))
                } else {
                    Ok(Self::BF16(values))
                }
            }
            dtype => Err(format!(
                "Vulkan models support F32, F16, or BF16 weights; tensor '{name}' uses {dtype:?}."
            )),
        }
    }

    /// The registered precision format id this payload resolves to
    /// (`chat::precision`): "f32" / "f16" / "bf16" for the original storage
    /// precisions, the quantized buffer's format id otherwise.
    fn format_id(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F16(_) => "f16",
            Self::BF16(_) => "bf16",
            Self::Quantized(buffer) => buffer.format_id(),
        }
    }

    /// The raw weight bytes: the typed payloads viewed as bytes (no copy), or
    /// the quantized buffer. The precision registry's operator tables take
    /// this uniform byte view; each format's implementation restores its own
    /// typed view (`precision::bytes`).
    fn as_bytes(&self) -> &[u8] {
        match self {
            Self::F32(values) => unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
            },
            Self::F16(values) | Self::BF16(values) => unsafe {
                std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2)
            },
            Self::Quantized(buffer) => buffer.buffer(),
        }
    }

    /// Flat scalar access, dequantized to f32; a quantized payload falls back to
    /// row 0 — flat accessors only serve diagnostics, real reads go through the
    /// operator tables.
    pub(crate) fn value(&self, index: usize) -> f32 {
        match self {
            Self::F32(values) => values[index],
            Self::F16(values) => half::f16::from_bits(values[index]).to_f32(),
            Self::BF16(values) => half::bf16::from_bits(values[index]).to_f32(),
            // Quantized buffers are only meaningful with row/col context;
            // callers that hit the flat accessor get the dequantized value
            // of row 0.
            Self::Quantized(matrix) => {
                matrix.dequant_value(0, index.min(matrix.cols().saturating_sub(1)))
            }
        }
    }

    /// Builds the `(1 + weight)` offset copy that gemma's norms apply.
    pub(crate) fn add_one(&self) -> Self {
        Self::F32(
            (0..self.len())
                .map(|index| self.value(index) + 1.0)
                .collect(),
        )
    }

    /// Logical element count of the payload.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) | Self::BF16(values) => values.len(),
            Self::Quantized(matrix) => matrix.element_count(),
        }
    }

    /// FFI view for shader operators: (pointer, weight_format, element_count).
    /// Format encoding matches shared_ops.comp: 0 = f32, 2 = f16, 3 = bf16;
    /// quantized formats report their registered code (q8 = 4) — they are
    /// only served by their dedicated matmul pipeline, not shared_ops.
    pub(crate) fn ffi_parts(&self) -> (*const std::os::raw::c_void, u32, usize) {
        match self {
            Self::F32(values) => (values.as_ptr().cast(), 0, values.len()),
            Self::F16(values) => (values.as_ptr().cast(), 2, values.len()),
            Self::BF16(values) => (values.as_ptr().cast(), 3, values.len()),
            Self::Quantized(matrix) => {
                let format = precision::vulkan_ops_for_buffer(matrix.as_ref())
                    .map(|ops| ops.weight_format)
                    .unwrap_or(0);
                (
                    matrix.buffer().as_ptr().cast(),
                    format,
                    matrix.buffer().len() / 4,
                )
            }
        }
    }

    /// RMSNorm shader per stored weight format; a quantized payload is an error
    /// — quantization applies to linear projections only, norms stay in their
    /// original precision.
    pub(crate) fn rms_norm(
        &self,
        input: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::F32(weights) => {
                cheetah::native::vulkan::rms_norm_f32(input, weights, rows, width, epsilon)
            }
            Self::F16(weights) => {
                cheetah::native::vulkan::rms_norm_fp16(input, weights, rows, width, epsilon)
            }
            Self::BF16(weights) => {
                cheetah::native::vulkan::rms_norm_bf16(input, weights, rows, width, epsilon)
            }
            Self::Quantized(matrix) => Err(format!(
                "{} quantization applies to linear projections only; norms stay in their original precision.",
                precision::format_label(matrix.format_id())
            )),
        }
    }

    /// LayerNorm shader (weight + bias, dtypes must match); quantized payloads
    /// error the same way as [`WeightData::rms_norm`].
    pub(crate) fn layer_norm(
        &self,
        bias: &Self,
        input: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        match (self, bias) {
            (Self::F32(weights), Self::F32(bias)) => {
                cheetah::native::vulkan::layer_norm_f32(input, weights, bias, rows, width, epsilon)
            }
            (Self::F16(weights), Self::F16(bias)) => {
                cheetah::native::vulkan::layer_norm_fp16(input, weights, bias, rows, width, epsilon)
            }
            (Self::BF16(weights), Self::BF16(bias)) => {
                cheetah::native::vulkan::layer_norm_bf16(input, weights, bias, rows, width, epsilon)
            }
            (Self::Quantized(_), _) | (_, Self::Quantized(_)) => Err(format!(
                "{} quantization applies to linear projections only; layer norms stay in their original precision.",
                match (self, bias) {
                    (Self::Quantized(matrix), _) => precision::format_label(matrix.format_id()),
                    (_, Self::Quantized(matrix)) => precision::format_label(matrix.format_id()),
                    _ => "Quantized",
                }
            )),
            _ => Err("Layer normalization weight and bias dtypes do not match.".into()),
        }
    }

    /// The streaming matmul core: row tiles (≤64) × column tiles (bounded by a
    /// 16M-element weight tile) for original precisions, one full-buffer pass
    /// per row tile for quantized formats (the C operator has no column offset);
    /// dimensions are validated up front.
    pub(crate) fn linear(
        &self,
        input: &[f32],
        rows: usize,
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<f32>, String> {
        let input_len = rows
            .checked_mul(input_size)
            .ok_or_else(|| "Vulkan linear input size overflow.".to_string())?;
        let output_len = rows
            .checked_mul(output_size)
            .ok_or_else(|| "Vulkan linear output size overflow.".to_string())?;
        if rows == 0 || input_size == 0 || output_size == 0 || input.len() != input_len {
            return Err("Invalid Vulkan linear dimensions or input length.".into());
        }
        const MAX_WEIGHT_TILE_ELEMENTS: usize = 16 * 1024 * 1024;
        const MAX_ROW_TILE: usize = 64;
        let mut output = vec![0.0; output_len];
        // Quantized formats stage their whole buffer in one pass (the C
        // operator has no column-offset parameter), so they run one
        // full-width matmul per row tile instead of the column-tiled loop
        // below.
        let quantized: Option<(precision::VulkanOps, &dyn PrecisionBuffer)> = match self {
            Self::Quantized(buffer) => {
                let ops = precision::vulkan_ops_for_buffer(buffer.as_ref()).ok_or_else(|| {
                    format!(
                        "Quantization format '{}' has no Vulkan operator table.",
                        buffer.format_id()
                    )
                })?;
                if buffer.rows() != output_size || buffer.cols() != input_size {
                    return Err(format!(
                        "Quantized buffer {}x{} does not match linear dimensions {}x{}.",
                        buffer.cols(),
                        buffer.rows(),
                        input_size,
                        output_size
                    ));
                }
                Some((ops, buffer.as_ref()))
            }
            _ => None,
        };
        let tile_size = output_size
            .min(16_384)
            .min((MAX_WEIGHT_TILE_ELEMENTS / input_size).max(1));
        for row_offset in (0..rows).step_by(MAX_ROW_TILE) {
            let tile_rows = (rows - row_offset).min(MAX_ROW_TILE);
            let input_start = row_offset * input_size;
            let input_tile = &input[input_start..input_start + tile_rows * input_size];
            if let Some((ops, buffer)) = quantized {
                let tile = (ops.matmul_streaming)(
                    input_tile,
                    buffer.buffer(),
                    tile_rows,
                    input_size,
                    output_size,
                )?;
                for row in 0..tile_rows {
                    let source = &tile[row * output_size..(row + 1) * output_size];
                    let destination_start = (row_offset + row) * output_size;
                    output[destination_start..destination_start + output_size]
                        .copy_from_slice(source);
                }
                continue;
            }
            // Original precision: the format's registered streaming matmul
            // reads its own stored payload (f32 pipeline for f32, the 16-bit
            // pipeline for f16/bf16) — no conversion into another width.
            let format_id = self.format_id();
            let ops = precision::vulkan_ops_for_format(format_id).ok_or_else(|| {
                format!("Weight precision '{format_id}' has no Vulkan operator table.")
            })?;
            let weight_bytes = self.as_bytes();
            let element_size = match self {
                Self::F32(_) => 4,
                Self::F16(_) | Self::BF16(_) => 2,
                Self::Quantized(_) => unreachable!("quantized formats run above the column loop"),
            };
            for column_offset in (0..output_size).step_by(tile_size) {
                let columns = (output_size - column_offset).min(tile_size);
                let weight_start = column_offset * input_size * element_size;
                let weight_end = (column_offset + columns) * input_size * element_size;
                let tile = (ops.matmul_streaming)(
                    input_tile,
                    &weight_bytes[weight_start..weight_end],
                    tile_rows,
                    input_size,
                    columns,
                )?;
                for row in 0..tile_rows {
                    let source = &tile[row * columns..(row + 1) * columns];
                    let destination_start = (row_offset + row) * output_size + column_offset;
                    output[destination_start..destination_start + columns].copy_from_slice(source);
                }
            }
        }
        Ok(output)
    }
}

pub(crate) struct NamedWeight {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: WeightData,
}

/// Ant residency slot for one weight tensor: Some = resident in the GPU
/// bucket (device-local when available, persistent host-coherent otherwise);
/// None = streaming path (weights re-staged per matmul from host memory).
#[derive(Clone)]
pub(crate) struct VulkanWeightSlot {
    /// Opaque handle returned by the format's registered `upload` operator;
    /// released through `weight_release` on eviction.
    pub(crate) handle: u64,
}

#[derive(Clone)]
pub(crate) struct Linear {
    pub(crate) name: String,
    pub(crate) input_size: usize,
    pub(crate) output_size: usize,
    pub(crate) weight: WeightData,
    pub(crate) bias: Option<WeightData>,
    pub(crate) slot: Option<VulkanWeightSlot>,
}

const MAX_ROW_TILE: usize = 64;

impl Linear {
    /// Linear dispatch: the resident-bucket path when a slot is bound (the
    /// format's registered resident matmul, row-tiled), otherwise the streaming
    /// path through [`WeightData::linear`]; adds the bias per row when present.
    pub(crate) fn forward(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, String> {
        let expected = rows
            .checked_mul(self.input_size)
            .ok_or_else(|| format!("Input size overflow in '{}'.", self.name))?;
        if rows == 0 || input.len() != expected {
            return Err(format!(
                "Invalid input for '{}': expected {expected} values, received {}.",
                self.name,
                input.len()
            ));
        }
        let mut output = if let Some(slot) = &self.slot {
            // Ant resident bucket: the weight stays bound in place on the GPU
            // (its registered format uploaded it) — only activation and output
            // tiles pass through the host pool.
            let format_id = self.weight.format_id();
            let ops = precision::vulkan_ops_for_format(format_id).ok_or_else(|| {
                format!("Weight precision '{format_id}' has no Vulkan operator table.")
            })?;
            let matmul_resident = ops.matmul_resident.ok_or_else(|| {
                format!("Weight precision '{format_id}' has no resident matmul operator.")
            })?;
            let mut output = vec![0.0; rows * self.output_size];
            for row_offset in (0..rows).step_by(MAX_ROW_TILE) {
                let tile_rows = (rows - row_offset).min(MAX_ROW_TILE);
                let input_tile =
                    &input[row_offset * self.input_size..][..tile_rows * self.input_size];
                let out_tile = (matmul_resident)(
                    slot.handle,
                    input_tile,
                    tile_rows,
                    self.input_size,
                    self.output_size,
                )?;
                output[row_offset * self.output_size..][..tile_rows * self.output_size]
                    .copy_from_slice(&out_tile);
            }
            output
        } else {
            // Host-streaming path: the format's registered streaming matmul
            // (quantized formats stage the whole buffer in one pass; original
            // precisions column-tile their stored payload).
            self.weight
                .linear(input, rows, self.input_size, self.output_size)?
        };
        if let Some(bias) = &self.bias {
            if bias.len() != self.output_size {
                return Err(format!("Invalid bias shape for '{}'.", self.name));
            }
            for row in output.chunks_exact_mut(self.output_size) {
                for (value, index) in row.iter_mut().zip(0..self.output_size) {
                    *value += bias.value(index);
                }
            }
        }
        Ok(output)
    }
}

/// Upload one linear weight into the resident bucket through its format's
/// registered `upload` operator (bf16 bits, or a quantized buffer with its
/// padded per-row stride); returns None when the backend cannot serve it
/// (then the streaming path stays active). Original precisions without a
/// registered upload (f32/f16) always stream.
pub(crate) fn resident_upload(weight: &WeightData, row_bytes: usize) -> Option<u64> {
    let ops = precision::vulkan_ops_for_format(weight.format_id())?;
    let upload = ops.upload?;
    upload(weight.as_bytes(), row_bytes)
}

/// `resident_upload` with the Ant page-in diagnostic line.
pub(crate) fn upload_slot(
    name: &str,
    weight: &WeightData,
    row_bytes: usize,
) -> Option<VulkanWeightSlot> {
    resident_upload(weight, row_bytes).map(|handle| {
        eprintln!(
            "{}",
            color::stderr(
                &format!("cheetah: Ant page-in '{name}' -> GPU bucket."),
                Color::Green
            )
        );
        VulkanWeightSlot { handle }
    })
}

/// Mmaps every shard, deserializes Safetensors, skips internal padding
/// tensors, and rejects duplicate tensor names across shards.
pub(crate) fn load_weights(paths: &[PathBuf]) -> Result<HashMap<String, NamedWeight>, String> {
    let mut weights = HashMap::new();
    for path in paths {
        let file = File::open(path).map_err(|error| {
            format!(
                "Cannot open Vulkan model shard '{}': {error}",
                path.display()
            )
        })?;
        let mapping = unsafe {
            memmap2::MmapOptions::new().map(&file).map_err(|error| {
                format!(
                    "Cannot mmap Vulkan model shard '{}': {error}",
                    path.display()
                )
            })?
        };
        let tensors = SafeTensors::deserialize(&mapping)
            .map_err(|error| format!("Invalid Vulkan model shard '{}': {error}", path.display()))?;
        for name in tensors.names() {
            if name.starts_with("__cheetah_padding_") {
                continue;
            }
            let view = tensors.tensor(name).map_err(|error| {
                format!(
                    "Cannot read tensor '{name}' from '{}': {error}",
                    path.display()
                )
            })?;
            let weight = NamedWeight {
                shape: view.shape().to_vec(),
                data: WeightData::from_view(name, &view)?,
            };
            if weights.insert(name.to_owned(), weight).is_some() {
                return Err(format!("Duplicate tensor '{name}' across model shards."));
            }
        }
    }
    Ok(weights)
}

/// Removes one tensor from the pool with exact shape validation.
pub(crate) fn take_weight(
    weights: &mut HashMap<String, NamedWeight>,
    name: &str,
    expected_shape: &[usize],
) -> Result<NamedWeight, String> {
    let weight = weights
        .remove(name)
        .ok_or_else(|| format!("Vulkan model checkpoint is missing tensor '{name}'."))?;
    if weight.shape != expected_shape {
        return Err(format!(
            "Tensor '{name}' has shape {:?}; expected {expected_shape:?}.",
            weight.shape
        ));
    }
    Ok(weight)
}

/// Shape assertion helper with a descriptive error.
pub(crate) fn validate_shape(
    name: &str,
    actual: &[usize],
    expected: &[usize],
) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "Tensor '{name}' has shape {actual:?}; expected {expected:?}."
        ));
    }
    Ok(())
}

/// Builds a [`Linear`]: shape-validated weight (plus a required bias when
/// `with_bias`), load-time quantization when a precision format is selected
/// (norms / biases / the embedding never quantize), and the Ant-bucket
/// resident upload for VRAM-tier units — upload failures fall back to the
/// streaming path with a warning.
pub(crate) fn take_linear(
    weights: &mut HashMap<String, NamedWeight>,
    plan: &BTreeMap<String, String>,
    prefix: &str,
    input_size: usize,
    output_size: usize,
    with_bias: bool,
) -> Result<Linear, String> {
    let weight_name = format!("{prefix}.weight");
    let resident = plan.get(&weight_name).map(String::as_str) == Some("vram");
    // Ant quantization: the format selected through the registered selector
    // environment variables (see chat::precision — CHEETAH_QUANT_Q8 selects "q8")
    // quantizes linear projections at load time through that format's
    // registered operator table, halving the weight footprint for both the
    // streaming and the resident path. Norms, biases and the embedding stay
    // in full precision.
    let quant_ops = precision::selected_vulkan_ops();
    let weight = take_weight(weights, &weight_name, &[output_size, input_size])?;
    let bias_name = format!("{prefix}.bias");
    let bias = match weights.remove(&bias_name) {
        Some(bias) => {
            if bias.shape != [output_size] {
                return Err(format!(
                    "Tensor '{bias_name}' has shape {:?}; expected [{output_size}].",
                    bias.shape
                ));
            }
            Some(bias.data)
        }
        None if with_bias => {
            return Err(format!(
                "Vulkan checkpoint is missing tensor '{bias_name}'."
            ));
        }
        None => None,
    };
    // Quantize before the resident upload so the GPU bucket holds the
    // quantized buffer (roughly half the footprint of BF16).
    let quantize = quant_ops
        .map(|ops| {
            ops.quantize.ok_or_else(|| {
                "The selected quantization format's Vulkan entry has no quantize operator."
                    .to_string()
            })
        })
        .transpose()?;
    let weight = match quantize {
        Some(quantize) => {
            let flat: Vec<f32> = (0..weight.data.len())
                .map(|index| weight.data.value(index))
                .collect();
            WeightData::Quantized(quantize(&flat, output_size, input_size)?)
        }
        None => weight.data,
    };
    // Ant bucketing: units planned into the VRAM tier are uploaded once into
    // the resident bucket (device-local when available); the host copy stays
    // authoritative for page-ins after an eviction. Upload failures fall back
    // to the per-step streaming path. `row_bytes` is the padded per-row
    // stride quantized uploads validate against (input_size = reduction
    // width); original precisions ignore it.
    let slot = if resident {
        let row_bytes = (input_size + 3) / 4 * 4;
        upload_slot(prefix, &weight, row_bytes)
    } else {
        None
    };
    if resident && slot.is_none() {
        eprintln!(
            "{}",
            color::stderr(
                &format!(
                    "cheetah: resident upload for '{prefix}' unavailable; keeping the streaming path."
                ),
                Color::Yellow
            )
        );
    }
    Ok(Linear {
        name: prefix.to_owned(),
        input_size,
        output_size,
        weight,
        bias,
        slot,
    })
}

#[cfg(test)]
mod cross_backend_tests {
    use super::*;
    use crate::chat::precision::{PrecisionBackend, PrecisionKind};

    /// Cross-backend coverage check for the precision registry: every
    /// registered quantized format must run on BOTH backends and agree —
    /// the Vulkan streaming matmul, the Vulkan resident matmul and the
    /// native CPU gemv all consume the same registered quantized buffer.
    /// A future format added to `src/chat/precision/registrations/` is
    /// covered by this test automatically.
    #[test]
    #[ignore = "requires a Vulkan compute device"]
    fn quantized_formats_match_across_backends() {
        cheetah::native::vulkan::initialize().unwrap();
        cheetah::native::vulkan::prepare().unwrap();

        let rows = 5usize;
        let cols = 11usize; // odd width: exercises row padding + nibble parities
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 19) as f32 - 9.0) * 0.35)
            .collect();
        let x: Vec<f32> = (0..cols)
            .map(|index| (index as f32 * 0.9 - 3.0).sin())
            .collect();
        let tolerance = 1e-3;

        for registration in precision::REGISTRY {
            if registration.kind != PrecisionKind::Quantized {
                continue;
            }
            let PrecisionBackend::Native(cpu) = registration.loader_for("native").unwrap() else {
                panic!(
                    "{}: native entry has the wrong loader kind",
                    registration.id
                );
            };
            let quantize = cpu.quantize.expect("quantized formats must quantize");
            let buffer = quantize(&values, rows, cols).unwrap();

            // Native CPU gemv over the registered buffer (the CPU side of
            // the contract).
            let mut cpu_output = vec![0f32; rows];
            (cpu.gemv)(
                buffer.buffer(),
                buffer.rows(),
                buffer.cols(),
                &x,
                &mut cpu_output,
            );

            // Vulkan streaming matmul over the same buffer.
            let PrecisionBackend::Vulkan(vulkan) = registration.loader_for("vulkan").unwrap()
            else {
                panic!(
                    "{}: vulkan entry has the wrong loader kind",
                    registration.id
                );
            };
            let streaming = (vulkan.matmul_streaming)(&x, buffer.buffer(), 1, cols, rows).unwrap();
            for (index, (gpu, reference)) in streaming.iter().zip(&cpu_output).enumerate() {
                assert!(
                    (gpu - reference).abs() <= tolerance,
                    "{} streaming row {index}: gpu {gpu} vs cpu {reference}",
                    registration.id
                );
            }

            // Vulkan resident matmul: upload once, multiply bound.
            let upload = vulkan
                .upload
                .expect("quantized formats must register an upload");
            let handle = upload(buffer.buffer(), buffer.row_bytes())
                .unwrap_or_else(|| panic!("{}: resident upload refused", registration.id));
            let resident = vulkan
                .matmul_resident
                .expect("quantized formats must register a resident matmul")(
                handle, &x, 1, cols, rows,
            )
            .unwrap();
            for (index, (gpu, reference)) in resident.iter().zip(&cpu_output).enumerate() {
                assert!(
                    (gpu - reference).abs() <= tolerance,
                    "{} resident row {index}: gpu {gpu} vs cpu {reference}",
                    registration.id
                );
            }
            cheetah::native::vulkan::weight_release(handle);
        }

        cheetah::native::vulkan::shutdown().unwrap();
    }
}
