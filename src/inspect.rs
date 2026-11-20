//! Safetensors checkpoint inspection.
//!
//! This module discovers the weight files that make up a model and produces a
//! serializable report describing their tensors and metadata. It is used by
//! the CLI to show what a checkpoint contains without loading tensor data into
//! application memory: files are memory-mapped and the Safetensors header is
//! parsed in place.
//!
//! Supported inputs:
//! - A single file whose extension is `safetensors` (case-insensitive).
//! - A directory containing `cheetah_manifest.json`. The manifest must have a
//!   `shards` array of objects with a `file` string. Each shard path must be
//!   relative, contain only normal path components, end in `.safetensors`, and
//!   exist.
//! - A directory containing `model.safetensors.index.json`. The `weight_map`
//!   object lists tensor names mapped to weight filenames, which are joined to
//!   the model directory.
//! - A directory containing `model.safetensors`.
//!
//! The manifest is preferred over the index, and the index is preferred over
//! the single-file fallback. A missing or unsupported path yields a
//! human-readable error string.
//!
//! Report format:
//! `inspect_model` returns a `ModelInfo`, which serializes to JSON with the
//! supplied model path, the combined size of all inspected files, the fixed
//! format label `Safetensors`, a name-sorted list of `TensorInfo` records, and
//! a key-sorted list of metadata key/value pairs. Each tensor reports its
//! name, dimensions, dtype (the Rust debug spelling such as `F32`), and byte
//! length. Internal alignment tensors whose names start with
//! `__cheetah_padding_` are omitted because they are an implementation detail
//! of shard padding, not model weights. Files are deduplicated after
//! discovery, and tensors from all shards are merged into one list.
//!
//! Security and trust:
//! Manifest shard paths receive explicit relative-path validation before they
//! are joined to the model directory. The `model.safetensors.index.json`
//! branch trusts the filenames in `weight_map` and joins them without the same
//! component-level checks, so an index file from an untrusted source should be
//! treated with caution. Neither branch canonicalizes symlinks in the final
//! shard paths.

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Description of one tensor found in a Safetensors checkpoint.
///
/// The shape is reported in the file's own dimension order. `dtype` is a
/// display string rather than a structured enum so the report remains stable
/// and easy to serialize. `bytes` is optional to leave room for formats that
/// cannot determine data length without reading tensor contents; for
/// Safetensors it is always populated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name as stored in the Safetensors header.
    pub name: String,
    /// Dimensions of the tensor, in stored order.
    pub shape: Vec<u64>,
    /// Data type label, for example `F32` or `BF16`.
    pub dtype: String,
    /// Size of the tensor payload in bytes, when known.
    pub bytes: Option<u64>,
}

/// Complete inspection report for a model, possibly spanning multiple shards.
///
/// The report is designed to be serialized directly to JSON by the CLI.
/// `path` preserves the spelling supplied by the caller rather than a
/// canonicalized form, so the output matches the user's invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model file or directory that was inspected, as supplied by the caller.
    pub path: PathBuf,
    /// Sum of the byte lengths of all inspected files.
    pub file_size: u64,
    /// Human-readable format name; currently always `Safetensors`.
    pub format: String,
    /// All non-padding tensors, sorted by name.
    pub tensors: Vec<TensorInfo>,
    /// Metadata pairs merged from all shards, sorted by key.
    pub metadata: Vec<(String, String)>,
}

/// Inspect a Safetensors model path and build a complete report.
///
/// `path` may be a single Safetensors file or a directory. Directory
/// resolution is delegated to `model_files`, which understands the Cheetah
/// manifest, the Hugging Face Safetensors index, and the single-file layout.
/// Every discovered shard is mapped and parsed; tensors and metadata are
/// accumulated across shards before being sorted for deterministic output.
///
/// Errors cover discovery failures, unreadable or unmappable files, invalid
/// Safetensors headers, header lengths that overflow the address space, and
/// cumulative file sizes that exceed `u64`.
pub fn inspect_model(path: &Path) -> Result<ModelInfo, String> {
    // Resolve the list of weight files first. This may validate manifest
    // entries or read an index, and it deduplicates repeated shard names.
    let files = model_files(path)?;
    let mut tensors = Vec::new();
    let mut metadata = BTreeMap::new();
    let mut file_size = 0u64;
    for file_path in &files {
        // Open read-only; the map below keeps the file contents accessible
        // without copying the tensor payload into the process heap.
        let file = File::open(file_path)
            .map_err(|error| format!("Failed to open {}: {error}", file_path.display()))?;
        // Safety: creating a shared memory map is only sound while the
        // underlying file is not truncated or otherwise mutated by another
        // process. Model files are treated as immutable inputs here; a
        // malicious concurrent writer could still invalidate this assumption.
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|error| format!("Failed to map {}: {error}", file_path.display()))?
        };
        // Deserialize the header and validate tensor offsets. This does not
        // copy tensor data; it borrows slices from the mapping.
        let safetensors = SafeTensors::deserialize(&mmap).map_err(|error| {
            format!(
                "Invalid Safetensors file '{}': {error}",
                file_path.display()
            )
        })?;
        // Accumulate logical file sizes with an explicit overflow check so
        // absurdly large or corrupted inputs cannot wrap the reported total.
        file_size = file_size
            .checked_add(mmap.len() as u64)
            .ok_or_else(|| "Model file size exceeds the supported range.".to_string())?;
        for (name, tensor) in safetensors.tensors() {
            // Padding tensors exist only to align shards and are not part of
            // the model's public weight set, so hide them from the report.
            if name.starts_with("__cheetah_padding_") {
                continue;
            }
            tensors.push(TensorInfo {
                name,
                // Convert header dimensions from usize to u64; the conversion
                // is lossless on all supported targets.
                shape: tensor
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as u64)
                    .collect(),
                // The debug representation is stable enough for display and
                // avoids adding a dependency on a Safetensors dtype mapping.
                dtype: format!("{:?}", tensor.dtype()),
                // Safetensors records exact data offsets, so the byte count is
                // always known for this format.
                bytes: Some(tensor.data().len() as u64),
            });
        }
        // Read the metadata map independently from the Safetensors crate.
        // Later shards overwrite earlier values for the same key because
        // `extend` replaces existing entries.
        metadata.extend(read_metadata(&mmap)?);
    }
    // Sorting by name makes the report deterministic even though directory
    // iteration and per-shard header order are not guaranteed. The sort is
    // stable, so duplicate names retain their relative discovery order.
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(ModelInfo {
        // Preserve the caller's spelling instead of canonicalizing the path.
        path: path.to_path_buf(),
        file_size,
        format: "Safetensors".into(),
        tensors,
        // BTreeMap already iterates in key order, yielding a sorted Vec.
        metadata: metadata.into_iter().collect(),
    })
}

/// Resolve `path` into the list of Safetensors files to inspect.
///
/// A direct Safetensors file is returned as a single-element list. For a
/// directory, resolution order is: `cheetah_manifest.json`, then
/// `model.safetensors.index.json`, then `model.safetensors`. Shard lists are
/// sorted and deduplicated before being returned. Errors describe whether the
/// path is missing, has an unsupported file extension, has a malformed or
/// unsafe manifest, or lacks any recognized model layout.
fn model_files(path: &Path) -> Result<Vec<PathBuf>, String> {
    // Direct file input must already be Safetensors; there is no format
    // probing or conversion in this module.
    if path.is_file() {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(format!("Unsupported model file format: {}", path.display()));
    }
    // Anything that is not a file and not a directory is treated as missing;
    // this also covers special filesystem objects.
    if !path.is_dir() {
        return Err(format!("Model path does not exist: {}", path.display()));
    }
    // The Cheetah manifest is the most specific layout and wins over the
    // generic Hugging Face index and single-file conventions.
    let manifest = path.join("cheetah_manifest.json");
    if manifest.is_file() {
        let manifest_data = std::fs::read(&manifest)
            .map_err(|error| format!("Failed to read {}: {error}", manifest.display()))?;
        // Parse into an untyped JSON value because only a few fields are
        // needed and extra fields must remain forward-compatible.
        let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_data)
            .map_err(|error| format!("Invalid Cheetah manifest: {error}"))?;
        let shards = manifest_value
            .get("shards")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "Cheetah manifest is missing its shards list.".to_string())?;
        let mut files = Vec::new();
        for shard in shards {
            // Every shard entry must be an object with a string file path.
            let file = shard
                .get("file")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "Cheetah manifest contains a shard without a file path.".to_string()
                })?;
            let relative = Path::new(file);
            // Reject absolute paths, parent traversal, Windows prefixes, and
            // other non-normal components so the shard stays under the model
            // directory. `CurDir` components such as a leading `./` are also
            // rejected because they are not `Normal`.
            if relative.as_os_str().is_empty()
                || !relative
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_)))
                || relative
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("safetensors")
            {
                return Err(format!(
                    "Cheetah manifest contains an unsafe or unsupported shard path: {file}"
                ));
            }
            let shard_path = path.join(relative);
            // The manifest is expected to be complete; a missing shard is an
            // error rather than something to silently skip. `is_file` follows
            // symlinks, so symlinked shards are allowed even if their target
            // lies outside the model directory.
            if !shard_path.is_file() {
                return Err(format!(
                    "Cheetah manifest shard does not exist: {}",
                    shard_path.display()
                ));
            }
            files.push(shard_path);
        }
        // Duplicate entries would otherwise be mapped and counted twice.
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err("Cheetah manifest contains no weight shards.".into());
        }
        return Ok(files);
    }
    // Fall back to the standard Hugging Face sharded index convention.
    let index = path.join("model.safetensors.index.json");
    if index.is_file() {
        let index_data = std::fs::read(&index)
            .map_err(|error| format!("Failed to read {}: {error}", index.display()))?;
        let index_value: serde_json::Value = serde_json::from_slice(&index_data)
            .map_err(|error| format!("Invalid Safetensors index: {error}"))?;
        let weight_map = index_value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| "Safetensors index is missing its weight_map.".to_string())?;
        // Collect the referenced filenames. Non-string values are ignored
        // rather than treated as fatal, matching the permissive parsing of the
        // surrounding index format.
        let mut files = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(|name| path.join(name))
            .collect::<Vec<_>>();
        // Multiple tensors usually map to the same file; deduplicate so each
        // shard is inspected and counted only once.
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err("Safetensors index contains no weight files.".into());
        }
        return Ok(files);
    }
    // Finally, accept the common single-file layout.
    let model_file = path.join("model.safetensors");
    if model_file.is_file() {
        return Ok(vec![model_file]);
    }
    // No supported layout: report the two primary names the user could create
    // to make the directory inspectable.
    Err(format!(
        "No model.safetensors or model.safetensors.index.json found in '{}'.",
        path.display()
    ))
}

/// Extract the `__metadata__` string map from a Safetensors file.
///
/// The Safetensors layout begins with an eight-byte little-endian header
/// length followed by a JSON header. This function parses that header again
/// instead of relying on the typed Safetensors API, because the API exposes
/// tensor descriptors but not the arbitrary metadata object. Missing
/// `__metadata__` yields an empty map. Metadata values that are not strings are
/// ignored rather than causing a failure, and keys and values are returned
/// verbatim without interpretation.
///
/// Errors cover a file shorter than the length prefix, a header length that
/// cannot be represented as `usize` or that runs past the mapped data, and
/// invalid JSON in the header.
fn read_metadata(data: &[u8]) -> Result<BTreeMap<String, String>, String> {
    // The fixed eight-byte prefix must exist before it can be decoded.
    if data.len() < 8 {
        return Err("Safetensors file is shorter than its header length prefix.".into());
    }
    // The slice length is known to be eight, so `try_into` cannot fail.
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap());
    // On 32-bit targets a header larger than usize::MAX cannot be addressed.
    let header_len = usize::try_from(header_len)
        .map_err(|_| "Safetensors header exceeds the supported size.".to_string())?;
    // Use checked arithmetic so a hostile length cannot wrap around, then
    // require the entire declared header to be present in the mapping.
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| "Safetensors header extends beyond the file.".to_string())?;
    let header: serde_json::Value = serde_json::from_slice(&data[8..header_end])
        .map_err(|error| format!("Invalid Safetensors metadata: {error}"))?;
    // Absence of metadata is valid and simply means there is nothing to show.
    let Some(metadata) = header
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };
    // Keep only string values; the Safetensors specification defines metadata
    // as strings, and silently dropping other JSON types keeps inspection
    // tolerant of slightly malformed files.
    Ok(metadata
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_owned())))
        .collect())
}

/// Unit tests for tensor inspection, padding filtering, error reporting, and
/// manifest-based shard resolution.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counter that keeps temporary filenames unique within the test process.
    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Build an in-memory Safetensors file from a raw JSON header and payload.
    ///
    /// The first eight bytes are the little-endian header length, matching the
    /// on-disk format that `inspect_model` and `read_metadata` expect.
    fn safetensors_file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(payload);
        data
    }

    /// A minimal valid file should report its size, tensor shape, dtype, and
    /// metadata without requiring any real weights.
    #[test]
    fn inspects_safetensors_header_and_tensor_shapes() {
        let path = std::env::temp_dir().join(format!(
            "cheetah-inspect-{}-{}.safetensors",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let data = safetensors_file(
            r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"format":"pt"}}"#,
            &[0; 8],
        );
        std::fs::write(&path, &data).unwrap();
        let info = inspect_model(&path).unwrap();
        assert_eq!(info.file_size, data.len() as u64);
        assert_eq!(info.tensors[0].name, "weight");
        assert_eq!(info.tensors[0].shape, vec![2]);
        assert_eq!(info.metadata, [("format".into(), "pt".into())]);
        std::fs::remove_file(path).unwrap();
    }

    /// Tensors whose names use the internal padding prefix must not appear in
    /// the public report, while real weights in the same file still do.
    #[test]
    fn hides_internal_aligned_shard_padding_tensors() {
        let path = std::env::temp_dir().join(format!(
            "cheetah-padding-{}-{}.safetensors",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let data = safetensors_file(
            r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"__cheetah_padding_000000":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#,
            &[0; 8],
        );
        std::fs::write(&path, data).unwrap();

        let info = inspect_model(&path).unwrap();
        assert_eq!(info.tensors.len(), 1);
        assert_eq!(info.tensors[0].name, "weight");
        std::fs::remove_file(path).unwrap();
    }

    /// Arbitrary non-Safetensors bytes must produce an inspection error that
    /// mentions the invalid Safetensors format.
    #[test]
    fn rejects_invalid_safetensors_files() {
        let path = std::env::temp_dir().join(format!(
            "cheetah-invalid-{}-{}.safetensors",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"not safetensors").unwrap();
        assert!(
            inspect_model(&path)
                .unwrap_err()
                .contains("Invalid Safetensors")
        );
        std::fs::remove_file(path).unwrap();
    }

    /// Manifest shard paths are resolved relative to the model directory and
    /// returned in sorted, deduplicated order regardless of manifest order.
    #[test]
    fn resolves_manifest_shards_for_inspection() {
        let directory = std::env::temp_dir().join(format!(
            "cheetah-inspect-manifest-{}-{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let weights = directory.join(".cheetah/weights");
        std::fs::create_dir_all(&weights).unwrap();
        std::fs::write(weights.join("layer_0.safetensors"), []).unwrap();
        std::fs::write(weights.join("layer_1.safetensors"), []).unwrap();
        std::fs::write(
            directory.join("cheetah_manifest.json"),
            r#"{"shards":[{"file":".cheetah/weights/layer_1.safetensors"},{"file":".cheetah/weights/layer_0.safetensors"}]}"#,
        )
        .unwrap();

        // The manifest lists layer_1 first, but the resolver sorts the paths.
        assert_eq!(
            model_files(&directory).unwrap(),
            vec![
                weights.join("layer_0.safetensors"),
                weights.join("layer_1.safetensors")
            ]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
