//! Safetensors checkpoint inspection.

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub file_size: u64,
    pub format: String,
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<(String, String)>,
}

pub fn inspect_model(path: &Path) -> Result<ModelInfo, String> {
    let files = model_files(path)?;
    let mut tensors = Vec::new();
    let mut metadata = BTreeMap::new();
    let mut file_size = 0u64;
    for file_path in &files {
        let file = File::open(file_path)
            .map_err(|error| format!("Failed to open {}: {error}", file_path.display()))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|error| format!("Failed to map {}: {error}", file_path.display()))?
        };
        let safetensors = SafeTensors::deserialize(&mmap).map_err(|error| {
            format!(
                "Invalid Safetensors file '{}': {error}",
                file_path.display()
            )
        })?;
        file_size = file_size
            .checked_add(mmap.len() as u64)
            .ok_or_else(|| "Model file size exceeds the supported range.".to_string())?;
        for (name, tensor) in safetensors.tensors() {
            if name.starts_with("__cheetah_padding_") {
                continue;
            }
            tensors.push(TensorInfo {
                name,
                shape: tensor
                    .shape()
                    .iter()
                    .map(|dimension| *dimension as u64)
                    .collect(),
                dtype: format!("{:?}", tensor.dtype()),
                bytes: Some(tensor.data().len() as u64),
            });
        }
        metadata.extend(read_metadata(&mmap)?);
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(ModelInfo {
        path: path.to_path_buf(),
        file_size,
        format: "Safetensors".into(),
        tensors,
        metadata: metadata.into_iter().collect(),
    })
}

fn model_files(path: &Path) -> Result<Vec<PathBuf>, String> {
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
    if !path.is_dir() {
        return Err(format!("Model path does not exist: {}", path.display()));
    }
    let manifest = path.join("cheetah_manifest.json");
    if manifest.is_file() {
        let manifest_data = std::fs::read(&manifest)
            .map_err(|error| format!("Failed to read {}: {error}", manifest.display()))?;
        let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_data)
            .map_err(|error| format!("Invalid Cheetah manifest: {error}"))?;
        let shards = manifest_value
            .get("shards")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "Cheetah manifest is missing its shards list.".to_string())?;
        let mut files = Vec::new();
        for shard in shards {
            let file = shard
                .get("file")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    "Cheetah manifest contains a shard without a file path.".to_string()
                })?;
            let relative = Path::new(file);
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
            if !shard_path.is_file() {
                return Err(format!(
                    "Cheetah manifest shard does not exist: {}",
                    shard_path.display()
                ));
            }
            files.push(shard_path);
        }
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err("Cheetah manifest contains no weight shards.".into());
        }
        return Ok(files);
    }
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
        let mut files = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(|name| path.join(name))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        if files.is_empty() {
            return Err("Safetensors index contains no weight files.".into());
        }
        return Ok(files);
    }
    let model_file = path.join("model.safetensors");
    if model_file.is_file() {
        return Ok(vec![model_file]);
    }
    Err(format!(
        "No model.safetensors or model.safetensors.index.json found in '{}'.",
        path.display()
    ))
}

fn read_metadata(data: &[u8]) -> Result<BTreeMap<String, String>, String> {
    if data.len() < 8 {
        return Err("Safetensors file is shorter than its header length prefix.".into());
    }
    let header_len = u64::from_le_bytes(data[..8].try_into().unwrap());
    let header_len = usize::try_from(header_len)
        .map_err(|_| "Safetensors header exceeds the supported size.".to_string())?;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|end| *end <= data.len())
        .ok_or_else(|| "Safetensors header extends beyond the file.".to_string())?;
    let header: serde_json::Value = serde_json::from_slice(&data[8..header_end])
        .map_err(|error| format!("Invalid Safetensors metadata: {error}"))?;
    let Some(metadata) = header
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(BTreeMap::new());
    };
    Ok(metadata
        .iter()
        .filter_map(|(key, value)| value.as_str().map(|value| (key.clone(), value.to_owned())))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn safetensors_file(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(header.as_bytes());
        data.extend_from_slice(payload);
        data
    }

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
