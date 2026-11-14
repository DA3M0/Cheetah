//! Native backend weight store: shard files are mmapped once and every atomic
//! unit resolves to a byte slice inside that mapping. Ant tiers act directly on
//! the mapped pages — resident tiers stay faulted in, disk-tier units are
//! dropped with `madvise(DONTNEED)` after each use so the process never holds
//! them resident (the OS page cache absorbs the re-reads).

use crate::ant::CHEETAH_UNITS_FILENAME;
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightDType {
    Bf16,
    F16,
    F32,
}

impl WeightDType {
    pub fn element_size(self) -> usize {
        match self {
            WeightDType::Bf16 | WeightDType::F16 => 2,
            WeightDType::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Kept resident across generation steps.
    Resident,
    /// Read on demand each step and dropped immediately afterwards.
    Disk,
}

struct UnitSlice {
    shard: usize,
    offset: usize,
    bytes: usize,
    dtype: WeightDType,
    shape: Vec<usize>,
}

struct ShardMapping {
    mmap: Mmap,
    data_start: usize,
}

pub struct MmapWeightStore {
    shards: Vec<ShardMapping>,
    units: HashMap<String, UnitSlice>,
    /// Units grouped by owning layer for per-step eviction of disk-tier units.
    units_by_layer: HashMap<i32, Vec<String>>,
    /// Tier state behind a mutex: the store lives in an Arc shared across
    /// runtime clones, so tiers are mutated through interior mutability.
    tiers: Mutex<HashMap<String, Tier>>,
}

impl MmapWeightStore {
    /// Map every shard referenced by the units manifest and index the units.
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        let manifest_path = model_dir.join(CHEETAH_UNITS_FILENAME);
        let bytes = std::fs::read(&manifest_path)
            .map_err(|error| format!("Failed to read {}: {error}", manifest_path.display()))?;
        let units: Vec<crate::ant::AntUnit> = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Invalid Ant units manifest: {error}"))?;
        if units.is_empty() {
            return Err("Ant units manifest is empty; run `cheetah init` first.".into());
        }
        let weights_dir = model_dir.join(".cheetah").join("weights");
        let mut shards: Vec<ShardMapping> = Vec::new();
        let mut shard_index: HashMap<String, usize> = HashMap::new();
        let mut index: HashMap<String, UnitSlice> = HashMap::new();
        let mut units_by_layer: HashMap<i32, Vec<String>> = HashMap::new();
        for unit in &units {
            if unit.unit_id.is_empty() || unit.source_file.is_empty() {
                continue;
            }
            let shard_idx = match shard_index.get(&unit.source_file) {
                Some(idx) => *idx,
                None => {
                    let path = weights_dir.join(&unit.source_file);
                    let file = std::fs::File::open(&path)
                        .map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
                    let mmap = unsafe { Mmap::map(&file) }
                        .map_err(|error| format!("Failed to map {}: {error}", path.display()))?;
                    // Safetensors layout: 8-byte LE header length, then the JSON
                    // header, then the payload. Unit offsets are relative to the
                    // payload start.
                    if mmap.len() < 8 {
                        return Err(format!("Shard {} is too small for a header.", path.display()));
                    }
                    let header_len = u64::from_le_bytes([
                        mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
                    ]) as usize;
                    let data_start = 8 + header_len;
                    if data_start > mmap.len() {
                        return Err(format!("Shard {} header exceeds the file size.", path.display()));
                    }
                    let idx = shards.len();
                    shards.push(ShardMapping {
                        mmap,
                        data_start,
                    });
                    shard_index.insert(unit.source_file.clone(), idx);
                    idx
                }
            };
            let dtype = match unit.dtype.as_str() {
                "BF16" | "bf16" => WeightDType::Bf16,
                "F16" | "f16" => WeightDType::F16,
                "F32" | "f32" => WeightDType::F32,
                other => {
                    return Err(format!(
                        "Native backend does not support dtype {other} for unit {}.",
                        unit.unit_id
                    ))
                }
            };
            let end = unit.source_offset as usize + unit.bytes as usize;
            if end > shards[shard_idx].mmap.len() {
                return Err(format!(
                    "Unit {} exceeds the bounds of shard {}.",
                    unit.unit_id, unit.source_file
                ));
            }
            units_by_layer
                .entry(unit.layer_id)
                .or_default()
                .push(unit.unit_id.clone());            index.insert(
                unit.unit_id.clone(),
                UnitSlice {
                    shard: shard_idx,
                    offset: shards[shard_idx].data_start + unit.source_offset as usize,
                    bytes: unit.bytes as usize,
                    dtype,
                    shape: unit.shape.iter().map(|dim| *dim as usize).collect(),
                },
            );
        }
        Ok(Self {
            shards,
            units: index,
            units_by_layer,
            tiers: Mutex::new(HashMap::new()),
        })
    }

    pub fn unit_ids(&self) -> impl Iterator<Item = &str> {
        self.units.keys().map(String::as_str)
    }

    /// Cloned shape of one unit, for parameter derivation from the manifest.
    pub fn unit_shape(&self, unit_id: &str) -> Option<Vec<usize>> {
        self.units.get(unit_id).map(|unit| unit.shape.clone())
    }

    /// Whether the units manifest carries a unit (e.g. optional biases, or the
    /// tied-vs-untied embedding/lm_head choice).
    pub fn contains(&self, unit_id: &str) -> bool {
        self.units.contains_key(unit_id)
    }

    /// Highest layer id in the manifest; the decoder layer count is id + 1
    /// (non-layer weights share layer 0 with the first decoder layer).
    pub fn max_layer_id(&self) -> i32 {
        self.units_by_layer.keys().copied().max().unwrap_or(0)
    }

    /// Raw byte slice of one unit inside its shard mapping.
    pub fn slice(&self, unit_id: &str) -> Result<(&[u8], WeightDType, &[usize]), String> {
        let unit = self
            .units
            .get(unit_id)
            .ok_or_else(|| format!("Native backend: unknown unit '{unit_id}'."))?;
        let shard = &self.shards[unit.shard];
        let start = unit.offset;
        let end = start + unit.bytes;
        if end > shard.mmap.len() {
            return Err(format!("Native backend: unit '{unit_id}' is out of bounds."));
        }
        Ok((&shard.mmap[start..end], unit.dtype, &unit.shape))
    }

    pub fn tier(&self, unit_id: &str) -> Tier {
        self.tiers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(unit_id)
            .copied()
            .unwrap_or(Tier::Resident)
    }

    pub fn set_tier(&self, unit_id: &str, tier: Tier) {
        let mut tiers = self
            .tiers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.units.contains_key(unit_id) {
            tiers.insert(unit_id.to_string(), tier);
            drop(tiers);
            match tier {
                Tier::Resident => self.page_in(unit_id),
                Tier::Disk => self.drop_pages(unit_id),
            }
        }
    }

    /// Units of one layer grouped by their current tier, for per-step eviction.
    pub fn layer_unit_ids(&self, layer_id: i32) -> Vec<String> {
        self.units_by_layer
            .get(&layer_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Fault a unit's pages back in (Ant page-in on the native backend).
    pub fn page_in(&self, unit_id: &str) {
        let (slice, _, _) = match self.slice(unit_id) {
            Ok(parts) => parts,
            Err(_) => return,
        };
        let ptr = slice.as_ptr();
        // Touch one byte per page so RSS reflects the tier immediately.
        const PAGE: usize = 4096;
        for offset in (0..slice.len()).step_by(PAGE) {
            // SAFETY: offset < slice.len(), reading a single byte from the map.
            let _ = unsafe { std::ptr::read_volatile(ptr.add(offset)) };
        }
        if slice.len() > 0 {
            let _ = unsafe { std::ptr::read_volatile(ptr.add(slice.len() - 1)) };
        }
    }

    /// Drop a byte range within one unit's mapping (chunked lm_head release).
    pub fn drop_range(&self, unit_id: &str, byte_start: usize, byte_len: usize) {
        let (slice, _, _) = match self.slice(unit_id) {
            Ok(parts) => parts,
            Err(_) => return,
        };
        if slice.is_empty() || byte_len == 0 {
            return;
        }
        let base = slice.as_ptr() as usize;
        let start = base + byte_start.min(slice.len());
        let end = (base + slice.len()).min(start.saturating_add(byte_len));
        if end <= start {
            return;
        }
        madvise_range(start, end - start, libc::MADV_DONTNEED);
    }

    /// Drop a unit's pages from the process (Ant evict on the native backend);
    /// the mapping itself stays valid and re-faults on the next access.
    pub fn drop_pages(&self, unit_id: &str) {
        let (slice, _, _) = match self.slice(unit_id) {
            Ok(parts) => parts,
            Err(_) => return,
        };
        if slice.is_empty() {
            return;
        }
        madvise_range(slice.as_ptr() as usize, slice.len(), libc::MADV_DONTNEED);
    }

    /// Drop every disk-tier unit of one layer after the layer's step completed.
    pub fn drop_disk_units_of_layer(&self, layer_id: i32) {
        for unit_id in self.layer_unit_ids(layer_id) {
            if self.tier(&unit_id) == Tier::Disk {
                self.drop_pages(&unit_id);
            }
        }
    }
}

/// madvise a page-aligned range; the kernel rounds as needed.
fn madvise_range(ptr: usize, len: usize, advice: i32) {
    if len == 0 {
        return;
    }
    let aligned_start = ptr & !4095;
    let aligned_end = (ptr + len).div_ceil(4096) * 4096;
    // SAFETY: the range stays inside the mapping; madvise only hints the kernel.
    unsafe {
        libc::madvise(
            aligned_start as *mut libc::c_void,
            aligned_end - aligned_start,
            advice,
        );
    }
}
