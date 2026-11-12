//! Ant: extremely fine-grained model memory optimization for Cheetah.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;
use cheetah::native::{self, AntFfiSettings};

pub const CHEETAH_UNITS_FILENAME: &str = "cheetah_units.json";
pub const CHEETAH_ANT_SETTINGS_FILENAME: &str = "cheetah_ant_settings.json";

#[derive(Debug, Clone)]
pub struct AntSettings {
    pub enabled: bool,
    pub vram_pool_bytes: u64,
    pub ram_budget_bytes: u64,
    pub disk_dir: Option<String>,
    pub resident_units: Vec<String>,
    pub eviction_policy: String,
    pub prefetch_stages: i32,
    pub allow_vram_swap: bool,
}

impl Default for AntSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            vram_pool_bytes: 8 * 1024 * 1024 * 1024,
            ram_budget_bytes: 16 * 1024 * 1024 * 1024,
            disk_dir: None,
            resident_units: Vec::new(),
            eviction_policy: "lru".to_string(),
            prefetch_stages: 2,
            allow_vram_swap: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AntUnit {
    pub unit_id: String,
    pub name: String,
    #[serde(default)] pub dtype: String,
    #[serde(default)] pub shape: Vec<u64>,
    #[serde(default)] pub bytes: u64,
    pub source_file: String,
    pub source_offset: u64,
    pub shard: String,
    #[serde(default)] pub layer_id: i32,
}

impl AntUnit {
    pub fn byte_size(&self) -> u64 {
        self.bytes
    }
}

/// Phase 2: per-layer residency breakdown derived from an atomic-unit plan.
/// A layer whose units span more than one tier is explicitly tracked as
/// `partial`, so "only part of a layer's weights reside in VRAM" becomes a
/// first-class, observable fact instead of an emergent side effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntLayerResidency {
    pub layer_id: i32,
    pub vram_units: u64,
    pub vram_bytes: u64,
    pub ram_units: u64,
    pub ram_bytes: u64,
    pub disk_units: u64,
    pub disk_bytes: u64,
}

impl AntLayerResidency {
    /// `full_vram`, `full_ram`, `full_disk`, or `partial` — partial means the
    /// layer's units span more than one tier.
    pub fn state(&self) -> &'static str {
        let in_vram = self.vram_units > 0;
        let in_ram = self.ram_units > 0;
        let on_disk = self.disk_units > 0;
        match (in_vram, in_ram, on_disk) {
            (true, false, false) => "full_vram",
            (false, true, false) => "full_ram",
            (false, false, true) => "full_disk",
            _ => "partial",
        }
    }
}

/// Phase 3: decide whether `unit_bytes` may swap into a pool that already holds
/// `used_bytes` under `budget_bytes`. When `allow_swap` is set, units are
/// evicted from `evictable` (in the given order) until the pool can hold the
/// swap-in; the budget is re-checked before each eviction. Returns:
/// - `Some(evictions)` when the swap may proceed after evicting the listed
///   units (empty when the pool already has room);
/// - `None` when the swap cannot fit (over budget with swapping disallowed, or
///   no eviction of any candidate leaves enough room).
pub fn ant_swap_fit(
    used_bytes: u64,
    budget_bytes: u64,
    unit_bytes: u64,
    allow_swap: bool,
    evictable: &[(u64, String)],
) -> Option<Vec<String>> {
    if budget_bytes == 0 {
        return None;
    }
    if used_bytes.saturating_add(unit_bytes) <= budget_bytes {
        return Some(Vec::new());
    }
    if !allow_swap {
        return None;
    }
    let mut evictions = Vec::new();
    let mut projected = used_bytes;
    for (bytes, unit_id) in evictable {
        if projected.saturating_add(unit_bytes) <= budget_bytes {
            break;
        }
        projected = projected.saturating_sub(*bytes);
        evictions.push(unit_id.clone());
    }
    if projected.saturating_add(unit_bytes) <= budget_bytes {
        Some(evictions)
    } else {
        None
    }
}

/// Keep only the requested unit ids that exist in the units manifest, warning
/// on (and dropping) unknown ids and duplicates.
pub fn ant_filter_known_units(requested: &[String], units: &[AntUnit]) -> Vec<String> {
    let known: HashSet<&str> = units.iter().map(|unit| unit.unit_id.as_str()).collect();
    let mut kept: Vec<String> = Vec::new();
    for requested_id in requested {
        let id = requested_id.trim();
        if id.is_empty() {
            continue;
        }
        if known.contains(id) {
            if !kept.iter().any(|existing| existing == id) {
                kept.push(id.to_string());
            }
        } else {
            eprintln!(
                "cheetah-ant: warning: unknown unit id '{id}' ignored (not in the units manifest)"
            );
        }
    }
    kept
}

pub struct Ant {
    handle: *mut std::os::raw::c_void,
    settings: AntSettings,
}

impl Ant {
    pub fn new() -> Result<Self, String> {
        let handle = native::ant_new()?;
        Ok(Self {
            handle,
            settings: AntSettings::default(),
        })
    }

    pub fn with_settings(settings: &AntSettings) -> Result<Self, String> {
        let mut ant = Self::new()?;
        ant.set_settings(settings)?;
        Ok(ant)
    }

    pub fn set_settings(&mut self, settings: &AntSettings) -> Result<(), String> {
        self.settings = settings.clone();
        let ffi = AntFfiSettings {
            enabled: i32::from(settings.enabled),
            vram_pool_bytes: settings.vram_pool_bytes,
            ram_budget_bytes: settings.ram_budget_bytes,
            disk_dir: std::ptr::null(),
            resident_units: std::ptr::null(),
            resident_units_count: 0,
            eviction_policy: std::ptr::null(),
            prefetch_stages: settings.prefetch_stages,
            allow_vram_swap: i32::from(settings.allow_vram_swap),
        };
        native::ant_set_settings(self.handle, &ffi)
    }

    pub fn settings(&self) -> &AntSettings {
        &self.settings
    }

    pub fn load_model(&self, dst_dir: &Path) -> Result<(), String> {
        native::ant_load_model(self.handle, dst_dir)
    }

    pub fn unit_tier(&self, dst_dir: &Path, unit_id: &str) -> Result<i32, String> {
        native::ant_unit_tier(self.handle, dst_dir, unit_id)
    }

    pub fn unit_pin(&self, unit_id: &str, pin: bool) -> Result<(), String> {
        native::ant_unit_pin(self.handle, unit_id, pin)
    }

    pub fn evict_unit(&self, dst_dir: &Path, unit_id: &str) -> Result<(), String> {
        native::ant_evict_unit(self.handle, dst_dir, unit_id)
    }

    pub fn page_in(&self, unit_id: &str, tier: i32) -> Result<(), String> {
        native::ant_page_in(self.handle, unit_id, tier)
    }

    pub fn model_is_initialized(&self, dst_dir: &Path) -> bool {
        native::ant_model_is_initialized(self.handle, dst_dir)
    }

    pub fn requires_init(&self) -> bool {
        native::ant_requires_init(self.handle)
    }

    pub fn not_initialized_warning(&self, model_id: &str, dst_dir: &Path) {
        native::ant_model_not_initialized_warning(model_id, dst_dir);
    }

    pub fn write_settings_for_model(&self, dst_dir: &Path, settings: &AntSettings) -> Result<(), String> {
        let path = dst_dir.join(CHEETAH_ANT_SETTINGS_FILENAME);
        let document = serde_json::json!({
            "enabled": settings.enabled,
            "vram_pool_bytes": settings.vram_pool_bytes,
            "ram_budget_bytes": settings.ram_budget_bytes,
            "disk_dir": settings.disk_dir,
            "resident_units": settings.resident_units,
            "eviction_policy": settings.eviction_policy,
            "prefetch_stages": settings.prefetch_stages,
            "allow_vram_swap": settings.allow_vram_swap,
        });
        let bytes = serde_json::to_vec_pretty(&document)
            .map_err(|error| format!("Failed to encode Ant settings: {error}"))?;
        std::fs::write(&path, bytes)
            .map_err(|error| format!("Failed to write {}: {error}", path.display()))
    }

    pub fn settings_for_model(&self, dst_dir: &Path) -> Result<AntSettings, String> {
        let path = dst_dir.join(CHEETAH_ANT_SETTINGS_FILENAME);
        let settings = if path.exists() {
            let bytes = std::fs::read(&path)
                .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|error| format!("Invalid Ant settings: {error}"))?;
            let mut settings = AntSettings::default();
            if let Some(object) = value.as_object() {
                if let Some(enabled) = object.get("enabled").and_then(serde_json::Value::as_bool) {
                    settings.enabled = enabled;
                }
                if let Some(vram) = object.get("vram_pool_bytes").and_then(serde_json::Value::as_u64) {
                    settings.vram_pool_bytes = vram;
                }
                if let Some(ram) = object.get("ram_budget_bytes").and_then(serde_json::Value::as_u64) {
                    settings.ram_budget_bytes = ram;
                }
                if let Some(disk) = object.get("disk_dir").and_then(serde_json::Value::as_str) {
                    settings.disk_dir = Some(disk.to_string());
                }
                if let Some(resident) = object.get("resident_units").and_then(serde_json::Value::as_array) {
                    settings.resident_units = resident
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string)
                        .collect();
                }
                if let Some(policy) = object.get("eviction_policy").and_then(serde_json::Value::as_str) {
                    settings.eviction_policy = policy.to_string();
                }
                if let Some(prefetch) = object.get("prefetch_stages").and_then(serde_json::Value::as_i64) {
                    settings.prefetch_stages = prefetch as i32;
                }
                if let Some(swap) = object.get("allow_vram_swap").and_then(serde_json::Value::as_bool) {
                    settings.allow_vram_swap = swap;
                }
            }
            settings
        } else {
            AntSettings::default()
        };
        Ok(settings)
    }

    /// Run an interactive preset configuration for a model directory and persist it.
    /// Ant: 交互式配置 Ant 加载策略并写入模型目录。
    pub fn run_interactive_setup(&self, model_dir: &Path) -> Result<AntSettings, String> {
        println!("Ant interactive setup for '{}':", model_dir.display());
        let units = self.units_for_model(model_dir).ok();
        if let Some(units) = &units {
            let total: u64 = units.iter().map(AntUnit::byte_size).sum();
            let layers = units
                .iter()
                .map(|unit| unit.layer_id)
                .collect::<HashSet<_>>()
                .len();
            println!(
                "  model units: {} across {} layer(s), total {}",
                units.len(),
                layers,
                crate::format_size(total)
            );
            println!(
                "  the VRAM pool fills units layer by layer; units that do not fit spill to RAM, then disk"
            );
        } else {
            println!(
                "  (units manifest not found in this directory; run `cheetah init` first for unit-level planning)"
            );
        }
        let mut settings = AntSettings::default();
        settings.enabled = true;
        settings.vram_pool_bytes = interactive_prompt_u64(
            &format!(
                "Ant interactive setup: VRAM pool size in bytes [default: {}]:",
                settings.vram_pool_bytes
            ),
            settings.vram_pool_bytes,
        )?;
        settings.ram_budget_bytes = interactive_prompt_u64(
            &format!(
                "Ant interactive setup: RAM budget in bytes [default: {}]:",
                settings.ram_budget_bytes
            ),
            settings.ram_budget_bytes,
        )?;
        let disk_dir = interactive_prompt_str(
            "Ant interactive setup: disk residency directory (empty for default):",
            "",
        )?;
        if !disk_dir.trim().is_empty() {
            settings.disk_dir = Some(disk_dir.trim().to_string());
        }
        let eviction_policy = interactive_prompt_str(
            &format!(
                "Ant interactive setup: eviction policy (lru/clock/clock2; default: {}):",
                settings.eviction_policy
            ),
            &settings.eviction_policy,
        )?;
        if !eviction_policy.trim().is_empty() {
            settings.eviction_policy = eviction_policy.trim().to_string();
        }
        let prefetch_stages = interactive_prompt_usize(
            &format!(
                "Ant interactive setup: prefetch stages (default: {}):",
                settings.prefetch_stages
            ),
            settings.prefetch_stages.max(0) as usize,
        )?;
        settings.prefetch_stages = prefetch_stages as i32;
        settings.allow_vram_swap = interactive_prompt_bool(
            &format!(
                "Ant interactive setup: allow VRAM swap (true/false; default: {}):",
                settings.allow_vram_swap
            ),
            settings.allow_vram_swap,
        )?;
        let pinned_input = interactive_prompt_str(
            "Ant interactive setup: pinned resident units (comma-separated unit ids; empty for none):",
            "",
        )?;
        let requested: Vec<String> = pinned_input
            .split(',')
            .map(|id| id.trim().to_string())
            .filter(|id| !id.is_empty())
            .collect();
        settings.resident_units = match units.as_deref() {
            Some(units) => ant_filter_known_units(&requested, units),
            None => requested,
        };
        let ant = Ant::with_settings(&settings)?;
        ant.write_settings_for_model(model_dir, &settings)?;
        println!(
            "Ant preset written to '{}'; Ant will run with these settings:",
            model_dir.join(CHEETAH_ANT_SETTINGS_FILENAME).display()
        );
        println!("  vram_pool_bytes:  {}", settings.vram_pool_bytes);
        println!("  ram_budget_bytes: {}", settings.ram_budget_bytes);
        println!(
            "  disk_dir:         {}",
            settings
                .disk_dir
                .as_deref()
                .unwrap_or("<model directory>")
        );
        println!("  eviction_policy:  {}", settings.eviction_policy);
        println!("  prefetch_stages:  {}", settings.prefetch_stages);
        println!("  allow_vram_swap:  {}", settings.allow_vram_swap);
        println!(
            "  resident_units:   {} pinned",
            settings.resident_units.len()
        );
        Ok(settings)
    }

    pub fn units_for_model(&self, dst_dir: &Path) -> Result<Vec<AntUnit>, String> {
        let path = dst_dir.join(CHEETAH_UNITS_FILENAME);
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
        let document: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Invalid Ant units manifest: {error}"))?;
        let array = document
            .as_array()
            .ok_or_else(|| "Ant units manifest must be a JSON array.".to_string())?;
        let mut units = Vec::with_capacity(array.len());
        for entry in array {
            let unit_id = entry
                .get("unit_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            if unit_id.is_empty() {
                continue;
            }
            let name = entry
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(&unit_id)
                .to_string();
            let dtype = entry
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("f32")
                .to_string();
            let shape = entry
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .map(|shape| {
                    shape
                        .iter()
                        .filter_map(serde_json::Value::as_u64)
                        .collect()
                })
                .unwrap_or_default();
            let bytes_size = entry
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let source_file = entry
                .get("source_file")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let source_offset = entry
                .get("source_offset")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let shard = entry
                .get("shard")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let layer_id = entry
                .get("layer_id")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0) as i32;
            units.push(AntUnit {
                unit_id,
                name,
                dtype,
                shape,
                bytes: bytes_size,
                source_file,
                source_offset,
                shard,
                layer_id,
            });
        }
        Ok(units)
    }

    /// Phase 2: summarize per-layer residency from the atomic-unit plan.
    /// Layers are ordered by layer_id; weights outside decoder layers
    /// (embeddings, norms, head) share layer_id 0 with the first decoder
    /// layer, matching the `cheetah_units.json` manifest.
    pub fn layer_residency_summary(
        units: &[AntUnit],
        plan: &BTreeMap<String, String>,
    ) -> Vec<AntLayerResidency> {
        let mut layers: BTreeMap<i32, AntLayerResidency> = BTreeMap::new();
        for unit in units {
            let tier = plan
                .get(&unit.unit_id)
                .map(String::as_str)
                .unwrap_or("vram");
            let entry = layers.entry(unit.layer_id).or_insert(AntLayerResidency {
                layer_id: unit.layer_id,
                vram_units: 0,
                vram_bytes: 0,
                ram_units: 0,
                ram_bytes: 0,
                disk_units: 0,
                disk_bytes: 0,
            });
            let (count, bytes) = match tier {
                "vram" => (&mut entry.vram_units, &mut entry.vram_bytes),
                "ram" => (&mut entry.ram_units, &mut entry.ram_bytes),
                _ => (&mut entry.disk_units, &mut entry.disk_bytes),
            };
            *count = count.saturating_add(1);
            *bytes = bytes.saturating_add(unit.byte_size());
        }
        layers.into_values().collect()
    }

    pub fn plan_residency(&self, units: &[AntUnit]) -> BTreeMap<String, String> {
        let mut plan = BTreeMap::new();
        if !self.settings.enabled {
            return plan;
        }
        let vram_pool = self.settings.vram_pool_bytes;
        let ram_budget = self.settings.ram_budget_bytes;
        let mut vram_used: u64 = 0;
        let mut ram_used: u64 = 0;
        let pinned: Vec<String> = self.settings.resident_units.clone();
        let pinned_set: HashSet<String> = pinned.iter().cloned().collect();
        let mut ordered: Vec<&AntUnit> = units.iter().collect();
        ordered.sort_by(|a, b| {
            let pa = pinned_set.contains(&a.unit_id);
            let pb = pinned_set.contains(&b.unit_id);
            if pa != pb {
                return pb.cmp(&pa);
            }
            // Phase 2: fill tiers layer by layer (in numeric layer order) so
            // the VRAM boundary falls inside one layer deterministically; that
            // boundary layer becomes a tracked partial layer.
            a.layer_id
                .cmp(&b.layer_id)
                .then_with(|| a.unit_id.cmp(&b.unit_id))
        });
        for unit in ordered {
            let bytes = unit.byte_size();
            let tier = if pinned_set.contains(&unit.unit_id) {
                "vram"
            } else if bytes > vram_pool {
                "disk"
            } else if vram_used + bytes <= vram_pool {
                vram_used += bytes;
                "vram"
            } else if ram_used + bytes <= ram_budget {
                ram_used += bytes;
                "ram"
            } else {
                "disk"
            };
            plan.insert(unit.unit_id.clone(), tier.to_string());
        }
        plan
    }
}

fn interactive_prompt(prompt: &str) -> Result<String, String> {
    print!("{prompt}");
    let mut stdout = std::io::stdout();
    let _ = stdout.flush();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|error| format!("Failed to read interactive setting: {error}"))?;
    Ok(line.trim().to_string())
}

fn interactive_prompt_u64(prompt: &str, default: u64) -> Result<u64, String> {
    let answer = interactive_prompt(prompt)?;
    if answer.is_empty() {
        return Ok(default);
    }
    answer
        .parse::<u64>()
        .map_err(|_| format!("{prompt} expects a non-negative integer; received '{answer}'."))
}

fn interactive_prompt_usize(prompt: &str, default: usize) -> Result<usize, String> {
    let answer = interactive_prompt(prompt)?;
    if answer.is_empty() {
        return Ok(default);
    }
    answer
        .parse::<usize>()
        .map_err(|_| format!("{prompt} expects a non-negative integer; received '{answer}'."))
}

fn interactive_prompt_bool(prompt: &str, default: bool) -> Result<bool, String> {
    let answer = interactive_prompt(prompt)?;
    if answer.is_empty() {
        return Ok(default);
    }
    match answer.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{prompt} expects a boolean value (true/false); received '{answer}'."
        )),
    }
}

fn interactive_prompt_str(prompt: &str, default: &str) -> Result<String, String> {
    let answer = interactive_prompt(prompt)?;
    if answer.is_empty() {
        return Ok(default.to_string());
    }
    Ok(answer)
}

impl Drop for Ant {
    fn drop(&mut self) {
        native::ant_delete(self.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(unit_id: &str, bytes: u64) -> AntUnit {
        AntUnit {
            unit_id: unit_id.to_string(),
            name: unit_id.to_string(),
            dtype: "F32".to_string(),
            shape: vec![1],
            bytes,
            source_file: String::new(),
            source_offset: 0,
            shard: String::new(),
            layer_id: 0,
        }
    }

    fn layered_unit(unit_id: &str, bytes: u64, layer_id: i32) -> AntUnit {
        AntUnit {
            layer_id,
            ..unit(unit_id, bytes)
        }
    }

    #[test]
    fn plan_residency_assigns_vram_then_ram_then_disk() {
        let ant = Ant::with_settings(&AntSettings::default()).expect("ant manager");
        let units = vec![
            unit("model.layers.0.mlp.gate_proj", 1024),
            unit("model.layers.0.mlp.up_proj", 1024),
        ];
        let plan = ant.plan_residency(&units);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan["model.layers.0.mlp.gate_proj"], "vram");
        assert_eq!(plan["model.layers.0.mlp.up_proj"], "vram");
    }

    #[test]
    fn plan_residency_fills_layers_in_order_and_tracks_partial_layers() {
        // Phase 2: the VRAM pool fills layer by layer in numeric order, so the
        // boundary layer keeps only part of its units in VRAM.
        let mut settings = AntSettings::default();
        settings.vram_pool_bytes = 1800;
        settings.ram_budget_bytes = 10_000;
        let ant = Ant::with_settings(&settings).expect("ant manager");
        let units = vec![
            layered_unit("model.layers.10.mlp.gate_proj", 600, 10),
            layered_unit("model.layers.2.mlp.up_proj", 600, 2),
            layered_unit("model.layers.2.mlp.gate_proj", 600, 2),
            layered_unit("model.layers.10.mlp.up_proj", 600, 10),
            layered_unit("model.layers.1.mlp.up_proj", 600, 1),
        ];
        let plan = ant.plan_residency(&units);
        // Layer 1 first (600), then layer 2 (1200) fills the pool exactly; the
        // numeric ordering must not be confused by lexicographic "10" < "2".
        assert_eq!(plan["model.layers.1.mlp.up_proj"], "vram");
        assert_eq!(plan["model.layers.2.mlp.gate_proj"], "vram");
        assert_eq!(plan["model.layers.2.mlp.up_proj"], "vram");
        assert_eq!(plan["model.layers.10.mlp.gate_proj"], "ram");
        assert_eq!(plan["model.layers.10.mlp.up_proj"], "ram");
        let summary = Ant::layer_residency_summary(&units, &plan);
        let layer = |layer_id: i32| {
            summary
                .iter()
                .find(|entry| entry.layer_id == layer_id)
                .expect("layer summary entry")
                .clone()
        };
        assert_eq!(layer(1).state(), "full_vram");
        assert_eq!(layer(2).state(), "full_vram");
        assert_eq!(layer(10).state(), "full_ram");
        // Now shrink the pool mid-layer: layer 2 must become partial.
        settings.vram_pool_bytes = 1500;
        let ant = Ant::with_settings(&settings).expect("ant manager");
        let plan = ant.plan_residency(&units);
        assert_eq!(plan["model.layers.1.mlp.up_proj"], "vram");
        assert_eq!(plan["model.layers.2.mlp.gate_proj"], "vram");
        assert_eq!(plan["model.layers.2.mlp.up_proj"], "ram");
        let summary = Ant::layer_residency_summary(&units, &plan);
        let layer_2 = summary
            .iter()
            .find(|entry| entry.layer_id == 2)
            .expect("layer summary entry");
        assert_eq!(layer_2.state(), "partial");
        assert_eq!(layer_2.vram_units, 1);
        assert_eq!(layer_2.vram_bytes, 600);
        assert_eq!(layer_2.ram_units, 1);
        assert_eq!(layer_2.ram_bytes, 600);
    }

    #[test]
    fn ant_filter_known_units_drops_unknown_and_duplicates() {
        let units = vec![
            unit("model.embed_tokens.weight", 272),
            layered_unit("model.layers.0.mlp.gate_proj", 600, 0),
            layered_unit("model.layers.1.mlp.gate_proj", 600, 1),
        ];
        let kept = ant_filter_known_units(
            &[
                "model.layers.0.mlp.gate_proj".to_string(),
                "model.layers.0.mlp.gate_proj".to_string(),
                " model.embed_tokens.weight ".to_string(),
                "model.missing.weight".to_string(),
                "".to_string(),
            ],
            &units,
        );
        assert_eq!(
            kept,
            vec![
                "model.layers.0.mlp.gate_proj".to_string(),
                "model.embed_tokens.weight".to_string(),
            ]
        );
    }

    #[test]
    fn ant_swap_fit_decision_rules() {
        let evictable = vec![
            (300u64, "b".to_string()),
            (500, "a".to_string()),
        ];
        // Fits without eviction.
        assert_eq!(
            ant_swap_fit(500, 2000, 300, true, &evictable),
            Some(Vec::new())
        );
        // Over budget with swapping disallowed.
        assert_eq!(ant_swap_fit(500, 1000, 600, false, &evictable), None);
        // Needs one eviction; the budget is re-checked before each pick.
        assert_eq!(
            ant_swap_fit(500, 1000, 600, true, &evictable),
            Some(vec!["b".to_string()])
        );
        // Needs both evictions, cheapest first, until the pool fits.
        assert_eq!(
            ant_swap_fit(500, 1200, 1100, true, &evictable),
            Some(vec!["b".to_string(), "a".to_string()])
        );
        // The unit itself exceeds the budget: no eviction can ever fit.
        assert_eq!(ant_swap_fit(500, 700, 800, true, &evictable), None);
        // Zero budget never admits a swap.
        assert_eq!(ant_swap_fit(0, 0, 1, true, &evictable), None);
    }

    #[test]
    fn plan_residency_respects_disabled_ant() {
        let mut settings = AntSettings::default();
        settings.enabled = false;
        let ant = Ant::with_settings(&settings).expect("ant manager");
        let units = vec![unit("model.layers.0.mlp.gate_proj", 1024)];
        let plan = ant.plan_residency(&units);
        assert!(plan.is_empty());
    }

    #[test]
    fn settings_for_model_reads_user_managed_file() {
        let ant = Ant::with_settings(&AntSettings::default()).expect("ant manager");
        let dir = std::env::temp_dir().join(format!("cheetah-ant-settings-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(CHEETAH_ANT_SETTINGS_FILENAME);
        std::fs::write(&path, r#"{"enabled":false,"vram_pool_bytes":1024}"#).unwrap();
        let settings = ant.settings_for_model(&dir).expect("settings");
        assert!(!settings.enabled);
        assert_eq!(settings.vram_pool_bytes, 1024);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ant_lifecycle_against_real_prepared_model() {
        let ant = Ant::with_settings(&AntSettings::default()).expect("ant manager");
        let dst = std::path::Path::new("/home/benny/models/qwen-0.5b-prepared");
        if !dst.exists() {
            // The lifecycle test needs a locally prepared model; skip elsewhere.
            eprintln!("skipping ant lifecycle test: {} is not prepared", dst.display());
            return;
        }
        assert!(ant.load_model(dst).is_ok());
        assert!(ant.model_is_initialized(dst));
        let units = ant.units_for_model(dst).expect("units");
        assert_eq!(units.len(), 290);
        let plan = ant.plan_residency(&units);
        assert_eq!(plan.len(), 290);
        let tier = ant.unit_tier(dst, "model.embed_tokens.weight").expect("tier");
        assert_eq!(tier, 0);
        ant.unit_pin("model.embed_tokens.weight", true).expect("pin");
        assert_eq!(
            ant.unit_tier(dst, "model.embed_tokens.weight").expect("tier"),
            1
        );
        ant.evict_unit(dst, "model.embed_tokens.weight").expect("evict");
        assert_eq!(
            ant.unit_tier(dst, "model.embed_tokens.weight").expect("tier"),
            2
        );
    }
}
