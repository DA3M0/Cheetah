//! Safe Rust adapters for C APIs declared by the root-level C headers.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Mutex;

static INIT_LOCK: Mutex<()> = Mutex::new(());

struct InitProgress {
    bars: MultiProgress,
    parse_bar: Option<ProgressBar>,
    write_bar: Option<ProgressBar>,
}

impl InitProgress {
    fn new() -> Self {
        Self {
            bars: MultiProgress::new(),
            parse_bar: None,
            write_bar: None,
        }
    }

    fn style() -> ProgressStyle {
        ProgressStyle::with_template("{pos:>7}/{len:<7} [{bar:40.cyan}] {msg}")
            .expect("valid indicatif style")
            .progress_chars("##-")
    }

    fn start_bar(&mut self, total: usize, message: &'static str) -> ProgressBar {
        let bar = self.bars.add(ProgressBar::new(total as u64));
        bar.set_style(Self::style());
        bar.set_message(message);
        bar.set_position(0);
        bar
    }
}

unsafe extern "C" {
    fn cheetah_init_model_with_progress(
        src_dir: *const c_char,
        dst_dir: *const c_char,
        progress_fn: extern "C" fn(*const c_char, usize, usize, *mut c_void),
        user_data: *mut c_void,
    ) -> i32;
}

#[repr(C)]
pub struct AntFfiSettings {
    pub enabled: i32,
    pub vram_pool_bytes: u64,
    pub ram_budget_bytes: u64,
    pub disk_dir: *const std::os::raw::c_char,
    pub resident_units: *const *const std::os::raw::c_char,
    pub resident_units_count: usize,
    pub eviction_policy: *const std::os::raw::c_char,
    pub prefetch_stages: i32,
    pub allow_vram_swap: i32,
}

unsafe extern "C" {
    fn cheetah_ant_new() -> *mut std::os::raw::c_void;
    fn cheetah_ant_delete(ant: *mut std::os::raw::c_void);
    fn cheetah_ant_set_settings(ant: *mut std::os::raw::c_void, settings: *const AntFfiSettings) -> i32;
    fn cheetah_ant_load_model(ant: *mut std::os::raw::c_void, dst_dir: *const std::os::raw::c_char) -> i32;
    fn cheetah_ant_unit_tier(ant: *const std::os::raw::c_void, dst_dir: *const std::os::raw::c_char, unit_id: *const std::os::raw::c_char, tier_out: *mut i32) -> i32;
    fn cheetah_ant_unit_pin(ant: *const std::os::raw::c_void, unit_id: *const std::os::raw::c_char, pin: i32) -> i32;
    fn cheetah_ant_evict_unit(ant: *const std::os::raw::c_void, dst_dir: *const std::os::raw::c_char, unit_id: *const std::os::raw::c_char) -> i32;
    fn cheetah_ant_page_in(ant: *const std::os::raw::c_void, unit_id: *const std::os::raw::c_char, tier: i32) -> i32;
    fn cheetah_ant_model_is_initialized(ant: *const std::os::raw::c_void, dst_dir: *const std::os::raw::c_char) -> i32;
    fn cheetah_ant_requires_init(ant: *const std::os::raw::c_void) -> i32;
    fn cheetah_ant_model_not_initialized_warning(model_id: *const std::os::raw::c_char, dst_dir: *const std::os::raw::c_char);
}

pub fn ant_new() -> Result<*mut std::os::raw::c_void, String> {
    let ptr = unsafe { cheetah_ant_new() };
    if ptr.is_null() {
        Err("Failed to create the Ant memory manager.".to_string())
    } else {
        Ok(ptr)
    }
}

pub fn ant_delete(ant: *mut std::os::raw::c_void) {
    unsafe { cheetah_ant_delete(ant) };
}

pub fn ant_set_settings(ant: *mut std::os::raw::c_void, settings: &AntFfiSettings) -> Result<(), String> {
    let status = unsafe { cheetah_ant_set_settings(ant, settings) };
    if status != 0 {
        return Err("Cheetah Ant set settings failed.".to_string());
    }
    Ok(())
}

pub fn ant_load_model(ant: *mut std::os::raw::c_void, dst_dir: &Path) -> Result<(), String> {
    let dst_dir_c = path_to_cstring(dst_dir)?;
    let status = unsafe { cheetah_ant_load_model(ant, dst_dir_c.as_ptr()) };
    if status != 0 {
        return Err("Cheetah Ant model load failed.".to_string());
    }
    Ok(())
}

pub fn ant_unit_tier(ant: *mut std::os::raw::c_void, dst_dir: &Path, unit_id: &str) -> Result<i32, String> {
    let dst_dir_c = path_to_cstring(dst_dir)?;
    let unit_id_c = CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let mut tier_out: std::os::raw::c_int = 0;
    let status = unsafe { cheetah_ant_unit_tier(ant, dst_dir_c.as_ptr(), unit_id_c.as_ptr(), &mut tier_out) };
    if status != 0 {
        return Err("Cheetah Ant unit tier query failed.".to_string());
    }
    Ok(tier_out as i32)
}

pub fn ant_unit_pin(ant: *mut std::os::raw::c_void, unit_id: &str, pin: bool) -> Result<(), String> {
    let unit_id_c = CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let status = unsafe { cheetah_ant_unit_pin(ant, unit_id_c.as_ptr(), if pin { 1i32 } else { 0i32 }) };
    if status != 0 {
        return Err("Cheetah Ant unit pin failed.".to_string());
    }
    Ok(())
}

pub fn ant_evict_unit(ant: *mut std::os::raw::c_void, dst_dir: &Path, unit_id: &str) -> Result<(), String> {
    let dst_dir_c = path_to_cstring(dst_dir)?;
    let unit_id_c = CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let status = unsafe { cheetah_ant_evict_unit(ant, dst_dir_c.as_ptr(), unit_id_c.as_ptr()) };
    if status != 0 {
        return Err("Cheetah Ant unit evict failed.".to_string());
    }
    Ok(())
}

pub fn ant_page_in(ant: *mut std::os::raw::c_void, unit_id: &str, tier: i32) -> Result<(), String> {
    let unit_id_c = CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let status = unsafe { cheetah_ant_page_in(ant, unit_id_c.as_ptr(), tier) };
    if status != 0 {
        return Err("Cheetah Ant unit page-in failed.".to_string());
    }
    Ok(())
}

pub fn ant_model_is_initialized(ant: *mut std::os::raw::c_void, dst_dir: &Path) -> bool {
    let dst_dir_c = path_to_cstring(dst_dir);
    let dst_ptr = dst_dir_c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null());
    let status = unsafe { cheetah_ant_model_is_initialized(ant, dst_ptr) };
    status == 1
}

pub fn ant_requires_init(ant: *mut std::os::raw::c_void) -> bool {
    let status = unsafe { cheetah_ant_requires_init(ant) };
    status != 0
}

pub fn ant_model_not_initialized_warning(model_id: &str, dst_dir: &Path) {
    let model_id_c = CString::new(model_id).ok();
    let dst_dir_c = path_to_cstring(dst_dir).ok();
    if let (Some(model_id_c), Some(dst_dir_c)) = (model_id_c, dst_dir_c) {
        unsafe { cheetah_ant_model_not_initialized_warning(model_id_c.as_ptr(), dst_dir_c.as_ptr()) };
    }
}

extern "C" fn report_init_progress(
    phase: *const c_char,
    current: usize,
    total: usize,
    user_data: *mut c_void,
) {
    if phase.is_null() || user_data.is_null() {
        return;
    }

    let progress = unsafe { &mut *user_data.cast::<InitProgress>() };
    let phase = unsafe { CStr::from_ptr(phase) }.to_string_lossy();

    match phase.as_ref() {
        // 解析源 safetensors：按源文件数更新。
        "parsing" => {
            let mut parse_bar = progress.parse_bar.take();
            if parse_bar.is_none() {
                parse_bar = Some(progress.start_bar(total, "Parsing safetensors files..."));
            }
            if let Some(bar) = parse_bar.as_mut() {
                bar.set_position(current as u64);
            }
            progress.parse_bar = parse_bar;
        }
        // 写入预处理分片：结束解析条，再创建写入条。
        "writing" => {
            let mut parse_bar = progress.parse_bar.take();
            if let Some(bar) = parse_bar.as_mut() {
                bar.finish_with_message("Parsing complete");
            }
            let write_bar = progress.start_bar(total, "Writing prepared shards...");
            progress.write_bar = Some(write_bar);
            if let Some(bar) = progress.write_bar.as_mut() {
                bar.set_position(current as u64);
            }
        }
        // 权重准备：保持当前条的状态，避免 1/1 短暂闪烁。
        "preparing" => {
            let mut parse_bar = progress.parse_bar.take();
            if let Some(bar) = parse_bar.as_mut() {
                bar.finish_with_message("Preparing weights...");
            }
        }
        // 完成：结束写入条并给出最终状态。
        "complete" => {
            let mut write_bar = progress.write_bar.take();
            if let Some(bar) = write_bar.as_mut() {
                bar.finish_with_message("Initialization complete");
            }
            progress.write_bar = write_bar;

            if progress.write_bar.is_none() {
                let mut parse_bar = progress.parse_bar.take();
                if let Some(bar) = parse_bar.as_mut() {
                    bar.finish_with_message("Initialization complete");
                }
                progress.parse_bar = parse_bar;
            }
        }
        _ => {}
    }
}

fn path_to_cstring(path: &Path) -> Result<CString, String> {
    let path = path
        .to_str()
        .ok_or_else(|| format!("Path is not valid UTF-8: {}", path.display()))?;
    CString::new(path).map_err(|_| "Path contains an interior NUL byte.".to_string())
}

pub struct ModelMetadata {
    pub chat_template: &'static str,
    pub stop_markers: &'static [&'static str],
}

/// Run the C static model-preparation stage and write its result to `dst_dir`.
pub fn init_model(src_dir: &Path, dst_dir: &Path) -> Result<(), String> {
    init_model_with_metadata(src_dir, dst_dir, None)
}

/// Run native preparation, then override chat metadata when the Rust registry recognizes the model.
pub fn init_model_with_metadata(
    src_dir: &Path,
    dst_dir: &Path,
    metadata: Option<ModelMetadata>,
) -> Result<(), String> {
    let dst_dir_path = dst_dir.to_path_buf();
    let src_dir = path_to_cstring(src_dir)?;
    let dst_dir = path_to_cstring(dst_dir)?;
    let _guard = INIT_LOCK
        .lock()
        .map_err(|_| "Native model initializer lock is poisoned.".to_string())?;

    let mut progress = InitProgress::new();
    let result = unsafe {
        cheetah_init_model_with_progress(
            src_dir.as_ptr(),
            dst_dir.as_ptr(),
            report_init_progress,
            (&mut progress as *mut InitProgress).cast(),
        )
    };

    if result == 0 {
        // The C path above normally ends with the "complete" callback; finish
        // any bar that is still open so the final status line is always shown.
        let mut write_bar = progress.write_bar.take();
        if let Some(bar) = write_bar.as_mut() {
            bar.finish_with_message("Initialization complete");
        }
        progress.write_bar = write_bar;

        let result = if let Some(metadata) = metadata {
            update_manifest_metadata(&dst_dir_path, metadata)
        } else {
            Ok(())
        };

        if result.is_ok() {
            for bar in [&mut progress.parse_bar, &mut progress.write_bar] {
                if let Some(b) = bar {
                    if !b.is_finished() {
                        b.finish_with_message("Initialization complete");
                    }
                }
            }
        } else {
            abandon_bars(&mut progress);
        }

        result
    } else {
        abandon_bars(&mut progress);
        Err("Cheetah model preparation failed; see the native diagnostic above.".into())
    }
}

fn abandon_bars(progress: &mut InitProgress) {
    for bar in [&mut progress.parse_bar, &mut progress.write_bar] {
        if let Some(b) = bar {
            b.abandon_with_message("Initialization failed");
        }
    }
}

fn update_manifest_metadata(dst_dir: &Path, metadata: ModelMetadata) -> Result<(), String> {
    let manifest_path = dst_dir.join("cheetah_manifest.json");
    let bytes = fs::read(&manifest_path)
        .map_err(|error| format!("Failed to read {}: {error}", manifest_path.display()))?;
    let mut manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid Cheetah manifest: {error}"))?;
    let object = manifest
        .as_object_mut()
        .ok_or_else(|| "Cheetah manifest root must be a JSON object.".to_string())?;
    let mut stop_markers = object
        .get("stop_markers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for marker in metadata.stop_markers {
        if !stop_markers.iter().any(|existing| existing == marker) {
            stop_markers.push((*marker).to_owned());
        }
    }
    object.insert(
        "chat_template".into(),
        serde_json::Value::String(metadata.chat_template.into()),
    );
    object.insert(
        "stop_markers".into(),
        serde_json::to_value(stop_markers)
            .map_err(|error| format!("Failed to encode model stop markers: {error}"))?,
    );

    let temporary_path = manifest_path.with_extension(format!("json.tmp.{}", std::process::id()));
    let updated = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("Failed to encode Cheetah manifest: {error}"))?;
    fs::write(&temporary_path, updated)
        .map_err(|error| format!("Failed to write {}: {error}", temporary_path.display()))?;
    if let Err(error) = fs::rename(&temporary_path, &manifest_path) {
        let _ = fs::remove_file(&temporary_path);
        return Err(format!(
            "Failed to replace {}: {error}",
            manifest_path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn initializer_progress_callback_tracks_phase_and_position() {
        let mut progress = InitProgress::new();
        let user_data = (&mut progress as *mut InitProgress).cast();
        let parsing = CString::new("parsing").unwrap();
        report_init_progress(parsing.as_ptr(), 2, 4, user_data);
        assert_eq!(
            progress.parse_bar.as_ref().map(|b| b.length()),
            Some(Some(4))
        );
        assert_eq!(progress.parse_bar.as_ref().map(|b| b.position()), Some(2));

        let writing = CString::new("writing").unwrap();
        report_init_progress(writing.as_ptr(), 1, 3, user_data);
        assert!(progress.parse_bar.is_none());
        assert_eq!(
            progress.write_bar.as_ref().map(|b| b.length()),
            Some(Some(3))
        );
        assert_eq!(progress.write_bar.as_ref().map(|b| b.position()), Some(1));
    }

    fn temporary_directory() -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cheetah-native-manifest-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn initializer_ffi_reports_native_failure() {
        let error = init_model(
            Path::new("/cheetah-test-path-that-does-not-exist"),
            Path::new("/tmp/cheetah-native-ffi-test"),
        )
        .unwrap_err();
        assert!(error.contains("native diagnostic"));
    }

    #[test]
    fn registry_metadata_updates_manifest_without_dropping_other_fields() {
        let directory = temporary_directory();
        let manifest_path = directory.join("cheetah_manifest.json");
        fs::write(
            &manifest_path,
            r#"{"model_type":"custom","stop_markers":["<|detected|>"],"shards":[]}"#,
        )
        .unwrap();

        update_manifest_metadata(
            &directory,
            ModelMetadata {
                chat_template: "custom-chat",
                stop_markers: &["<|stop|>", "</s>"],
            },
        )
        .unwrap();

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["model_type"], "custom");
        assert_eq!(manifest["chat_template"], "custom-chat");
        assert_eq!(
            manifest["stop_markers"],
            serde_json::json!(["<|detected|>", "<|stop|>", "</s>"])
        );
        fs::remove_dir_all(directory).unwrap();
    }
}

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use std::os::raw::{c_int, c_void};
    use std::sync::Mutex;

    static VULKAN_LOCK: Mutex<()> = Mutex::new(());

    unsafe extern "C" {
        fn init_vulkan_device() -> c_int;
        fn shutdown_vulkan_device();
        fn vk_qwen_prepare() -> c_int;
        fn vk_matmul_checked(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn vk_matmul_transposed_checked(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn vk_matmul_bf16_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn vk_matmul_bf16_transposed_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn cheetah_vulkan_weight_upload_bf16(weights: *const u16, element_count: u64) -> u64;
        fn cheetah_vulkan_weight_release(handle: u64) -> c_int;
        fn vk_matmul_bf16_transposed_resident_checked(
            handle: u64,
            a: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn vk_matmul_fp16_transposed_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        fn vk_qwen_rms_norm(
            input: *const f32,
            weight: *const c_void,
            weight_format: u32,
            output: *mut f32,
            rows: u32,
            width: u32,
            epsilon: f32,
        ) -> c_int;
        fn vk_qwen_layer_norm(
            input: *const f32,
            weight: *const c_void,
            bias: *const c_void,
            weight_format: u32,
            output: *mut f32,
            rows: u32,
            width: u32,
            epsilon: f32,
        ) -> c_int;
        fn vk_qwen_elementwise(
            a: *const f32,
            b: *const f32,
            output: *mut f32,
            count: usize,
            operation: u32,
        ) -> c_int;
        fn vk_qwen_rotary(
            query: *const f32,
            keys: *const f32,
            query_output: *mut f32,
            key_output: *mut f32,
            sequence: u32,
            offset: u32,
            query_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            rotary_dim: u32,
            theta: f32,
        ) -> c_int;
        fn vk_qwen_attention(
            query: *const f32,
            keys: *const f32,
            values: *const f32,
            output: *mut f32,
            sequence: u32,
            offset: u32,
            cache_len: u32,
            query_heads: u32,
            kv_heads: u32,
            head_dim: u32,
            sliding_window: u32,
        ) -> c_int;
    }

    /// Initialize the C Vulkan backend and its mapped memory pool.
    pub fn initialize() -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe { init_vulkan_device() };
        if result == 0 {
            Ok(())
        } else {
            Err("Vulkan backend initialization failed; see the native diagnostic above.".into())
        }
    }

    /// Release the Vulkan backend. No allocations or submitted work may remain.
    pub fn shutdown() -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        unsafe { shutdown_vulkan_device() }
        Ok(())
    }

    /// Validate that the Qwen2 compute shader and pipeline are available.
    pub fn prepare() -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        check_native_result(unsafe { vk_qwen_prepare() }, "Vulkan Qwen2 pipeline setup")
    }

    /// Compute row-major `C[M,N] = A[M,K] * B[K,N]` using the Vulkan backend.
    pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), b.len(), |dims, output| unsafe {
            vk_matmul_checked(a.as_ptr(), b.as_ptr(), output, dims[0], dims[1], dims[2])
        })
    }

    /// Multiply an FP32 activation matrix by a row-major BF16 weight matrix.
    pub fn matmul_bf16(
        a: &[f32],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), b.len(), |dims, output| unsafe {
            vk_matmul_bf16_checked(a.as_ptr(), b.as_ptr(), output, dims[0], dims[1], dims[2])
        })
    }

    /// Multiply an FP32 activation matrix by the transpose of a BF16 weight matrix.
    pub fn matmul_bf16_transposed(
        a: &[f32],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), b.len(), |dims, output| unsafe {
            vk_matmul_bf16_transposed_checked(
                a.as_ptr(),
                b.as_ptr(),
                output,
                dims[0],
                dims[1],
                dims[2],
            )
        })
    }

    pub fn matmul_fp16_transposed(
        a: &[f32],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), b.len(), |dims, output| unsafe {
            vk_matmul_fp16_transposed_checked(
                a.as_ptr(),
                b.as_ptr(),
                output,
                dims[0],
                dims[1],
                dims[2],
            )
        })
    }

    pub fn matmul_f32_transposed(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), b.len(), |dims, output| unsafe {
            vk_matmul_transposed_checked(a.as_ptr(), b.as_ptr(), output, dims[0], dims[1], dims[2])
        })
    }

    /// Upload one BF16 weight tensor into the resident bucket (device-local
    /// memory when available, a persistent host-coherent allocation otherwise).
    /// Returns the resident-weight handle, or None when the upload cannot be
    /// served and the caller should keep the per-step streaming path.
    pub fn weight_upload_bf16(weights: &[u16]) -> Option<u64> {
        let guard = VULKAN_LOCK.lock();
        let _guard = guard.unwrap_or_else(|poisoned| poisoned.into_inner());
        let handle = unsafe {
            cheetah_vulkan_weight_upload_bf16(weights.as_ptr(), weights.len() as u64)
        };
        if handle == 0 { None } else { Some(handle) }
    }

    /// Release a resident weight bucket (device-local allocation or the
    /// persistent host-coherent one, depending on where it was uploaded).
    pub fn weight_release(handle: u64) {
        let guard = VULKAN_LOCK.lock();
        let _guard = guard.unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe { cheetah_vulkan_weight_release(handle) };
    }

    /// C[M,N] = A[M,K] * weight^T over a resident weight buffer: the weight
    /// stays bound in device/host persistent memory and only A/C pass through
    /// the pool. Caller must hold a valid upload handle (weight_upload_bf16).
    pub fn matmul_bf16_transposed_resident(
        handle: u64,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_with(m, k, n, a.len(), n, |dims, output| unsafe {
            vk_matmul_bf16_transposed_resident_checked(
                handle,
                a.as_ptr(),
                output,
                dims[0],
                dims[1],
                dims[2],
            )
        })
    }

    pub fn rms_norm_f32(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        rms_norm(
            input,
            weight.as_ptr().cast(),
            weight.len(),
            0,
            rows,
            width,
            epsilon,
        )
    }

    pub fn rms_norm_bf16(
        input: &[f32],
        weight: &[u16],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        rms_norm(
            input,
            weight.as_ptr().cast(),
            weight.len(),
            1,
            rows,
            width,
            epsilon,
        )
    }

    pub fn layer_norm_f32(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        layer_norm(
            input,
            weight.as_ptr().cast(),
            bias.as_ptr().cast(),
            weight.len(),
            bias.len(),
            0,
            rows,
            width,
            epsilon,
        )
    }

    pub fn layer_norm_fp16(
        input: &[f32],
        weight: &[u16],
        bias: &[u16],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        layer_norm(
            input,
            weight.as_ptr().cast(),
            bias.as_ptr().cast(),
            weight.len(),
            bias.len(),
            2,
            rows,
            width,
            epsilon,
        )
    }

    pub fn layer_norm_bf16(
        input: &[f32],
        weight: &[u16],
        bias: &[u16],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        layer_norm(
            input,
            weight.as_ptr().cast(),
            bias.as_ptr().cast(),
            weight.len(),
            bias.len(),
            1,
            rows,
            width,
            epsilon,
        )
    }

    fn layer_norm(
        input: &[f32],
        weight: *const c_void,
        bias: *const c_void,
        weight_len: usize,
        bias_len: usize,
        format: u32,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| "Layer normalization dimensions overflow usize.".to_string())?;
        if rows == 0
            || width == 0
            || input.len() != expected
            || weight_len != width
            || bias_len != width
            || !epsilon.is_finite()
            || epsilon < 0.0
        {
            return Err("Invalid Vulkan layer normalization input.".into());
        }
        let (rows, width) = (to_u32(rows)?, to_u32(width)?);
        let mut output = vec![0.0; expected];
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_qwen_layer_norm(
                input.as_ptr(),
                weight,
                bias,
                format,
                output.as_mut_ptr(),
                rows,
                width,
                epsilon,
            )
        };
        check_native_result(result, "Vulkan layer normalization")?;
        Ok(output)
    }

    pub fn rms_norm_fp16(
        input: &[f32],
        weight: &[u16],
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        rms_norm(
            input,
            weight.as_ptr().cast(),
            weight.len(),
            2,
            rows,
            width,
            epsilon,
        )
    }

    fn rms_norm(
        input: &[f32],
        weight: *const c_void,
        weight_len: usize,
        format: u32,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| "RMS normalization dimensions overflow usize.".to_string())?;
        if rows == 0
            || width == 0
            || input.len() != expected
            || weight_len != width
            || !epsilon.is_finite()
            || epsilon < 0.0
        {
            return Err("Invalid Vulkan RMS normalization input.".into());
        }
        let (rows, width) = (to_u32(rows)?, to_u32(width)?);
        let mut output = vec![0.0; expected];
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_qwen_rms_norm(
                input.as_ptr(),
                weight,
                format,
                output.as_mut_ptr(),
                rows,
                width,
                epsilon,
            )
        };
        check_native_result(result, "Vulkan RMS normalization")?;
        Ok(output)
    }

    pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        elementwise(a, b, 1)
    }

    pub fn silu_mul(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
        elementwise(gate, up, 2)
    }

    pub fn gelu_mul(gate: &[f32], up: &[f32], approximate: bool) -> Result<Vec<f32>, String> {
        elementwise(gate, up, if approximate { 6 } else { 5 })
    }

    fn elementwise(a: &[f32], b: &[f32], operation: u32) -> Result<Vec<f32>, String> {
        if a.is_empty() || a.len() != b.len() {
            return Err("Vulkan elementwise operands must have equal non-zero lengths.".into());
        }
        let mut output = vec![0.0; a.len()];
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_qwen_elementwise(
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
                a.len(),
                operation,
            )
        };
        check_native_result(result, "Vulkan elementwise operation")?;
        Ok(output)
    }

    pub fn rotary(
        query: &[f32],
        keys: &[f32],
        sequence: usize,
        offset: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let query_count = sequence
            .checked_mul(query_heads)
            .and_then(|count| count.checked_mul(head_dim))
            .ok_or_else(|| "Vulkan rotary query shape overflows usize.".to_string())?;
        let key_count = sequence
            .checked_mul(kv_heads)
            .and_then(|count| count.checked_mul(head_dim))
            .ok_or_else(|| "Vulkan rotary key shape overflows usize.".to_string())?;
        if query.len() != query_count || keys.len() != key_count {
            return Err("Vulkan rotary input lengths do not match dimensions.".into());
        }
        let mut query_output = vec![0.0; query_count];
        let mut key_output = vec![0.0; key_count];
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_qwen_rotary(
                query.as_ptr(),
                keys.as_ptr(),
                query_output.as_mut_ptr(),
                key_output.as_mut_ptr(),
                to_u32(sequence)?,
                to_u32(offset)?,
                to_u32(query_heads)?,
                to_u32(kv_heads)?,
                to_u32(head_dim)?,
                to_u32(rotary_dim)?,
                theta,
            )
        };
        check_native_result(result, "Vulkan rotary embedding")?;
        Ok((query_output, key_output))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        sequence: usize,
        offset: usize,
        cache_len: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sliding_window: usize,
    ) -> Result<Vec<f32>, String> {
        let query_count = sequence
            .checked_mul(query_heads)
            .and_then(|count| count.checked_mul(head_dim))
            .ok_or_else(|| "Vulkan attention query shape overflows usize.".to_string())?;
        let cache_count = cache_len
            .checked_mul(kv_heads)
            .and_then(|count| count.checked_mul(head_dim))
            .ok_or_else(|| "Vulkan attention cache shape overflows usize.".to_string())?;
        if query.len() != query_count || keys.len() != cache_count || values.len() != cache_count {
            return Err("Vulkan attention input lengths do not match dimensions.".into());
        }
        let mut output = vec![0.0; query_count];
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_qwen_attention(
                query.as_ptr(),
                keys.as_ptr(),
                values.as_ptr(),
                output.as_mut_ptr(),
                to_u32(sequence)?,
                to_u32(offset)?,
                to_u32(cache_len)?,
                to_u32(query_heads)?,
                to_u32(kv_heads)?,
                to_u32(head_dim)?,
                to_u32(sliding_window)?,
            )
        };
        check_native_result(result, "Vulkan attention")?;
        Ok(output)
    }

    fn to_u32(value: usize) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| "Vulkan dimension exceeds 32-bit indexing.".into())
    }

    fn check_native_result(result: c_int, operation: &str) -> Result<(), String> {
        if result == 0 {
            Ok(())
        } else {
            Err(format!(
                "{operation} failed; see the native diagnostic above."
            ))
        }
    }

    fn matmul_with(
        m: usize,
        k: usize,
        n: usize,
        a_len: usize,
        b_len: usize,
        dispatch: impl FnOnce([c_int; 3], *mut f32) -> c_int,
    ) -> Result<Vec<f32>, String> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Vulkan matrix dimensions must be positive.".into());
        }
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| "A matrix element count overflows usize.".to_string())?;
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| "B matrix element count overflows usize.".to_string())?;
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "C matrix element count overflows usize.".to_string())?;
        if a_len != expected_a || b_len != expected_b {
            return Err(format!(
                "Vulkan matmul expected A/B lengths {expected_a}/{expected_b}, received {}/{}.",
                a_len, b_len
            ));
        }
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let dimensions = [m, k, n]
            .into_iter()
            .map(|dimension| {
                c_int::try_from(dimension)
                    .map_err(|_| "Vulkan matrix dimension exceeds the C API limit.".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Internal Vulkan dimension conversion failed.".to_string())?;
        let mut output = vec![0.0; output_len];
        let result = dispatch(dimensions, output.as_mut_ptr());
        if result == 0 {
            Ok(output)
        } else {
            Err("Vulkan matrix multiplication failed; see the native diagnostic above.".into())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn matmul_rejects_invalid_shapes_before_calling_vulkan() {
            assert!(matmul(&[], &[], 0, 1, 1).is_err());
            assert!(matmul(&[1.0], &[], 1, 1, 1).is_err());
        }

        #[test]
        #[ignore = "requires a Vulkan compute device"]
        fn bf16_matmul_matches_reference() {
            initialize().unwrap();
            let b = [5.0f32, 6.0, 7.0, 8.0]
                .map(half::bf16::from_f32)
                .map(half::bf16::to_bits);
            let output = matmul_bf16(&[1.0, 2.0, 3.0, 4.0], &b, 2, 2, 2).unwrap();
            shutdown().unwrap();
            assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
        }

        #[test]
        #[ignore = "requires a Vulkan compute device"]
        fn qwen_operators_match_cpu_reference() {
            initialize().unwrap();
            prepare().unwrap();

            let epsilon = 1e-5;
            let normalized = rms_norm_f32(&[3.0, 4.0], &[1.0, 2.0], 1, 2, epsilon).unwrap();
            let scale = (12.5f32 + epsilon).sqrt().recip();
            assert!((normalized[0] - 3.0 * scale).abs() < 1e-5);
            assert!((normalized[1] - 8.0 * scale).abs() < 1e-5);
            let bf16_weights = [1.0f32, 2.0]
                .map(half::bf16::from_f32)
                .map(half::bf16::to_bits);
            let fp16_weights = [1.0f32, 2.0]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits);
            assert_eq!(
                rms_norm_bf16(&[3.0, 4.0], &bf16_weights, 1, 2, epsilon).unwrap(),
                normalized
            );
            assert_eq!(
                rms_norm_fp16(&[3.0, 4.0], &fp16_weights, 1, 2, epsilon).unwrap(),
                normalized
            );

            let transposed =
                matmul_f32_transposed(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2)
                    .unwrap();
            assert_eq!(transposed, vec![17.0, 23.0, 39.0, 53.0]);
            let bf16_matrix = [5.0f32, 6.0, 7.0, 8.0]
                .map(half::bf16::from_f32)
                .map(half::bf16::to_bits);
            assert_eq!(
                matmul_bf16_transposed(&[1.0, 2.0, 3.0, 4.0], &bf16_matrix, 2, 2, 2).unwrap(),
                transposed
            );
            let fp16_matrix = [5.0f32, 6.0, 7.0, 8.0]
                .map(half::f16::from_f32)
                .map(half::f16::to_bits);
            assert_eq!(
                matmul_fp16_transposed(&[1.0, 2.0, 3.0, 4.0], &fp16_matrix, 2, 2, 2).unwrap(),
                transposed
            );

            assert_eq!(add(&[1.0, 2.0], &[3.0, 4.0]).unwrap(), vec![4.0, 6.0]);
            let gated = silu_mul(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
            assert!((gated[0]).abs() < 1e-6);
            assert!((gated[1] - 3.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-5);

            let (query, keys) =
                rotary(&[1.0, 0.0], &[1.0, 0.0], 1, 1, 1, 1, 2, 2, 10_000.0).unwrap();
            assert!((query[0] - 1.0f32.cos()).abs() < 1e-5);
            assert!((query[1] - 1.0f32.sin()).abs() < 1e-5);
            assert!((keys[0] - query[0]).abs() < 1e-6);
            assert!((keys[1] - query[1]).abs() < 1e-6);

            let query = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
            let keys = [1.0, 0.0, 0.0, 1.0];
            let values = [2.0, 3.0, 5.0, 7.0];
            let output = attention(&query, &keys, &values, 2, 0, 2, 2, 1, 2, 0).unwrap();
            assert_eq!(&output[0..2], &[2.0, 3.0]);
            assert_eq!(&output[2..4], &[2.0, 3.0]);
            let second_weight = 1.0 / (1.0 + (-std::f32::consts::FRAC_1_SQRT_2).exp());
            let expected = [
                2.0 * second_weight + 5.0 * (1.0 - second_weight),
                3.0 * second_weight + 7.0 * (1.0 - second_weight),
                2.0 * (1.0 - second_weight) + 5.0 * second_weight,
                3.0 * (1.0 - second_weight) + 7.0 * second_weight,
            ];
            for (actual, expected) in output[4..].iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "attention output {actual} differs from expected {expected}"
                );
            }

            shutdown().unwrap();
        }

        #[test]
        fn matmul_returns_native_failure_when_backend_is_uninitialized() {
            assert!(matmul(&[1.0], &[2.0], 1, 1, 1).is_err());
        }

        #[test]
        #[ignore = "requires a Vulkan compute device and compiled matmul.spv"]
        fn matmul_computes_on_vulkan_device() {
            initialize().unwrap();
            let output = matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2).unwrap();
            shutdown().unwrap();
            assert_eq!(output, vec![19.0, 22.0, 43.0, 50.0]);
        }
    }
}
