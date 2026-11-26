//! Safe Rust adapters for C APIs declared by the root-level C headers.
//!
//! This module is the single Rust-side boundary for the project's native
//! translation units. It exposes three groups of wrappers:
//!
//! * Model preparation, implemented in `cheetah_init.c` / `cheetah_init.h`.
//!   `init_model` and `init_model_with_metadata` convert a source directory
//!   of safetensors files into the prepared layout consumed by the runtime,
//!   while a C-to-Rust progress callback drives `indicatif` bars. After a
//!   successful native run the manifest can be patched with chat metadata
//!   known only to the Rust model registry.
//! * The "Ant" memory manager, implemented in `cheetah_memory.c` /
//!   `cheetah_memory.h`. These wrappers cover its lifecycle, tuning
//!   settings, model loading, residency queries, pinning, eviction, and
//!   page-in operations. `AntFfiSettings` mirrors the C `CheetahAntSettings`
//!   struct field for field.
//! * The optional Vulkan compute backend (feature `vulkan`), implemented in
//!   `cheetah_vulkan.c` / `cheetah_vulkan.h` together with the SPIR-V
//!   compute shaders (`matmul*.comp`, `shared_ops.comp`). The nested
//!   `vulkan` module wraps device setup, matrix multiplication in several
//!   storage formats, resident weight uploads, and the Qwen2 building-block
//!   operators.
//!
//! # Ownership and lifetimes
//!
//! Every native object is represented by a raw pointer or an integer handle
//! and remains owned by C. The Rust wrappers deliberately do not implement
//! `Drop` for native handles: callers must invoke the matching destructor
//! (`ant_delete`, `weight_release`, or `shutdown`) exactly once and only
//! after all dependent operations have finished. `ant_new` returns a pointer
//! that is freed only by `ant_delete`; an Ant pointer must not be used after
//! that call. Resident weight handles returned by the upload functions are
//! borrowed tokens into a process-wide C table and are invalidated by
//! `weight_release` or by backend shutdown.
//!
//! `CString` values created in these wrappers live only until the foreign
//! call returns. The C side is expected to copy anything it keeps; none of
//! the wrappers retain a pointer to a temporary Rust allocation. Input
//! slices are borrowed for the duration of a call and are never freed or
//! resized by C. Outputs are always freshly allocated `Vec`s whose length is
//! validated or derived before the call, so the native code writes only
//! within the buffer Rust provides. `ModelMetadata` uses `'static` strings,
//! which keeps manifest updates independent of any shorter-lived caller
//! borrow.
//!
//! # Threading model
//!
//! The C libraries keep process-global state (Vulkan device objects, resident
//! weight tables, and progress reporting resources), so the Rust API
//! serializes entry points with module-private mutexes:
//! `INIT_LOCK` covers model preparation and `VULKAN_LOCK` covers every
//! Vulkan operation, including initialization and shutdown. The locks guard
//! the foreign calls rather than Rust data, therefore lock poisoning does
//! not imply that native memory is corrupted; most wrappers recover the
//! inner guard so a panic elsewhere cannot permanently disable FFI. Model
//! preparation and Vulkan calls are synchronous with respect to the caller,
//! but the locks are what make sharing the module across threads safe at the
//! wrapper level. The progress callback is invoked synchronously by the C
//! initializer and must never unwind across the FFI boundary, so its body
//! avoids panicking operations and ignores malformed null arguments.
//!
//! # Error reporting
//!
//! C functions use `0` for success and nonzero (typically `-1`) for failure,
//! and most print the detailed diagnostic to stderr themselves. The wrappers
//! translate only the status code into a short `Err(String)`; the native
//! diagnostic is intentionally left on stderr so callers can log both. A few
//! helper APIs return `bool` or `Option` where the C contract makes failure
//! non-fatal (for example, a resident upload that cannot be served). String
//! conversion errors (interior NUL bytes, non-UTF-8 paths) are reported
//! before any foreign call is made.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::Mutex;

/// Process-wide serialization for native model-preparation runs.
///
/// The C initializer emits progress through a callback that borrows a stack
/// allocated [`InitProgress`], and it also touches shared native bookkeeping.
/// Holding this mutex for the whole call keeps two preparation sessions from
/// interleaving their bars or racing in the C code. The guard is acquired
/// after the source and destination paths have been converted, so a bad path
/// fails early without contention. A poisoned lock is treated as an error
/// because the initializer must not run with partially observed state.
static INIT_LOCK: Mutex<()> = Mutex::new(());

/// Progress UI state shared with the C initializer for one invocation.
///
/// The C callback receives a pointer to this value as its opaque `user_data`.
/// Only one initializer call can own the value at a time because
/// [`INIT_LOCK`] is held for the duration of the foreign call. All methods
/// are driven from the initializer thread; the `indicatif` handles themselves
/// are internally synchronized, but this struct is not shared beyond the
/// callback contract.
struct InitProgress {
    /// Container that renders all active bars together, keeping ordering
    /// stable when a phase switch adds a new bar.
    bars: MultiProgress,
    /// Bar for the "parsing" phase. It is taken out of the option while a
    /// bar is being mutated and put back afterwards, which is the pattern
    /// used throughout the callback to avoid borrowing conflicts.
    parse_bar: Option<ProgressBar>,
    /// Bar for the "writing" phase, created when the C side transitions from
    /// parsing to writing prepared shards.
    write_bar: Option<ProgressBar>,
}

/// Construction and display helpers for the initializer's progress bars.
impl InitProgress {
    /// Create empty progress state with no bars installed yet.
    ///
    /// Bars are created lazily on the first callback for a phase so that the
    /// UI reflects the phases the native code actually enters.
    fn new() -> Self {
        Self {
            bars: MultiProgress::new(),
            parse_bar: None,
            write_bar: None,
        }
    }

    /// Return the shared template used by every preparation bar.
    ///
    /// The template shows a right-aligned completed/total counter, a cyan
    /// bar, and the phase message. The style string is a compile-time
    /// constant, so failure to parse it would be a programming error and is
    /// reported with `expect`.
    fn style() -> ProgressStyle {
        ProgressStyle::with_template("{pos:>7}/{len:<7} [{bar:40.cyan}] {msg}")
            .expect("valid indicatif style")
            .progress_chars("##-")
    }

    /// Register a new bar of `total` units and set its initial message.
    ///
    /// `total` is widened to `u64` because that is what `indicatif` uses;
    /// the C callback reports `size_t` counts, which fit on all supported
    /// targets. The bar starts at position 0 and is displayed even before
    /// the first progress update.
    fn start_bar(&mut self, total: usize, message: &'static str) -> ProgressBar {
        let bar = self.bars.add(ProgressBar::new(total as u64));
        bar.set_style(Self::style());
        bar.set_message(message);
        bar.set_position(0);
        bar
    }
}

// Raw declarations from `cheetah_init.c`.
//
// These are the only preparation symbols linked from the Rust side. The
// foreign signatures use C ABI types (`*const c_char`, `usize`, `void*`) and
// return `i32` status codes. Wrappers below add path/string conversion,
// locking, and progress management; they never change pointer ownership.
unsafe extern "C" {
    /// Prepare a model with periodic progress callbacks.
    ///
    /// `src_dir` and `dst_dir` are NUL-terminated UTF-8 paths owned by the
    /// caller for the whole call. `progress_fn` is a non-null function
    /// pointer that the C code invokes synchronously for each phase update;
    /// `user_data` is passed through unchanged and only needs to remain valid
    /// for the duration of the call. Returns 0 on success and -1 on failure,
    /// printing a diagnostic to stderr.
    fn cheetah_init_model_with_progress(
        src_dir: *const c_char,
        dst_dir: *const c_char,
        progress_fn: extern "C" fn(*const c_char, usize, usize, *mut c_void),
        user_data: *mut c_void,
    ) -> i32;
}

/// FFI mirror of the C `CheetahAntSettings` structure.
///
/// The `#[repr(C)]` attribute is mandatory: the native code reads this struct
/// by raw layout, so field order and types must stay identical to
/// `cheetah_memory.h`. Boolean-like fields are `i32` because C uses `int`
/// there. All pointer fields are borrowed only for the duration of
/// [`ant_set_settings`]; the C side copies or interprets them synchronously
/// and does not take ownership. Build these values with pointers obtained
/// from live `CString`s and keep those owners alive across the call.
#[repr(C)]
pub struct AntFfiSettings {
    /// Nonzero to enable the Ant memory manager, zero to disable it.
    pub enabled: i32,
    /// Size in bytes of the VRAM pool the manager may use.
    pub vram_pool_bytes: u64,
    /// Size in bytes of the host RAM budget.
    pub ram_budget_bytes: u64,
    /// NUL-terminated directory used for disk-resident units. May be null if
    /// the native implementation does not need a disk tier for this model.
    pub disk_dir: *const std::os::raw::c_char,
    /// Array of `resident_units_count` NUL-terminated unit identifiers that
    /// should be kept resident regardless of eviction pressure. The array and
    /// every pointed-to string are borrowed only for the settings call.
    pub resident_units: *const *const std::os::raw::c_char,
    /// Number of entries in `resident_units`.
    pub resident_units_count: usize,
    /// NUL-terminated eviction-policy selector understood by the native
    /// manager (for example a policy name such as LRU). Interpreted
    /// synchronously and not retained.
    pub eviction_policy: *const std::os::raw::c_char,
    /// Number of prefetch stages the manager may run ahead of demand.
    pub prefetch_stages: i32,
    /// Nonzero to allow the manager to place overflow units in VRAM swap
    /// space instead of failing or evicting to RAM.
    pub allow_vram_swap: i32,
}

// Raw declarations from `cheetah_memory.c` for the Ant memory manager.
//
// Every `ant` parameter is an opaque pointer previously returned by
// `cheetah_ant_new`; passing null or a pointer that has already been deleted
// is undefined behavior. The wrappers in this module preserve that contract
// and expose safe-looking Rust signatures, but they cannot validate pointer
// provenance. Status-returning functions use 0 for success and negative
// values for failure, matching the C header. Functions taking `*const`
// handles are logically read-only but may still update internal caches.
unsafe extern "C" {
    /// Allocate a new Ant memory manager.
    ///
    /// Returns an opaque pointer owned by the caller, or null when allocation
    /// fails. The returned handle must eventually be passed to
    /// `cheetah_ant_delete` exactly once.
    fn cheetah_ant_new() -> *mut std::os::raw::c_void;
    /// Destroy an Ant manager and release all of its native resources.
    ///
    /// `ant` must be a live handle from `cheetah_ant_new`; after this call
    /// every other pointer returned by or associated with the manager is
    /// invalid. No Rust wrapper can enforce the single-use rule, so callers
    /// must own the lifetime explicitly.
    fn cheetah_ant_delete(ant: *mut std::os::raw::c_void);
    /// Apply memory-manager settings.
    ///
    /// `settings` is borrowed only for the call; the C side validates and
    /// copies the values it needs. Returns 0 on success and nonzero on
    /// failure.
    fn cheetah_ant_set_settings(
        ant: *mut std::os::raw::c_void,
        settings: *const AntFfiSettings,
    ) -> i32;
    /// Load the prepared model rooted at `dst_dir` into the manager.
    ///
    /// `dst_dir` is a NUL-terminated path to a directory previously produced
    /// by the initializer. The native code reads the manifest/index during
    /// this call and may fail if the layout is incomplete.
    fn cheetah_ant_load_model(
        ant: *mut std::os::raw::c_void,
        dst_dir: *const std::os::raw::c_char,
    ) -> i32;
    /// Query the current tier of one unit.
    ///
    /// `dst_dir` selects the loaded model and `unit_id` identifies a tensor
    /// unit. On success the native tier enum (0 = VRAM, 1 = RAM, 2 = disk) is
    /// written to `tier_out`; the pointer must be valid and writable. The
    /// function returns 0 on success.
    fn cheetah_ant_unit_tier(
        ant: *const std::os::raw::c_void,
        dst_dir: *const std::os::raw::c_char,
        unit_id: *const std::os::raw::c_char,
        tier_out: *mut i32,
    ) -> i32;
    /// Pin or unpin a unit so the eviction logic keeps it resident.
    ///
    /// `pin` is interpreted as a C boolean: nonzero pins, zero unpins. The
    /// handle is `const` because pinning changes residency bookkeeping, not
    /// the manager's identity.
    fn cheetah_ant_unit_pin(
        ant: *const std::os::raw::c_void,
        unit_id: *const std::os::raw::c_char,
        pin: i32,
    ) -> i32;
    /// Evict a unit from its current tier according to the active policy.
    ///
    /// `dst_dir` identifies the model whose unit table should be consulted.
    /// Returns 0 on success or a negative status when the unit cannot be
    /// evicted (for example because it is pinned).
    fn cheetah_ant_evict_unit(
        ant: *const std::os::raw::c_void,
        dst_dir: *const std::os::raw::c_char,
        unit_id: *const std::os::raw::c_char,
    ) -> i32;
    /// Page a unit into the requested tier.
    ///
    /// `tier` uses the native enum values (0 = VRAM, 1 = RAM, 2 = disk). The
    /// manager may reject the request when the target tier lacks capacity.
    fn cheetah_ant_page_in(
        ant: *const std::os::raw::c_void,
        unit_id: *const std::os::raw::c_char,
        tier: i32,
    ) -> i32;
    /// Report whether the model at `dst_dir` has been initialized.
    ///
    /// Returns 1 when initialized, 0 when not initialized or when the query
    /// cannot be answered. This is a tri-state-free C convention, which the
    /// Rust wrapper maps to `bool`.
    fn cheetah_ant_model_is_initialized(
        ant: *const std::os::raw::c_void,
        dst_dir: *const std::os::raw::c_char,
    ) -> i32;
    /// Report whether the manager still requires an explicit initialization
    /// step before it can manage a model. Nonzero means initialization is
    /// required.
    fn cheetah_ant_requires_init(ant: *const std::os::raw::c_void) -> i32;
    /// Emit the standard "model not initialized" warning.
    ///
    /// This is a diagnostic-only call: it returns nothing and is used to keep
    /// the user-facing message consistent between Rust and C. Both strings
    /// are read synchronously; no ownership is transferred.
    fn cheetah_ant_model_not_initialized_warning(
        model_id: *const std::os::raw::c_char,
        dst_dir: *const std::os::raw::c_char,
    );
}

/// Create a new Ant memory manager.
///
/// On success the returned pointer is an opaque native handle. The caller
/// owns it and must pass it to [`ant_delete`] exactly once when done; Rust
/// provides no automatic cleanup because the handle may outlive this module's
/// call stack. A null return from C is translated into an `Err` so callers
/// never receive a dangling null handle wrapped in `Ok`.
pub fn ant_new() -> Result<*mut std::os::raw::c_void, String> {
    // The C constructor reports failure only through a null pointer.
    let ptr = unsafe { cheetah_ant_new() };
    if ptr.is_null() {
        Err("Failed to create the Ant memory manager.".to_string())
    } else {
        Ok(ptr)
    }
}

/// Destroy an Ant manager previously returned by [`ant_new`].
///
/// This is the sole owner-side cleanup point for the C object. The pointer
/// must be live and non-null; the native destructor frees all manager state,
/// after which the handle is dangling. Callers must ensure no other thread is
/// still using the manager (the module's wrappers do not add lifetime
/// tracking), and must not call this function twice for the same pointer.
pub fn ant_delete(ant: *mut std::os::raw::c_void) {
    // Ownership is transferred back to C here; nothing else remains to free.
    unsafe { cheetah_ant_delete(ant) };
}

/// Apply tuning and residency settings to an Ant manager.
///
/// The settings value is passed by reference and borrowed only for the
/// duration of the call; any pointers inside it must reference live
/// NUL-terminated strings. A nonzero native status is reported as a generic
/// error because the C layer has already printed the specific diagnostic.
pub fn ant_set_settings(
    ant: *mut std::os::raw::c_void,
    settings: &AntFfiSettings,
) -> Result<(), String> {
    // `settings` is a Rust reference to a `#[repr(C)]` value, so its address
    // is directly consumable by C without copying.
    let status = unsafe { cheetah_ant_set_settings(ant, settings) };
    if status != 0 {
        return Err("Cheetah Ant set settings failed.".to_string());
    }
    Ok(())
}

/// Load a prepared model into the Ant manager.
///
/// `dst_dir` must point at an initialized model directory (the same layout
/// produced by [`init_model`]). The path is converted to a temporary
/// `CString`; failure to represent the path as UTF-8 or the presence of an
/// interior NUL is reported before entering C, so no partial native state is
/// created on conversion errors.
pub fn ant_load_model(ant: *mut std::os::raw::c_void, dst_dir: &Path) -> Result<(), String> {
    // Keep `dst_dir_c` alive until the C call returns; C does not retain it.
    let dst_dir_c = path_to_cstring(dst_dir)?;
    let status = unsafe { cheetah_ant_load_model(ant, dst_dir_c.as_ptr()) };
    if status != 0 {
        return Err("Cheetah Ant model load failed.".to_string());
    }
    Ok(())
}

/// Query the tier currently holding a unit.
///
/// Both the model directory and the unit identifier are converted to
/// temporary C strings for the call. `tier_out` is initialized to zero before
/// the call; the native function only writes it on success, and the initial
/// value makes the failure path deterministic. On success the native tier enum
/// is returned as `i32` (0 = VRAM, 1 = RAM, 2 = disk).
pub fn ant_unit_tier(
    ant: *mut std::os::raw::c_void,
    dst_dir: &Path,
    unit_id: &str,
) -> Result<i32, String> {
    let dst_dir_c = path_to_cstring(dst_dir)?;
    // Unit ids are UTF-8 strings with no interior NUL by construction; reject
    // anything that cannot be passed to C verbatim.
    let unit_id_c =
        CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let mut tier_out: std::os::raw::c_int = 0;
    // The out-pointer aliases the local integer, which outlives the call.
    let status = unsafe {
        cheetah_ant_unit_tier(ant, dst_dir_c.as_ptr(), unit_id_c.as_ptr(), &mut tier_out)
    };
    if status != 0 {
        return Err("Cheetah Ant unit tier query failed.".to_string());
    }
    Ok(tier_out as i32)
}

/// Pin or unpin a unit in the eviction policy.
///
/// `pin` is a Rust `bool` marshalled as the C convention 1/0. The unit id is
/// validated for interior NUL bytes before the call. Pinning is advisory to
/// the eviction logic; the native layer decides whether the request can be
/// honored and reports failure through its status code.
pub fn ant_unit_pin(
    ant: *mut std::os::raw::c_void,
    unit_id: &str,
    pin: bool,
) -> Result<(), String> {
    let unit_id_c =
        CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    // Explicit 1/0 keeps the ABI independent of Rust's `bool` layout.
    let status =
        unsafe { cheetah_ant_unit_pin(ant, unit_id_c.as_ptr(), if pin { 1i32 } else { 0i32 }) };
    if status != 0 {
        return Err("Cheetah Ant unit pin failed.".to_string());
    }
    Ok(())
}

/// Evict one unit from its current tier.
///
/// `dst_dir` identifies the loaded model's unit table. The native code may
/// refuse to evict pinned or in-use units, which surfaces as an `Err`; the
/// wrapper does not retry or change policy, so callers can decide how to
/// react to pressure.
pub fn ant_evict_unit(
    ant: *mut std::os::raw::c_void,
    dst_dir: &Path,
    unit_id: &str,
) -> Result<(), String> {
    let dst_dir_c = path_to_cstring(dst_dir)?;
    let unit_id_c =
        CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    // Both temporaries remain in scope across the call.
    let status = unsafe { cheetah_ant_evict_unit(ant, dst_dir_c.as_ptr(), unit_id_c.as_ptr()) };
    if status != 0 {
        return Err("Cheetah Ant unit evict failed.".to_string());
    }
    Ok(())
}

/// Page a unit into the requested tier.
///
/// `tier` uses the native enum values (0 = VRAM, 1 = RAM, 2 = disk) and is
/// passed through without translation. The manager may reject the request if
/// the target tier has no capacity; that condition is reported as an error and
/// leaves the unit in its previous tier.
pub fn ant_page_in(ant: *mut std::os::raw::c_void, unit_id: &str, tier: i32) -> Result<(), String> {
    let unit_id_c =
        CString::new(unit_id).map_err(|_| "Unit id contains a NUL byte.".to_string())?;
    let status = unsafe { cheetah_ant_page_in(ant, unit_id_c.as_ptr(), tier) };
    if status != 0 {
        return Err("Cheetah Ant unit page-in failed.".to_string());
    }
    Ok(())
}

/// Return whether the model at `dst_dir` is initialized for Ant management.
///
/// This helper is intentionally infallible: a path that cannot be represented
/// as a C string is mapped to a null pointer, and the native query then
/// reports "not initialized". That matches the caller's intent (the model
/// cannot be managed without a valid directory) and avoids introducing an
/// error type into a yes/no probe.
pub fn ant_model_is_initialized(ant: *mut std::os::raw::c_void, dst_dir: &Path) -> bool {
    // Preserve the distinction between conversion failure and a real path:
    // only a successful conversion contributes a non-null pointer.
    let dst_dir_c = path_to_cstring(dst_dir);
    let dst_ptr = dst_dir_c
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());
    let status = unsafe { cheetah_ant_model_is_initialized(ant, dst_ptr) };
    // The C API returns 1 for initialized; every other value is treated as
    // "not initialized".
    status == 1
}

/// Return whether the manager still needs an explicit initialization step.
///
/// The native function returns a C boolean-style integer; any nonzero value
/// means initialization is required.
pub fn ant_requires_init(ant: *mut std::os::raw::c_void) -> bool {
    let status = unsafe { cheetah_ant_requires_init(ant) };
    status != 0
}

/// Emit the standard warning for an uninitialized model.
///
/// This is best-effort diagnostics: if either string cannot be converted to a
/// C string, the warning is silently skipped rather than panicking or
/// inventing a different message. Both temporary `CString`s live through the
/// call, and the native function retains neither pointer.
pub fn ant_model_not_initialized_warning(model_id: &str, dst_dir: &Path) {
    let model_id_c = CString::new(model_id).ok();
    let dst_dir_c = path_to_cstring(dst_dir).ok();
    // Only call C when both arguments are representable; a partial warning
    // would be misleading.
    if let (Some(model_id_c), Some(dst_dir_c)) = (model_id_c, dst_dir_c) {
        unsafe {
            cheetah_ant_model_not_initialized_warning(model_id_c.as_ptr(), dst_dir_c.as_ptr())
        };
    }
}

/// C progress callback used by [`init_model_with_metadata`].
///
/// The native initializer calls this synchronously with a NUL-terminated
/// phase name, the current and total counts for that phase, and the opaque
/// pointer supplied as `user_data`. Here that pointer always comes from a
/// stack-local [`InitProgress`] that outlives the C call, so the cast back to
/// `&mut InitProgress` is valid. The function is `extern "C"`: it must not
/// unwind, and a panic would cross the FFI boundary and abort the process.
/// Null phase or user data is ignored defensively because the callback
/// contract assumes both are non-null but a malformed native build should not
/// cause undefined behavior in Rust.
extern "C" fn report_init_progress(
    phase: *const c_char,
    current: usize,
    total: usize,
    user_data: *mut c_void,
) {
    // Both pointers are required by the protocol; bail out before forming
    // references if either is missing.
    if phase.is_null() || user_data.is_null() {
        return;
    }

    // The pointer is known to refer to the `InitProgress` owned by the
    // initializer call that registered this callback. No other callback can
    // run concurrently because `INIT_LOCK` serializes preparation.
    let progress = unsafe { &mut *user_data.cast::<InitProgress>() };
    // Copy the phase name into an owned, lossily decoded string so the C
    // buffer only needs to remain valid for this instant.
    let phase = unsafe { CStr::from_ptr(phase) }.to_string_lossy();

    match phase.as_ref() {
        // Parsing source safetensors: update by source-file count.
        "parsing" => {
            // The bar is taken out of the option so the mutable borrow ends
            // before the field is assigned again.
            let mut parse_bar = progress.parse_bar.take();
            if parse_bar.is_none() {
                // First update for this phase: create the bar with the total
                // supplied by C.
                parse_bar = Some(progress.start_bar(total, "Parsing safetensors files..."));
            }
            if let Some(bar) = parse_bar.as_mut() {
                // `current` is a byte/object count, always within `total` as
                // far as this UI is concerned; no extra clamping is needed.
                bar.set_position(current as u64);
            }
            progress.parse_bar = parse_bar;
        }
        // Writing prepared shards: finish the parse bar, then create the
        // write bar.
        "writing" => {
            // Close the previous phase first so the two bars do not appear
            // active at the same time.
            let mut parse_bar = progress.parse_bar.take();
            if let Some(bar) = parse_bar.as_mut() {
                bar.finish_with_message("Parsing complete");
            }
            // The native callback emits this transition once; a fresh bar is
            // created each time the phase is entered.
            let write_bar = progress.start_bar(total, "Writing prepared shards...");
            progress.write_bar = Some(write_bar);
            if let Some(bar) = progress.write_bar.as_mut() {
                bar.set_position(current as u64);
            }
        }
        // Preparing weights: keep the current bar state to avoid a brief 1/1
        // flicker.
        "preparing" => {
            // There is no separate bar for this short phase. If parsing never
            // transitioned to writing, finish that bar with a phase label
            // instead of leaving it running.
            let mut parse_bar = progress.parse_bar.take();
            if let Some(bar) = parse_bar.as_mut() {
                bar.finish_with_message("Preparing weights...");
            }
        }
        // Complete: finish the write bar and show the final status.
        "complete" => {
            let mut write_bar = progress.write_bar.take();
            if let Some(bar) = write_bar.as_mut() {
                bar.finish_with_message("Initialization complete");
            }
            progress.write_bar = write_bar;

            // Fallback for very small runs that never entered the writing
            // phase: finish whichever parse bar may still be open.
            if progress.write_bar.is_none() {
                let mut parse_bar = progress.parse_bar.take();
                if let Some(bar) = parse_bar.as_mut() {
                    bar.finish_with_message("Initialization complete");
                }
                progress.parse_bar = parse_bar;
            }
        }
        // Unknown phase names are ignored so a newer C initializer can add
        // phases without breaking this UI.
        _ => {}
    }
}

/// Convert a filesystem path into a NUL-terminated C string.
///
/// C APIs in this module expect UTF-8 paths with no interior NUL. Rust paths
/// are not guaranteed to be UTF-8 on Unix, so non-UTF-8 paths are rejected
/// with a message that includes the lossy display form (useful for diagnosis,
/// although that form is not round-trippable). The returned `CString` owns its
/// allocation and must be kept alive by the caller for as long as C may read
/// it; all wrappers here do so only for the duration of a single call.
fn path_to_cstring(path: &Path) -> Result<CString, String> {
    let path = path
        .to_str()
        .ok_or_else(|| format!("Path is not valid UTF-8: {}", path.display()))?;
    // `CString::new` rejects embedded NULs, which would otherwise truncate the
    // path silently inside C.
    CString::new(path).map_err(|_| "Path contains an interior NUL byte.".to_string())
}

/// Chat metadata that the Rust registry can contribute after native
/// preparation.
///
/// The native initializer derives most of the manifest, but chat templates and
/// stop markers are model-family knowledge owned by the Rust registry. These
/// fields are `'static` so they can be stored in registry tables and passed to
/// [`init_model_with_metadata`] without lifetime plumbing. They are copied into
/// the manifest, so the C layer and the resulting file never borrow them.
pub struct ModelMetadata {
    /// Jinja-style chat template to write into `chat_template` in the
    /// manifest, replacing any value produced natively.
    pub chat_template: &'static str,
    /// Additional stop markers to merge into the manifest's `stop_markers`
    /// array. Existing markers are preserved and duplicates are skipped.
    pub stop_markers: &'static [&'static str],
}

/// Run the C static model-preparation stage and write its result to `dst_dir`.
///
/// This is the metadata-free entry point: it forwards to
/// [`init_model_with_metadata`] with `None`, so the native manifest is left
/// untouched after preparation. Progress is rendered through the module's
/// `indicatif` UI. Returns a short error on native failure; the detailed
/// diagnostic is printed by C to stderr.
pub fn init_model(src_dir: &Path, dst_dir: &Path) -> Result<(), String> {
    init_model_with_metadata(src_dir, dst_dir, None)
}

/// Run native preparation, then override chat metadata when the Rust registry
/// recognizes the model.
///
/// The two paths are converted to C strings before the global initializer
/// lock is acquired, so path errors fail fast. The lock is held for the whole
/// native call and the subsequent manifest patch: this keeps the progress
/// callback's stack-local state private and prevents a second initializer from
/// rewriting the same destination while the manifest is being updated.
///
/// On success, if `metadata` is present, `cheetah_manifest.json` is updated in
/// place (via a temporary file and atomic rename). Progress bars are always
/// finalized or abandoned before returning so the terminal is left in a clean
/// state on every path, including manifest-update failures.
pub fn init_model_with_metadata(
    src_dir: &Path,
    dst_dir: &Path,
    metadata: Option<ModelMetadata>,
) -> Result<(), String> {
    // Keep an owned copy of the destination path: the `CString` conversion
    // below consumes the borrowed path, but the manifest update needs the
    // original directory afterwards.
    let dst_dir_path = dst_dir.to_path_buf();
    let src_dir = path_to_cstring(src_dir)?;
    let dst_dir = path_to_cstring(dst_dir)?;
    // Serialize the entire native run plus manifest patch. A poisoned lock
    // means a previous initializer panicked while holding it, which may have
    // left the native side inconsistent; refuse to continue.
    let _guard = INIT_LOCK
        .lock()
        .map_err(|_| "Native model initializer lock is poisoned.".to_string())?;

    let mut progress = InitProgress::new();
    // The callback receives a raw pointer to `progress`; `_guard` and the
    // local binding keep both the lock and the pointee alive across the call.
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

        // Metadata patching is strictly post-processing: the native model is
        // already valid, so a failure here is reported without retrying or
        // deleting the prepared output.
        let result = if let Some(metadata) = metadata {
            update_manifest_metadata(&dst_dir_path, metadata)
        } else {
            Ok(())
        };

        if result.is_ok() {
            // Defensive final sweep: `parsing` and `writing` may both have
            // bars in unusual phase orderings. Only unfinished bars are
            // touched so completed messages are not overwritten.
            for bar in [&mut progress.parse_bar, &mut progress.write_bar] {
                if let Some(b) = bar
                    && !b.is_finished()
                {
                    b.finish_with_message("Initialization complete");
                }
            }
        } else {
            // Manifest failure is a failure of the overall operation from the
            // caller's perspective, so mark any remaining bars accordingly.
            abandon_bars(&mut progress);
        }

        result
    } else {
        // The C diagnostic has already been printed; only the progress UI
        // needs cleanup here.
        abandon_bars(&mut progress);
        Err("Cheetah model preparation failed; see the native diagnostic above.".into())
    }
}

/// Mark every outstanding preparation bar as failed.
///
/// Used on all error paths. `abandon_with_message` leaves the bar visible with
/// a failure label rather than removing it, which preserves the user's record
/// of how far preparation progressed. Bars that were already finished are
/// still abandoned; `flatten` skips `None` slots safely.
fn abandon_bars(progress: &mut InitProgress) {
    for b in [&mut progress.parse_bar, &mut progress.write_bar]
        .into_iter()
        .flatten()
    {
        b.abandon_with_message("Initialization failed");
    }
}

/// Merge registry-provided chat metadata into a prepared model manifest.
///
/// The native initializer writes `cheetah_manifest.json` with model facts it
/// can discover from the weights. This function loads that JSON, preserves all
/// unrelated fields, appends any new stop markers without duplicating existing
/// ones, and replaces `chat_template` with the registry's authoritative value.
/// The manifest is then written to a sibling temporary file and renamed into
/// place. Keeping the temporary file in the same directory ensures the final
/// rename stays on one filesystem, which is what makes the replacement atomic
/// and prevents readers from observing a partially written manifest.
///
/// Any I/O or JSON error is returned with enough path/context for diagnosis.
/// On rename failure the temporary file is removed on a best-effort basis so
/// repeated attempts do not accumulate stale files.
fn update_manifest_metadata(dst_dir: &Path, metadata: ModelMetadata) -> Result<(), String> {
    let manifest_path = dst_dir.join("cheetah_manifest.json");
    // Reading the whole manifest is safe: preparation just produced it and the
    // initializer lock prevents concurrent updates through this module.
    let bytes = fs::read(&manifest_path)
        .map_err(|error| format!("Failed to read {}: {error}", manifest_path.display()))?;
    let mut manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Invalid Cheetah manifest: {error}"))?;
    // A JSON object is required for field insertion; arrays or scalars are
    // structurally invalid manifests.
    let object = manifest
        .as_object_mut()
        .ok_or_else(|| "Cheetah manifest root must be a JSON object.".to_string())?;
    // Start from the markers already in the manifest. Missing or non-array
    // fields simply contribute no initial entries; non-string elements are
    // ignored rather than causing a hard failure, which keeps the updater
    // tolerant of forward-compatible manifest additions.
    let mut stop_markers = object
        .get("stop_markers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for marker in metadata.stop_markers {
        // Preserve native/discovered markers and append only new ones. The
        // order is native markers first, then registry order.
        if !stop_markers.iter().any(|existing| existing == marker) {
            stop_markers.push((*marker).to_owned());
        }
    }
    // Insertion replaces any previous values for these two keys while leaving
    // every other manifest field untouched.
    object.insert(
        "chat_template".into(),
        serde_json::Value::String(metadata.chat_template.into()),
    );
    object.insert(
        "stop_markers".into(),
        serde_json::to_value(stop_markers)
            .map_err(|error| format!("Failed to encode model stop markers: {error}"))?,
    );

    // The process id keeps concurrent processes from colliding on the
    // temporary name; the sibling directory keeps the final rename atomic.
    let temporary_path = manifest_path.with_extension(format!("json.tmp.{}", std::process::id()));
    let updated = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("Failed to encode Cheetah manifest: {error}"))?;
    fs::write(&temporary_path, updated)
        .map_err(|error| format!("Failed to write {}: {error}", temporary_path.display()))?;
    if let Err(error) = fs::rename(&temporary_path, &manifest_path) {
        // Best-effort cleanup: the original manifest is still intact, and the
        // caller only needs the rename error.
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
        // Initializes the C Vulkan backend (device selection + memory pool).
        fn init_vulkan_device() -> c_int;
        // Tears the backend down; all uploads must be released first.
        fn shutdown_vulkan_device();
        // Builds the shader pipelines the qwen operator set needs.
        fn vk_qwen_prepare() -> c_int;
        // Row-major f32 matmul C[M,N] = A[M,K] · B[K,N].
        fn vk_matmul_checked(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Matmul against the transpose view of a row-major f32 weight.
        fn vk_matmul_transposed_checked(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // FP32 activation × row-major BF16 weight matmul.
        fn vk_matmul_bf16_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // FP32 activation × transpose-of-BF16 weight matmul.
        fn vk_matmul_bf16_transposed_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Uploads BF16 bits into the resident bucket; returns the handle.
        fn cheetah_vulkan_weight_upload_bf16(weights: *const u16, element_count: u64) -> u64;
        // Uploads a q8 buffer (scale head + padded int8 rows).
        fn cheetah_vulkan_weight_upload_q8(bytes: *const u8, byte_size: u64, row_bytes: u32)
        -> u64;
        // Uploads a q4 buffer (scale head + padded nibble rows).
        fn cheetah_vulkan_weight_upload_q4(bytes: *const u8, byte_size: u64, row_bytes: u32)
        -> u64;
        // Releases one resident upload handle.
        fn cheetah_vulkan_weight_release(handle: u64) -> c_int;
        // Resident bf16 matmul reading the uploaded weight in place.
        fn vk_matmul_bf16_transposed_resident_checked(
            handle: u64,
            a: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Streaming q8 matmul (weights staged through the pool per tile).
        fn vk_matmul_q8_transposed_checked(
            a: *const f32,
            q8: *const u8,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Streaming q4 matmul.
        fn vk_matmul_q4_transposed_checked(
            a: *const f32,
            q4: *const u8,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Resident q4 matmul from the uploaded handle.
        fn vk_matmul_q4_transposed_resident_checked(
            handle: u64,
            a: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Resident q8 matmul from the uploaded handle.
        fn vk_matmul_q8_transposed_resident_checked(
            handle: u64,
            a: *const f32,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // FP32 × transpose-of-FP16 matmul (shared 16-bit pipeline, fp16 decode).
        fn vk_matmul_fp16_transposed_checked(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: c_int,
            k: c_int,
            n: c_int,
        ) -> c_int;
        // Embedding gather: one row per token id, optional sqrt(hidden) scale.
        fn vk_qwen_gather(
            token_ids: *const u32,
            table: *const c_void,
            weight_format: u32,
            table_element_count: u64,
            output: *mut f32,
            token_count: usize,
            hidden_size: usize,
            scale: u32,
        ) -> c_int;
        // Adds a bias tensor per `period` lanes in place.
        fn vk_qwen_bias_add(
            values: *const f32,
            bias: *const c_void,
            bias_format: u32,
            total_count: usize,
            period: usize,
            output: *mut f32,
        ) -> c_int;
        // RMSNorm shader over the stored weight format.
        fn vk_qwen_rms_norm(
            input: *const f32,
            weight: *const c_void,
            weight_format: u32,
            output: *mut f32,
            rows: u32,
            width: u32,
            epsilon: f32,
        ) -> c_int;
        // LayerNorm shader (weight + bias) over the stored weight format.
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
        // Two-operand elementwise shader (add / silu-mul / gelu-mul).
        fn vk_qwen_elementwise(
            a: *const f32,
            b: *const f32,
            output: *mut f32,
            count: usize,
            operation: u32,
        ) -> c_int;
        // NeoX rotate-half rotary over the leading rotary dim of each head.
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
        // GQA attention shader with causal masking + sliding window.
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
        // Generic catalog dispatch (xxx_ops, op codes >= 10) on the shared_ops
        // pipeline. parameters[0] = op code; sources/destinations/byte_sizes
        // are the 5 storage-buffer slots (a slot may be both, for state).
        fn vk_run_op(
            parameters: *const u32,
            sources: *const *const std::os::raw::c_void,
            destinations: *const *mut std::os::raw::c_void,
            byte_sizes: *const usize,
            invocation_count: u32,
        ) -> c_int;
        // Custom-shader catalog dispatch (op codes >= 100). The SPIR-V is taken
        // from shader_env when set, else default_spv; the shader must conform to
        // the shared_ops buffer/push-constant layout.
        fn vk_run_custom_op(
            shader_env: *const std::os::raw::c_char,
            default_spv: *const std::os::raw::c_char,
            parameters: *const u32,
            sources: *const *const std::os::raw::c_void,
            destinations: *const *mut std::os::raw::c_void,
            byte_sizes: *const usize,
            invocation_count: u32,
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

    /// Raw generic catalog dispatch (`xxx_ops`): binds `sources` /
    /// `destinations` to the 5 `shared_ops` storage slots, pushes `parameters`
    /// (`parameters[0]` = op code) and launches `invocation_count` threads on
    /// the shared compute pipeline. A slot may appear in both `sources` and
    /// `destinations` (recurrent state read-modify-write); its host contents
    /// are preserved across the round trip.
    pub fn run_op_raw(
        parameters: [u32; 8],
        sources: [*const c_void; 5],
        destinations: [*mut c_void; 5],
        byte_sizes: [usize; 5],
        invocation_count: usize,
    ) -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let result = unsafe {
            vk_run_op(
                parameters.as_ptr(),
                sources.as_ptr(),
                destinations.as_ptr(),
                byte_sizes.as_ptr(),
                to_u32(invocation_count)?,
            )
        };
        check_native_result(result, "Vulkan catalog operator dispatch")
    }

    /// Raw custom-shader catalog dispatch (`xxx_ops`, op code >= 100): like
    /// [`run_op_raw`] but binds a custom SPIR-V pipeline. The shader path is
    /// taken from the `shader_env` environment variable when set, otherwise
    /// `default_spv` (the build-time-compiled artifact under `OUT_DIR`).
    pub fn run_custom_op_raw(
        shader_env: &str,
        default_spv: &str,
        parameters: [u32; 8],
        sources: [*const c_void; 5],
        destinations: [*mut c_void; 5],
        byte_sizes: [usize; 5],
        invocation_count: usize,
    ) -> Result<(), String> {
        let _guard = VULKAN_LOCK
            .lock()
            .map_err(|_| "Vulkan backend lock is poisoned.".to_string())?;
        let shader_env_c = std::ffi::CString::new(shader_env)
            .map_err(|_| format!("shader_env name contains a NUL byte: {shader_env}"))?;
        let default_c = std::ffi::CString::new(default_spv)
            .map_err(|_| format!("default_spv path contains a NUL byte: {default_spv}"))?;
        let result = unsafe {
            vk_run_custom_op(
                shader_env_c.as_ptr(),
                default_c.as_ptr(),
                parameters.as_ptr(),
                sources.as_ptr(),
                destinations.as_ptr(),
                byte_sizes.as_ptr(),
                to_u32(invocation_count)?,
            )
        };
        check_native_result(result, "Vulkan custom catalog operator dispatch")
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

    /// Multiply an FP32 activation matrix by the transpose of an FP16 weight
    /// matrix on the shared 16-bit pipeline.
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

    /// Multiply an FP32 activation matrix by the transpose of a row-major FP32
    /// weight matrix (the f32 precision format's streaming path).
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
        let handle =
            unsafe { cheetah_vulkan_weight_upload_bf16(weights.as_ptr(), weights.len() as u64) };
        if handle == 0 { None } else { Some(handle) }
    }

    /// Upload one q8 quantized weight buffer into the resident bucket.
    /// `bytes` covers the f32 scale head plus the packed int8 rows;
    /// `row_bytes` is the padded per-row stride the resident matmul validates
    /// against. Returns None when the upload cannot be served and the caller
    /// should keep the per-step streaming path.
    pub fn weight_upload_q8(bytes: &[u8], row_bytes: u32) -> Option<u64> {
        let guard = VULKAN_LOCK.lock();
        let _guard = guard.unwrap_or_else(|poisoned| poisoned.into_inner());
        let handle = unsafe {
            cheetah_vulkan_weight_upload_q8(bytes.as_ptr(), bytes.len() as u64, row_bytes)
        };
        if handle == 0 { None } else { Some(handle) }
    }

    /// Upload one q4 quantized weight buffer into the resident bucket; same
    /// contract as `weight_upload_q8` (f32 scale head + packed nibble rows).
    pub fn weight_upload_q4(bytes: &[u8], row_bytes: u32) -> Option<u64> {
        let guard = VULKAN_LOCK.lock();
        let _guard = guard.unwrap_or_else(|poisoned| poisoned.into_inner());
        let handle = unsafe {
            cheetah_vulkan_weight_upload_q4(bytes.as_ptr(), bytes.len() as u64, row_bytes)
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

    /// Ant op 8: gather embedding rows on the GPU. `table` is the embedding
    /// tensor in `table_format` (0 = f32, 2 = bf16, 3 = f16); the result holds
    /// one row per token, optionally scaled by sqrt(hidden) (gemma-style).
    pub fn embedding_gather(
        token_ids: &[u32],
        table: *const std::os::raw::c_void,
        table_format: u32,
        table_element_count: usize,
        hidden_size: usize,
        scale: bool,
    ) -> Result<Vec<f32>, String> {
        if token_ids.is_empty() || hidden_size == 0 {
            return Err("Vulkan embedding gather requires tokens and a hidden size.".to_string());
        }
        let output_len = token_ids.len() * hidden_size;
        let mut output = vec![0.0; output_len];
        let _guard = VULKAN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = unsafe {
            vk_qwen_gather(
                token_ids.as_ptr(),
                table,
                table_format,
                table_element_count as u64,
                output.as_mut_ptr(),
                token_ids.len(),
                hidden_size,
                if scale { 1u32 } else { 0u32 },
            )
        };
        check_native_result(result, "Vulkan embedding gather")?;
        Ok(output)
    }

    /// Ant op 9: broadcast bias add on the GPU over a matmul output
    /// (`values` has `rows * period` elements; the bias cycles per row).
    /// In-place on `values`.
    pub fn bias_add(
        values: &mut [f32],
        bias: *const std::os::raw::c_void,
        bias_format: u32,
        period: usize,
    ) -> Result<(), String> {
        if values.is_empty() || period == 0 {
            return Err("Vulkan bias add requires values and a bias period.".to_string());
        }
        let _guard = VULKAN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = unsafe {
            vk_qwen_bias_add(
                values.as_ptr(),
                bias,
                bias_format,
                values.len(),
                period,
                values.as_mut_ptr(),
            )
        };
        check_native_result(result, "Vulkan bias add")?;
        Ok(())
    }

    /// Host-streaming q8 matmul: C[M,N] = A[M,K] * W^T where W is the
    /// quantized buffer (f32 scale head + packed int8 rows). Roughly half
    /// the PCIe traffic of the BF16 streaming path.
    pub fn matmul_q8_transposed(
        a: &[f32],
        q8: &[u8],
        _weight_rows: usize,
        _weight_cols: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Vulkan matrix dimensions must be positive.".to_string());
        }
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| "A matrix element count overflows usize.".to_string())?;
        let row_bytes = (k + 3) / 4 * 4;
        let expected_q8 = n * 4 + n * row_bytes;
        if q8.len() != expected_q8 {
            return Err(format!(
                "Vulkan q8 matmul expected {expected_q8} bytes, received {}.",
                q8.len()
            ));
        }
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "C matrix element count overflows usize.".to_string())?;
        if a.len() != expected_a {
            return Err(format!(
                "Vulkan q8 matmul expected A length {expected_a}, received {}.",
                a.len()
            ));
        }
        let mut output = vec![0.0; output_len];
        let _guard = VULKAN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = unsafe {
            vk_matmul_q8_transposed_checked(
                a.as_ptr(),
                q8.as_ptr(),
                output.as_mut_ptr(),
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };
        check_native_result(result, "Vulkan q8 matmul")?;
        Ok(output)
    }

    /// Host-streaming q4 matmul: C[M,N] = A[M,K] * W^T over the quantized
    /// buffer (f32 scale head + packed int4 rows); a quarter of the BF16
    /// footprint and half of q8.
    pub fn matmul_q4_transposed(
        a: &[f32],
        q4: &[u8],
        _weight_rows: usize,
        _weight_cols: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Vulkan matrix dimensions must be positive.".to_string());
        }
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| "A matrix element count overflows usize.".to_string())?;
        let row_bytes = (k + 1).div_ceil(2).div_ceil(4) * 4;
        let expected_q4 = n * 4 + n * row_bytes;
        if q4.len() != expected_q4 {
            return Err(format!(
                "Vulkan q4 matmul expected {expected_q4} bytes, received {}.",
                q4.len()
            ));
        }
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "C matrix element count overflows usize.".to_string())?;
        if a.len() != expected_a {
            return Err(format!(
                "Vulkan q4 matmul expected A length {expected_a}, received {}.",
                a.len()
            ));
        }
        let mut output = vec![0.0; output_len];
        let _guard = VULKAN_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = unsafe {
            vk_matmul_q4_transposed_checked(
                a.as_ptr(),
                q4.as_ptr(),
                output.as_mut_ptr(),
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };
        check_native_result(result, "Vulkan q4 matmul")?;
        Ok(output)
    }

    /// Ant resident q4 matmul: the quantized weight buffer stays bound on
    /// the GPU; only activations and outputs pass through the pool.
    /// Caller must hold a valid q4 upload handle (weight_upload_q4).
    pub fn matmul_q4_transposed_resident(
        handle: u64,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_common(m, k, n, a.len(), |dims, output| unsafe {
            vk_matmul_q4_transposed_resident_checked(
                handle,
                a.as_ptr(),
                output,
                dims[0],
                dims[1],
                dims[2],
            )
        })
    }

    /// Ant resident q8 matmul: the quantized weight buffer stays bound on
    /// the GPU; only activations and outputs pass through the pool.
    /// Caller must hold a valid q8 upload handle (weight_upload_q8).
    pub fn matmul_q8_transposed_resident(
        handle: u64,
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        matmul_common(m, k, n, a.len(), |dims, output| unsafe {
            vk_matmul_q8_transposed_resident_checked(
                handle,
                a.as_ptr(),
                output,
                dims[0],
                dims[1],
                dims[2],
            )
        })
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
        matmul_common(m, k, n, a.len(), |dims, output| unsafe {
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

    /// RMSNorm shader with an f32 weight, f32 activations in, f32 out.
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

    /// RMSNorm shader with a BF16 weight (weights stay in their stored width).
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

    /// LayerNorm shader with f32 weight and bias.
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

    /// LayerNorm shader with FP16 weight and bias.
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

    /// LayerNorm shader with BF16 weight and bias.
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

    /// Format-generic LayerNorm dispatch behind the typed wrappers: resolves the
    /// shared-ops weight-format code from the payload and forwards to C.
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

    /// RMSNorm shader with an FP16 weight.
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

    /// Format-generic RMSNorm dispatch behind the typed wrappers.
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

    /// Elementwise add shader (residual updates on the GPU side).
    pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        elementwise(a, b, 1)
    }

    /// SwiGLU elementwise shader: silu(gate) ⊙ up.
    pub fn silu_mul(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
        elementwise(gate, up, 2)
    }

    /// GELU elementwise shader; `approximate` selects the tanh approximation.
    pub fn gelu_mul(gate: &[f32], up: &[f32], approximate: bool) -> Result<Vec<f32>, String> {
        elementwise(gate, up, if approximate { 6 } else { 5 })
    }

    /// Generic two-operand elementwise dispatch behind add / silu_mul / gelu_mul.
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

    /// NeoX rotate-half rotary embedding over the leading rotary dim of each
    /// head: rotates `query` and `keys` for `sequence` positions starting at
    /// `offset`.
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

    /// GQA attention shader: every `query_heads / kv_heads` query heads read one
    /// KV head, scores use the 1/sqrt(head_dim) scale, keys older than the
    /// sliding window are masked, and the softmax-weighted value mix is
    /// returned for the whole sequence.
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

    /// usize → c_int range check for FFI dimension arguments.
    fn to_u32(value: usize) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| "Vulkan dimension exceeds 32-bit indexing.".into())
    }

    /// Maps a C status code to a descriptive Rust error naming the operation.
    fn check_native_result(result: c_int, operation: &str) -> Result<(), String> {
        if result == 0 {
            Ok(())
        } else {
            Err(format!(
                "{operation} failed; see the native diagnostic above."
            ))
        }
    }

    /// Shared matmul driver: validates the dimensions and A, drives the
    /// dispatch once, assembles the output. Resident matmuls use this
    /// directly — their weight is bound on the GPU behind a handle and no B
    /// buffer flows through the pool.
    fn matmul_common(
        m: usize,
        k: usize,
        n: usize,
        a_len: usize,
        dispatch: impl FnOnce([c_int; 3], *mut f32) -> c_int,
    ) -> Result<Vec<f32>, String> {
        if m == 0 || k == 0 || n == 0 {
            return Err("Vulkan matrix dimensions must be positive.".into());
        }
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| "A matrix element count overflows usize.".to_string())?;
        let output_len = m
            .checked_mul(n)
            .ok_or_else(|| "C matrix element count overflows usize.".to_string())?;
        if a_len != expected_a {
            return Err(format!(
                "Vulkan matmul expected A length {expected_a}, received {a_len}."
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

    /// Streaming matmul driver: additionally validates the B buffer against
    /// `k * n` before touching the device.
    fn matmul_with(
        m: usize,
        k: usize,
        n: usize,
        a_len: usize,
        b_len: usize,
        dispatch: impl FnOnce([c_int; 3], *mut f32) -> c_int,
    ) -> Result<Vec<f32>, String> {
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| "B matrix element count overflows usize.".to_string())?;
        if b_len != expected_b {
            return Err(format!(
                "Vulkan matmul expected B length {expected_b}, received {b_len}."
            ));
        }
        matmul_common(m, k, n, a_len, dispatch)
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
