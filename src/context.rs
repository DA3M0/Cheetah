//! Named, portable chat context archives.
//!
//! This module owns the on-disk representation of saved chat sessions. A
//! context archive couples a list of user/assistant turns with the identity of
//! the model that produced them and the inference options that were used. The
//! goal is that restoring a context reproduces the same conversation settings
//! instead of silently running with a different model or a different sampling
//! configuration.
//!
//! Data model:
//! - `ConversationTurn` is a single exchange: one user message and the
//!   assistant reply that followed it.
//! - `ContextArchive` is the complete persisted record. It carries a schema
//!   version, the user-visible context name, the canonical model path, the
//!   saved `ChatOptions`, all turns, and creation/update timestamps in whole
//!   Unix seconds.
//! - `ContextStore` is the filesystem-facing handle. It either uses an
//!   explicitly supplied directory or resolves a platform-specific default
//!   user data directory on demand.
//!
//! Persistence:
//! Each context is stored as a single pretty-printed JSON file named
//! `<name>.json` inside the store directory. The JSON format is the wire
//! format as well as the disk format, so archives can be copied between
//! machines. The format is versioned with `SCHEMA_VERSION`; archives written
//! by an incompatible version are rejected rather than partially interpreted.
//!
//! Security:
//! Context names are restricted to a small ASCII alphabet and explicitly
//! exclude `.` and `..`, which keeps the constructed archive path inside the
//! context directory. On Unix, the default context directory is created with
//! mode 0700 and temporary archive files with mode 0600 so that conversation
//! contents are not exposed to other local users. A directory supplied
//! explicitly with `--context-dir` is left with whatever permissions the user
//! configured. Archives are plain JSON and are not encrypted; their
//! confidentiality depends on the surrounding filesystem permissions.
//!
//! Atomicity:
//! Saving never writes directly over an existing archive. The complete JSON
//! payload is first written to a unique sibling temporary file, flushed with
//! `sync_all`, and then renamed over the target. Because the temporary file
//! lives in the same directory, the rename is a single filesystem operation
//! on the same filesystem, so readers observe either the old complete archive
//! or the new complete archive, never a truncated mixture. Temporary files are
//! removed on write or rename failures. The containing directory is not
//! fsynced, so this protects against torn writes, not against every possible
//! power-loss scenario.

use crate::options::ChatOptions;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(unix)]
use std::{
    fs::Permissions,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
};

/// On-disk schema version written into every newly created archive.
///
/// `ContextArchive::validate` rejects any archive whose version differs from
/// this constant. Bumping the version is therefore the signal that the
/// serialized layout changed in a way older or newer code cannot safely read.
const SCHEMA_VERSION: u32 = 1;

/// Monotonic counter used to make temporary archive filenames unique.
///
/// The process id is included in temporary names as well, so separate
/// processes normally cannot collide. The counter separates successive save
/// attempts within one process and avoids immediately reusing a name left
/// behind by an earlier failed write. `Relaxed` ordering is sufficient because
/// the counter is only a uniqueness token, not a synchronization mechanism.
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A single user/assistant exchange inside a saved conversation.
///
/// Turns are stored verbatim as UTF-8 strings and are not interpreted by this
/// module. Keeping the pair together preserves the conversational structure
/// that chat templates expect when the archive is replayed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Text sent by the user for this turn.
    pub user: String,
    /// Text produced by the assistant in response to `user`.
    pub assistant: String,
}

/// The complete persisted representation of a named chat context.
///
/// An archive is self-describing: it records which schema version produced it,
/// which model it belongs to, which inference options were active, and the
/// full conversation. Serialization uses the field names as JSON object keys,
/// so renaming a field is a schema-breaking change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextArchive {
    /// Version of the archive layout; must equal `SCHEMA_VERSION` to be used.
    pub schema_version: u32,
    /// User-visible context name and the file stem used on disk.
    pub name: String,
    /// Canonical filesystem path of the bound model directory or file.
    pub model_path: String,
    /// Inference options captured when the context was last saved.
    pub options: ChatOptions,
    /// Ordered conversation history, oldest turn first.
    pub turns: Vec<ConversationTurn>,
    /// Creation time as whole seconds since the Unix epoch.
    pub created_at: u64,
    /// Last successful save time as whole seconds since the Unix epoch.
    pub updated_at: u64,
}

/// Construction and validation logic for individual archives.
impl ContextArchive {
    /// Create an empty archive bound to `model_path`.
    ///
    /// The path must name an existing directory or a Safetensors file. It is
    /// canonicalized before being stored so that equivalent spellings, relative
    /// paths, and symlinks all resolve to the same identity during later
    /// launch-time checks.
    ///
    /// The returned archive has an empty name. Callers must assign a valid
    /// name before saving or validating, because an unnamed archive cannot be
    /// persisted.
    ///
    /// Errors are returned as human-readable strings suitable for the CLI.
    pub fn new(model_path: &Path, options: ChatOptions) -> Result<Self, String> {
        // Reject obviously unsupported inputs early, before attempting any
        // filesystem resolution with potentially surprising error messages.
        ensure_model_path(model_path)?;
        // Canonicalization can still fail (for example for a missing
        // `.safetensors` path or an unreadable directory), so surface the
        // original spelling in the error to keep it recognizable.
        let model_path = model_path.canonicalize().map_err(|error| {
            format!(
                "Failed to resolve model path '{}': {error}",
                model_path.display()
            )
        })?;
        // Both timestamps start at the same instant; later saves only move
        // `updated_at` forward.
        let now = unix_timestamp()?;
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            name: String::new(),
            model_path: model_path.to_string_lossy().into_owned(),
            options,
            turns: Vec::new(),
            created_at: now,
            updated_at: now,
        })
    }

    /// Check that the archive is internally consistent and safe to use.
    ///
    /// Validation covers the schema version, the archive name (which doubles
    /// as a filename component), and the inference options. It does not check
    /// that the bound model still exists; that is deliberately deferred to
    /// `validate_for_launch`, where the current model path is available.
    pub fn validate(&self) -> Result<(), String> {
        // Refuse unknown layouts instead of guessing how to interpret fields.
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "Context '{}' uses unsupported data version {} (currently supported: {}).",
                self.name, self.schema_version, SCHEMA_VERSION
            ));
        }
        // The name is also used as a path component, so validate it even when
        // loading a file that already exists on disk.
        validate_name(&self.name)?;
        // Reject nonsensical values and non-finite floats. NaN fails the
        // `is_finite` checks, and `temperature < 0.0` also catches negative
        // infinities. Keeping this list in sync with option semantics avoids
        // restoring an archive that would make sampling fail later.
        if self.options.max_tokens == 0
            || self.options.ctx_size == 0
            || self.options.threads == Some(0)
            || !self.options.temperature.is_finite()
            || self.options.temperature < 0.0
            || !(0.0..=1.0).contains(&self.options.top_p)
            || !self.options.repeat_penalty.is_finite()
            || self.options.repeat_penalty < 1.0
        {
            return Err(format!(
                "Context '{}' contains invalid inference options.",
                self.name
            ));
        }
        Ok(())
    }

    /// Validate that this archive may be resumed with the model and options
    /// currently requested by the user, and return the options to use.
    ///
    /// A saved conversation is bound to the canonical model path to avoid
    /// silently resuming it on another model. If the user explicitly supplied
    /// inference options on the command line, those options must exactly match
    /// the archived ones; otherwise the archive's options are restored. On
    /// success the options that should drive the resumed session are returned.
    pub fn validate_for_launch(
        &self,
        model_path: &Path,
        requested_options: &ChatOptions,
        options_specified: bool,
    ) -> Result<ChatOptions, String> {
        // Bind a saved conversation to the canonical model path to avoid silently resuming on another model.
        // The same validation and canonicalization used during archive creation
        // is applied here so both sides of the comparison use identical paths.
        ensure_model_path(model_path)?;
        let resolved_path = model_path.canonicalize().map_err(|error| {
            format!(
                "Failed to resolve model path '{}': {error}",
                model_path.display()
            )
        })?;
        // Compare the stored string with the resolved path. Both were produced
        // by canonicalization, so ordinary spelling differences do not matter.
        if resolved_path.to_string_lossy().as_ref() != self.model_path.as_str() {
            return Err(format!(
                "Context '{}' is bound to model '{}', not '{}'.",
                self.name,
                self.model_path,
                resolved_path.display()
            ));
        }
        // Only enforce option equality when the user actually asked for
        // specific options. With no explicit options, the archived settings
        // are the desired settings and are restored below.
        if options_specified && requested_options != &self.options {
            return Err(format!(
                "Inference options saved in context '{}' do not match the command-line options; omit inference options to restore the archived settings, or use the same options.",
                self.name
            ));
        }
        // Return a clone so the caller can move the options into the runtime
        // without borrowing the archive or mutating the saved record.
        Ok(self.options.clone())
    }
}

/// Filesystem-facing handle for loading, saving, listing, and deleting
/// context archives.
///
/// The store deliberately does not hold any open handles or cached data. Every
/// operation resolves the directory, validates names, and touches the
/// filesystem afresh, which keeps concurrent CLI invocations from observing
/// stale in-memory state.
#[derive(Debug, Clone)]
pub struct ContextStore {
    /// Explicit archive directory, or `None` to use the platform default.
    directory: Option<PathBuf>,
}

/// Storage operations for context archives.
impl ContextStore {
    /// Create a store that writes either to `directory` or to the platform
    /// default when `None` is supplied.
    ///
    /// This constructor is purely in-memory; it does not create or otherwise
    /// touch the directory. Creation happens lazily on the first save.
    pub fn new(directory: Option<PathBuf>) -> Self {
        Self { directory }
    }

    /// Load and validate a single named archive.
    ///
    /// The name is validated before being turned into a path. The file must
    /// contain valid JSON for the current schema, and the name stored inside
    /// the archive must match the requested name. Requiring both checks
    /// prevents a renamed or misplaced file from being loaded under a
    /// misleading identity.
    pub fn load(&self, name: &str) -> Result<ContextArchive, String> {
        // Resolve the directory on every call so an explicitly configured
        // store and the platform default behave consistently.
        let directory = self.directory()?;
        // `archive_path` rejects unsafe names before any filesystem access.
        let path = Self::archive_path(&directory, name)?;
        // A missing file or unreadable path becomes a descriptive I/O error.
        let bytes = fs::read(&path).map_err(|error| {
            format!(
                "Failed to read context '{name}' ({}): {error}",
                path.display()
            )
        })?;
        // Deserialize first, then validate the semantic constraints. Parsing
        // errors and unsupported versions get distinct messages.
        let archive: ContextArchive = serde_json::from_slice(&bytes)
            .map_err(|error| format!("Context '{name}' is not a valid JSON archive: {error}"))?;
        archive.validate()?;
        // The on-disk filename and the embedded name must agree; otherwise a
        // copied or hand-edited archive could be resumed under the wrong name.
        if archive.name != name {
            return Err(format!(
                "Name in context file '{}' does not match the requested name.",
                path.display()
            ));
        }
        Ok(archive)
    }

    /// Persist an archive, updating its modification timestamp.
    ///
    /// The caller's archive is not mutated: a clone receives the fresh
    /// `updated_at` value. The directory is created if necessary, and the file
    /// is replaced atomically via a temporary sibling file. Validation runs
    /// before any filesystem mutation, so an invalid archive never creates a
    /// directory or temp file.
    pub fn save(&self, archive: &ContextArchive) -> Result<(), String> {
        // Work on a clone so stamping `updated_at` does not surprise callers
        // that still hold the archive they passed in.
        let mut archive = archive.clone();
        archive.updated_at = unix_timestamp()?;
        // Validate after stamping so the persisted record is exactly what was
        // checked.
        archive.validate()?;
        let directory = self.directory()?;
        // Create the full directory chain; this is harmless if it already
        // exists and makes the default location work on a first save.
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "Failed to create context directory '{}': {error}",
                directory.display()
            )
        })?;
        // Tighten permissions only for the implicit platform directory. A
        // user-provided --context-dir may intentionally have broader or
        // managed permissions, so it is left untouched.
        #[cfg(unix)]
        if self.directory.is_none() {
            fs::set_permissions(&directory, Permissions::from_mode(0o700)).map_err(|error| {
                format!(
                    "Failed to restrict permissions on the default context directory '{}': {error}",
                    directory.display()
                )
            })?;
        }
        // Re-validate the name through `archive_path` before using it in a
        // path join; this is the last line of defense against traversal.
        let target = Self::archive_path(&directory, &archive.name)?;
        // Pretty JSON is used for readability and easy manual inspection. The
        // complete payload is held in memory so a short write cannot leave a
        // valid-looking prefix behind.
        let bytes = serde_json::to_vec_pretty(&archive)
            .map_err(|error| format!("Failed to serialize context '{}': {error}", archive.name))?;
        // Stage the complete archive in a private sibling file before replacing the target.
        // The file is created with `create_new`, so an existing path (including
        // a leftover from a crashed process) is never truncated. On collision we
        // generate another unique name and retry.
        let (temp_path, mut file) = loop {
            let temp_path = Self::temporary_path(&directory, &archive.name);
            let mut open_options = OpenOptions::new();
            open_options.write(true).create_new(true);
            // Restrict temp-file permissions up front on Unix; relying on the
            // process umask alone could briefly expose conversation data.
            #[cfg(unix)]
            open_options.mode(0o600);
            match open_options.open(&temp_path) {
                Ok(file) => break (temp_path, file),
                // A collision is expected under concurrency; pick a new name.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(format!(
                        "Failed to create temporary context file '{}': {error}",
                        temp_path.display()
                    ));
                }
            }
        };
        // Write the whole buffer and flush it to stable storage before the
        // rename. If either step fails, remove the partial temp file and leave
        // the previous archive untouched.
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "Failed to write context '{}': {error}",
                archive.name
            ));
        }
        // Close the handle explicitly before renaming. This matters on
        // platforms that do not permit renaming an open file, and it also
        // makes the ownership boundary clear.
        drop(file);
        // Renaming within the same directory replaces the target in one
        // atomic filesystem step. Clean up the temp file if the rename fails
        // so a failed save does not leave clutter behind.
        if let Err(error) = fs::rename(&temp_path, &target) {
            let _ = fs::remove_file(&temp_path);
            return Err(format!(
                "Failed to update context '{}' ({}): {error}",
                archive.name,
                target.display()
            ));
        }
        Ok(())
    }

    /// List all valid archives in the store directory, sorted by name.
    ///
    /// Non-JSON entries are ignored. A missing directory is treated as an
    /// empty store, but a present directory containing an invalid archive
    /// produces an error instead of silently hiding the problem.
    pub fn list(&self) -> Result<Vec<ContextArchive>, String> {
        let directory = self.directory()?;
        // A never-used store has no directory yet; that is not an error for
        // listing, so map only `NotFound` to an empty result.
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!(
                    "Failed to read context directory '{}': {error}",
                    directory.display()
                ));
            }
        };
        let mut archives = Vec::new();
        // Iterate eagerly over directory entries. The order supplied by the
        // filesystem is unspecified, hence the sort near the end.
        for entry in entries {
            let entry = entry
                .map_err(|error| format!("Failed to read context directory entry: {error}"))?;
            let path = entry.path();
            // Only regular `.json` names are candidates. The extension check is
            // exact and case-sensitive, which also excludes temporary `.tmp`
            // staging files from interrupted saves.
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path).map_err(|error| {
                format!("Failed to read context file '{}': {error}", path.display())
            })?;
            let archive: ContextArchive = serde_json::from_slice(&bytes).map_err(|error| {
                format!(
                    "Context file '{}' is not a valid JSON archive: {error}",
                    path.display()
                )
            })?;
            // Validate before adding to the result so the CLI never presents
            // an archive that could not later be loaded.
            archive.validate()?;
            // The file stem must match the embedded name; otherwise the
            // listing would report a name that `load` cannot use.
            if path.file_stem().and_then(|stem| stem.to_str()) != Some(archive.name.as_str()) {
                return Err(format!(
                    "Filename of context file '{}' does not match the archive name.",
                    path.display()
                ));
            }
            archives.push(archive);
        }
        // Stable ordering makes the CLI listing predictable across filesystem implementations.
        // Sorting by the embedded name (which must match the file stem) gives
        // users a deterministic view regardless of directory iteration order.
        archives.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(archives)
    }

    /// Delete the named archive from the store.
    ///
    /// The name is validated before path construction. Missing files are
    /// reported as errors so the caller can distinguish a successful deletion
    /// from a typo or an already-removed context.
    pub fn delete(&self, name: &str) -> Result<(), String> {
        let directory = self.directory()?;
        // This call also performs the traversal-safety name validation.
        let path = Self::archive_path(&directory, name)?;
        // Unlike listing, deletion has no special case for a missing file:
        // the I/O error carries the path for a useful diagnostic.
        fs::remove_file(&path).map_err(|error| {
            format!(
                "Failed to delete context '{name}' ({}): {error}",
                path.display()
            )
        })
    }

    /// Resolve the configured directory or fall back to the platform default.
    ///
    /// A custom directory is returned as-is and is not canonicalized or
    /// checked for existence here; filesystem operations report any problems.
    fn directory(&self) -> Result<PathBuf, String> {
        match &self.directory {
            Some(directory) => Ok(directory.clone()),
            None => default_context_directory(),
        }
    }

    /// Build the archive path for `name` after validating it as a safe
    /// filename component.
    ///
    /// Because `validate_name` rejects path separators, `.` and `..`, the
    /// resulting path is always a direct child of `directory`.
    fn archive_path(directory: &Path, name: &str) -> Result<PathBuf, String> {
        validate_name(name)?;
        Ok(directory.join(format!("{name}.json")))
    }

    /// Generate a unique temporary filename in the target directory.
    ///
    /// The leading dot hides the file on Unix, and the `.tmp` extension keeps
    /// it out of `list`. Uniqueness combines the process id with a
    /// process-local counter. A collision can still occur if a process id is
    /// reused after a crash; the caller's `create_new` loop handles that case.
    fn temporary_path(directory: &Path, name: &str) -> PathBuf {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        directory.join(format!(".{name}.{}.{}.tmp", std::process::id(), counter))
    }
}

/// Verify that `path` looks like a supported model location.
///
/// A directory is accepted directly. A file is accepted only when its
/// extension is `safetensors`, compared case-insensitively. The extension
/// check intentionally does not require the file to exist: canonicalization by
/// the caller will surface a missing or unreadable path with a clearer
/// context-specific error. Symlinks are followed by the later `is_dir` check
/// and canonicalization.
fn ensure_model_path(path: &Path) -> Result<(), String> {
    let is_safetensors_file = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
    if !path.is_dir() && !is_safetensors_file {
        return Err(format!(
            "Chat contexts require a model directory or Safetensors file: {}",
            path.display()
        ));
    }
    Ok(())
}

/// Validate a context name for safe use as a filename component.
///
/// Allowed names are 1 to 64 bytes long and consist only of ASCII letters,
/// digits, periods, hyphens, and underscores. The special components `.` and
/// `..` are explicitly rejected. There are no path separators, so a validated
/// name cannot escape the store directory. The check is platform-neutral:
/// names that pass here may still be rejected by the underlying filesystem,
/// for example Windows device names or names with trailing periods.
fn validate_name(name: &str) -> Result<(), String> {
    // The length is measured in bytes, but because only ASCII is allowed the
    // byte length and character length are the same for valid names.
    if name.is_empty()
        || name.len() > 64
        || name == "."
        || name == ".."
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "Context names must be 1-64 ASCII letters, digits, periods, hyphens, or underscores, and cannot be '.' or '..'."
                .into(),
        );
    }
    Ok(())
}

/// Compute the platform-specific default directory for context archives.
///
/// Windows uses `LOCALAPPDATA` and falls back to `APPDATA`. macOS uses
/// `$HOME/Library/Application Support`. Other Unix systems honor an absolute
/// `XDG_DATA_HOME` and otherwise use `$HOME/.local/share`. All variants place
/// archives under a `cheetah/contexts` subdirectory. The directory is not
/// created here; `ContextStore::save` does that with the correct permissions.
fn default_context_directory() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    {
        let base = std::env::var_os("LOCALAPPDATA")
            .or_else(|| std::env::var_os("APPDATA"))
            .ok_or_else(|| {
                "Unable to determine the user data directory; specify one with --context-dir."
                    .to_string()
            })?;
        return Ok(PathBuf::from(base).join("cheetah").join("contexts"));
    }

    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            "Unable to determine the user data directory; specify one with --context-dir."
                .to_string()
        })?;
        return Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("cheetah")
            .join("contexts"));
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // The XDG specification says relative values must be ignored, so only
        // an absolute `XDG_DATA_HOME` is honored.
        if let Some(base) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
            && base.is_absolute()
        {
            return Ok(base.join("cheetah").join("contexts"));
        }
        let home = std::env::var_os("HOME").ok_or_else(|| {
            "Unable to determine the user data directory; specify one with --context-dir."
                .to_string()
        })?;
        return Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("cheetah")
            .join("contexts"));
    }

    // Compilation targets not covered above have no known data-directory
    // convention. The attribute keeps the fallback reachable in the compiler's
    // view on supported platforms without introducing warnings.
    #[allow(unreachable_code)]
    Err("Unable to determine the user data directory on this platform; specify one with --context-dir.".into())
}

/// Return the current time as whole seconds since the Unix epoch.
///
/// Timestamps have one-second resolution, so two saves within the same second
/// may receive equal values. That is acceptable for display purposes; ordering
/// of dependent changes should not rely on these timestamps. A system clock
/// set before the epoch is reported as an error rather than wrapping around.
fn unix_timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("System time is earlier than the Unix epoch; unable to create context archive: {error}"))
}

/// Unit tests for archive validation, persistence, and model binding.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::DEFAULT_CHAT_OPTIONS;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counter that keeps per-test temporary directories unique within the
    /// test process, even when tests run in parallel.
    static TEST_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create an isolated store backed by a unique temporary directory.
    ///
    /// Any leftover directory with the same generated name is removed first.
    /// The caller is responsible for cleaning up the directory at the end of
    /// the test.
    fn test_store() -> (ContextStore, PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "cheetah-context-test-{}-{}",
            std::process::id(),
            TEST_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        (ContextStore::new(Some(path.clone())), path)
    }

    /// Build a valid, fully populated archive for tests.
    ///
    /// The fake model path is never touched because this helper only exercises
    /// in-memory validation and serialization.
    fn test_archive(name: &str) -> ContextArchive {
        ContextArchive {
            schema_version: SCHEMA_VERSION,
            name: name.into(),
            model_path: "/models/test.safetensors".into(),
            options: DEFAULT_CHAT_OPTIONS,
            turns: vec![ConversationTurn {
                user: "hello".into(),
                assistant: "world".into(),
            }],
            created_at: 1,
            updated_at: 1,
        }
    }

    /// Names containing traversal, separators, spaces, or non-ASCII bytes must
    /// be rejected, while a normal name with dots and punctuation is allowed.
    #[test]
    fn rejects_unsafe_context_names() {
        for name in ["", ".", "..", "../outside", "a/b", "name with spaces", "é"] {
            assert!(validate_name(name).is_err(), "accepted {name:?}");
        }
        assert!(validate_name("valid_name-1.json").is_ok());
    }

    /// Saving twice replaces the same archive, loading returns the latest
    /// turns, listing sees it, and deletion removes it from subsequent lists.
    #[test]
    fn saves_lists_loads_and_replaces_contexts() {
        let (store, directory) = test_store();
        let mut archive = test_archive("session");
        store.save(&archive).unwrap();
        // The second save must replace the first file atomically rather than
        // appending or creating a second archive.
        archive.turns[0].assistant = "updated".into();
        store.save(&archive).unwrap();
        let loaded = store.load("session").unwrap();
        assert_eq!(loaded.turns, archive.turns);
        // Timestamps have one-second granularity, so equality is possible.
        assert!(loaded.updated_at >= archive.updated_at);
        assert_eq!(store.list().unwrap()[0].name, "session");
        store.delete("session").unwrap();
        assert!(store.list().unwrap().is_empty());
        fs::remove_dir_all(directory).unwrap();
    }

    /// Malformed JSON is reported as a parse error, and a newer schema version
    /// is reported as unsupported instead of being interpreted best-effort.
    #[test]
    fn reports_corrupt_and_unsupported_archives() {
        let (store, directory) = test_store();
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("broken.json"), b"{").unwrap();
        assert!(store.load("broken").unwrap_err().contains("valid JSON"));
        fs::remove_file(directory.join("broken.json")).unwrap();

        let mut archive = test_archive("unsupported");
        archive.schema_version = SCHEMA_VERSION + 1;
        fs::write(
            directory.join("unsupported.json"),
            serde_json::to_vec(&archive).unwrap(),
        )
        .unwrap();
        assert!(
            store
                .load("unsupported")
                .unwrap_err()
                .contains("unsupported")
        );
        fs::remove_dir_all(directory).unwrap();
    }

    /// Older option JSON without `ctx_size` must still deserialize, using the
    /// documented default context size.
    #[test]
    fn older_archives_default_context_size_to_8192() {
        // Remove the field from a serialized current default so the test
        // exercises the deserialization default rather than a hand-built value.
        let mut options = serde_json::to_value(DEFAULT_CHAT_OPTIONS).unwrap();
        options.as_object_mut().unwrap().remove("ctx_size");
        let restored: ChatOptions = serde_json::from_value(options).unwrap();
        assert_eq!(restored.ctx_size, crate::options::DEFAULT_CONTEXT_TOKENS);
    }

    /// A store whose directory does not exist lists nothing, and load/delete
    /// still fail with normal I/O errors.
    #[test]
    fn empty_missing_directory_lists_no_contexts() {
        let (store, directory) = test_store();
        assert!(store.list().unwrap().is_empty());
        assert!(store.load("missing").is_err());
        assert!(store.delete("missing").is_err());
        let _ = fs::remove_dir_all(directory);
    }

    /// Launch validation binds an archive to one canonical model path and
    /// rejects a different model or mismatched explicitly supplied options.
    #[test]
    fn validates_model_identity_and_saved_options() {
        let (_, directory) = test_store();
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("model.safetensors");
        fs::write(&model_path, b"").unwrap();
        let mut archive = ContextArchive::new(&model_path, DEFAULT_CHAT_OPTIONS).unwrap();
        archive.name = "session".into();
        assert_eq!(
            archive
                .validate_for_launch(&model_path, &DEFAULT_CHAT_OPTIONS, false)
                .unwrap(),
            DEFAULT_CHAT_OPTIONS
        );

        // A different canonical path must not be accepted as the same model.
        let other_model = directory.join("other.safetensors");
        fs::write(&other_model, b"").unwrap();
        assert!(
            archive
                .validate_for_launch(&other_model, &DEFAULT_CHAT_OPTIONS, false)
                .unwrap_err()
                .contains("not")
        );
        // Explicitly supplied options that differ from the archive are an
        // error rather than a silent override.
        let changed_options = ChatOptions {
            max_tokens: DEFAULT_CHAT_OPTIONS.max_tokens - 1,
            ..DEFAULT_CHAT_OPTIONS
        };
        assert!(
            archive
                .validate_for_launch(&model_path, &changed_options, true)
                .unwrap_err()
                .contains("Inference options")
        );

        // Negative temperature is outside the accepted range and must fail
        // basic archive validation.
        archive.options.temperature = -1.0;
        assert!(archive.validate().unwrap_err().contains("invalid"));
        let _ = fs::remove_dir_all(directory);
    }

    /// Context creation accepts a Safetensors file but rejects other file
    /// extensions and nonexistent directory paths.
    #[test]
    fn accepts_safetensors_paths_and_rejects_other_formats() {
        let path = std::env::temp_dir().join(format!(
            "cheetah-context-model-{}.safetensors",
            std::process::id()
        ));
        fs::write(&path, b"").unwrap();
        assert!(ContextArchive::new(&path, DEFAULT_CHAT_OPTIONS).is_ok());
        assert!(ContextArchive::new(Path::new("model.gguf"), DEFAULT_CHAT_OPTIONS).is_err());
        fs::remove_file(path).unwrap();
        assert!(ContextArchive::new(Path::new("model-directory"), DEFAULT_CHAT_OPTIONS).is_err());
    }
}
