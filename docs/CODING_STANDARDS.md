# Coding standards & complete function reference

This document defines Cheetah's coding conventions and contains the analysis
of **every function in the codebase** (section 10): 931 functions across 40
source files — runtime functions, FFI declarations, and tests — each with a
one-entry explanation of what it does and why it exists the way it does.

Companion guides: [CODEBASE.md](CODEBASE.md) (repository map),
[REGISTRY.md](REGISTRY.md) (model resolution),
[ANT_MEMORY.md](ANT_MEMORY.md) (Ant memory tiers).

---

## 1. Language, formatting & layout

1. **Rust, stable toolchain, one crate.** The crate is primarily a binary
   (`main.rs`); `lib.rs` exposes only the FFI module (`native`) that wraps
   the three C translation units.
2. **`cargo fmt` is law.** Every commit is formatted; no exceptions. Comment
   lines wrap at ~100 columns to match rustfmt's default width.
3. **Flat module layout.** No source file lives more than two directories
   below `src/` (`src/chat/registrations/llama.rs`, `src/chat/native/ops.rs`).
   A new subsystem is a new sibling directory or a new file — never a new
   nesting level.
4. **One family, one file.** Each model family (llama, qwen2, ...) is a
   single registration file under `src/chat/registrations/` that carries BOTH
   its declarative capability (architectures, template, stop markers,
   backends, family constants) AND its forward assembly. Adding a family is
   adding one file; no core-code edits.
5. **One format, one file.** Weight precisions (f32/f16/bf16/q4/q8) follow
   the same rule under `src/chat/precision/`: operators + registration entry
   in one file; build.rs scans the directory.
6. **Generated code is never edited.** `OUT_DIR/model_registry.rs` and
   `OUT_DIR/precision_registry.rs` are produced by build.rs from the
   registration directories; the checked-in sources of truth are the
   registration files.
7. **Test modules live at the end of a file** under `#[cfg(test)] mod tests`
   (or `mod <name>_tests`). Nothing below that marker is runtime code.
8. **C code follows the Rust docs.** `cheetah_init.c`, `cheetah_memory.c`,
   `cheetah_vulkan.c` implement exactly the contracts the Rust FFI module
   (`src/native.rs`) documents; headers `cheetah_*.h` carry the same story.

## 2. Naming conventions

| Item | Convention | Examples |
|------|-----------|----------|
| Types / structs / enums | `PascalCase`; runtime structs carry the family suffix | `OpsToolbox`, `LlamaNative`, `VulkanPhiExecutor`, `MmapWeightStore` |
| Functions / methods | `snake_case`, verb-first; predicates read as questions | `run_forward`, `sample_token`, `is_model_directory`, `ant_within_budget` |
| Ant page-level ops | keep the historical names (`page_in_unit`, `evict_unit`, `ant_swap_in_unit`) — they are part of the frozen contract | |
| Constants | `SCREAMING_SNAKE_CASE` | `GEMV_CHUNK_ROWS`, `DEFAULT_CONTEXT_TOKENS`, `SSE_HEARTBEAT_INTERVAL` |
| Environment variables | `CHEETAH_*` prefix, documented in CODEBASE.md §4 | `CHEETAH_DEVICE`, `CHEETAH_QUANT_Q8` |
| Unit ids | exact Hugging Face tensor names, including the `.weight` suffix — the units manifest and the Vulkan layout strings must agree bit for bit | `model.layers.0.self_attn.q_proj.weight` |
| Test functions | full snake_case sentences naming the guarantee, not the mechanism | `dense_forward_is_parity_clean_across_residency_transitions` |
| Fixture helpers | `ref_*` prefix for reference math, `write_*` for fixture builders | `ref_mat`, `write_manifest` |

**Frozen names.** The low-level math kernels (`gemv`, `rmsnorm`,
`rmsnorm_offset`, `layernorm`, `rope_in_place`, `rope_heads`, `silu_multiply`,
`gelu`, `gelu_multiply`, `softmax`, `scalar_at`, `dequant_values`, `gemv_q8`,
`gemv_q4`) are frozen: the precision registry wires them by name and
`src/chat/native/kernels.rs` re-exports them for compatibility. Do not
rename them; extend the toolbox with new names instead.

## 3. Comment & documentation conventions (English)

Every runtime function carries an English `///` doc comment (extern FFI
declarations carry `//` lines — doc comments never attach inside extern
blocks). The rules:

1. **One summary line first.** A single sentence, third person, that says
   what the function does — never a restatement of the name
   (`/// Reject an empty token batch and ids beyond the vocabulary.`).
2. **Paragraphs for non-obvious semantics.** Contracts, error conditions,
   memory behavior, and argument units go in the following lines. If a
   function's doc is only its name repeated, the doc is wrong.
3. **Math is written as math.** Formulas appear inline with Unicode:
   `x / sqrt(mean(x²) + eps) · weight`, `0.5 · x · (1 + tanh(√(2/π) · (x +
   0.044715 x³)))`.
4. **Traps get a `NOTE`.** Historical bugs that could recur carry an explicit
   warning — the canonical example is
   `// NOTE: f32::sin_cos returns (sin, cos) — do not swap these.`
5. **Unsafe blocks get a `// SAFETY:` line** stating the invariant that makes
   the call sound (alignment, length, lifetime), e.g. the byte-view helpers.
6. **Modules open with `//!`** describing ownership and layout: what the
   module owns, what it deliberately does not (e.g. ops.rs: "operators and
   state, not model structure — no family branch anywhere here").
7. **Cross-references by path, not vibes.** Docs point at the counterpart
   module or shader (`the same math the matmul_q8.comp shader performs`).
8. **Tests are sentences.** A test name states the guarantee; a `///` doc is
   added only when setup context matters (env-var gating, fixture quirks).
9. **Bilingual inline comments are tolerated** where a reviewer wrote them
   (serve/, queue.rs), but new doc comments are English-only.

## 4. Visibility conventions

- **Private by default.** A helper reachable from one module stays private.
- **`pub(super)` / `pub(crate)`** for the deliberate internal surface:
  boundary types in `registry.rs` are `pub(crate)` so the lookup functions
  can carry them; toolbox ops are `pub(crate)` because registration files
  consume them.
- **`pub` only at the crate edge**: `chat::chat` (public entry),
  `native::native` FFI module, `options::ChatOptions`, CLI option structs.
- **The dyn boundary is one**: `Box<dyn ModelForward>` at the engine seam.
  Everything below is static dispatch; no `Box<dyn>` in hot paths, no
  trait objects for operators (the precision tables are `fn`-pointer consts,
  copied by value — one table lookup, zero vtable calls per tile).

## 5. Error handling conventions

1. **`Result<T, String>` inside the library**, `RunError` at the CLI edge
   (`Usage` → exit code 2, `Runtime` → exit code 1).
2. **Errors carry their subsystem prefix**: `"Native backend: ..."`,
   `"Vulkan {model} ..."`, `"Ant ..."`, so a log line identifies the layer
   without a backtrace.
3. **Fail loudly, never silently.** A missing manifest field, an ambiguous
   model name, an inconsistent expert count — all abort with a message that
   says what to do (run `cheetah init`, keep config.json, ...). Silent
   fallbacks are only acceptable where the doc states the fallback contract
   (e.g. resident-upload failure falls back to streaming, with a yellow
   warning line).
4. **Validate at the edge.** Request parsing (serve/openai.rs), CLI parsing
   (cli.rs), and config resolution (`DecoderConfig::resolve`,
   `VulkanTransformerConfig::from_raw`) reject bad input before any model
   work starts; HTTP errors map to status codes at that boundary
   (`prompt_error_status`).
5. **Mutex poisoning never panics the server**: lock helpers unwrap with
   `unwrap_or_else(|poisoned| poisoned.into_inner())`.

## 6. Performance & memory conventions

1. **Zero per-token allocation on the hot path.** Activation buffers are
   preallocated once in `Scratch::new` and reused via `clear` + `resize`;
   the only per-forward allocation is the returned logits vector.
2. **Weights are never copied.** The native backend consumes mmap bytes in
   place; the Vulkan backend uploads once (resident bucket) or streams tiles.
3. **Disk tier means real released pages** (`madvise(MADV_DONTNEED)`); any
   change that keeps disk-tier bytes resident breaks the Ant contract. Every
   layer step calls `drop_disk_units_of_layer`; the lm head sweeps in 16k-row
   chunks and drops each chunk.
4. **Every swap re-checks the budget before executing** (`ant_swap_in_unit`,
   `ant_swap_fit`) — CODEBASE.md §6 invariant 4.
5. **Kernel changes require the isolation tests** (`dot_row_bf16_fused` vs
   dequant+dot on exact bf16 integers) AND a native forward-parity run.
6. **Bit-exactness over cleverness.** bf16 widening is a bit shift; the fused
   AVX2 path must track the portable reference to a few ULPs (measured by
   test, not assumed).

## 7. Architecture conventions (control flow vs. operators)

1. **The main program contains no family branches.** No
   `if model_type == "qwen"` anywhere outside registration files; the native
   OpsToolbox and the Vulkan executors are family-free.
2. **Registration files own structure; the toolbox owns math.** Residual
   topology, projection naming, norm flavor, activation — all live in the
   family's registration file, composed from `OpsToolbox` ops
   (`linear`, `rmsnorm` / `rmsnorm_offset` / `layernorm`, `qk_norm`,
   `add_bias`, `rope_heads`, `cache_append`, `attention`, `lm_head_into`,
   `drop_disk_units_of_layer`).
3. **The registry is generated.** build.rs scans `src/chat/registrations/`
   and `src/chat/precision/`; every lookup (loaders, templates, stop
   markers, family constants) derives from the generated `REGISTRY` tables.
4. **Unit-optional ops keep compositions branch-free.** `add_bias` and
   `qk_norm` no-op when the unit is absent, so llama (no biases) and qwen2
   (biases) share one code shape without flags.
5. **Config-data dispatch, not name dispatch.** Dense-vs-MoE switches on
   `num_local_experts > 0`; the sliding window is resolved config data
   flowing into the attention op. `DenseFeatures` remains only as
   resolution metadata for the parameter fallback chain.
6. **Tools have one protocol shape and one execution point.** The OpenAI
   function-calling JSON shapes live in `chat/toolcalls.rs` (shared by the
   HTTP API, the REPL, and the MCP layer); tool EXECUTION lives in the MCP
   client (`mcp.rs`) — the engine's agentic loop
   (`Engine::generate_tool_chat`) is MCP-agnostic and only routes calls to
   `Option<&McpRegistry>`. Client-declared (non-MCP) tool calls keep the
   client-driven OpenAI contract: they return to the caller.

## 8. Testing conventions

1. **Test modules at file end**, `#[cfg(test)]`; helper fns inside them are
   test-context code.
2. **Reference math is the oracle.** gemma/phi suites re-implement the whole
   forward with plain tensor math and compare to 1e-4; the MoE suite
   re-derives router softmax + expert mix independently.
3. **Parity across residency transitions.** Dense and sparse paths must
   produce bit-identical logits across evict-to-disk / page-back-in cycles.
4. **Device-gated tests opt in via env** (`CHEETAH_TEST_VULKAN_Q8`) or skip
   when artifacts are absent (`ant_lifecycle_against_real_prepared_model`).
5. **Fixtures are deterministic** (no RNG): the `pattern()` helper varies
   tensors so masking/routing bugs cannot hide behind constant inputs.
6. **Cross-backend contract tests** (weights.rs `quantized_formats_match_across_backends`)
   automatically cover any newly registered precision.

## 9. Documentation map

| Document | Covers |
|----------|--------|
| CODEBASE.md | repository tree, data flows, module index, env vars, test matrix |
| REGISTRY.md | declarative registration mechanism, lookups, add-a-family checklist |
| ANT_MEMORY.md | Ant tiering deep dive: planner, page control, forward pipeline |
| CODING_STANDARDS.md (this file) | conventions + complete per-function reference |


---

## 10. Complete function reference

Every function in the codebase, module by module. `*(test)*` marks
test-context functions (inside `#[cfg(test)]` modules); everything else is
runtime code. Analyses state what the function does and the contract it
enforces — the source doc comments carry the same story in place.

### 10.1 `src/main.rs` — CLI dispatch (16 runtime + 4 tests)

| Function | Analysis |
|---|---|
| `main() -> ExitCode` | Entry point: delegates to `run`, prints the error, maps the [`RunError`] kind to the process exit code (2 usage / 1 runtime). |
| `run(args) -> Result<(), RunError>` | Argument dispatch for every subcommand (chat/serve/init/inspect/context/formats/help); also handles the bare-invocation chat-from-cwd case. |
| `run_init_command(args)` | `cheetah init <src> <dst>`: native model preparation (C `cheetah_init_model`), manifest metadata, then Ant settings resolution and page-in of the planned resident units. |
| `resolve_ant_settings(model_dir, ant_options)` | Resolves effective Ant settings through the chain defaults → user preset file (`--setup` wizard) → CLI overrides; every execution consumes exactly this resolved value. |
| `split_ant_args(args)` | Value-aware extraction of `--ant*`/`--setup` flags from a mixed argument list, so `serve` can parse the rest while Ant options travel to the engine. |
| `run_inspect_command(args)` | `cheetah inspect`: loads model metadata (shards/manifest), renders text or JSON, optionally writes `-o <file>`. |
| `write_report(path, report)` | Writes an inspect report to a file (overwrites), appending a trailing newline. |
| `apply_ant_overrides(settings, ant_options)` | Applies parsed CLI Ant flags onto a settings struct (defaults → preset → CLI order). |
| `run_chat(path, requested, context, ant)` | `cheetah chat`: loads the engine with a progress spinner, validates/loads the context archive, hands everything to `chat::chat`. |
| `run_context_command(operation, directory)` | `cheetah context list/delete`: manages saved conversation archives in the archive directory. |
| `runtime(result)` | Adapts a backend `Result<_, String>` into [`RunError::Runtime`]. |
| `print_formats()` | Prints the supported model formats (`cheetah formats`). |
| `print_help()` | Prints the full CLI help text (`cheetah help` / `--help`). |
| `render_model(info, show_all)` | Terminal rendering of an inspect report; the tensor table is bounded to `DEFAULT_TENSOR_LIMIT` entries unless `show_all`. |
| `render_json(info)` | JSON rendering of an inspect report (unbounded tensor list, unlike the bounded terminal view). |
| `format_size(size)` | Formats a byte count in binary units (B/KB/MB/GB/TB, 1024-based). |
| `inspect_text_export_limits_tensors_by_default` *(test)* | Verifies the terminal tensor table is bounded while `show_all` lifts the bound. |
| `inspect_json_export_preserves_structured_schema` *(test)* | Verifies the JSON export keeps the full structured schema. |
| `inspect_report_export_overwrites_existing_file` *(test)* | Verifies `-o` overwrites an existing report file. |
| `no_arguments_launches_chat_from_the_current_directory` *(test)* | Verifies the bare-invocation behavior documented in `run`. |

### 10.2 `src/cli.rs` — argument parsing (20 runtime + 26 tests)

| Function | Analysis |
|---|---|
| `RunError::default()` | Derives a Display/std::error impl default; the safest out-of-the-box posture for the two error kinds. |
| `RunError::usage(message)` | Convenience constructor for the usage-error variant. |
| `RunError::exit_code()` | Maps the error kind to the process exit code (2 = usage, 1 = runtime). |
| `fmt()` (Display impl) | Renders the error message; usage vs runtime share one presentation. |
| `parse_chat_args(args)` | Parses `cheetah chat <model> [flags]` into the model path, chat options, context options and Ant options (defaults → flags). |
| `parse_context_args(args)` | Parses `cheetah context list \| delete <name> [--context-dir <path>]`. |
| `parse_serve_args(args)` | Parses `cheetah serve` options (host/port/key/queue limits) plus the per-request generation defaults. |
| `parse_inspect_args(args)` | Parses `cheetah inspect <model-path> [--all] [--json] [-o <file>]`. |
| `parse_positive_option(args, i, option)` | Parses an option value that must be a strictly positive integer (> 0). |
| `parse_nonnegative_option(args, i, option)` | Parses an option value that must be a non-negative integer (>= 0). |
| `parse_u64_option(args, i, option)` | Parses the next argument as u64 (Ant budgets and seeds). |
| `parse_text_option(args, i, option)` | Consumes the next argument as a raw text value (paths, unit ids, prompts). |
| `is_boolean_literal(value)` | Whether a token is a bool literal — lets flags like `--ant` stay value-free while `--ant=true` still works. |
| `parse_bool_option(args, i, option)` | Parses a boolean flag value from the accepted literal set (see [`is_boolean_literal`]); anything else is a usage error. |
| `AntCliOptions::resolve()` | Collapses the parsed Ant flags into concrete settings (unspecified fields fall back to the built-in defaults). |
| `parse_setup_flag(args, i, option)` | Parses the `--setup`/`--ant-setup` interactive flag without consuming the next token (the caller's loop still applies its trailing `index += 1`). |
| `parse_ant_args(args)` | Parses all `--ant*` flags into [`AntCliOptions`]; used by both chat and the serve-side pre-extraction. |
| `parse_float_option(args, i, option, min)` | Parses a float with a lower bound `min` (NaN rejected). |
| `parse_bounded_float_option(args, i, option, min, max)` | Parses a float constrained to the inclusive range `[min, max]`. |
| `parse_float_raw(args, i, option)` | Parses the next argument as a finite f64, rejecting NaN up front. |
| 29 `parses_*` / `rejects_*` / `accepts_*` tests *(test)* | Cover every parser: chat speed/sampling/system flags, deterministic defaults, out-of-range rejections, Ant flag shapes (bare flag, `=value`, preset override, `--setup` aliases), context commands, serve defaults + overrides + ephemeral port + key generation, mutually exclusive keys, invalid ports/limits, multiple models, inspect output options, and the usage-error variant mapping. |

### 10.3 `src/color.rs` — terminal coloring (4 runtime + 1 test)

| Function | Analysis |
|---|---|
| `Color::ansi_code()` | Maps the color enum to its ANSI escape sequence (respecting NO_COLOR at the call sites below). |
| `stdout(text, color)` | Colorizes text destined for stdout (no-op when the target is not a TTY or `NO_COLOR` is set). |
| `stderr(text, color)` | Colorizes text destined for stderr; same rules as [`stdout`]. |
| `colorize(text, color, is_terminal, no_color)` | The shared implementation: applies the ANSI wrapper only when the target is a terminal AND color has not been disabled. |
| `applies_color_only_when_terminal_color_is_enabled` *(test)* | Verifies the TTY/NO_COLOR gating matrix. |

### 10.4 `src/options.rs` — generation options (2 runtime)

| Function | Analysis |
|---|---|
| `default_context_tokens()` | Serde default for archives written before `ctx_size` existed (returns `DEFAULT_CONTEXT_TOKENS`). |
| `ChatOptions::default()` | Interactive-chat defaults: identical to `DEFAULT_CHAT_OPTIONS` except that interactive chat injects the system prompt by default (the HTTP server does not). |

### 10.5 `src/context.rs` — conversation archive/store (15 runtime + 9 tests)

| Function | Analysis |
|---|---|
| `ContextArchive::new(model_path, options)` | Creates a fresh archive bound to a model path + options; errors are human-readable strings suitable for the CLI. |
| `ContextArchive::validate()` | Validates identity fields in isolation (no model path needed); deeper cross-checks live in `validate_for_launch`. |
| `ContextArchive::validate_for_launch(...)` | Cross-checks the archive against the model being launched and the explicitly supplied options; on success returns the options that should drive the resumed session. |
| `ContextStore::new(directory)` | Builds a store over an optional directory; never touches the directory. Creation happens lazily on the first save. |
| `load(name)` | Reads and parses one archive; a mismatched model id inside the file is reported rather than silently adopted (misleading identity). |
| `save(archive)` | Atomic save through a temp file + rename, so a crash never leaves a half-written archive in the directory. |
| `list()` | Lists all stored archives (sorted); unreadable entries produce an error instead of silently hiding the problem. |
| `delete(name)` | Deletes one archive; unknown names are distinct errors so typos are visible. |
| `directory()` | Resolves the effective store directory; existence is not checked here — filesystem operations report any problems. |
| `archive_path(directory, name)` | Joins a validated archive name into a direct child path (no traversal). |
| `temporary_path(directory, name)` | Temp-file path for atomic saves; stale temp files are reused after a crash via the caller's `create_new` loop. |
| `ensure_model_path(path)` | Validates that an archive's model path is absolute-ish and canonicalization succeeds. |
| `validate_name(name)` | Rejects path-unsafe archive names (traversal, separators; e.g. Windows device names or trailing periods). |
| `default_context_directory()` | Resolves the per-user default context directory; not created here — `ContextStore::save` does that with the correct permissions. |
| `unix_timestamp()` | Current UNIX time as u64; a clock set before the epoch is reported as an error rather than wrapping around. |
| 9 tests *(test)* | Cover unsafe-name rejection, save/list/load/replace cycles, corrupt + unsupported archive reporting, the pre-`ctx_size` default, empty stores, model-identity/option validation, and safetensors-path acceptance. |

### 10.6 `src/inspect.rs` — model inspection (3 runtime + 5 tests)

| Function | Analysis |
|---|---|
| `inspect_model(path)` | Loads a model's metadata report (files, tensors, sizes); errors on oversized cumulative sizes exceeding u64. |
| `model_files(path)` | Resolves the shard/weight file set for a model directory or file, rejecting unsafe manifests or unrecognized layouts. |
| `read_metadata(data)` | Parses a Safetensors header into a metadata map; errors on truncated or invalid JSON headers. |
| 5 tests *(test)* | Cover header parsing with synthetic safetensors files, hiding internal `__cheetah_padding_` tensors, invalid-format rejection, and manifest-driven shard resolution (sorted + deduplicated). |

### 10.7 `src/native.rs` — FFI wrappers over the C substrate (93 runtime — including 36 `extern` FFI declarations — + 9 tests)

| Function | Analysis |
|---|---|
| `InitProgress::new()` | Builds the progress tracker with the phase bars the native code actually enters. |
| `InitProgress::style()` | Shared indicatif style for the progress bars (the template is static; failures are impossible and reported with `expect`). |
| `InitProgress::start_bar(total, message)` | Starts (or restarts) a phase bar so the UI reflects work from the first progress update. |
| `cheetah_init_model_with_progress(...)` (extern) | C entry: prepares a model with periodic progress callbacks; non-zero status means failure (C prints the diagnostic to stderr). |
| `cheetah_ant_new()` (extern) | Creates the C Ant manager; the returned handle must reach `cheetah_ant_delete` exactly once. |
| `cheetah_ant_delete(ant)` (extern) | Frees the C manager; the Rust wrapper owns the lifetime explicitly. |
| `cheetah_ant_set_settings(ant, settings)` (extern) | Pushes settings into the C manager; status-code failure. |
| `cheetah_ant_load_model(ant, dst_dir)` (extern) | Loads a prepared model into the manager's unit index; requires `cheetah init` layout. |
| `cheetah_ant_unit_tier(...)` (extern) | Queries one unit's tier (0=vram, 1=ram, 2=disk); writes through the out-pointer, returns 0 on success. |
| `cheetah_ant_unit_pin(ant, unit_id, pin)` (extern) | Pins/unpins a unit, protecting it from eviction; the pin travels with the manager's identity. |
| `cheetah_ant_evict_unit(...)` (extern) | Evicts a unit to the disk tier; fails when the unit cannot be evicted (e.g. pinned). |
| `cheetah_ant_page_in(ant, unit_id, tier)` (extern) | Moves a unit into a target tier; the manager may reject the request when the tier lacks capacity. |
| `cheetah_ant_model_is_initialized(...)` (extern) | Whether the manager already loaded the model (manifest + units); Rust maps to bool. |
| `cheetah_ant_requires_init(ant)` (extern) | Whether the C manager still needs a model. |
| `cheetah_ant_model_not_initialized_warning(...)` (extern) | Prints the C-side advisory that the model was not prepared for Ant; strings are read synchronously, no ownership transfer. |
| `ant_new()` | Rust wrapper: creates the C manager, never returning a dangling null handle wrapped in `Ok`. |
| `ant_delete(ant)` | Rust wrapper: frees the manager exactly once (the `Drop` impl on `Ant` enforces this). |
| `ant_set_settings(ant, settings)` | Rust wrapper over `cheetah_ant_set_settings`; the generic error is enough because C already printed the specific diagnostic. |
| `ant_load_model(ant, dst_dir)` | Rust wrapper: loads the prepared model; CString conversion errors surface here (none are created on the happy path). |
| `ant_unit_tier(ant, dst_dir, unit_id)` | Rust wrapper returning the tier as i32 (0/1/2). |
| `ant_unit_pin(ant, unit_id, pin)` | Rust wrapper; C reports whether the pin was honored via its status code. |
| `ant_evict_unit(ant, dst_dir, unit_id)` | Rust wrapper: evicts to disk so budget enforcement can react to pressure. |
| `ant_page_in(ant, unit_id, tier)` | Rust wrapper: page-in; failure leaves the unit in its previous tier. |
| `ant_model_is_initialized(ant, dst_dir)` | Rust wrapper collapsing Result into a yes/no probe. |
| `ant_requires_init(ant)` | Rust wrapper; true means initialization is required. |
| `ant_model_not_initialized_warning(model_id, dst_dir)` | Rust wrapper printing the C advisory; strings only live for the call. |
| `report_init_progress(phase, current, total, user_data)` (extern "C" callback) | Invoked by C during init; converts the phase string to a bar index and must not panic into C (a panic across FFI is undefined behavior in Rust). |
| `path_to_cstring(path)` | Converts a path for FFI; wrappers hold the allocation only for the duration of one call. |
| `init_model(src_dir, dst_dir)` | Prepares a model without metadata; failures print a C-side diagnostic to stderr. |
| `init_model_with_metadata(src, dst, metadata)` | Prepares a model and merges chat metadata into the manifest afterwards; progress bars are cleaned up on every path including manifest-update failures. |
| `abandon_bars(progress)` | Abandons all phase bars after a failure (`flatten` skips `None` slots safely). |
| `update_manifest_metadata(dst_dir, metadata)` | Merges chat template/stop markers into `cheetah_manifest.json`, removing stale files first so repeated attempts do not accumulate. |
| `initialize()` | Rust wrapper: initializes the C Vulkan backend and its mapped memory pool. |
| `shutdown()` | Releases the Vulkan backend; no allocations or submitted work may remain. |
| `prepare()` | Validates that the Qwen2 compute shader and pipeline are available. |
| `matmul(a, b, m, k, n)` | Row-major `C[M,N] = A[M,K]·B[K,N]` through the Vulkan backend. |
| `matmul_bf16(a, b, m, k, n)` | FP32 activation × row-major BF16 weight matmul. |
| `matmul_bf16_transposed(a, b, m, k, n)` | FP32 × transpose-of-BF16 matmul (the executor's row-major weight layout). |
| `matmul_fp16_transposed(a, b, m, k, n)` | FP32 × transpose-of-FP16 matmul on the shared 16-bit pipeline. |
| `matmul_f32_transposed(a, b, m, k, n)` | FP32 × transpose-of-FP32 matmul (f32 precision format). |
| `weight_upload_bf16(weights)` | Uploads BF16 bits into the resident bucket; None = not served, keep streaming. |
| `weight_upload_q8(bytes, row_bytes)` | Uploads a q8 buffer (scale head + padded int8 rows); None keeps streaming. |
| `weight_upload_q4(bytes, row_bytes)` | Uploads a q4 buffer under the same layout contract as q8. |
| `weight_release(handle)` | Releases a resident upload (device-local or persistent host-coherent, depending on where it landed). |
| `embedding_gather(token_ids, table, ...)` | Gathers embedding rows as a shader op — one row per token, optionally scaled by sqrt(hidden) (gemma-style). |
| `bias_add(values, bias, format, period)` | Adds a bias tensor per `period` lanes in place on `values`. |
| `matmul_q8_transposed(a, q8, m, k, n)` | Streaming q8 matmul — weights cross PCIe once per tile, cheaper than the BF16 streaming path. |
| `matmul_q4_transposed(a, q4, m, k, n)` | Streaming q4 matmul (quarter of BF16 footprint, half of q8). |
| `matmul_q4_transposed_resident(handle, a, m, k, n)` | Resident q4 matmul from an uploaded handle. |
| `matmul_q8_transposed_resident(handle, a, m, k, n)` | Resident q8 matmul from an uploaded handle. |
| `matmul_bf16_transposed_resident(handle, a, m, k, n)` | Resident bf16 matmul from an uploaded handle. |
| `rms_norm_f32 / rms_norm_fp16 / rms_norm_bf16` | Shared-ops RMSNorm shaders per stored weight format. |
| `rms_norm(input, weight_ptr, format, ...)` (private) | Format-generic RMSNorm dispatch used by the typed wrappers. |
| `layer_norm_f32 / layer_norm_fp16 / layer_norm_bf16` | Shared-ops LayerNorm shaders (weight + bias) per stored format. |
| `layer_norm(...)` (private) | Format-generic LayerNorm dispatch. |
| `add(a, b)` | Elementwise add shader. |
| `silu_mul(gate, up)` | SwiGLU elementwise shader. |
| `gelu_mul(gate, up, approximate)` | GELU elementwise shader (`approximate` selects tanh-GELU). |
| `elementwise(a, b, operation)` | Generic two-operand elementwise dispatch behind add/silu/gelu. |
| `rotary(query, keys, ...)` | NeoX rotate-half rotary shader over the leading rotary dim. |
| `attention(query, keys, values, ...)` | GQA attention shader with causal masking + sliding window. |
| `to_u32(value)` | usize → c_int range check for FFI dimensions. |
| `check_native_result(result, operation)` | Maps a C status code to a descriptive Rust error. |
| `matmul_common(m, k, n, a_len, dispatch)` | Shared matmul plumbing: validates shapes, allocates the output, ensures the activation buffer flows through the pool. |
| `matmul_with(m, k, n, a_len, b_len, dispatch)` | Two-operand matmul plumbing that also validates the weight length `k * n` before touching the device. |
| 24 `extern` FFI declarations inside `pub mod vulkan` (init_vulkan_device, shutdown_vulkan_device, vk_qwen_prepare, the vk_*_checked matmul/operator bindings, cheetah_vulkan_weight_upload_*, cheetah_vulkan_weight_release, vk_qwen_gather/bias_add/rms_norm/layer_norm/elementwise/rotary/attention) | The raw C bindings the typed wrappers below call — one declaration per C entry point, all under the feature-gated Vulkan module. |
| 9 tests *(test)* | Cover progress-callback phases, FFI failure reporting, manifest metadata merging, matmul shape rejection, bf16/qwen operator reference parity, backend-uninitialized errors, and device-gated Vulkan matmul. |

### 10.8 `src/ant.rs` — Ant planner & manager (29 runtime + 9 tests)

| Function | Analysis |
|---|---|
| `AntSettings::default()` | Built-in defaults: two prefetch stages, VRAM swap allowed. |
| `AntUnit::byte_size()` | Payload size used by the planner and budget checks. |
| `AntUnit::state()` | Effective residency state string; partial layers occur when a layer's units span more than one tier. |
| `ant_swap_fit(used, budget, unit, allow_swap, evictable)` | Pure swap decision used by the engine's budget-checked swap — decides evictions BEFORE executing (CODEBASE §6 invariant 4). |
| `ant_filter_known_units(requested, units)` | Filters a requested unit-id list down to known units, deduplicated — used by the `--setup` wizard to validate user-typed pinned ids. |
| `Ant::new()` | Creates a manager with default settings. |
| `Ant::with_settings(settings)` | Creates a manager and immediately pushes `settings` to the C side. |
| `Ant::set_settings(settings)` | Updates settings (cached Rust copy + C pointers). |
| `Ant::settings()` | Currently active settings (the cached Rust-side copy). |
| `Ant::load_model(dst_dir)` | Loads a prepared model into the C manager's unit index. |
| `Ant::unit_tier(dst_dir, unit_id)` | Queries the current tier of one unit (0=vram, 1=ram, 2=disk). |
| `Ant::unit_pin(unit_id, pin)` | Pins/unpins a unit, protecting it from eviction. |
| `Ant::evict_unit(dst_dir, unit_id)` | Evicts a unit down to the disk tier (releases its resident pages). |
| `Ant::page_in(unit_id, tier)` | Moves a unit into the given tier (0=vram, 1=ram, 2=disk). |
| `Ant::model_is_initialized(dst_dir)` | Whether the C manager already loaded the model (manifest + units). |
| `Ant::requires_init()` | True when the C manager still needs a model to be loaded. |
| `Ant::not_initialized_warning(model_id, dst_dir)` | Prints the C-side "model not prepared for Ant" advisory. |
| `Ant::write_settings_for_model(dst_dir, settings)` | Writes the user-managed Ant settings file (`cheetah_ant_settings.json`), pretty-printed JSON. |
| `Ant::settings_for_model(dst_dir)` | Reads the model's settings file; missing file = pure defaults (only preset tiers are legal). |
| `Ant::run_interactive_setup(model_dir)` | The `--setup` wizard: walks unit tiers interactively, starting from the saved preset when one exists (unknown ids warn and are dropped). |
| `Ant::units_for_model(dst_dir)` | Reads the units manifest through the C index (same data `cheetah init` still loads). |
| `Ant::layer_residency_summary(units, plan)` | Per-layer residency summary for the startup banner, keyed by layer id matching the `cheetah_units.json` manifest. |
| `Ant::plan_residency(units)` | Layer-ordered greedy tier fill (vram → ram → disk); returns an empty plan when Ant is disabled. |
| `interactive_prompt(prompt)` | Prints a prompt, flushes, reads one trimmed stdin line. |
| `interactive_prompt_u64(prompt, default)` | Prompt variant returning u64; empty input takes the default. |
| `interactive_prompt_usize(prompt, default)` | Prompt variant returning usize; empty input takes the default. |
| `interactive_prompt_bool(prompt, default)` | Prompt variant accepting true/1/yes/on vs false/0/no/off. |
| `interactive_prompt_str(prompt, default)` | Prompt variant returning the raw trimmed string; empty input takes the default. |
| `Ant::drop()` | Releases the C manager handle exactly once (the FFI ownership contract). |
| 9 tests *(test)* | Cover the tier-fill order and partial-layer tracking, unknown-id filtering, all `ant_swap_fit` decision rules, disabled-Ant plans, user-managed settings files, and the real-model lifecycle. |

### 10.9 `src/chat/mod.rs` — chat entry (2 runtime + 4 tests)

| Function | Analysis |
|---|---|
| `chat(path, options, archive, store, ant)` | Public chat entry: loads the engine and runs the interactive REPL; the model path must be a directory or a `.safetensors` file. |
| `model_metadata_for_init(source)` | Derives init-time metadata (chat template id + stop markers) from a model's `config.json` architecture through the generated REGISTRY; `None` when the config names nothing resolvable. |
| 4 `init_metadata_*` tests *(test)* | Cover template/stop-marker resolution for qwen2/mistral/mixtral/qwen3 and the absent-metadata case for unknown architectures. |

### 10.10 `src/chat/engine.rs` — the inference engine (46 runtime + 12 tests)

| Function | Analysis |
|---|---|
| `ChatMessage::new(role, content)` | Builds a chat message; takes `impl Into<String>` so an owned `String` is already available. |
| `log_timestamp()` | Millisecond timestamp for engine log lines; `unwrap_or_default` is used because logging must never fail the process. |
| `create_spinner(message)` | Auto-ticking progress spinner that finishes itself, so loads without requiring the caller to drive it manually. |
| `load_engine_with_progress(...)` | Spinner-wrapped `Engine::load`; on failure the spinner is abandoned and the original error string is returned unchanged. |
| `print_model_info(engine)` | Startup banner (backends, ant plan/layers/swap); print errors are ignored so printing cannot turn a successful load into an error. |
| `TextSink::push(text)` | Sink contract: deliver one streamed chunk; ownership of the text stays with the caller; implementors reserve errors for real I/O failures. |
| `TextSink::finish()` | Ends the stream (newline for stdout, nothing for SSE); default is a no-op. |
| `TextSink::capture_text()` | Whether the sink needs the generation text retained (most callers do, for history or non-streaming responses). |
| `TextSink::interrupted()` | Whether the consumer aborted (client disconnect, Ctrl-C style interrupts). |
| `FinishReason::as_str()` | API-facing finish-reason string used in responses and logs. |
| `Generation::tokens_per_second()` | Decode speed over the measured decode window; `None` lets callers distinguish "no measurable speed" from a real zero rate. |
| `IdentityParser::parse(text, partial)` | Pass-through tool parser (used when no template parser is registered); `_partial` is accepted for API compatibility but does not change behavior. |
| `default_context_size()` | Serde default for models whose config omits `max_position_embeddings`. |
| `ant_tier_priority(tier)` | Numeric eviction priority per tier (disk first); sorts descending when choosing victims. |
| `ant_resident_weight_paths(weight_paths, units, plan)` | Narrows the weight-path list handed to a backend to the units the Ant plan keeps resident — so the backend does not fault disk-tier pages in during loading. |
| `ant_apply_budget(manager, dst_dir, units, plan, budget)` | Applies the resolved budget at load time, evicting over-budget units; returns the evicted unit ids so callers can update their bookkeeping. |
| `Engine::load(path, threads, ctx_size, ant)` | The load pipeline: config.json (or manifest-synthesized), backend resolution through the registry, Ant plan install, KV seed; consumed directly by CLI and server error paths. |
| `Engine::model_id()` | Loaded model's display id. |
| `Engine::model_architecture()` | Resolved architecture name (None when the config named nothing). |
| `Engine::device_name()` | Backend device label forwarded from the `ModelForward` runtime. |
| `Engine::ant_within_budget()` | Whether current VRAM+RAM occupancy (tracked by the Phase 3 bookkeeping) stays within the configured budget. |
| `Engine::ant_apply_real_tier(page_ins, evictions)` | Pushes tier transitions into real pages when the backend owns mmap weights (native); host-resident models keep only the C-side bookkeeping. |
| `Engine::ant_lock_runtime()` | Locks the Ant bookkeeping mutex, recovering poisoned locks instead of panicking. |
| `Engine::ant_swap_in_unit(manager, unit, tier, allow_evict)` | Phase 3: pages one unit in at `tier_value`, re-checking the budget before the swap and again before each eviction; VRAM swaps evict only when both VRAM swap and `allow_evict` permit. Returns true when the swap-in executed. |
| `Engine::apply_ant_request_overrides(overrides)` | Applies per-request resident-unit overrides: disk-tier or evicted units named in the request are swapped back in (budget-checked, evictions allowed). |
| `Engine::prefetch_ant_units()` | Generation hook: swaps up to `prefetch_stages` evicted resident-tier units back in, ordered by tier priority then layer — each swap budget-checked. |
| `Engine::promote_ant_units()` | Phase 3: promotes queued RAM-tier units into VRAM; the walk stops at the first unit that does not fit without eviction, so later layers are never promoted at the expense of earlier ones. |
| `Engine::enforce_ant_budget()` | Generation hook: pages evicted resident-tier units back in, then evicts lowest-priority residents (RAM first, highest layers first; pinned never move) until the budget holds. |
| `Engine::new_context()` | Creates an inference context (stateless placeholder — KV state lives in the model runtime). |
| `Engine::prepare(messages, system, max_tokens)` | Renders + tokenizes the prompt; truncates oldest turns (with a warning) until the input plus reserved generation tokens fits the context limit; errors when a single turn cannot fit. |
| `Engine::prepare_tool_chat(messages_json, tools_json, choice, max_tokens)` | Builds the tool-calling prompt: injects the JSON tool instruction per `ToolChoice` (none/auto/required), maps developer→system, tokenizes; returns tokens + chat params. |
| `Engine::generate(context, prompt, options, extra_stop, sink, resident_overrides)` | Public generation entry (ignores the context placeholder); delegates to `generate_inner`. |
| `Engine::generate_with_chat_params(...)` | Generation entry that extends the stop list with the chat template's own stop markers parsed from `ChatParams`. |
| `Engine::generate_inner(prompt, options, extra_stop, sink, resident_overrides)` | The core loop: validates limits, runs the four Ant hooks (overrides → prefetch → promote → enforce), clones the model out of the mutex, clears KV, prefills, then samples token by token (stop markers checked on the decoded text, character-boundary guarded), feeding each token back at its absolute position. |
| `resolve_model_files(path)` | Resolves the weight paths + manifest stop markers for a model path (directory or file). |
| `synth_config_from_manifest(model_dir, read_error)` | Rebuilds a minimal config.json from the manifest snapshot when the copied config is missing; merges family fallback fields so a prepared directory stays loadable on its own. |
| `read_model_manifest(model_dir)` | Reads `cheetah_manifest.json` if present (None otherwise); parse errors are loud. |
| `parse_manifest_stop_markers(manifest)` | Extracts the manifest's stop_markers list; a missing list or non-string marker is an error. |
| `resolve_weight_files(path, manifest)` | Weight resolution chain: direct safetensors file → manifest shards (path-traversal-checked, sorted, deduplicated) → `model.safetensors.index.json` → single `model.safetensors`. |
| `random_seed()` | Clock-derived seed (nanoseconds since epoch, defaulting to 0 pre-epoch). |
| `sample_token(logits, history, options, rng)` | Sampling pipeline: repeat penalty over the last 64 tokens → greedy when temperature = 0 → temperature scaling → optional top-k → top-p nucleus truncation → weighted sample. |
| `chat(path, options, archive, store, ant)` | Interactive chat entry: validates the path kind, loads the engine with progress, runs the REPL loop. |
| `StdoutSink::push(text)` | Prints one streamed chunk and flushes stdout; write failures are errors. |
| `StdoutSink::finish()` | Ends the turn with a newline. |
| `format_generation_stats(generation)` | Formats the post-turn stats line (tokens + tok/s, or "speed unavailable"). |
| `chat_loop(engine, options, archive, store)` | The REPL: slash commands (`/exit` `/help` `/clear` `/save`), history replay from the archive, per-turn prepare → generate → stats, and archive bookkeeping. |
| 12 tests *(test)* | Cover manifest shard resolution (+ unsafe path rejection), direct-weight + metadata paths, qwen/llama/mistral/llama3 prompt rendering, Ant resident-path narrowing (enabled + disabled), fixed-seed sampling determinism, and generation stats formatting. |

### 10.11 `src/chat/moe.rs` — shared MoE routing (1 runtime + 3 tests)

| Function | Analysis |
|---|---|
| `select_experts(logits, num_experts_per_tok)` | The one routing rule every backend shares: max-subtracted softmax over router logits → top-k by score → selected weights renormalized to sum 1; errors on non-finite scores or inconsistent counts. |
| 3 tests *(test)* | Cover top-k ordering + renormalization, the top-k == full-softmax identity, and rejection of invalid logits/counts. |

### 10.12 `src/chat/registry.rs` — boundary + lookups (30 runtime + 10 tests)

| Function | Analysis |
|---|---|
| `ModelForward::forward(tokens, offset)` | The backend-agnostic inference boundary: runs tokens from absolute position `offset`, returns final-position logits. |
| `ModelForward::clear_cache()` | Drops the runtime's KV state between generations. |
| `ModelForward::device_name()` | Backend label for the banner and logs. |
| `ModelForward::as_ant_tier_control()` | Exposes real page-level tier control on backends that own mmap weights; host-resident backends report None. |
| `AntTierControl::page_in_unit(unit_id, tier)` | Faults one unit's pages back in (native backend); bool = manifest knows the unit. |
| `AntTierControl::evict_unit(unit_id)` | Drops one unit's pages to the disk tier; bool = manifest knows the unit. |
| `ModelForwardClone::clone_box()` (blanket impl) | Clones through the trait object: any `ModelForward + Clone` type can be duplicated behind the Box. |
| `ModelForwardClone::clone_box()` (Box impl) | The dyn-side entry the `Box<dyn ModelForward>` Clone impl calls. |
| `Clone for Box<dyn ModelForward>` | Makes the boxed runtime clonable via `clone_box` (the engine clones the model per generation). |
| `ModelFamily::profile_for(hidden, vocab)` | Looks up the family's `(q_heads, kv_heads, head_dim)` profile for a known size; vocab keys disambiguate llama2/llama3-style same-hidden variants. |
| `ModelRegistration::serves(architecture)` | Whether this registration claims an architecture name (normalized: case-insensitive, `ForCausalLM` suffix stripped on both sides). |
| `ModelRegistration::loader_for(backend)` | Backend loader by name, case-insensitive. |
| `ModelRegistration::backend_names()` | Registered backend names in declared order (what the banner reports). |
| `registration_for_architecture(arch)` | The lookup entry: finds the registration serving a normalized architecture name or lists all registered families in the error. |
| `vulkan_backends()` *(test-support, vulkan)* | Lists the architectures served by the Vulkan backend, derived from the registration table (the native side asserts capability through `backend_loader_for_architecture` directly). |
| `architectures_serving(backend)` | Shared derivation behind the two views above. |
| `architecture_has_backend(arch, backend)` | Bool probe used by capability tests and the Vulkan fallback check. |
| `normalize_architecture(architecture)` | Lowercases and strips the `ForCausalLM` suffix so `Qwen2ForCausalLM` and `qwen2` resolve identically. |
| `registered_backends(arch)` | Backend names for an architecture in declared order (e.g. `["native", "vulkan"]`). |
| `backend_loader_for_architecture(arch, backend)` | Loader-table probe: returns the backend's loader fn for one architecture. |
| `family_for_architecture(arch)` | Family constants for a native backend family, straight from the registration file. |
| `chat_template_for_architecture(arch)` | Chat prompt renderer resolved through the registration (init-time metadata keys prompt families by architecture). |
| `stop_markers_for_architecture(arch)` | Stop marker names resolved through the registration; an unknown architecture stops on nothing. |
| `load_model(weight_paths, config_bytes, raw, arch, plan)` | The `CHEETAH_DEVICE` dispatch entry: cpu → native-guidance error (Candle CPU retired), vulkan → `load_vulkan_or_fallback`. |
| `no_native_backend_message(arch)` | The guidance error shown when an architecture has no native backend in this build. |
| `backend_device(selected)` | Maps a device name to the `Device` descriptor (both backends are host-resident today). |
| `load_model_for_device(...)` | Device-parameterized load used by the engine; honors the `CHEETAH_DEVICE` override for Vulkan. |
| `vulkan_loader_for_architecture(arch)` | Vulkan loader from the registration table — a miss is a registration-file bug, not a dispatch gap. |
| `load_vulkan_or_fallback(...)` | Loads the Vulkan executor for an architecture, with the capability check, failure diagnostics, and the CPU-guidance fallback message. |
| 10 tests *(test)* | Cover backend registration, device mapping, architecture normalization across all families, native capability per family, family presence per registration, architecture-name uniqueness, Vulkan capability/loader agreement, and per-registration Vulkan loader feature selection. |

### 10.13 `src/chat/templates.rs` — prompt renderers (8 runtime)

All renderers take `(messages, system)` and return the raw prompt string the
tokenizer consumes; each implements one family's HF chat format.

| Function | Analysis |
|---|---|
| `qwen(messages, system)` | Qwen ChatML: `<\|im_start\|>{role}\n...<\|im_end\|>` per turn, optional system turn first, `<\|im_start\|>assistant\n` tail. |
| `gemma(messages, system)` | Gemma turns: `<start_of_turn>{user\|model}` with the system prompt folded into the first user turn; ends with `<start_of_turn>model\n`. |
| `llama2(messages, system)` | Llama-2 `[INST] <<SYS>>...<</SYS>> ... [/INST]` instruction format (delegates to `instruction`). |
| `mistral(messages, system)` | Mistral instruction format — identical shape to llama2 (delegates to `instruction`). |
| `mixtral(messages, system)` | Mixtral instruction format — identical shape to llama2 (delegates to `instruction`). |
| `instruction(messages, system)` | The shared `[INST]` renderer: folds the system prompt into the first user turn's `<<SYS>>` block, separates assistant turns with ` </s>`, opens with `<s>`. |
| `llama3(messages, system)` | Llama-3 header format: `<\|begin_of_text\|>` + `<\|start_header_id\|>{role}<\|end_header_id\|>` turns closed by `<\|eot_id\|>`. |
| `phi(messages, system)` | Phi-2 format: `<\|system\|>` / `<\|user\|>` / `<\|assistant\|>` turns closed by `<\|end\|>`, assistant prompt tail. |

### 10.14 `src/chat/registrations/` — one family, one file

Each file has the same skeleton; only the family-specific pieces differ.
`*(family)*` below names the struct each row belongs to.

| Function | Analysis |
|---|---|
| `model_backends()` (all 7) | The `const` backend table: the native loader always, plus the family's Vulkan loader behind `#[cfg(feature = "vulkan")]`. |
| `<family>_family()` (all 7) | The family constants: rope theta, eps, tied embeddings, size profiles, extra config JSON, MoE layout, DenseFeatures metadata. |
| `<Family>Native::with_store(cfg, store)` (llama/mixtral/gemma/phi; cfg(test)) | Fixture/test constructor around an already-resolved config + store. |
| `<Family>Native::new(ctx)` (all 7) | Loader-path constructor: clones the resolved config out of the toolbox and preallocates the scratch buffers. |
| `<Family>Native::run_forward(tokens, offset)` (all 7) | The family's full forward shell: validates tokens, walks positions, sweeps the lm head (tied vs untied per family constant), drops layer-0 disk pages. |
| `<Family>Native::forward_position(token, position)` (all 7) | One token position: embedding row (gemma adds the sqrt(hidden) scale), every layer, final norm (phi names it `final_layernorm` and uses LayerNorm; gemma uses offset RMSNorm). |
| `<Family>Native::forward_layer(layer, position)` (all 7) | One decoder block in the family's topology: llama/qwen/mistral/qwen3/mixtral run sequential residuals (norm → attn → residual → norm → FFN → residual); gemma swaps in offset norms + GeGLU; phi runs the parallel residual `x + attn + mlp` off one shared input norm. |
| `<Family>Native::forward_attention(prefix, layer, position)` (all 7) | The attention block: split q/k/v projections (qwen3 adds per-head q/k RMSNorm before rope; phi fuses them into `qkv_proj` and names the output `dense`), family-unit-optional bias adds, full or partial rope, KV-cache append, GQA attention through the toolbox (sliding window from config), output projection. |
| `<Family>Native::forward_mlp(prefix)` / `(prefix, layer)` (all 7) | The feed-forward block: SwiGLU (`silu_multiply`) for llama/qwen2/qwen3/mistral; GeGLU (`gelu_multiply`) for gemma; phi's plain `fc1`/`gelu`/`fc2` with biases; mixtral dispatches dense-vs-MoE on config counts. |
| `MixtralNative::forward_moe_block(prefix, layer)` | The sparse block: router gemv → shared `select_experts` → routing-aware expert residency → per-expert gate/up/down with route-weighted accumulation into `mlp_out`. |
| `MixtralNative::forward_dense_mlp(prefix)` | Dense fallback for a mixtral-shaped manifest without routed experts. |
| `LlamaNative::debug_attention_out(tokens)` (cfg(test)) | Test hook: captures layer-0 attention scores + pre-projection mix at the last position for the sliding-window test. |
| `<Family>Native::forward` (ModelForward impl, all 7) | Trait entry delegating to `run_forward`. |
| `<Family>Native::clear_cache` (all 7) | Clears the toolbox KV caches. |
| `<Family>Native::device_name` (all 7) | Reports `"native-cpu"`. |
| `<Family>Native::as_ant_tier_control` (all 7) | Exposes the family runtime for real Ant page-level tier transitions. |
| `<Family>Native::page_in_unit` / `evict_unit` (AntTierControl impl, all 7) | Forward to the toolbox's unit page-in / evict; the bool reports whether the units manifest knows the unit. |
| `<family>_native_loader(dir, bytes, raw, plan, family)` (all 7) | The registration's native entry: `OpsToolbox::open` (store → config resolve → Ant plan install) → family struct → `Box<dyn ModelForward>`. |
| `<family>_vulkan_config(raw)` (llama/qwen2/qwen3/mistral/gemma/mixtral) | Builds the family's `TransformerFeatures` (biases, qk-norm, head-dim override, norm offsets, MoE counts, tied head) and validates the config for the generic Vulkan executor. |
| `<family>_vulkan_loader(paths, bytes, raw, plan)` (all 7) | The registration's Vulkan entry: phi builds the dedicated `VulkanPhiExecutor`; the others build the generic transformer executor around their feature set. |
| phi vulkan executor: `VulkanPhiConfig::parse` / `PhiNorm::forward` / `take_phi_norm` / `PhiLayer` / `VulkanPhiExecutor::load` / `load_initialized` / `forward` (+ trait impls) | phi's dedicated GPU compute graph, hosted in the family file (one family, one file on both backends): biased LayerNorms, parallel residual `x + attn + mlp`, partial rotary, plain `fc1`/`fc2` GELU MLP, split projections with `dense` output; bootstraps the substrate through `load_with_substrate` and dispatches operators through the shared weights layer |
| family Vulkan tests *(test)* | Per-family feature assertions: qwen2 bias/tie + dimension rejection, qwen3 dense layout / window disable / rope-scaling rejection / attention bias, gemma norm+embedding defaults, mixtral MoE layout + count rejection. |

### 10.15 `src/chat/native/ops.rs` — the OpsToolbox (48 runtime + 14 tests)

| Function | Analysis |
|---|---|
| `bf16_f32(bits)` | Widens a BF16 bit pattern to f32 by shifting into the high half — exact, no rounding, which is what makes the fused dot kernels correct. |
| `f16_f32(bits)` | Decodes an IEEE half bit pattern to f32 through the `half` crate. |
| `widen_bf16(lo, hi)` | Little-endian u16 assembly + widen, used by the scalar tails of the fused kernels. |
| `scalar_at(bytes, index, dtype)` | Reads one stored-precision scalar (BF16/F16/F32) from raw weight bytes. |
| `dequant_bf16_row(row_bytes, out)` | Fast bf16→f32 row widening: two lanes per u32 load, pure bit arithmetic (auto-vectorizes). |
| `dequant_row(row_bytes, dtype, out)` | Dequantizes one weight row to f32 for any dtype (the non-bf16 gemv path). |
| `dot_f32(a, b)` | f32 dot with an 8-lane accumulator array — the fixed-width inner loop lowers to wide vector FMA sequences. |
| `dot_row_bf16_fused(row, x)` | The hot bf16 dot: AVX2+FMA intrinsics when the CPU supports them, else the portable fused loop; widening stays fused into the accumulators. |
| `dot_row_bf16_fused_portable(row, x)` | Portable fused bf16 dot: 4 lanes per 8-byte chunk, no intermediate buffer. |
| `dot_row_bf16_avx2(row, x)` (unsafe) | AVX2+FMA path: `cvtepu16_epi32 + slli_epi32(16)` widening (exact `bf16 << 16` bitcast) + two FMAs per 16 lanes; caller must have verified CPU features. |
| `gemv(weight, dtype, rows, cols, x, y)` | Row-parallel matrix-vector: bf16 rows take the fused kernel, other dtypes dequant-then-dot; ≤128 rows sequential, larger matrices split into rayon chunks. |
| `gemv_q8(weight, rows, cols, x, y)` | Quantized-domain q8 matvec: per-row f32 scale × int8 codes, f32 accumulate, 4-byte row padding ignored — same math as `matmul_q8.comp`. |
| `gemv_q4(weight, rows, cols, x, y)` | Quantized-domain q4 matvec: two nibbles per byte (low nibble = even column), scale × code × x accumulated in f32 — same math as `matmul_q4.comp`. |
| `dequant_values(bytes, dtype, count)` | Bulk-dequantizes a contiguous weight array to f32 — feeds the registered quantize operators at load time. |
| `rmsnorm(x, weight, dtype, eps)` | RMSNorm `x / sqrt(mean(x²) + eps) · weight` in place, weights read from stored bytes. |
| `rmsnorm_offset(x, weight, dtype, eps)` | gemma-flavored RMSNorm applying the affine as `(1 + weight)`. |
| `layernorm(x, weight, w_dtype, bias, b_dtype, eps)` | LayerNorm with affine weight + bias (phi); an empty bias slice skips the shift. |
| `rope_in_place(values, heads, head_dim, position, theta)` | NeoX rotate-half rotary: splits each head `[x1 \| x2]`, pairs angle i with both halves (`sin_cos` order noted in a NOTE). |
| `rope_heads(values, heads, head_dim, rotary_dim, position, theta)` | Per-head rotary over the leading `rotary_dim` dims — full span for llama-shaped families, partial for phi. |
| `silu_multiply(gate, up)` | SwiGLU activation `silu(gate) ⊙ up`, written into `gate`. |
| `gelu(values)` | tanh-approximation GELU in place. |
| `gelu_multiply(gate, up)` | GeGLU activation `gelu(gate) ⊙ up` (gemma). |
| `softmax(scores)` | Numerically stable in-place softmax (max-subtracted) over a masked score vector. |
| `Scratch::new(cfg)` | Preallocates every activation buffer at its maximum capacity for this config. |
| `OpsToolbox::clone()` | Clone semantics: weights shared through the Arc'd store, but a fresh KV cache, quantized-buffer table, expert-residency map and debug state (the engine clones per generation). |
| `OpsToolbox::with_store(cfg, store)` | Test/fixture constructor around an already-resolved config + store. |
| `OpsToolbox::open(dir, raw_config, plan, family)` | Load-path constructor: opens the mmap store, resolves the config through the fallback chain, installs the Ant residency plan. |
| `OpsToolbox::validate_tokens(tokens)` | Rejects an empty token batch and ids beyond the vocabulary. |
| `OpsToolbox::embedding_row_into(token, out)` | Copies one embedding row in stored precision; any scaling (gemma) is the caller's model logic. |
| `OpsToolbox::rmsnorm(unit, target)` | RMSNorm against `{unit}.weight` bytes. |
| `OpsToolbox::rmsnorm_offset(unit, target)` | Offset RMSNorm against `{unit}.weight`. |
| `OpsToolbox::layernorm(unit, target)` | LayerNorm against `{unit}.weight` (+ `.bias` when present). |
| `OpsToolbox::qk_norm(values, heads, head_dim, unit)` | Per-head RMSNorm on q/k states (qwen3); no-op when the unit is absent so families call it unconditionally. |
| `OpsToolbox::add_bias(values, unit)` | Adds `{unit}.bias` values; no-op when absent (families without biases). |
| `OpsToolbox::linear(unit, rows, cols, x, y)` | The linear dispatch: quantized buffer (fast path) → lazy load-time quantization (dequant once, quantize, drop source pages) → stored-precision gemv via the precision registry. |
| `OpsToolbox::cache_append(layer, k, v)` | Appends one position's K/V rows to a layer's cache. |
| `OpsToolbox::attention(layer, position, q, out)` | GQA attention over the cached keys: per-head scores with `1/sqrt(head_dim)` scale, sliding-window masking (window start clamped so softmax is never all-masked), softmax-weighted value mix; captures head-0 scores when the debug hook is armed. |
| `OpsToolbox::update_expert_residency(layer, picked)` | RC2-B: evicts experts the router moved away from, pages in newly selected ones; keyed `(layer, expert)`, persists across positions. |
| `OpsToolbox::set_expert_resident(layer, expert, resident)` | Moves one expert's gate/up/down units together between tiers; unknown unit ids are ignored. |
| `OpsToolbox::lm_head_into(head_unit, hidden, logits)` | Vocabulary logits via a 16k-row chunked sweep that releases each chunk's pages when the head sits on the disk tier. |
| `OpsToolbox::clear_kv()` | Clears every layer's KV cache. |
| `OpsToolbox::apply_tiers(plan)` | Installs the Ant residency plan: resident tiers faulted in, disk tiers dropped. |
| `OpsToolbox::page_in_unit(unit_id)` / `evict_unit(unit_id)` | Ant page control on single units; bool = the manifest knows the unit. |
| `OpsToolbox::drop_disk_units_of_layer(layer_id)` | Drops every disk-tier unit of one layer after its step, keeping the transient footprint near one layer. |
| `OpsToolbox::arm_debug_capture()` / `take_debug_capture()` (cfg(test)) | Test hooks around the layer-0 attention capture. |
| `QuantState::selected()` | Reads the precision registry for the selected quantization format's native operators; inert when none is selected. |
| 14 tests *(test)* | Cover bf16 widening, gemv (f32+bf16), q8/q4 rowwise-scale references, quantize-operator cross-consumption, RMSNorm, rope (magnitude + rotate-half reference), the `rope_heads` partial span, silu, softmax shift invariance, and the AVX2-vs-dequant dot isolation at 8 and 896 lanes. |

### 10.16 `src/chat/native/store.rs` — mmap weight store (16 runtime)

| Function | Analysis |
|---|---|
| `WeightDType::element_size()` | Byte width per stored element (BF16/F16 = 2, F32 = 4). |
| `WeightDType::format_id()` | Maps the dtype to its registered precision format id ("bf16"/"f16"/"f32") — the id the linear dispatch uses. |
| `MmapWeightStore::open(model_dir)` | Maps every shard referenced by the units manifest, parses the Safetensors header (8-byte length + JSON), and indexes each unit's byte range + dtype + shape. |
| `MmapWeightStore::unit_ids()` | Iterator over all indexed unit ids. |
| `MmapWeightStore::unit_shape(unit_id)` | Cloned shape of one unit, for parameter derivation from the manifest. |
| `MmapWeightStore::contains(unit_id)` | Whether the manifest carries a unit (optional biases, q/k norms, tied-vs-untied head choice). |
| `MmapWeightStore::max_layer_id()` | Highest layer id in the manifest; the decoder layer count is id + 1 (non-layer weights share layer 0). |
| `MmapWeightStore::slice(unit_id)` | Zero-copy `(bytes, dtype, shape)` view of one unit inside its shard mapping. |
| `MmapWeightStore::tier(unit_id)` | Current Ant tier of a unit (default Resident). |
| `MmapWeightStore::set_tier(unit_id, tier)` | Sets the tier behind the interior-mutability lock and immediately drives real page control (page-in or madvise drop). |
| `MmapWeightStore::layer_unit_ids(layer_id)` | Units of one layer, for per-step eviction. |
| `MmapWeightStore::page_in(unit_id)` | Faults a unit's pages back in by touching one byte per page, so RSS reflects the tier immediately. |
| `MmapWeightStore::drop_range(unit_id, start, len)` | Drops a byte range within one unit (the chunked lm-head release). |
| `MmapWeightStore::drop_pages(unit_id)` | Drops a unit's pages (`madvise(MADV_DONTNEED)`); the mapping stays valid and re-faults on next access. |
| `MmapWeightStore::drop_disk_units_of_layer(layer_id)` | Drops every disk-tier unit of one layer after its step completed. |
| `madvise_range(ptr, len, advice)` | Page-aligned madvise helper; the kernel rounds as needed. |

### 10.17 `src/chat/native/decoder.rs` — config resolution + tests (5 runtime + 36 tests)

| Function | Analysis |
|---|---|
| `MoEExpertLayout::router_unit(prefix)` | Router unit id for one layer (`.weight` suffix included — the manifest names whole tensors). |
| `MoEExpertLayout::expert_unit(prefix, expert, projection)` | One expert projection's unit id, e.g. `model.layers.3.block_sparse_moe.experts.1.w3.weight`. |
| `DenseFeatures::default()` | The llama-shaped baseline feature set (every structural delta off, full rope). |
| `DecoderConfig::resolve(raw, store, family)` | The parameter fallback chain: config.json fields → units-manifest shape derivation → family constants; fails loudly when a shape cannot be recovered (silent guessing would corrupt every matvec). Also enforces the parallel-residual/MoE mutual exclusion and resolves the rotary dim + activation. |
| `count_layer_experts(store, layout)` | Counts layer-0 experts from the manifest (highest index + 1, contiguous-from-zero assumed); zero selects the dense feed-forward path. |
| 33 tests *(test)* | The native forward test suite: fixtures (dense/MoE/phi tensor sets + shard/manifest writers), residency-parity cycles for dense and MoE, the router-weight mixing reference, router-driven expert residency transitions, sliding-window masking (capture + logits), config resolution for sparse/MoE fields and dense family features, and the gemma/phi full-forward reference-math suites. |

### 10.18 `src/chat/precision.rs` + `src/chat/precision/` — precision registry

Shared module (21 runtime + 8 tests), then one file per format.

| Function | Analysis |
|---|---|
| `PrecisionBuffer` trait methods (`format_id`, `rows`, `cols`, `buffer`, `row_bytes`, `dequant_value`, `element_count`) | The quantized-buffer contract: registered format id, matmul shape (rows = output features, cols = reduction width), the complete byte buffer (scale head + packed payloads), the padded per-row stride, and a dequantized element for tests/diagnostics. |
| `PrecisionRegistration::loader_for(backend)` | Backend operator table by name, case-insensitive. |
| `PrecisionRegistration::backend_names()` | Registered backend names in declared order (registry tests). |
| `f32_view(bytes)` / `u16_view(bytes)` | Typed views over the registry's uniform byte payloads — SAFETY-checked alignment/length instead of undefined behavior on misuse; vulkan-only. |
| `registration_for_format(format)` | The format-id lookup; errors list every registered id so selector typos are visible. |
| `format_label(format)` | Display label for diagnostics. |
| `vulkan_ops_for_format(format)` | Vulkan operator table for a format; None surfaces a registration-file bug as a dispatch error. |
| `vulkan_ops_for_buffer(buffer)` | Vulkan operators for an existing quantized buffer (dispatch by its registered id). |
| `native_ops_for_format(format)` | Native operator table for a format id. |
| `selector_enabled(value)` | Selector semantics: "1" or "true", case-insensitive; anything else keeps the format off. |
| `selected_format()` | The quantization format enabled via its selector env (first registry match wins); quantization is strictly opt-in. |
| `selected_vulkan_ops()` / `selected_native_ops()` | The selected format's per-backend operator tables. |
| `registered_format_ids()` (cfg(test)) | All registered format ids for the registry tests. |
| 8 tests *(test)* | Cover original+quantized registration, id/selector uniqueness, full backend/operator-table presence, quantize-presence matching the kind, selectors only on quantized formats, both-backend coverage, lookup normalization, and selector literals. |

**Per-format files** (`f32.rs`, `f16.rs`, `bf16.rs`, `q4.rs`, `q8.rs`):

| Function | Analysis |
|---|---|
| `f32::matmul_streaming` | Host-streaming f32 matmul on the dedicated f32 pipeline. |
| `f32::gemv` | `y = W·x` over stored f32 rows (f32 multiply-accumulate). |
| `f32::precision_backends` | The format's declarative backend table (no quantize, no resident upload). |
| `f16::matmul_streaming` | Streaming f16 matmul on the shared 16-bit pipeline (fp16 decode mode). |
| `f16::gemv` | `y = W·x` over stored f16 rows (half decoded per row, f32 accumulate). |
| `f16::precision_backends` | Backend table (no quantize/upload). |
| `bf16::matmul_streaming` | Streaming bf16 matmul on the dedicated bf16 pipeline. |
| `bf16::upload` | The only original precision with a resident upload (u16 payload served directly by the C substrate). |
| `bf16::matmul_resident` | Resident bf16 matmul from an uploaded handle. |
| `bf16::gemv` | `y = W·x` over stored bf16 rows (fused bf16 kernel, f32 accumulate). |
| `bf16::precision_backends` | Backend table with the resident pair. |
| `q4::row_stride(cols)` | Padded per-row byte stride of the q4 payload (what the FFI validates). |
| `q4::decode_code(byte, half)` | Unpacks one int4 code — `half == 0` the low nibble (even column), `half == 1` the high nibble. |
| `Q4Buffer` trait impls (`format_id`..`dequant_value`) | The PrecisionBuffer contract for q4 (layout + scale/code dequant). |
| `q4::quantize(values, rows, cols)` | Row-wise symmetric int4 quantization: `scale = max\|w\|/7`, two codes per byte, padded stride — rejects size mismatches before building a buffer. |
| `q4::upload(buffer, row_bytes)` | Resident upload validated against the padded stride the format table guards. |
| `q4::matmul_streaming(...)` | Streaming q4 matmul; a shader/FFI layout drift fails loudly at the length check. |
| `q4::matmul_resident(handle, ...)` | Resident q4 matmul reading descriptor 1 of the resident record (no per-call pool staging). |
| `q4::gemv(weight, rows, cols, x, y)` | Quantized-domain q4 matvec kept decoupled from the kernel module layout. |
| `q4::precision_backends` | Backend table: native quantize+gemv, Vulkan full operator set. |
| `q8::Q8Buffer` trait impls | The PrecisionBuffer contract for q8. |
| `q8::quantize(values, rows, cols)` | Row-wise int8 quantization: `scale = max\|w\|/127`, codes clamped ±127, zero rows keep scale 1.0 (never divide by zero). |
| `q8::upload(buffer, row_bytes)` | Resident q8 upload; failure keeps the streaming path. |
| `q8::matmul_streaming(...)` | Streaming q8 matmul (row-padded; the C kernel computes the stride from `k`). |
| `q8::matmul_resident(handle, ...)` | Resident q8 matmul tiling `m` to bound pool usage. |
| `q8::gemv(weight, rows, cols, x, y)` | Quantized-domain q8 matvec. |
| `q8::precision_backends` | Backend table: native quantize+gemv, Vulkan full operator set. |
| format tests *(test)* | FFI layout contracts, quantize round-trips within one quantization step, nibble column parity, size-mismatch rejection. |

### 10.19 `src/chat/toolcalls.rs` — shared tool-call shapes (9 runtime + 2 tests)

The OpenAI function-calling JSON shapes are the de-facto tool protocol for
both the HTTP API and the local REPL; this module holds those shapes and the
shape-tolerant parsing, pure serde_json + std (no HTTP types), so the engine
layer runs tool loops without reaching into `serve`. `serve/openai.rs`
re-exports the surface.

| Function | Analysis |
|---|---|
| `parse_tool_response(parsed, raw_text, tools, choice)` | Tool-output resolution ladder: template-parsed `tool_calls` → compatible raw JSON shapes → free-text JSON extraction validated against the tools and choice → plain-text fallback. |
| `parse_generated_tool_calls(text)` | Parses tool calls from free model text (compact, OpenAI, or flat shapes) after fence/brace extraction. |
| `extract_generated_tool_call_json(text)` | Strips code fences and slices the outermost JSON object. |
| `convert_tool_calls(calls, tools, choice)` | Converts parsed calls to the response shape, filtering by tool existence and a named `tool_choice`. |
| `normalize_tool_arguments(arguments)` | Accepts arguments as a JSON object or a JSON-encoded string; both validate to object text. |
| `tool_call_id()` | Generates `call_<hex>` ids for outgoing tool calls. |
| `random_hex(bytes)` | OS-entropy-seeded hex string (keys and ids). |
| `next_tool_round(messages_json, calls, results)` | Builds the next tool-loop round's messages: the assistant's serialized tool calls (the same history convention the request parser applies) plus one `role: "tool"` result per call. |
| types (`ToolDefinition` / `FunctionDefinition` / `ResolvedToolChoice` / `ToolResponse` / `ToolCall` / `ToolCallFunction` / `Parsed*`) | The request/response/history tool shapes, moved verbatim from `serve/openai.rs` when the REPL gained tools. |
| 2 tests *(test)* | Cover the next-round message shape (assistant serialization + tool results) and invalid-history rejection. |

### 10.20 `src/mcp.rs` + `src/mcp/client.rs` — MCP client (25 runtime + 14 tests)

The Model Context Protocol client: Cheetah connects to local MCP servers
over stdio, discovers their tools, exposes them to the model, and executes
their tool calls in the agentic loop (`Engine::generate_tool_chat`).

| Function | Analysis |
|---|---|
| `parse_server_spec(spec)` | Parses one `--mcp-server` spec: `name=command args...` (whitespace-split; quoted arguments belong in a config file). |
| `server_configs(specs, config_file)` | Resolves the configured servers: flag specs plus the `--mcp-config` JSON file entries; names unique across both. |
| `registry_from_flags(specs, config_file)` | `None` when nothing is configured (the zero-cost MCP-free posture); otherwise connects everything. |
| `McpRegistry::connect(configs)` | Spawns every server, runs the `initialize` handshake, lists tools; any failure fails the whole registry loudly — a half-configured tool surface would silently corrupt tool calls. |
| `McpRegistry::tool_definitions()` | Renders discovered tools as OpenAI-style definitions (`inputSchema` → `parameters`). |
| `McpRegistry::has_tool(name)` | Whether a call targets a connected MCP server. |
| `McpRegistry::call(name, arguments)` | Executes `tools/call` on the owning server (per-client mutex serializes calls; `Arc`-shared across request workers); MCP-level `isError` surfaces as `[error] ...` text the model can react to. |
| `McpRegistry::shutdown()` + `Drop` | Kills every child; `Drop` guarantees no leaked processes. |
| `McpClient::connect(config)` | Spawns the server and runs the handshake. |
| `McpClient::request(method, params)` | JSON-RPC request with id correlation over the newline transport; notifications ignored, server→client requests answered with a protocol error (so well-behaved servers never stall), 30s deadline. |
| `McpClient::initialize()` | The `initialize` handshake + `notifications/initialized` acknowledgment. |
| `McpClient::list_tools()` | `tools/list` with `nextCursor` pagination. |
| `McpClient::call_tool(name, arguments)` | `tools/call`, text content blocks joined. |
| `LineTransport` / `ChildTransport` | The newline transport trait + the real child-piped implementation (reader thread pumps stdout so a slow server never blocks past the deadline; stderr inherits). |
| 14 tests *(test)* | Scripted-transport protocol tests (handshake shape, pagination, content joining, `isError`, protocol-error replies, server-error rejection) + config/spec parsing + a real-stdio end-to-end test against a live Python MCP server (gated on python3). |

### 10.20b `src/mcp_server.rs` — Cheetah as an MCP server (4 runtime + 6 tests)

`cheetah mcp <model-path>` exposes the loaded model to MCP agents over
stdio: JSON-RPC 2.0 (newline-delimited) in, responses out. The engine loads
through `Engine::load` directly — the protocol channel carries JSON frames
only; diagnostics go to stderr.

| Function | Analysis |
|---|---|
| `run(path, options, ant)` | Server lifecycle: `Engine::load` → the stdio request loop (parse → dispatch → respond) until the agent closes stdin; malformed JSON lines answer a parse error (id null) and keep serving. |
| `handle_request(message, state)` | Protocol dispatch: notifications (no id) return nothing; `initialize` → protocol version + tools capability + serverInfo; `ping` → empty result; `tools/list` → the two-tool catalog; `tools/call` → routed through the `McpServerStateLike` invoker; unknown methods → `-32601`. Generic over the invoker so tests stub generation. |
| `McpServerState::invoke_tool(params)` (via `McpServerStateLike`) | Tool dispatch: `chat_generate` (validate messages → apply max_tokens/temperature overrides → chat template → generate → text) and `model_info` (id/architecture/device); unknown tools → `-32602`. Execution failures are `isError: true` MCP results — the agent sees them. |
| `parse_tool_messages(arguments)` | Validates the agent's `messages` array (roles/content strings, non-empty) into engine chat messages. |
| `initialize_result()` / `tools_list_result()` | The static protocol payloads (protocol version, capabilities, two-tool catalog with JSON Schema). |
| `DiscardSink` | Drops streamed deltas — the MCP result carries `generation.text` (capture defaults to true). |
| 6 tests *(test)* | Cover the initialize shape, the two-tool catalog, call routing through the invoker, unknown-tool/unknown-method protocol errors, notification handling, and chat_generate argument validation — all without loading a model (the invoker is stubbed). |

### 10.21 `src/chat/vulkan/` — the Vulkan backend

`mod.rs` (2 runtime), `transformer.rs` (17 runtime + 9 tests), `weights.rs`
(17 runtime + 1 test). The phi executor lives in its registration file
(`src/chat/registrations/phi.rs`, section 10.14) — one family, one file, on
both backends.

| Function | Analysis |
|---|---|
| `required_usize(config, name)` | Reads a required usize field from a model config, with a uniform error message. |
| `load_with_substrate(load_initialized)` (vulkan) | Bootstraps the Vulkan substrate for one executor load and guarantees symmetric teardown: initialize → validate pipelines → run the caller's loader → shut the substrate down on any failure (reporting a second failure if the cleanup itself errors). Both executors' `load` entries call this. |
| `TransformerFeatures::standard(model_name)` | The baseline executor feature set for one family name (no biases, no qk-norm, dense FFN, default weight layout). |
| `TransformerWeightLayout::default()` | The standard llama-shaped tensor naming (embedding/norms/projections/MLP suffixes) — families override pieces via their registration. |
| `VulkanTransformerConfig::from_raw(raw, features)` | Config parsing + validation: derives head_dim/kv heads, applies family defaults, honors `use_sliding_window`, maps `hidden_act`, rejects RoPE scaling and inconsistent MoE counts, and fails loudly on any non-positive or non-finite dimension. |
| `FeedForward::forward(input, rows, hidden, expert_residency, layer)` | The feed-forward dispatch: dense = gate/up/activation/down; MoE = router logits → per-row `select_experts` → rows bucketed BY EXPERT → per-expert batched matmul → scatter-add weighted outputs; also drives the GPU-bucket expert residency (evict unpicked, upload picked, keyed `(layer, expert)`). |
| `apply_activation(activation, gate, up)` | Applies the configured activation (silu / gelu / tanh-gelu) to the gate·up pair as a shader call. |
| `VulkanTransformerExecutor::forward` (ModelForward impl) | Trait entry delegating to the inherent forward. |
| `VulkanTransformerExecutor::clear_cache` (trait impl) | Clears every layer's KV cache. |
| `VulkanTransformerExecutor::device_name` | Reports `"Vulkan"`. |
| `VulkanTransformerExecutor::as_ant_tier_control` | Exposes the executor's GPU-bucket page control. |
| `VulkanTransformerExecutor::page_weight(unit_id, resident)` | Ant bucketing: page one tensor between the GPU bucket (persistent upload) and the host-streaming bucket; the host copy in `WeightData` is always authoritative, so eviction only releases the device allocation and a page-in re-uploads from it. |
| `VulkanTransformerExecutor::page_in_unit` / `evict_unit` (AntTierControl impl) | Forward to `page_weight` for the engine's tier hooks. |
| `VulkanTransformerExecutor::load(paths, config, plan)` | Public load: initializes the substrate, runs `prepare`, then `load_initialized`; on any failure it also shuts the backend down (and reports if that cleanup fails too). |
| `VulkanTransformerExecutor::load_initialized(paths, config, plan)` | The load graph: take embedding/final norm (offset applied per feature), resolve tied-vs-untied lm head, then per layer take norms (q/k norms by declaration or presence), projections and the dense-or-MoE feed-forward with Ant-bucket uploads per the plan. |
| `Layer::clear_cache` (inherent) | Per-layer KV reset used by the trait impl. |
| `VulkanTransformerExecutor::forward(tokens, offset)` (inherent) | The transformer graph: validates ids/positions/cache-offset consistency, embedding gather as a shader, then per layer: input RMSNorm → q/k/v (+ optional qk-norm) → rotary → cache append → attention (sliding window) → o_proj → residual → post norm → feed-forward → residual; final norm + lm head on the last position. |
| `load_moe_feed_forward(weights, plan, prefix, config, experts, top_k)` | Loads the Mixtral-style sparse feed-forward (router + per-expert gate/up/down) under the fixed MOE_* layout constants. |
| 9 tests *(test)* | Cover common + MoE config parsing, feature overrides, MoE weight loading from the configured layout, row-tile bounding, tiny dense/MoE forward + cache reset, the env-gated q8 resident forward, and fixture config bytes. |
| `WeightData::from_view(name, view)` | Converts a Safetensors tensor view into the typed payload (F32/F16/BF16); other dtypes are rejected loudly. |
| `WeightData::format_id()` | The registered precision id this payload resolves to (quantized buffers report their own). |
| `WeightData::as_bytes()` | The raw byte view (no copy) the precision-registry operator tables consume. |
| `WeightData::value(index)` | Flat scalar access (dequantized); quantized payloads fall back to row 0 (flat accessors only serve diagnostics). |
| `WeightData::add_one()` | Builds the `(1 + weight)` offset copy gemma's norms apply. |
| `WeightData::len()` | Logical element count. |
| `WeightData::ffi_parts()` | FFI view for shader operators: `(pointer, weight_format, element_count)` with the shared_ops format codes (0/2/3; quantized formats report their registered code and are served only by their dedicated matmul pipeline). |
| `WeightData::rms_norm(input, rows, width, eps)` | RMSNorm shader per stored format; quantized payloads error (norms stay in original precision by design). |
| `WeightData::layer_norm(bias, input, rows, width, eps)` | LayerNorm shader (weight + bias, dtypes must match); quantized payloads error the same way. |
| `WeightData::linear(input, rows, input_size, output_size)` | The streaming matmul core: row tiles (≤64) × column tiles (bounded by a 16M-element weight tile) for original precisions; quantized buffers stage whole-buffer per row tile through their format's registered streaming operator; validates dimensions up front. |
| `Linear::forward(input, rows)` | Linear dispatch: resident-bucket path when a slot is bound (format's registered resident matmul, row-tiled), else the streaming path; adds the bias per row when present. |
| `resident_upload(weight, row_bytes)` | Uploads one weight into the resident bucket through its format's registered operator; None when not servable (streaming stays active). |
| `upload_slot(name, weight, row_bytes)` | `resident_upload` with the Ant page-in diagnostic line. |
| `load_weights(paths)` | Mmaps every shard, deserializes Safetensors, skips internal padding tensors, rejects duplicate tensor names across shards. |
| `take_weight(weights, name, expected_shape)` | Removes one tensor from the pool with exact shape validation. |
| `validate_shape(name, actual, expected)` | Shape assertion helper. |
| `take_linear(weights, plan, prefix, in, out, with_bias)` | Builds a `Linear`: shape-validated weight (+ required bias when `with_bias`), load-time quantization when a format is selected (norms/biases/embedding never quantize), and the Ant-bucket resident upload for VRAM-tier units (fallback to streaming with a warning). |
| `quantized_formats_match_across_backends` *(test)* | The cross-backend contract: every registered quantized format must agree across Vulkan streaming, Vulkan resident and the native gemv. |

### 10.22 `src/serve/` — OpenAI-compatible API (82 runtime + 39 tests)

`mod.rs` (9 runtime), `http.rs` (37 runtime), `openai.rs` (27 runtime),
`queue.rs` (10 runtime).

| Function | Analysis |
|---|---|
| `serve(options, chat_options, ant_options, mcp_options)` | Server lifecycle: refuses unauthenticated non-loopback listeners, resolves the model + Ant settings, loads the engine, resolves/generates the API key, installs the Ctrl-C handler, builds the shared state, and loops on `recv_timeout` spawning one worker per request (generation requests take a queue ticket first). |
| `resolve_model(options)` | Model resolution: direct safetensors file → prepared model directory → name match inside `--model-dir` (ambiguous/missing errors list what is available). |
| `model_name_for_match(model)` | Strips directories + the `.safetensors` suffix to get the matchable name. |
| `model_id_from_path(path)` | The API-visible model id: file stem for files, directory name for directories. |
| `is_model_directory(path)` | Whether a directory looks like a model (config.json + tokenizer.json + a recognized weight layout). |
| `list_safetensors_models(dir)` | Sorted listing of safetensors files and model directories — embedded in the not-found error so users can pick the right name. |
| `resolve_api_key(options)` | Explicit `--api-key` wins; `--generate-key` creates one; otherwise no auth (loopback-only enforced upstream). |
| `is_loopback(host)` | Whether the bind address is loopback-only. |
| `display_url(host, port)` | The startup banner URL (localhost for loopback, bracketed IPv6). |
| `is_generation_request(request)` | Whether an incoming request is a generation endpoint (POST `/v1/chat/completions` or `/v1/completions`) — decides whether a queue ticket is taken. |
| `handle(state, request, generation_ticket)` | The request dispatcher: OPTIONS → CORS preflight, `/health` auth-free, Bearer auth check, then route to models/chat/completions handlers. |
| `route(method, path)` | Route table: known paths with wrong methods → 405, unknown → 404. |
| `authorized(request, expected)` | Bearer-token check; no expected key means allow. |
| `constant_time_eq(a, b)` | Length-checked constant-time byte comparison so token checks do not leak the first differing position. |
| `health_body()` | The static `{"status":"ok"}` body. |
| `model_list_string(state)` | Serializes the OpenAI `/v1/models` list (single loaded model). |
| `single_model_string(state)` | Serializes the `/v1/models/{id}` entry. |
| `cors_headers()` | The CORS header set (origin/methods/headers). |
| `options_response()` | 204 preflight response with CORS + Allow headers. |
| `json_headers()` | CORS + `application/json`. |
| `respond_json(request, status, body)` | Sends a JSON response with explicit length. |
| `respond_error(request, status, message)` | Sends an OpenAI-shaped error body (`error_body`). |
| `read_body(request)` | Drains the request body. |
| `handle_chat_completions(state, request, ticket)` | The chat endpoint: parse → model match → tool-choice validation → message building → option resolution; tool requests route through the tool prompt path, then stream (SSE with disconnect flag) or respond non-streaming; every failure maps to a status + error body before generation starts. |
| `chat_completion_parts(text, finish_reason)` | Splits a finished generation into content/tool-calls/finish-reason (tool JSON detected in the text). |
| `build_tool_prompt_inputs(request, tools, choice, default_system)` | Serializes messages (+ injected default system when none present) and the (choice-filtered) tools for the template's tool parser. |
| `generate_tool_chat(state, guard, input)` | Runs a tool-mode generation (null sink), parses the output through the template parser, then through `parse_tool_response` → text or calls. |
| `handle_completions(state, request, ticket)` | The legacy `/v1/completions` endpoint (non-streaming only; echo unsupported): prompt → single user message → generate → text completion response. |
| `generate_chat(state, guard, tokens, options, stop, sink, resident_overrides)` | Shared generation core: new context + `Engine::generate` under the queue guard. |
| `prompt_error_status(error)` | Maps context-overflow prompt errors to 400; everything else is 500. |
| `NullSink::push` | Discards streamed text (non-streaming requests still stream internally). |
| `SseSink::new / new_with_disconnect / new_inner` | Sink construction with optional client-disconnect tracking. |
| `SseSink::send_chunk(...)` | Emits one OpenAI `chat.completion.chunk` frame; a closed channel maps to "Client disconnected". |
| `SseSink::send_tool_calls(calls)` | Emits the tool-call delta frame (indexed, id/name/arguments) with the assistant role on first emission. |
| `SseSink::send_done()` | Emits `data: [DONE]`. |
| `sse_frame(...)` | Serializes one SSE frame (None fields skipped per the OpenAI shape). |
| `SseSink::push(text)` (TextSink impl) | Streams one content delta; empty chunks ignored; a flagged disconnect aborts generation. |
| `SseSink::capture_text` | Always false — SSE text is never retained server-side. |
| `SseSink::interrupted` | True once the client disconnected. |
| `sse_response(receiver, heartbeat, disconnected)` | Builds the `text/event-stream` response around a `ChannelReader`. |
| `ChannelReader::new / new_with_disconnect` | Reader construction (disconnect flag shared with the sink). |
| `ChannelReader::drop()` | Marks the client disconnected — the generation sink sees it and stops. |
| `ChannelReader::read(buf)` | The streaming pump: blocks only when no bytes were returned yet this call (heartbeat frame on idle timeout), otherwise drains queued chunks without waiting; EOF when the sender drops. |
| `ApiError::bad_request(message)` | 400 error constructor. |
| `PromptValue::default()` | Empty-text default for the untagged prompt field. |
| `role_kind(role)` | Role validation: system/developer (→ system), user, assistant, tool; anything else is a 400. |
| `parse_content(content)` | Joins content parts; non-text parts are rejected (no multimodal input). |
| `build_chat_messages(request)` | Validates + converts the OpenAI message list: tool_calls only in assistant messages, tool messages need ids, assistant tool-call history is serialized into content, system/developer content split out, at least one chat message required. |
| `parse_tool_response(parsed, raw_text, tools, choice)` | Tool-output resolution ladder: template-parsed tool_calls → compatible raw JSON shapes → free-text JSON extraction (validated against tools + choice) → plain text fallback. |
| `parse_generated_tool_calls(text)` | Parses tool calls from free model text (compact, OpenAI, or flat shapes) after fence/brace extraction. |
| `extract_generated_tool_call_json(text)` | Strips code fences and slices the outermost JSON object. |
| `convert_tool_calls(calls, tools, choice)` | Converts parsed calls to the response shape, filtering by tool existence + named choice; None → fall back to text. |
| `normalize_tool_arguments(arguments)` | Accepts arguments as an object or a JSON-encoded string; both validate to object text. |
| `tool_call_id()` | Generates `call_<hex>` ids. |
| `resolve_tool_choice(tools, tool_choice)` | Validates tool declarations (function type only, unique names, object parameters) together with the choice, so an invalid combination never reaches generation. |
| `build_completion_message(request)` | Converts a completion prompt (text or array) into one user message; empty prompts rejected. |
| `apply_sampling(options, ...)` | Merges request sampling fields with range validation (temperature ≥ 0, top_p ∈ [0,1]). |
| `resolve_chat_options(defaults, request)` | Merges chat request options; `max_tokens` and `max_completion_tokens` are mutually exclusive; `n > 1` rejected. |
| `resolve_completion_options(defaults, request)` | Completion variant (no top_k, echo unsupported). |
| `stop_strings(stop)` | Normalizes the untagged stop value (string or array) to a vec. |
| `ensure_model_matches(requested, loaded)` | 404 when the request names a different model than the one loaded (OpenAI-compatible model pinning). |
| `error_kind(status)` | OpenAI error type string per status class. |
| `error_body(status, message)` | Serializes the OpenAI error envelope (with a constant fallback body if serialization ever fails). |
| `Usage::new(prompt, completion)` | Totals prompt + completion tokens. |
| `completion_id()` / `text_completion_id()` | `chatcmpl-` / `cmpl-` prefixed random ids. |
| `now_unix()` | Current UNIX seconds (0 pre-epoch). |
| `generate_api_key()` | `cheetah-` prefixed 16-byte hex key. |
| `random_hex(bytes)` | OS-entropy-seeded hex string (via `random_u64`) for keys/ids. |
| `random_u64()` | Entropy source built from `RandomState` (per-process randomized hasher seed). |
| `FifoQueue::new(max_active)` | Builds the concurrency-limited FIFO queue; zero rejected. |
| `FifoQueue::enqueue()` | Takes the next ticket number (wrapping) and returns a cancellable ticket. |
| `FifoQueue::acquire(ticket)` | Blocks until the ticket is at the head AND an active slot is free; cancelled tickets are skipped only at the head (no queue jumping); waking all waiters on progress. |
| `FifoQueue::cancel(ticket)` | Marks a ticket cancelled — head tickets advance immediately, others join the cancelled set. |
| `FifoQueue::lock_state()` | Mutex lock with poisoned-lock recovery. |
| `advance_cancelled(state)` | Skips abandoned tickets only when they reach the head, so later requests cannot jump the queue. |
| `QueueTicket::acquire()` | Consumes the ticket into a queue guard. |
| `QueueTicket::drop()` | An unacquired dropped ticket cancels itself (abandoned requests never hold a slot). |
| `QueueGuard::drop()` | Releases one active slot and wakes waiters. |
| `FifoQueue::default()` | One-slot queue (single concurrent generation). |
| 39 serve tests *(test)* | Cover constant-time token comparison, context-overflow status mapping, routing (known/405/404), the SSE channel reader (EOF, chunk pumping, batching, heartbeat, unblocking on drop), sink interruption after reader drop, tool-call delta fields, streaming + non-streaming chat paths, empty-chunk tolerance, request parsing (content arrays, tool histories, tool_choice validation, generated-call shapes), option resolution conflicts, completion prompt building, model matching, error bodies, id/key formats, and queue fairness/cancellation/concurrency limits. |

### 10.23 Coverage notes & regeneration

- **Coverage**: the tables above cover all 921 functions extracted from
  `src/` — 682 runtime functions (including the 36 `extern` FFI declarations
  that bind the C substrate) and 246 test-context functions. Every runtime
  function carries an English `///` doc comment in source (extern
  declarations a `//` line), per section 3. Where a
  codebase repeats one shape across families (the 7 × 6 trait-impl methods in
  the registration files, the per-family backend tables) the reference lists
  the repeated rows explicitly with the family named, so every function is
  addressed at least once.
- **C side**: the three C translation units (`cheetah_init.c`,
  `cheetah_memory.c`, `cheetah_vulkan.c`) implement exactly the contracts the
  FFI wrappers in section 10.7 document; their entry points are the `cheetah_*`
  extern declarations listed there.
- **Regeneration of the skeleton**: the function list can be re-derived at any
  time with
  `grep -rn -E '^\s*(pub([^ ]* )?)?fn ' src --include='*.rs'` or by re-running
  the audit this document was written from. When you add a function, add its
  row here in the same commit — the reference is review surface, not generated
  output.
- **When to update**: any new function, any contract change (error conditions,
  memory behavior, dispatch rules), and any rename of a frozen name (§2).
  A PR that changes behavior without touching the corresponding row here is
  incomplete.

### 10.24 Test-function index (every test named)

Section 10 groups test suites into analyzed rows; this index names every
one of the 246 test functions so the reference covers the full
921-function inventory (681 runtime + 240 tests + the MCP server's
protocol-surface stubs).

**`src/ant.rs`** (9):
  `unit` `layered_unit` `plan_residency_assigns_vram_then_ram_then_disk`
  `plan_residency_fills_layers_in_order_and_tracks_partial_layers`
  `ant_filter_known_units_drops_unknown_and_duplicates` `ant_swap_fit_decision_rules`
  `plan_residency_respects_disabled_ant` `settings_for_model_reads_user_managed_file`
  `ant_lifecycle_against_real_prepared_model`

**`src/chat/engine.rs`** (12):
  `temporary_directory` `resolves_safetensors_from_cheetah_manifest`
  `rejects_unsafe_manifest_shard_paths`
  `resolves_direct_weight_file_and_parent_manifest_metadata`
  `renders_qwen_chatml_prompt_with_system_message` `renders_llama_instruction_prompt`
  `renders_mistral_instruction_prompt` `renders_llama3_header_prompt`
  `ant_resident_weight_paths_keep_only_resident_units`
  `ant_resident_weight_paths_keep_everything_when_ant_disabled`
  `sampling_is_deterministic_for_fixed_seed`
  `generation_stats_report_decode_speed_and_handle_unavailable_speed`

**`src/chat/mod.rs`** (4):
  `init_metadata_comes_from_generated_model_architectures`
  `init_metadata_is_absent_for_unsupported_architecture`
  `init_metadata_comes_from_mistral_and_mixtral_architectures`
  `init_metadata_comes_from_qwen3_architecture`

**`src/chat/moe.rs`** (3):
  `selects_top_experts_and_renormalizes_weights`
  `full_top_k_equals_softmax_of_the_router_logits` `rejects_invalid_scores_and_expert_count`

**`src/chat/native/decoder.rs`** (33):
  `pattern` `new` `units` `tensors` `new` `phi` `drop` `write_shard` `write_manifest`
  `dense_config` `moe_config` `logits_of`
  `dense_forward_is_parity_clean_across_residency_transitions`
  `q8_quantized_forward_terminates_and_populates_quantized_units`
  `moe_forward_is_parity_clean_across_residency_transitions`
  `moe_block_mixes_selected_experts_by_router_weight`
  `moe_router_drives_per_expert_residency_transitions`
  `sparse_attention_masks_positions_outside_the_window`
  `sparse_window_changes_the_logits_only_when_it_cuts_the_cache`
  `decoder_config_resolves_sparse_window_and_moe_expert_fields` `phi_units` `phi_tensors`
  `gemma_config` `phi_config` `ref_vec` `ref_mat` `ref_rope` `ref_gelu` `ref_rmsnorm_offset`
  `ref_layernorm` `gemma_forward_matches_reference_math` `phi_forward_matches_reference_math`
  `decoder_config_resolves_dense_family_features`

**`src/chat/native/ops.rs`** (14):
  `bf16_widening_matches_bit_shift` `gemv_multiplies_rows_in_parallel`
  `gemv_q8_matches_rowwise_scale_code_reference` `gemv_q4_matches_rowwise_scale_code_reference`
  `gemv_q4_consumes_the_registered_quantize_output`
  `gemv_q8_consumes_the_registered_quantize_output` `rmsnorm_matches_reference_computation`
  `rope_preserves_magnitude_and_matches_reference`
  `rope_heads_rotates_leading_dims_of_every_head` `silu_multiply_matches_reference`
  `softmax_sums_to_one_and_is_shift_invariant` `rope_matches_reference_rotate_half`
  `dot_row_bf16_fused_matches_dequant_dot` `dot_row_bf16_fused_full_width_matches_reference`

**`src/chat/precision.rs`** (9):
  `registry_contains_original_and_quantized_formats`
  `registrations_have_unique_ids_and_selectors` `every_registration_carries_backends`
  `selectors_belong_to_quantized_formats_only` `every_format_is_registered_for_both_backends`
  `format_lookup_normalizes_and_reports_unknown` `selector_accepts_only_one_and_true`
  `views_round_trip_values` `views_reject_bad_lengths`

**`src/chat/precision/q4.rs`** (5):
  `buffer_layout_matches_ffi_contract` `quantize_round_trip_within_one_step`
  `nibble_packing_matches_column_parity` `buffer_dequant_scale`
  `quantize_rejects_size_mismatch`

**`src/chat/precision/q8.rs`** (3):
  `buffer_layout_matches_ffi_contract` `quantize_round_trip_within_one_step`
  `quantize_rejects_size_mismatch`

**`src/chat/registrations/gemma.rs`** (1):
  `gemma_vulkan_features_configure_norm_and_embedding_behavior`

**`src/chat/registrations/mixtral.rs`** (2):
  `mixtral_vulkan_features_configure_moe_tensor_layout`
  `mixtral_vulkan_features_reject_invalid_expert_counts`

**`src/chat/registrations/qwen2.rs`** (2):
  `qwen2_features_include_attention_bias_and_tied_embeddings`
  `qwen2_features_reject_inconsistent_attention_dimensions`

**`src/chat/registrations/qwen3.rs`** (5):
  `qwen3_raw` `qwen3_vulkan_features_configure_dense_tensor_layout`
  `qwen3_vulkan_disables_the_sliding_window_when_the_config_asks`
  `qwen3_vulkan_rejects_scaled_rope` `qwen3_vulkan_honors_a_configured_attention_bias`

**`src/chat/registry.rs`** (10):
  `registers_cpu_and_vulkan_backends` `backend_device_rejects_unknown_backend`
  `registration_lookup_normalizes_architecture_names`
  `native_backend_serves_llama_and_qwen_families` `native_backend_serves_mixtral_moe_family`
  `native_backend_serves_dense_delta_families` `every_native_registration_carries_a_family`
  `registrations_cover_disjoint_architectures` `vulkan_capability_table_and_loader_table_agree`
  `vulkan_loaders_reach_family_feature_selection`

**`src/chat/vulkan/transformer.rs`** (9):
  `identity_linear` `tiny_executor` `parses_common_and_moe_transformer_configurations`
  `feature_values_override_common_configuration`
  `loads_moe_weights_from_configured_tensor_layout`
  `linear_splits_large_sequences_into_bounded_row_tiles`
  `tiny_dense_and_moe_models_forward_and_reset_cache` `vulkan_q8_quantized_forward_terminates`
  `config_bytes`

**`src/chat/vulkan/weights.rs`** (1):
  `quantized_formats_match_across_backends`

**`src/cli.rs`** (26):
  `args` `parses_chat_speed_options` `parses_sampling_and_system_options`
  `defaults_are_deterministic` `rejects_out_of_range_sampling_values`
  `rejects_zero_and_missing_numeric_options` `rejects_missing_model_path`
  `parses_ant_flags_into_chat_options` `parses_chat_ant_flags` `rejects_unknown_ant_flag`
  `bare_ant_flag_enables_preset_ant` `ant_flag_with_boolean_value_overrides_preset`
  `parses_ant_setup_interactive_flag` `parses_setup_alias_and_bare_interactive_flags`
  `reports_usage_errors_with_usage_variant` `parses_context_options_and_management_commands`
  `parses_serve_defaults` `parses_serve_options_and_generation_overrides`
  `accepts_ephemeral_port_and_generate_key` `serve_rejects_mutually_exclusive_keys`
  `serve_rejects_missing_model_and_invalid_port`
  `serve_rejects_invalid_or_repeated_generation_limit`
  `parses_positional_serve_model_and_options` `serve_rejects_multiple_model_arguments`
  `parses_inspect_output_options` `inspect_rejects_missing_or_repeated_output_path`

**`src/color.rs`** (1):
  `applies_color_only_when_terminal_color_is_enabled`

**`src/context.rs`** (9):
  `test_store` `test_archive` `rejects_unsafe_context_names`
  `saves_lists_loads_and_replaces_contexts` `reports_corrupt_and_unsupported_archives`
  `older_archives_default_context_size_to_8192` `empty_missing_directory_lists_no_contexts`
  `validates_model_identity_and_saved_options`
  `accepts_safetensors_paths_and_rejects_other_formats`

**`src/inspect.rs`** (5):
  `safetensors_file` `inspects_safetensors_header_and_tensor_shapes`
  `hides_internal_aligned_shard_padding_tensors` `rejects_invalid_safetensors_files`
  `resolves_manifest_shards_for_inspection`

**`src/chat/native/decoder.rs`** — KV 复用基础测试（3，计入 decoder 测试组）:
  `kv_continuation_matches_full_prefill_bit_exact` `kv_cache_does_not_leak_across_generations`
  `clone_shares_weights_but_starts_with_empty_kv`

**`src/chat/toolcalls.rs`** (2):
  `next_tool_round_appends_calls_and_results` `next_tool_round_rejects_invalid_history`

**`src/mcp.rs`** (3):
  `parses_server_specs_and_config_files` `connects_to_a_real_stdio_server_end_to_end`
  `tool_definitions_map_and_duplicate_names_reject`

**`src/mcp/client.rs`** (5):
  `initialize_sends_handshake_and_notification` `list_tools_extracts_and_paginates`
  `call_tool_joins_text_and_flags_errors` `server_requests_are_answered_with_protocol_error`
  `server_error_rejects_the_request`

**`src/cli.rs`** — MCP flag tests (3, inside the cli tests group):
  `parses_mcp_flags_for_chat` `rejects_repeated_mcp_config` `parses_mcp_flags_for_serve`

**`src/main.rs`** (4):
  `inspect_text_export_limits_tensors_by_default`
  `inspect_json_export_preserves_structured_schema`
  `inspect_report_export_overwrites_existing_file`
  `no_arguments_launches_chat_from_the_current_directory`

**`src/native.rs`** (9):
  `initializer_progress_callback_tracks_phase_and_position` `temporary_directory`
  `initializer_ffi_reports_native_failure`
  `registry_metadata_updates_manifest_without_dropping_other_fields`
  `matmul_rejects_invalid_shapes_before_calling_vulkan` `bf16_matmul_matches_reference`
  `qwen_operators_match_cpu_reference`
  `matmul_returns_native_failure_when_backend_is_uninitialized`
  `matmul_computes_on_vulkan_device`

**`src/serve/http.rs`** (14):
  `constant_time_compares_bearer_tokens` `prompt_context_overflow_maps_to_bad_request`
  `routes_known_paths` `routes_return_405_wrong_method_and_404_unknown`
  `channel_reader_eof_on_disconnect` `channel_reader_pumps_chunks`
  `channel_reader_batches_ready_chunks` `dropping_reader_unblocks_sse_sink`
  `sse_sink_stops_generation_after_reader_drop` `channel_reader_sends_heartbeat_when_idle`
  `sse_tool_call_delta_uses_openai_fields`
  `chat_stream_emits_each_generated_chunk_without_copying_text`
  `non_stream_chat_converts_tool_json_and_falls_back_to_text`
  `sse_sink_ignores_empty_generated_chunks`

**`src/serve/mod.rs`** (10):
  `model_name_strips_safetensors` `model_id_from_stem` `loopback_checking`
  `display_url_defaults_to_localhost` `resolve_model_by_exact_file`
  `resolve_model_by_name_in_dir` `resolve_model_by_model_directory_name`
  `resolve_model_directory_with_cheetah_manifest` `resolve_model_ambiguous_errors`
  `resolve_model_not_found_shows_listing`

**`src/serve/openai.rs`** (18):
  `parse` `parses_chat_request_with_content_array` `parses_tool_calls_and_tool_result_history`
  `validates_tool_choice_and_tool_definitions`
  `parses_valid_tool_output_and_falls_back_for_invalid_output`
  `parses_generated_tool_calls_in_compact_and_openai_shapes`
  `parses_generated_tool_calls_from_fenced_json_and_surrounding_prose`
  `parses_generated_tool_calls_in_flat_format`
  `rejects_invalid_generated_tool_calls_for_text_fallback` `rejects_non_text_and_unknown_roles`
  `maps_developer_role_to_system` `normalizes_stop_values`
  `resolves_options_and_rejects_conflicts` `builds_completion_prompt`
  `model_matching_and_error_body` `sse_chunk_delta_omits_none_fields`
  `serializes_tool_calls_and_omits_them_from_text_responses`
  `api_key_uses_cheetah_prefix_and_hex`

**`src/serve/queue.rs`** (4):
  `serves_waiters_in_ticket_order` `cancelled_request_does_not_block_following_requests`
  `permits_only_the_configured_number_of_active_tickets` `rejects_a_zero_generation_limit`
