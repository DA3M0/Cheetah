//! Generic operator catalog (`xxx_ops`) under the declarative registration
//! mode.
//!
//! Every file in `src/chat/ops/` declares a group of basic operators — each
//! with a stable id and a per-backend implementation (native CPU + Vulkan) —
//! so any model architecture can be assembled by combining registered
//! operators. `build.rs` collects them into the generated
//! `OUT_DIR/ops_registry.rs` (`REGISTRY`, included below); both backends
//! derive every dispatch from that table: adding an operator is adding a
//! file, no core dispatch edits.
//!
//! This catalog carries only **non-matmul** operators (activations,
//! elementwise, norms, position encoding, stateful recurrent steps). Large
//! linear projections stay in the precision registry (`src/chat/precision`),
//! which computes them in the weight's stored precision. Catalog operators
//! operate on f32 buffers: small parameters (norm weights, biases) are
//! dequantized to f32 by the caller, which keeps native and Vulkan symmetric
//! and lets the parity tests compare buffers directly.
//!
//! How dispatch flows:
//!
//! ```text
//! model registration forward ──► ops::native_op_for("act.relu")  [CPU]
//!                              └► ops::vulkan_op_for("act.relu") [GPU, feature = vulkan]
//!        └─ both lookups scan the generated REGISTRY; a miss lists
//!           every registered op id so typos surface with the valid set.
//! ```

// The catalog is additive: model forwards do not consume it yet, so in a
// non-test binary the lookup functions and entries are intentionally unused.
#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Stable op codes shared with `shared_ops.comp` and `cheetah_vulkan.c`.
// Existing codes (0..9) are reused with the f32 weight-format code (0) so the
// catalog reads f32 buffers uniformly; new codes (10+) are catalog-only.
// ---------------------------------------------------------------------------

/// rms_norm (b0=x, b1=weight f32, b2=out): format param = 0 ⇒ f32.
pub(crate) const OP_RMS_NORM: u32 = 0;
/// elementwise add (`lhs += rhs`) on f32 buffers.
pub(crate) const OP_ELEMENTWISE_ADD: u32 = 1;
/// elementwise silu(gate) ⊙ rhs (swiglu) on f32 buffers.
pub(crate) const OP_ELEMENTWISE_SILU_RHS: u32 = 2;
/// rotary embedding (b0=query, b1=key, b2=q-out, b3=k-out).
pub(crate) const OP_ROTARY: u32 = 3;
/// elementwise gelu_erf(gate) ⊙ rhs on f32 buffers.
pub(crate) const OP_ELEMENTWISE_GELU_ERF_RHS: u32 = 5;
/// elementwise gelu_tanh(gate) ⊙ rhs (geglu) on f32 buffers.
pub(crate) const OP_ELEMENTWISE_GELU_TANH_RHS: u32 = 6;
/// layer_norm (b0=x, b1=weight f32, b4=bias f32, b2=out): format param = 0.
pub(crate) const OP_LAYER_NORM: u32 = 7;
/// embedding gather (b0=token ids u32, b1=table f32, b2=out): format = 0.
pub(crate) const OP_EMBEDDING_GATHER: u32 = 8;
/// broadcast bias add (b0=x, b1=bias f32, b2=out): format param = 0.
pub(crate) const OP_BIAS_ADD: u32 = 9;

// New catalog op codes — every one reads/writes raw f32 buffers directly.
pub(crate) const OP_ACTIVATION: u32 = 10;
pub(crate) const OP_BINARY: u32 = 11;
pub(crate) const OP_RMS_NORM_OFFSET: u32 = 12;
pub(crate) const OP_GROUP_NORM: u32 = 13;
pub(crate) const OP_L2_NORM: u32 = 14;
pub(crate) const OP_SOFTMAX: u32 = 15;
pub(crate) const OP_CONV1D_STEP: u32 = 16;
pub(crate) const OP_GATED_DELTA_STEP: u32 = 17;
/// Reduction over one axis of a row-major f32 buffer (sub-op picks
/// sum/mean/max/argmax; axis rides a push constant).
pub(crate) const OP_REDUCE: u32 = 18;

/// Op codes >= this base route to a registration-supplied custom SPIR-V shader
/// (see [`CustomShader`]) instead of the shared `shared_ops` pipeline. This is
/// the escape hatch for operations that don't fit the shared dispatch: adding
/// such an op is still "add a file" (a `.rs` registration plus a `.comp` under
/// `src/chat/ops/shaders/`), with no core dispatch edits.
pub(crate) const CUSTOM_OP_BASE: u32 = 100;

// Sub-op selectors packed into push constants (OP_ACTIVATION / OP_BINARY).
pub(crate) const ACT_RELU: u32 = 0;
pub(crate) const ACT_LEAKY_RELU: u32 = 1;
pub(crate) const ACT_TANH: u32 = 2;
pub(crate) const ACT_SIGMOID: u32 = 3;
pub(crate) const ACT_SILU: u32 = 4;
pub(crate) const ACT_GELU_TANH: u32 = 5;
pub(crate) const ACT_GELU_ERF: u32 = 6;
pub(crate) const ACT_SOFTPLUS: u32 = 7;

pub(crate) const BIN_ADD: u32 = 0;
pub(crate) const BIN_SUB: u32 = 1;
pub(crate) const BIN_MUL: u32 = 2;
pub(crate) const BIN_DIV: u32 = 3;
pub(crate) const BIN_MAX: u32 = 4;
pub(crate) const BIN_MIN: u32 = 5;

// Sub-op selectors for OP_REDUCE.
pub(crate) const RED_SUM: u32 = 0;
pub(crate) const RED_MEAN: u32 = 1;
pub(crate) const RED_MAX: u32 = 2;
pub(crate) const RED_ARGMAX: u32 = 3;

/// Operator shape: tells the generic Vulkan dispatcher how to bind buffers
/// and push constants for a registered op. Mirrors the `NativeOp` variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// In-place unary elementwise: `x[i] = f(x[i], param)`.
    Unary,
    /// In-place binary: `a[i] = op(a[i], b[i])` (swiglu / geglu).
    InPlaceBinop,
    /// Binary elementwise into `out`: `out[i] = lhs[i] op rhs[i % period]`.
    Binary,
    /// Vector norm (rms / layer / group) with f32 weight & bias.
    Norm,
    /// Per-head L2 norm.
    NormL2,
    /// Row-wise softmax.
    Softmax,
    /// Rotary embedding.
    Rope,
    /// Embedding table gather.
    Embedding,
    /// Stateful causal conv1d step.
    Conv1dStep,
    /// Stateful gated delta-rule step.
    GatedDeltaStep,
    /// Reduction over one axis (sum / mean / max / argmax).
    Reduce,
}

/// One native CPU operator, tagged by its calling signature. The variants are
/// plain `fn` pointers stored in `Copy` consts, so a dispatch costs one table
/// lookup — no vtable call, matching the precision registry's cost model.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NativeOp {
    /// `x[i] = f(x[i], param)` in place.
    Unary(fn(&mut [f32], f32)),
    /// `a[i] = op(a[i], b[i])` in place (swiglu / geglu).
    InPlaceBinop(fn(&mut [f32], &[f32])),
    /// `out[i] = lhs[i] op rhs[i % period]`; `period == 0` ⇒ full-length rhs.
    Binary(fn(&[f32], &[f32], &mut [f32], usize)),
    /// In-place norm over one vector: f32 `weight` & `bias` (bias may be
    /// empty); `p1`/`p2` carry variant params (groupnorm groups/channels),
    /// ignored by rmsnorm / layernorm.
    Norm(fn(&mut [f32], &[f32], &[f32], f32, u32, u32)),
    /// In-place per-head L2 norm: `x` is `heads * head_dim`.
    NormL2(fn(&mut [f32], u32, u32)),
    /// In-place row-wise softmax: `x` is `rows * width`.
    Softmax(fn(&mut [f32], u32)),
    /// In-place rotary embedding.
    Rope(fn(&mut [f32], u32, u32, u32, u32, f32)),
    /// `out = table[row * width ..][..width]`.
    Embedding(fn(&[f32], usize, usize, &mut [f32])),
    /// One causal-conv1d step with an in-place state shift.
    Conv1dStep(fn(&[f32], &[f32], u32, &mut [f32], &mut [f32])),
    /// One gated-delta-rule step with recurrent state.
    GatedDeltaStep(
        fn(&[f32], &[f32], &[f32], &[f32], &[f32], &mut [f32], &mut [f32], u32, u32, u32, u32),
    ),
    /// Reduction over one axis of a row-major `[rows, width]` buffer:
    /// `(x, rows, width, axis, out)`; the sub-op picks the reduction.
    Reduce(fn(&[f32], u32, u32, u32, &mut [f32])),
}

/// A registration-supplied compute shader (op codes >= `CUSTOM_OP_BASE`).
/// The `.comp` lives under `src/chat/ops/shaders/` and is compiled by
/// `build.rs`; it must follow the `shared_ops` layout (5 storage buffers,
/// 8 uint push constants, `local_size_x = 64`, entry point `main`).
#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CustomShader {
    /// Shader file stem under `src/chat/ops/shaders/` (build-time default).
    pub(crate) name: &'static str,
    /// Environment variable overriding the compiled SPIR-V path at runtime.
    pub(crate) env: &'static str,
}

/// One Vulkan operator: the op code to dispatch, the operator shape
/// (buffer/param binding), an optional sub-op selector, and — for custom op
/// codes (>= `CUSTOM_OP_BASE`) — the custom shader it runs. `custom_shader:
/// None` means the built-in `shared_ops.spv` pipeline.
#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VulkanOp {
    pub(crate) op_code: u32,
    pub(crate) kind: OpKind,
    pub(crate) sub_op: u32,
    pub(crate) custom_shader: Option<CustomShader>,
}

/// One backend implementation for an operator. Mirrors the precision
/// registry's `PrecisionBackend` — a new backend kind adds a variant plus
/// per-op entries in the registration files.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OpBackend {
    Native(NativeOp),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanOp),
}

/// One registered operator: its stable id and the per-backend implementations.
pub(crate) struct OpEntry {
    pub(crate) id: &'static str,
    pub(crate) backends: &'static [OpBackend],
}

/// One declarative operator-group registration (one per file in
/// `src/chat/ops/`). Entries live in those files and carry no runtime code;
/// build.rs assembles them into `REGISTRY` (generated below).
pub(crate) struct OpRegistration {
    /// Source file stem (diagnostics / docs).
    #[allow(dead_code)]
    pub(crate) source: &'static str,
    /// Operators declared by this file, in declared order.
    pub(crate) entries: &'static [OpEntry],
}

impl OpEntry {
    /// Backend operator by kind, scanning the declared backends in order.
    pub(crate) fn native(&self) -> Option<NativeOp> {
        self.backends.iter().find_map(|backend| match backend {
            OpBackend::Native(op) => Some(*op),
            #[cfg(feature = "vulkan")]
            _ => None,
        })
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn vulkan(&self) -> Option<VulkanOp> {
        self.backends.iter().find_map(|backend| match backend {
            OpBackend::Vulkan(op) => Some(*op),
            _ => None,
        })
    }

    /// Registered backend names, in declared order (registry invariant tests).
    #[allow(dead_code)]
    pub(crate) fn backend_names(&self) -> Vec<&'static str> {
        self.backends
            .iter()
            .map(|backend| match backend {
                OpBackend::Native(_) => "native",
                #[cfg(feature = "vulkan")]
                OpBackend::Vulkan(_) => "vulkan",
            })
            .collect()
    }
}

// Generated by build.rs from src/chat/ops/: the `<group>` modules (each file
// = operators + its declarative registration) plus the assembled `REGISTRY`.
include!(concat!(env!("OUT_DIR"), "/ops_registry.rs"));

// Generated by build.rs from src/chat/ops/shaders/ (vulkan only): lists every
// custom op shader that compiled, mapping its stem to the absolute default
// SPIR-V path under OUT_DIR. Absent without the vulkan feature.
#[cfg(feature = "vulkan")]
include!(concat!(env!("OUT_DIR"), "/custom_op_shaders.rs"));

/// Every `(registration, entry)` pair in registry order.
fn all_entries() -> impl Iterator<Item = (&'static OpRegistration, &'static OpEntry)> {
    REGISTRY
        .iter()
        .copied()
        .flat_map(|registration| registration.entries.iter().map(move |e| (registration, e)))
}

/// All registered op ids in registry order — the fixed-point assertion for
/// the registry-content tests.
pub(crate) fn registered_op_ids() -> Vec<&'static str> {
    all_entries().map(|(_, entry)| entry.id).collect()
}

/// The entry serving an op id, or `None`. A miss surfaces the registered id
/// list through `op_listing` so callers can name the valid set in errors.
pub(crate) fn entry_for_op(id: &str) -> Option<&'static OpEntry> {
    all_entries()
        .find(|(_, entry)| entry.id == id)
        .map(|(_, entry)| entry)
}

/// Registered op ids joined for "unknown op" error messages.
pub(crate) fn op_listing() -> String {
    format!("{:?}", registered_op_ids())
}

/// Native CPU operator for an op id, from the registration's backend table.
pub(crate) fn native_op_for(id: &str) -> Option<NativeOp> {
    entry_for_op(id).and_then(OpEntry::native)
}

/// Vulkan operator for an op id, from the registration's backend table.
#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_op_for(id: &str) -> Option<VulkanOp> {
    entry_for_op(id).and_then(OpEntry::vulkan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_non_empty() {
        assert!(
            !registered_op_ids().is_empty(),
            "ops REGISTRY must not be empty"
        );
    }

    #[test]
    fn unknown_op_lookup_is_none_and_lists_registered() {
        assert!(native_op_for("definitely.not.an.op").is_none());
        let listing = op_listing();
        assert!(
            listing.contains("act.relu"),
            "op listing should name registered ids, got: {listing}"
        );
    }

    #[test]
    fn known_bootstrap_op_resolves_native() {
        let op = native_op_for("act.relu").expect("act.relu is registered");
        let mut values = vec![-1.0, 0.0, 2.0, -3.5];
        match op {
            NativeOp::Unary(f) => {
                f(&mut values, 0.0);
                assert_eq!(values, [0.0, 0.0, 2.0, 0.0]);
            }
            other => panic!("act.relu should be a Unary op, got {other:?}"),
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn known_bootstrap_op_resolves_vulkan() {
        let op = vulkan_op_for("act.relu").expect("act.relu has a vulkan backend");
        assert_eq!(op.op_code, OP_ACTIVATION);
        assert_eq!(op.kind, OpKind::Unary);
        assert_eq!(op.sub_op, ACT_RELU);
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_probe {
    #[test]
    fn device_probe_runs_one_op() {
        let init = crate::native::vulkan::initialize();
        let prep = init.and(crate::native::vulkan::prepare());
        match prep {
            Ok(()) => {
                let vk = crate::chat::ops::vulkan_op_for("act.relu").expect("act.relu registered");
                let out = crate::chat::vulkan::ops::run_vulkan_op(
                    &vk,
                    crate::chat::vulkan::ops::OpCall::Unary {
                        x: &[-2.0, 0.5, 3.0],
                        param: 0.0,
                    },
                )
                .expect("relu dispatch must succeed when device is present");
                eprintln!("PROBE OK vulkan relu = {:?}", out.primary);
                assert_eq!(out.primary, [0.0, 0.5, 3.0]);
            }
            Err(e) => {
                eprintln!(
                    "PROBE SKIP: no usable Vulkan device ({e}); vulkan parity tests will skip"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registry invariants: ids are unique, every op has both backends, and the
// Vulkan op kind matches the native op shape.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod invariants {
    use super::*;

    #[test]
    fn registered_op_ids_are_unique() {
        let ids = registered_op_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            ids.len(),
            "duplicate op id registered; all ids = {ids:?}"
        );
    }

    #[test]
    fn every_registered_op_has_native_backend() {
        for id in registered_op_ids() {
            assert!(
                native_op_for(id).is_some(),
                "op {id} is missing a native backend"
            );
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn every_registered_op_has_vulkan_backend() {
        for id in registered_op_ids() {
            assert!(
                vulkan_op_for(id).is_some(),
                "op {id} is missing a vulkan backend"
            );
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_kind_matches_native_shape() {
        for id in registered_op_ids() {
            let n = native_op_for(id).expect(id);
            let v = vulkan_op_for(id).expect(id);
            let expected = match n {
                NativeOp::Unary(_) => OpKind::Unary,
                NativeOp::InPlaceBinop(_) => OpKind::InPlaceBinop,
                NativeOp::Binary(_) => OpKind::Binary,
                NativeOp::Norm(_) => OpKind::Norm,
                NativeOp::NormL2(_) => OpKind::NormL2,
                NativeOp::Softmax(_) => OpKind::Softmax,
                NativeOp::Rope(_) => OpKind::Rope,
                NativeOp::Embedding(_) => OpKind::Embedding,
                NativeOp::Conv1dStep(_) => OpKind::Conv1dStep,
                NativeOp::GatedDeltaStep(_) => OpKind::GatedDeltaStep,
                NativeOp::Reduce(_) => OpKind::Reduce,
            };
            assert_eq!(v.kind, expected, "op {id} kind mismatch");
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn custom_op_codes_and_shaders_are_consistent() {
        let mut custom_codes = std::collections::HashSet::new();
        for id in registered_op_ids() {
            let v = vulkan_op_for(id).expect(id);
            if v.op_code >= CUSTOM_OP_BASE {
                assert!(
                    custom_codes.insert(v.op_code),
                    "custom op code {} is reused (op {id})",
                    v.op_code
                );
                let shader = v
                    .custom_shader
                    .unwrap_or_else(|| panic!("custom op {id} declares no CustomShader"));
                assert!(
                    !shader.name.is_empty(),
                    "custom op {id} has an empty shader name"
                );
                assert!(
                    !shader.env.is_empty(),
                    "custom op {id} has an empty shader env"
                );
            } else {
                assert!(
                    v.custom_shader.is_none(),
                    "built-in op {id} must not carry a CustomShader"
                );
            }
        }
        assert!(
            !custom_codes.is_empty(),
            "the custom-shader family must register at least one op"
        );
    }
}

// ---------------------------------------------------------------------------
// Native ↔ Vulkan parity: every registered op runs on both backends with
// identical inputs and the buffers must match within tolerance. Tests skip
// (pass vacuously) when no Vulkan device is available, so the suite stays
// green on headless CI without a GPU.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "vulkan"))]
mod parity {
    use super::*;
    use crate::chat::vulkan::ops::{OpCall, run_vulkan_op};

    fn device_available() -> bool {
        crate::native::vulkan::initialize()
            .and_then(|_| crate::native::vulkan::prepare())
            .is_ok()
    }

    fn cmp(label: &str, native: &[f32], vulkan: &[f32], tol: f32) {
        assert_eq!(
            native.len(),
            vulkan.len(),
            "{label}: length mismatch native={} vulkan={}",
            native.len(),
            vulkan.len()
        );
        for (i, (a, b)) in native.iter().zip(vulkan).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: native={a} vulkan={b} diff={diff} tol={tol}"
            );
        }
    }

    #[test]
    fn parity_unary_activations() {
        if !device_available() {
            return;
        }
        let x: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let cases: &[(&str, f32, f32)] = &[
            ("act.relu", 0.0, 1e-6),
            ("act.leaky_relu", 0.01, 1e-6),
            ("act.tanh", 0.0, 1e-5),
            ("act.sigmoid", 0.0, 1e-5),
            ("act.silu", 0.0, 1e-5),
            ("act.gelu_tanh", 0.0, 1e-5),
            ("act.gelu_erf", 0.0, 1e-5),
            ("act.softplus", 0.0, 1e-5),
        ];
        for (id, param, tol) in cases {
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Unary(f) => f(&mut n_out, *param),
                other => panic!("{id} should be Unary, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Unary {
                    x: &x,
                    param: *param,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, *tol);
        }
    }

    #[test]
    fn parity_custom_shader_op() {
        if !device_available() {
            return;
        }
        // A failed custom-shader compile is a build warning, not an error; the
        // runtime override is the documented recovery path. Skip when neither
        // the build artifact nor an override is available.
        if custom_op_shader_default("mish").is_none()
            && std::env::var_os("CHEETAH_MISH_OPS_SPV").is_none()
        {
            eprintln!("SKIP act.mish: custom shader not compiled and no CHEETAH_MISH_OPS_SPV");
            return;
        }
        let x: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
        let native = native_op_for("act.mish").expect("act.mish");
        let vulkan = vulkan_op_for("act.mish").expect("act.mish");
        let mut n_out = x.to_vec();
        match native {
            NativeOp::Unary(f) => f(&mut n_out, 0.0),
            other => panic!("act.mish should be Unary, got {other:?}"),
        }
        let v_out = run_vulkan_op(&vulkan, OpCall::Unary { x: &x, param: 0.0 })
            .expect("act.mish custom-shader dispatch")
            .primary;
        cmp("act.mish", &n_out, &v_out, 1e-5);
    }

    #[test]
    fn parity_in_place_binop() {
        if !device_available() {
            return;
        }
        // swiglu / geglu
        let gate = [-2.0f32, -0.5, 0.5, 2.0];
        let up = [1.0f32, 2.0, 3.0, 4.0];
        for id in ["act.swiglu", "act.geglu"] {
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = gate.to_vec();
            match native {
                NativeOp::InPlaceBinop(f) => f(&mut n_out, &up),
                other => panic!("{id} should be InPlaceBinop, got {other:?}"),
            }
            let v_out = run_vulkan_op(&vulkan, OpCall::InPlaceBinop { a: &gate, b: &up })
                .expect(id)
                .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }
        // bias.add
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let bias = [10.0f32, 20.0];
        let native = native_op_for("bias.add").expect("bias.add");
        let vulkan = vulkan_op_for("bias.add").expect("bias.add");
        let mut n_out = x.to_vec();
        match native {
            NativeOp::InPlaceBinop(f) => f(&mut n_out, &bias),
            other => panic!("bias.add should be InPlaceBinop, got {other:?}"),
        }
        let v_out = run_vulkan_op(&vulkan, OpCall::InPlaceBinop { a: &x, b: &bias })
            .expect("bias.add")
            .primary;
        cmp("bias.add", &n_out, &v_out, 1e-6);
    }

    #[test]
    fn parity_binary_elementwise() {
        if !device_available() {
            return;
        }
        let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let rhs = [10.0f32, 20.0];
        let ids = ["ew.add", "ew.sub", "ew.mul", "ew.div", "ew.max", "ew.min"];
        for id in ids {
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = vec![0.0f32; lhs.len()];
            match native {
                NativeOp::Binary(f) => f(&lhs, &rhs, &mut n_out, 0),
                other => panic!("{id} should be Binary, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Binary {
                    lhs: &lhs,
                    rhs: &rhs,
                    period: 0,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-6);
        }
    }

    #[test]
    fn parity_norms() {
        if !device_available() {
            return;
        }
        let width = 6usize;
        let x = [1.0f32, -2.0, 3.0, -0.5, 2.5, -1.0];
        let weight = [0.5f32, 1.0, 1.5, 0.8, 1.2, 0.3];
        let bias = [0.1f32, -0.2, 0.3, 0.0, 0.5, -0.1];
        let eps = 1e-5;

        // rmsnorm (no bias)
        for id in ["norm.rmsnorm", "norm.rmsnorm_offset"] {
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Norm(f) => f(&mut n_out, &weight, &[], eps, 0, 0),
                other => panic!("{id} should be Norm, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Norm {
                    x: &x,
                    weight: &weight,
                    bias: &[],
                    eps,
                    rows: 1,
                    width,
                    groups: 0,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }

        // layernorm (with bias)
        {
            let id = "norm.layernorm";
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Norm(f) => f(&mut n_out, &weight, &bias, eps, 0, 0),
                other => panic!("{id} should be Norm, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Norm {
                    x: &x,
                    weight: &weight,
                    bias: &bias,
                    eps,
                    rows: 1,
                    width,
                    groups: 0,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }

        // groupnorm (width=6, groups=2 → group_size=3)
        {
            let id = "norm.groupnorm";
            let groups = 2u32;
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Norm(f) => f(&mut n_out, &weight, &bias, eps, groups, width as u32),
                other => panic!("{id} should be Norm, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Norm {
                    x: &x,
                    weight: &weight,
                    bias: &bias,
                    eps,
                    rows: 1,
                    width,
                    groups: groups as usize,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }
    }

    #[test]
    fn parity_l2_softmax_rope_embedding() {
        if !device_available() {
            return;
        }

        // l2norm: heads=2, head_dim=4
        {
            let id = "norm.l2norm";
            let x = [1.0f32, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5];
            let (heads, head_dim) = (2u32, 4u32);
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::NormL2(f) => f(&mut n_out, heads, head_dim),
                other => panic!("{id} should be NormL2, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::NormL2 {
                    x: &x,
                    heads: heads as usize,
                    head_dim: head_dim as usize,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }

        // softmax: rows=2, width=4
        {
            let id = "norm.softmax";
            let x = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
            let (rows, width) = (2usize, 4usize);
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Softmax(f) => f(&mut n_out, width as u32),
                other => panic!("{id} should be Softmax, got {other:?}"),
            }
            let v_out = run_vulkan_op(&vulkan, OpCall::Softmax { x: &x, rows, width })
                .expect(id)
                .primary;
            cmp(id, &n_out, &v_out, 1e-5);
        }

        // rope: query-only span, heads=2, head_dim=8, rotary_dim=4
        {
            let id = "pos.rope";
            let x = [
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6,
                -0.7, -0.8,
            ];
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = x.to_vec();
            match native {
                NativeOp::Rope(f) => f(&mut n_out, 2, 8, 4, 3, 10000.0),
                other => panic!("{id} should be Rope, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Rope {
                    x: &x,
                    heads: 2,
                    head_dim: 8,
                    rotary_dim: 4,
                    position: 3,
                    theta: 10000.0,
                    scaling: &crate::chat::native::ops::RopeScaling::None,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-4);
        }

        // embedding gather: 3 rows x width 4, gather row 1
        {
            let id = "embedding.gather";
            let table = [
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let (row, width) = (1usize, 4usize);
            let native = native_op_for(id).expect(id);
            let vulkan = vulkan_op_for(id).expect(id);
            let mut n_out = vec![0.0f32; width];
            match native {
                NativeOp::Embedding(f) => f(&table, row, width, &mut n_out),
                other => panic!("{id} should be Embedding, got {other:?}"),
            }
            let v_out = run_vulkan_op(
                &vulkan,
                OpCall::Embedding {
                    table: &table,
                    row,
                    width,
                },
            )
            .expect(id)
            .primary;
            cmp(id, &n_out, &v_out, 1e-6);
        }
    }

    #[test]
    fn parity_conv1d_step_multi_token() {
        if !device_available() {
            return;
        }
        let id = "conv.causal1d_step";
        let (channels, kernel) = (4usize, 2usize);
        let weight = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]; // channels*kernel
        let inputs: [&[f32]; 3] = [
            &[1.0, 2.0, 3.0, 4.0],
            &[0.5, -1.0, 2.0, 0.0],
            &[-2.0, 1.5, 0.0, 3.0],
        ];
        let native = native_op_for(id).expect(id);
        let vulkan = vulkan_op_for(id).expect(id);
        let mut state_n = vec![0.0f32; channels * (kernel - 1)];
        let mut state_v = vec![0.0f32; channels * (kernel - 1)];
        for (step, input) in inputs.iter().enumerate() {
            let mut out_n = vec![0.0f32; channels];
            match native {
                NativeOp::Conv1dStep(f) => {
                    f(input, &weight, kernel as u32, &mut state_n, &mut out_n)
                }
                other => panic!("{id} should be Conv1dStep, got {other:?}"),
            }
            let out_v = run_vulkan_op(
                &vulkan,
                OpCall::Conv1dStep {
                    input,
                    weight: &weight,
                    kernel,
                    state: &mut state_v,
                },
            )
            .expect(id)
            .primary;
            cmp(&format!("{id} out step {step}"), &out_n, &out_v, 1e-5);
            cmp(&format!("{id} state step {step}"), &state_n, &state_v, 1e-5);
        }
    }

    #[test]
    fn parity_reduce() {
        if !device_available() {
            return;
        }
        // 5 rows × 130 columns: a width that is not a multiple of the
        // workgroup size (64) exercises the shader's tail on both axes.
        let (rows, width) = (5usize, 130usize);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        };
        let x: Vec<f32> = (0..rows * width).map(|_| next()).collect();
        let ids = ["reduce.sum", "reduce.mean", "reduce.max", "reduce.argmax"];
        for id in ids {
            for axis in [0u32, 1] {
                let out_len = if axis == 0 { rows } else { width };
                let native = native_op_for(id).expect(id);
                let vulkan = vulkan_op_for(id).expect(id);
                let mut n_out = vec![f32::NAN; out_len];
                match native {
                    NativeOp::Reduce(f) => f(&x, rows as u32, width as u32, axis, &mut n_out),
                    other => panic!("{id} should be Reduce, got {other:?}"),
                }
                let v_out = run_vulkan_op(
                    &vulkan,
                    OpCall::Reduce {
                        x: &x,
                        rows,
                        width,
                        axis,
                    },
                )
                .expect(id)
                .primary;
                // argmax outputs are exact indices; sums over 130 addends of
                // magnitude < 1 stay far inside 1e-4.
                let label = format!("{id} axis {axis}");
                cmp(&label, &n_out, &v_out, 1e-4);
            }
        }
    }

    #[test]
    fn parity_gated_delta_step_multi_token() {
        if !device_available() {
            return;
        }
        let id = "attn.gated_delta_step";
        let (num_k, num_v, k_dim, v_dim) = (1usize, 1usize, 4usize, 3usize);
        let state_len = num_v * k_dim * v_dim;
        let native = native_op_for(id).expect(id);
        let vulkan = vulkan_op_for(id).expect(id);
        let mut state_n = vec![0.0f32; state_len];
        let mut state_v = vec![0.0f32; state_len];

        // (query, key, value, gate, beta) per step
        let steps: [(&[f32], &[f32], &[f32], f32, f32); 3] = [
            (
                &[0.1, 0.2, -0.1, 0.3],
                &[0.2, -0.1, 0.4, 0.0],
                &[1.0, 2.0, 3.0],
                -0.5,
                0.9,
            ),
            (
                &[0.3, -0.2, 0.1, 0.5],
                &[0.0, 0.5, -0.3, 0.2],
                &[0.5, -1.0, 2.0],
                -0.3,
                0.7,
            ),
            (
                &[-0.2, 0.4, 0.6, -0.1],
                &[0.1, 0.1, 0.1, 0.1],
                &[2.0, 0.0, 1.0],
                -0.8,
                1.0,
            ),
        ];
        for (step, (query, key, value, gate, beta)) in steps.iter().enumerate() {
            let gate = [*gate];
            let beta = [*beta];
            let mut out_n = vec![0.0f32; num_v * v_dim];
            match native {
                NativeOp::GatedDeltaStep(f) => f(
                    query,
                    key,
                    value,
                    &gate,
                    &beta,
                    &mut state_n,
                    &mut out_n,
                    num_k as u32,
                    num_v as u32,
                    k_dim as u32,
                    v_dim as u32,
                ),
                other => panic!("{id} should be GatedDeltaStep, got {other:?}"),
            }
            let out_v = run_vulkan_op(
                &vulkan,
                OpCall::GatedDeltaStep {
                    query,
                    key,
                    value,
                    gate: &gate,
                    beta: &beta,
                    state: &mut state_v,
                    num_k_heads: num_k,
                    num_v_heads: num_v,
                    head_k_dim: k_dim,
                    head_v_dim: v_dim,
                },
            )
            .expect(id)
            .primary;
            cmp(&format!("{id} out step {step}"), &out_n, &out_v, 1e-4);
            cmp(&format!("{id} state step {step}"), &state_n, &state_v, 1e-4);
        }
    }
}
