//! Reduction operators (sum / mean / max / argmax) over one axis of a
//! row-major f32 buffer, native CPU. The Vulkan side is op code 18 in
//! `shared_ops.comp`; the sub-op selector picks the reduction and the axis
//! rides a push constant.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};
use crate::chat::native::ops as native_ops;

#[cfg(feature = "vulkan")]
use super::{
    OpKind, VulkanOp, OP_REDUCE, RED_ARGMAX, RED_MAX, RED_MEAN, RED_SUM,
};

fn reduce_sum(x: &[f32], rows: u32, width: u32, axis: u32, out: &mut [f32]) {
    native_ops::reduce_sum(x, rows as usize, width as usize, axis, out);
}

fn reduce_mean(x: &[f32], rows: u32, width: u32, axis: u32, out: &mut [f32]) {
    native_ops::reduce_mean(x, rows as usize, width as usize, axis, out);
}

fn reduce_max(x: &[f32], rows: u32, width: u32, axis: u32, out: &mut [f32]) {
    native_ops::reduce_max(x, rows as usize, width as usize, axis, out);
}

fn reduce_argmax(x: &[f32], rows: u32, width: u32, axis: u32, out: &mut [f32]) {
    native_ops::reduce_argmax(x, rows as usize, width as usize, axis, out);
}

const fn sum_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Reduce(reduce_sum)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_REDUCE,
                kind: OpKind::Reduce,
                sub_op: RED_SUM,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Reduce(reduce_sum))]
    }
}

const fn mean_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Reduce(reduce_mean)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_REDUCE,
                kind: OpKind::Reduce,
                sub_op: RED_MEAN,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Reduce(reduce_mean))]
    }
}

const fn max_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Reduce(reduce_max)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_REDUCE,
                kind: OpKind::Reduce,
                sub_op: RED_MAX,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Reduce(reduce_max))]
    }
}

const fn argmax_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Reduce(reduce_argmax)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_REDUCE,
                kind: OpKind::Reduce,
                sub_op: RED_ARGMAX,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Reduce(reduce_argmax))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "reductions",
    entries: &[
        OpEntry { id: "reduce.sum", backends: sum_backends() },
        OpEntry { id: "reduce.mean", backends: mean_backends() },
        OpEntry { id: "reduce.max", backends: max_backends() },
        OpEntry { id: "reduce.argmax", backends: argmax_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str, x: &[f32], rows: u32, width: u32, axis: u32, out_len: usize) -> Vec<f32> {
        let mut out = vec![f32::NAN; out_len];
        match super::super::native_op_for(id).unwrap() {
            NativeOp::Reduce(f) => f(x, rows, width, axis, &mut out),
            other => panic!("{id} should be Reduce, got {other:?}"),
        }
        out
    }

    const X: [f32; 10] = [1.0, -2.0, 3.0, 0.5, 4.0, -1.5, 2.5, -0.5, 1.5, 2.0];

    #[test]
    fn sum_and_mean_over_both_axes() {
        // axis 0: one output per row (2 rows × 5).
        assert_eq!(run("reduce.sum", &X, 2, 5, 0, 2), [6.5, 4.0]);
        assert_eq!(run("reduce.mean", &X, 2, 5, 0, 2), [1.3, 0.8]);
        // axis 1: one output per column (mean pooling over the rows).
        assert_eq!(
            run("reduce.sum", &X, 2, 5, 1, 5),
            [-0.5, 0.5, 2.5, 2.0, 6.0]
        );
        assert_eq!(
            run("reduce.mean", &X, 2, 5, 1, 5),
            [-0.25, 0.25, 1.25, 1.0, 3.0]
        );
    }

    #[test]
    fn max_and_argmax_over_both_axes() {
        assert_eq!(run("reduce.max", &X, 2, 5, 0, 2), [4.0, 2.5]);
        assert_eq!(run("reduce.max", &X, 2, 5, 1, 5), [1.0, 2.5, 3.0, 1.5, 4.0]);
        // axis 0 argmax: column indices of each row's maximum.
        assert_eq!(run("reduce.argmax", &X, 2, 5, 0, 2), [4.0, 1.0]);
        // axis 1 argmax: row indices of each column's maximum.
        assert_eq!(run("reduce.argmax", &X, 2, 5, 1, 5), [0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn argmax_breaks_ties_toward_the_later_index() {
        // Mirrors engine::sample_token's greedy path:
        // `max_by(|(_, l), (_, r)| l.total_cmp(r))` returns the LAST maximal
        // element, and NaN ranks above every number under `total_cmp`.
        let tied = [1.0f32, 5.0, 5.0, 5.0, -1.0];
        assert_eq!(run("reduce.argmax", &tied, 1, 5, 0, 1), [3.0]);
        // NaN wins the scan; the output is the NaN's index (1) as an f32.
        let with_nan = [1.0f32, f32::NAN, 5.0];
        assert_eq!(run("reduce.argmax", &with_nan, 1, 3, 0, 1), [1.0]);
    }

    #[test]
    fn nan_propagates_through_sum_mean_max() {
        let with_nan = [1.0f32, f32::NAN, 3.0];
        assert!(run("reduce.sum", &with_nan, 1, 3, 0, 1)[0].is_nan());
        assert!(run("reduce.mean", &with_nan, 1, 3, 0, 1)[0].is_nan());
        assert!(run("reduce.max", &with_nan, 1, 3, 0, 1)[0].is_nan());
    }

    #[test]
    fn width_that_is_not_a_multiple_of_the_workgroup() {
        // 131 columns (local_size_x = 64 would tail at 128): rows of a
        // ragged width must reduce completely on both axes.
        let x: Vec<f32> = (0..3 * 131).map(|i| (i % 17) as f32 - 8.0).collect();
        let sums = run("reduce.sum", &x, 3, 131, 0, 3);
        for (row, value) in sums.iter().enumerate() {
            let expected: f32 = x[row * 131..(row + 1) * 131].iter().sum();
            assert_eq!(*value, expected, "row {row}");
        }
        let maxima = run("reduce.max", &x, 3, 131, 1, 131);
        for (column, value) in maxima.iter().enumerate() {
            let expected = (0..3)
                .map(|row| x[row * 131 + column])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(*value, expected, "column {column}");
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn reduce_entries_resolve_the_expected_codes_and_sub_ops() {
        let cases = [
            ("reduce.sum", RED_SUM),
            ("reduce.mean", RED_MEAN),
            ("reduce.max", RED_MAX),
            ("reduce.argmax", RED_ARGMAX),
        ];
        for (id, sub_op) in cases {
            let op = super::super::vulkan_op_for(id).expect(id);
            assert_eq!(op.op_code, OP_REDUCE);
            assert_eq!(op.kind, OpKind::Reduce);
            assert_eq!(op.sub_op, sub_op);
            assert!(op.custom_shader.is_none());
        }
    }

    /// PyTorch cross-verification. The fixture is generated externally
    /// (`torch.sum/mean/max/argmax` over a seeded random `[rows, width]`
    /// buffer) and pointed at through `CHEETAH_REDUCE_FIXTURE`; without it
    /// the test skips, keeping the suite self-contained (same pattern as the
    /// Ant real-model tests). When a Vulkan device is present the GPU path is
    /// checked against the same reference.
    #[test]
    fn pytorch_reference_fixture_matches() {
        let Some(path) = std::env::var_os("CHEETAH_REDUCE_FIXTURE") else {
            eprintln!(
                "SKIP: CHEETAH_REDUCE_FIXTURE not set — generate the torch reference fixture to run the cross-verification"
            );
            return;
        };
        let raw = std::fs::read_to_string(&path).expect("read reduce fixture");
        let fixture: serde_json::Value = serde_json::from_str(&raw).expect("parse reduce fixture");
        let rows = fixture["rows"].as_u64().expect("rows") as u32;
        let width = fixture["width"].as_u64().expect("width") as u32;
        let x: Vec<f32> = fixture["x"]
            .as_array()
            .expect("x")
            .iter()
            .map(|v| v.as_f64().expect("x element") as f32)
            .collect();
        assert_eq!(x.len(), rows as usize * width as usize);

        let ids = [
            ("reduce.sum", "sum"),
            ("reduce.mean", "mean"),
            ("reduce.max", "max"),
            ("reduce.argmax", "argmax"),
        ];
        #[cfg(feature = "vulkan")]
        let device_ready = crate::native::vulkan::initialize()
            .and_then(|_| crate::native::vulkan::prepare())
            .is_ok();
        #[cfg(not(feature = "vulkan"))]
        let device_ready = false;
        for (id, key) in ids {
            for axis in [0u32, 1] {
                let out_len = if axis == 0 { rows as usize } else { width as usize };
                let expected: Vec<f32> = fixture["expected"][format!("{key}_axis{axis}")]
                    .as_array()
                    .unwrap_or_else(|| panic!("expected[{key}_axis{axis}]"))
                    .iter()
                    .map(|v| v.as_f64().expect("expected element") as f32)
                    .collect();
                assert_eq!(expected.len(), out_len);

                let mut n_out = vec![f32::NAN; out_len];
                match super::super::native_op_for(id).unwrap() {
                    NativeOp::Reduce(f) => f(&x, rows, width, axis, &mut n_out),
                    other => panic!("{id} should be Reduce, got {other:?}"),
                }
                // Sequential f32 accumulation vs torch's reduction order:
                // relative tolerance; argmax indices must match exactly.
                for (i, (got, want)) in n_out.iter().zip(&expected).enumerate() {
                    let diff = (got - want).abs();
                    let tol = if id == "reduce.argmax" {
                        1e-6
                    } else {
                        1e-4 * want.abs().max(1.0)
                    };
                    assert!(
                        diff <= tol,
                        "{id} axis {axis}[{i}]: got {got} want {want} (diff {diff})"
                    );
                }

                #[cfg(feature = "vulkan")]
                {
                    if device_ready {
                        let op = super::super::vulkan_op_for(id).expect(id);
                        let v_out = crate::chat::vulkan::ops::run_vulkan_op(
                            &op,
                            crate::chat::vulkan::ops::OpCall::Reduce {
                                x: &x,
                                rows: rows as usize,
                                width: width as usize,
                                axis,
                            },
                        )
                        .expect(id)
                        .primary;
                        for (i, (got, want)) in v_out.iter().zip(&expected).enumerate() {
                            let diff = (got - want).abs();
                            let tol = if id == "reduce.argmax" {
                                1e-6
                            } else {
                                1e-4 * want.abs().max(1.0)
                            };
                            assert!(
                                diff <= tol,
                                "vulkan {id} axis {axis}[{i}]: got {got} want {want} (diff {diff})"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "PyTorch reference match: native{} all sub-ops × both axes",
            if device_ready { " + vulkan" } else { "" }
        );
    }
}
