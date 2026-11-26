//! Elementwise binary operators, native CPU: `out[i] = lhs[i] op rhs[i %
//! period]`. `period == 0` means "broadcast by `rhs.len()`", so a length-1
//! `rhs` is a scalar and a length-`last_dim` `rhs` broadcasts over rows.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};

#[cfg(feature = "vulkan")]
use super::{
    BIN_ADD, BIN_DIV, BIN_MAX, BIN_MIN, BIN_MUL, BIN_SUB, OpKind, VulkanOp, OP_BINARY,
};

fn binary(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize, op: fn(f32, f32) -> f32) {
    let p = if period == 0 { rhs.len() } else { period };
    assert!(p > 0, "binary rhs period must be non-zero");
    let n = lhs.len();
    assert!(out.len() >= n, "binary output too short");
    for i in 0..n {
        out[i] = op(lhs[i], rhs[i % p]);
    }
}

fn ew_add(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, |a, b| a + b)
}
fn ew_sub(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, |a, b| a - b)
}
fn ew_mul(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, |a, b| a * b)
}
fn ew_div(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, |a, b| a / b)
}
fn ew_max(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, f32::max)
}
fn ew_min(lhs: &[f32], rhs: &[f32], out: &mut [f32], period: usize) {
    binary(lhs, rhs, out, period, f32::min)
}

/// Generate a `const fn` backend constructor for one binary op. The array
/// literal is built from const items (the op fn path + a sub-op constant), so
/// it is const-promotable to `&'static` — a parameterized `const fn` could not
/// return that.
macro_rules! def_binary_backends {
    ($ctor:ident, $op:path, $sub:expr) => {
        const fn $ctor() -> &'static [OpBackend] {
            #[cfg(feature = "vulkan")]
            {
                &[
                    OpBackend::Native(NativeOp::Binary($op)),
                    OpBackend::Vulkan(VulkanOp {
                        op_code: OP_BINARY,
                        kind: OpKind::Binary,
                        sub_op: $sub,
                        custom_shader: None,
                    }),
                ]
            }
            #[cfg(not(feature = "vulkan"))]
            {
                &[OpBackend::Native(NativeOp::Binary($op))]
            }
        }
    };
}

def_binary_backends!(ew_add_backends, ew_add, BIN_ADD);
def_binary_backends!(ew_sub_backends, ew_sub, BIN_SUB);
def_binary_backends!(ew_mul_backends, ew_mul, BIN_MUL);
def_binary_backends!(ew_div_backends, ew_div, BIN_DIV);
def_binary_backends!(ew_max_backends, ew_max, BIN_MAX);
def_binary_backends!(ew_min_backends, ew_min, BIN_MIN);

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "elementwise",
    entries: &[
        OpEntry { id: "ew.add", backends: ew_add_backends() },
        OpEntry { id: "ew.sub", backends: ew_sub_backends() },
        OpEntry { id: "ew.mul", backends: ew_mul_backends() },
        OpEntry { id: "ew.div", backends: ew_div_backends() },
        OpEntry { id: "ew.max", backends: ew_max_backends() },
        OpEntry { id: "ew.min", backends: ew_min_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str, lhs: &[f32], rhs: &[f32], period: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; lhs.len()];
        match super::super::native_op_for(id).unwrap() {
            NativeOp::Binary(f) => f(lhs, rhs, &mut out, period),
            other => panic!("{id} should be Binary, got {other:?}"),
        }
        out
    }

    #[test]
    fn elementwise_add_full_length() {
        assert_eq!(run("ew.add", &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0], 0), [11.0, 22.0, 33.0]);
    }

    #[test]
    fn broadcast_scalar() {
        assert_eq!(run("ew.mul", &[1.0, 2.0, 3.0], &[2.0], 0), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcast_last_dim_over_rows() {
        let lhs = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let rhs = [1.0, 1.0, 1.0];
        assert_eq!(run("ew.sub", &lhs, &rhs, 3), [0.0, 1.0, 2.0, 9.0, 19.0, 29.0]);
    }

    #[test]
    fn max_min() {
        assert_eq!(run("ew.max", &[1.0, 5.0], &[3.0, 2.0], 0), [3.0, 5.0]);
        assert_eq!(run("ew.min", &[1.0, 5.0], &[3.0, 2.0], 0), [1.0, 2.0]);
    }
}
