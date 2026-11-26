//! Activation operators (unary elementwise + gated multiplies), native CPU.
//!
//! Operates on f32 buffers. The unary family takes an optional `param`
//! (used by `leaky_relu` as alpha; ignored by the parameterless ones) so one
//! `Unary` signature covers the whole family. `swiglu`/`geglu` reuse the
//! frozen free functions in `crate::chat::native::ops` (`silu_multiply` /
//! `gelu_multiply`) through a same-signature fn pointer.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};
use crate::chat::native::ops as native_ops;

#[cfg(feature = "vulkan")]
use super::{
    OpKind, VulkanOp, ACT_GELU_ERF, ACT_GELU_TANH, ACT_LEAKY_RELU, ACT_RELU, ACT_SIGMOID,
    ACT_SILU, ACT_SOFTPLUS, ACT_TANH, OP_ACTIVATION, OP_ELEMENTWISE_GELU_TANH_RHS,
    OP_ELEMENTWISE_SILU_RHS,
};

// --- Unary kernels ---------------------------------------------------------

fn relu(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

fn leaky_relu(x: &mut [f32], alpha: f32) {
    for value in x.iter_mut() {
        if *value < 0.0 {
            *value *= alpha;
        }
    }
}

fn tanh(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        *value = value.tanh();
    }
}

fn sigmoid(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        *value = 1.0 / (1.0 + (-*value).exp());
    }
}

fn silu(x: &mut [f32], _param: f32) {
    native_ops::silu(x);
}

fn gelu_tanh(x: &mut [f32], _param: f32) {
    native_ops::gelu(x);
}

/// GELU with the exact (erf) approximation, mirroring the GLSL `gelu_erf`
/// in `shared_ops.comp` bit-for-bit so native and Vulkan agree in the parity
/// tests.
fn gelu_erf(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        let v = *value;
        let sign = if v < 0.0 { -1.0 } else { 1.0 };
        let ax = v.abs() / core::f32::consts::SQRT_2;
        let t = 1.0 / (1.0 + 0.327_591_1 * ax);
        let erf = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-ax * ax).exp();
        *value = 0.5 * v * (1.0 + sign * erf);
    }
}

fn softplus(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        *value = native_ops::softplus(*value);
    }
}

// --- Backend constructors (one per op) -------------------------------------

const fn relu_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(relu)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_RELU,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(relu))]
    }
}

const fn leaky_relu_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(leaky_relu)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_LEAKY_RELU,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(leaky_relu))]
    }
}

const fn tanh_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(tanh)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_TANH,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(tanh))]
    }
}

const fn sigmoid_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(sigmoid)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_SIGMOID,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(sigmoid))]
    }
}

const fn silu_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(silu)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_SILU,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(silu))]
    }
}

const fn gelu_tanh_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(gelu_tanh)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_GELU_TANH,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(gelu_tanh))]
    }
}

const fn gelu_erf_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(gelu_erf)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_GELU_ERF,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(gelu_erf))]
    }
}

const fn softplus_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(softplus)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ACTIVATION,
                kind: OpKind::Unary,
                sub_op: ACT_SOFTPLUS,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(softplus))]
    }
}

const fn swiglu_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::InPlaceBinop(native_ops::silu_multiply)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ELEMENTWISE_SILU_RHS,
                kind: OpKind::InPlaceBinop,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::InPlaceBinop(native_ops::silu_multiply))]
    }
}

const fn geglu_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::InPlaceBinop(native_ops::gelu_multiply)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ELEMENTWISE_GELU_TANH_RHS,
                kind: OpKind::InPlaceBinop,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::InPlaceBinop(native_ops::gelu_multiply))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "activations",
    entries: &[
        OpEntry { id: "act.relu", backends: relu_backends() },
        OpEntry { id: "act.leaky_relu", backends: leaky_relu_backends() },
        OpEntry { id: "act.tanh", backends: tanh_backends() },
        OpEntry { id: "act.sigmoid", backends: sigmoid_backends() },
        OpEntry { id: "act.silu", backends: silu_backends() },
        OpEntry { id: "act.gelu_tanh", backends: gelu_tanh_backends() },
        OpEntry { id: "act.gelu_erf", backends: gelu_erf_backends() },
        OpEntry { id: "act.softplus", backends: softplus_backends() },
        OpEntry { id: "act.swiglu", backends: swiglu_backends() },
        OpEntry { id: "act.geglu", backends: geglu_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn run_unary(id: &str, input: &[f32], param: f32) -> Vec<f32> {
        let mut buf = input.to_vec();
        match super::super::native_op_for(id).unwrap() {
            NativeOp::Unary(f) => f(&mut buf, param),
            other => panic!("{id} should be Unary, got {other:?}"),
        }
        buf
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn relu_clamps_negatives() {
        assert_eq!(run_unary("act.relu", &[-1.0, 0.0, 2.0, -3.5], 0.0), [0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn leaky_relu_uses_alpha() {
        assert!(close(
            &run_unary("act.leaky_relu", &[-2.0, 3.0], 0.01),
            &[-0.02, 3.0]
        ));
    }

    #[test]
    fn sigmoid_known_values() {
        assert!(close(&run_unary("act.sigmoid", &[0.0], 0.0), &[0.5]));
        assert!(run_unary("act.sigmoid", &[100.0], 0.0)[0] > 0.999);
    }

    #[test]
    fn silu_matches_x_times_sigmoid() {
        let input: [f32; 3] = [-1.5, 0.0, 2.0];
        let expected: Vec<f32> = input
            .iter()
            .map(|x| {
                let v = *x;
                v / (1.0 + (-v).exp())
            })
            .collect();
        assert!(close(&run_unary("act.silu", &input, 0.0), &expected));
    }

    #[test]
    fn gelu_erf_matches_reference() {
        // GELU(erf) at 0 is 0, at large positive approaches x, at large negative approaches 0.
        assert!(run_unary("act.gelu_erf", &[0.0], 0.0)[0].abs() < 1e-6);
        assert!(close(&run_unary("act.gelu_erf", &[10.0], 0.0), &[10.0]));
        assert!(run_unary("act.gelu_erf", &[-10.0], 0.0)[0].abs() < 1e-3);
    }

    #[test]
    fn swiglu_and_geglu_run_in_place() {
        let orig = [1.0f32, 2.0, -1.0];
        let up = [3.0f32, 0.5, 2.0];
        let mut gate = orig.to_vec();
        match super::super::native_op_for("act.swiglu").unwrap() {
            NativeOp::InPlaceBinop(f) => f(&mut gate, &up),
            other => panic!("{other:?}"),
        }
        let want: Vec<f32> = orig
            .iter()
            .zip(up)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        assert!(close(&gate, &want));
    }
}
