//! Stateful recurrent operators, native CPU: a causal conv1d step and a
//! gated-delta-rule (linear attention) step. Both reuse the frozen free
//! functions in `crate::chat::native::ops`; the wrappers only bridge the
//! `u32`/`usize` dimension types. Their Vulkan peers land in
//! `shared_ops.comp` (op codes 16 / 17) and are verified against these in the
//! parity tests.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};
use crate::chat::native::ops as native_ops;

#[cfg(feature = "vulkan")]
use super::{OpKind, VulkanOp, OP_CONV1D_STEP, OP_GATED_DELTA_STEP};

fn conv1d_step_wrap(
    input: &[f32],
    weight: &[f32],
    kernel: u32,
    state: &mut [f32],
    output: &mut [f32],
) {
    native_ops::causal_conv1d_step(input, weight, kernel as usize, state, output);
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_step_wrap(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    num_k_heads: u32,
    num_v_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
) {
    native_ops::gated_delta_step(
        query,
        key,
        value,
        gate,
        beta,
        state,
        output,
        num_k_heads as usize,
        num_v_heads as usize,
        head_k_dim as usize,
        head_v_dim as usize,
    );
}

const fn conv1d_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Conv1dStep(conv1d_step_wrap)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_CONV1D_STEP,
                kind: OpKind::Conv1dStep,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Conv1dStep(conv1d_step_wrap))]
    }
}

const fn gated_delta_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::GatedDeltaStep(gated_delta_step_wrap)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_GATED_DELTA_STEP,
                kind: OpKind::GatedDeltaStep,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::GatedDeltaStep(gated_delta_step_wrap))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "stateful",
    entries: &[
        OpEntry { id: "conv.causal1d_step", backends: conv1d_backends() },
        OpEntry { id: "attn.gated_delta_step", backends: gated_delta_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv1d_step_shifts_state_and_matches_inner() {
        let width = 4usize;
        let kernel = 2usize;
        let weight = vec![0.5f32; width * kernel];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = vec![0.0f32; width * (kernel - 1)];
        let mut out = vec![0.0f32; width];
        // Reference via the inner free function.
        let mut ref_state = state.clone();
        let mut ref_out = out.clone();
        native_ops::causal_conv1d_step(&input, &weight, kernel, &mut ref_state, &mut ref_out);
        // Via the registry.
        match super::super::native_op_for("conv.causal1d_step").unwrap() {
            NativeOp::Conv1dStep(f) => f(&input, &weight, kernel as u32, &mut state, &mut out),
            other => panic!("{other:?}"),
        }
        assert_eq!(out, ref_out);
        assert_eq!(state, ref_state);
    }
}
