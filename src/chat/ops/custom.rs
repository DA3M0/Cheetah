//! Custom-shader operators (op codes >= `CUSTOM_OP_BASE`).
//!
//! These are the escape hatch for operations that do not fit the shared
//! `shared_ops` dispatch: the Vulkan backend runs a registration-supplied
//! `.comp` from `src/chat/ops/shaders/` (build.rs compiles it and publishes
//! the default SPIR-V path), while the native backend keeps a plain Rust
//! implementation for CPU-only builds and parity tests. Adding another custom
//! op is still "add a file" — no edits to `shared_ops.comp`,
//! `cheetah_vulkan.c`, or the generic Rust dispatcher.
//!
//! Custom shaders receive the shared_ops layout unchanged: op code in
//! `params.op`, op-specific push constants in `params.p1..p7`, and the five
//! storage buffers `b0..b4`. They must declare `local_size_x = 64` and an
//! entry point named `main`.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp, CUSTOM_OP_BASE};
use crate::chat::native::ops as native_ops;

#[cfg(feature = "vulkan")]
use super::{CustomShader, OpKind, VulkanOp};

/// Stable op code for `act.mish` (first custom-shader slot).
const OP_MISH: u32 = CUSTOM_OP_BASE;

/// Mish activation: `x · tanh(softplus(x))`, matching the custom shader's
/// numerically stable softplus (linear branch above 20).
fn mish(x: &mut [f32], _param: f32) {
    for value in x.iter_mut() {
        *value *= native_ops::softplus(*value).tanh();
    }
}

const fn mish_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Unary(mish)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_MISH,
                kind: OpKind::Unary,
                sub_op: 0,
                custom_shader: Some(CustomShader {
                    name: "mish",
                    env: "CHEETAH_MISH_OPS_SPV",
                }),
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Unary(mish))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "custom",
    entries: &[OpEntry { id: "act.mish", backends: mish_backends() }],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mish_matches_reference() {
        let input = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
        let mut out = input;
        match super::super::native_op_for("act.mish").unwrap() {
            NativeOp::Unary(f) => f(&mut out, 0.0),
            other => panic!("act.mish should be Unary, got {other:?}"),
        }
        for (value, actual) in input.iter().zip(out) {
            let expected = value * native_ops::softplus(*value).tanh();
            assert!(
                (actual - expected).abs() < 1e-6,
                "mish({value}) = {actual}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn mish_declares_its_custom_shader() {
        let op = super::super::vulkan_op_for("act.mish").expect("act.mish has a vulkan backend");
        assert!(op.op_code >= CUSTOM_OP_BASE);
        let shader = op.custom_shader.expect("custom op code must carry a shader");
        assert_eq!(shader.name, "mish");
        assert_eq!(shader.env, "CHEETAH_MISH_OPS_SPV");
    }
}
