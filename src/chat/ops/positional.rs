//! Position / embedding / bias operators, native CPU. These already have
//! Vulkan implementations in `shared_ops.comp` (op codes 3 / 8 / 9); here we
//! register symmetric native wrappers and point the Vulkan backend at the
//! existing codes.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};
use crate::chat::native::ops as native_ops;

#[cfg(feature = "vulkan")]
use super::{OpKind, VulkanOp, OP_BIAS_ADD, OP_EMBEDDING_GATHER, OP_ROTARY};

fn rope_wrap(x: &mut [f32], heads: u32, head_dim: u32, rotary_dim: u32, position: u32, theta: f32) {
    native_ops::rope_heads(
        x,
        heads as usize,
        head_dim as usize,
        rotary_dim as usize,
        position as usize,
        theta,
    );
}

fn embedding_gather(table: &[f32], row: usize, width: usize, out: &mut [f32]) {
    assert!(out.len() >= width);
    let start = row * width;
    assert!(start + width <= table.len(), "embedding row out of bounds");
    out[..width].copy_from_slice(&table[start..start + width]);
}

fn bias_add(x: &mut [f32], bias: &[f32]) {
    let period = bias.len();
    for (i, value) in x.iter_mut().enumerate() {
        *value += bias[i % period];
    }
}

const fn rope_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Rope(rope_wrap)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_ROTARY,
                kind: OpKind::Rope,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Rope(rope_wrap))]
    }
}

const fn embedding_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Embedding(embedding_gather)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_EMBEDDING_GATHER,
                kind: OpKind::Embedding,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Embedding(embedding_gather))]
    }
}

const fn bias_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::InPlaceBinop(bias_add)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_BIAS_ADD,
                kind: OpKind::InPlaceBinop,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::InPlaceBinop(bias_add))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "positional",
    entries: &[
        OpEntry { id: "pos.rope", backends: rope_backends() },
        OpEntry { id: "embedding.gather", backends: embedding_backends() },
        OpEntry { id: "bias.add", backends: bias_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_gather_copies_row() {
        let table = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]; // 2 rows × 3
        let mut out = [0.0f32; 3];
        match super::super::native_op_for("embedding.gather").unwrap() {
            NativeOp::Embedding(f) => f(&table, 1, 3, &mut out),
            other => panic!("{other:?}"),
        }
        assert_eq!(out, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn bias_add_broadcasts_over_period() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let bias = [10.0, 20.0]; // period 2
        match super::super::native_op_for("bias.add").unwrap() {
            NativeOp::InPlaceBinop(f) => f(&mut x, &bias),
            other => panic!("{other:?}"),
        }
        assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);
    }
}
