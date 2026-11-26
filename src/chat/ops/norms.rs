//! Normalization operators, native CPU. All operate on f32 buffers — small
//! norm weights/biases are dequantized to f32 by the caller, keeping native
//! and Vulkan symmetric (the Vulkan side passes weight-format code 0).
//!
//! `Norm` variant signature: `(x, weight, bias, eps, p1, p2)` where `p1`/`p2`
//! carry variant-specific params (groupnorm groups/channels) and are ignored
//! by rmsnorm / layernorm.

use super::{OpBackend, OpEntry, OpRegistration, NativeOp};

#[cfg(feature = "vulkan")]
use super::{
    OpKind, VulkanOp, OP_GROUP_NORM, OP_L2_NORM, OP_LAYER_NORM, OP_RMS_NORM, OP_RMS_NORM_OFFSET,
    OP_SOFTMAX,
};

fn rmsnorm_f32(x: &mut [f32], weight: &[f32], _bias: &[f32], eps: f32, _p1: u32, _p2: u32) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean_square + eps).sqrt().recip();
    for (value, index) in x.iter_mut().zip(0..) {
        *value *= inv * weight[index];
    }
}

fn rmsnorm_offset_f32(x: &mut [f32], weight: &[f32], _bias: &[f32], eps: f32, _p1: u32, _p2: u32) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mean_square + eps).sqrt().recip();
    for (value, index) in x.iter_mut().zip(0..) {
        *value *= inv * (1.0 + weight[index]);
    }
}

fn layernorm_f32(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32, _p1: u32, _p2: u32) {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inv = (variance + eps).sqrt().recip();
    for (value, index) in x.iter_mut().zip(0..) {
        let mut normalized = (*value - mean) * inv * weight[index];
        if !bias.is_empty() {
            normalized += bias[index];
        }
        *value = normalized;
    }
}

fn groupnorm_f32(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32, groups: u32, channels: u32) {
    let groups = groups as usize;
    let channels = channels as usize;
    assert!(channels > 0 && groups > 0 && channels % groups == 0);
    let group_size = channels / groups;
    for g in 0..groups {
        let start = g * group_size;
        let slice = &mut x[start..start + group_size];
        let mean = slice.iter().sum::<f32>() / group_size as f32;
        let variance = slice.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
        let inv = (variance + eps).sqrt().recip();
        for (offset, value) in slice.iter_mut().enumerate() {
            let channel = start + offset;
            let mut normalized = (*value - mean) * inv * weight[channel];
            if !bias.is_empty() {
                normalized += bias[channel];
            }
            *value = normalized;
        }
    }
}

fn l2norm_heads_wrap(x: &mut [f32], heads: u32, head_dim: u32) {
    crate::chat::native::ops::l2norm_heads(x, heads as usize, head_dim as usize);
}

fn softmax_rows(x: &mut [f32], width: u32) {
    let width = width as usize;
    assert!(width > 0);
    for row in x.chunks_mut(width) {
        crate::chat::native::ops::softmax(row);
    }
}

macro_rules! def_norm_backends {
    ($ctor:ident, $op:path, $code:expr) => {
        const fn $ctor() -> &'static [OpBackend] {
            #[cfg(feature = "vulkan")]
            {
                &[
                    OpBackend::Native(NativeOp::Norm($op)),
                    OpBackend::Vulkan(VulkanOp {
                        op_code: $code,
                        kind: OpKind::Norm,
                        sub_op: 0,
                        custom_shader: None,
                    }),
                ]
            }
            #[cfg(not(feature = "vulkan"))]
            {
                &[OpBackend::Native(NativeOp::Norm($op))]
            }
        }
    };
}

def_norm_backends!(rmsnorm_backends, rmsnorm_f32, OP_RMS_NORM);
def_norm_backends!(rmsnorm_offset_backends, rmsnorm_offset_f32, OP_RMS_NORM_OFFSET);
def_norm_backends!(layernorm_backends, layernorm_f32, OP_LAYER_NORM);
def_norm_backends!(groupnorm_backends, groupnorm_f32, OP_GROUP_NORM);

const fn l2_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::NormL2(l2norm_heads_wrap)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_L2_NORM,
                kind: OpKind::NormL2,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::NormL2(l2norm_heads_wrap))]
    }
}

const fn softmax_backends() -> &'static [OpBackend] {
    #[cfg(feature = "vulkan")]
    {
        &[
            OpBackend::Native(NativeOp::Softmax(softmax_rows)),
            OpBackend::Vulkan(VulkanOp {
                op_code: OP_SOFTMAX,
                kind: OpKind::Softmax,
                sub_op: 0,
                custom_shader: None,
            }),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[OpBackend::Native(NativeOp::Softmax(softmax_rows))]
    }
}

pub(crate) const REGISTRATION: OpRegistration = OpRegistration {
    source: "norms",
    entries: &[
        OpEntry { id: "norm.rmsnorm", backends: rmsnorm_backends() },
        OpEntry { id: "norm.rmsnorm_offset", backends: rmsnorm_offset_backends() },
        OpEntry { id: "norm.layernorm", backends: layernorm_backends() },
        OpEntry { id: "norm.groupnorm", backends: groupnorm_backends() },
        OpEntry { id: "norm.l2norm", backends: l2_backends() },
        OpEntry { id: "norm.softmax", backends: softmax_backends() },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn run_norm(id: &str, x: &[f32], w: &[f32], b: &[f32], eps: f32, p1: u32, p2: u32) -> Vec<f32> {
        let mut buf = x.to_vec();
        match super::super::native_op_for(id).unwrap() {
            NativeOp::Norm(f) => f(&mut buf, w, b, eps, p1, p2),
            other => panic!("{id} should be Norm, got {other:?}"),
        }
        buf
    }

    fn close(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn rmsnorm_scales_by_weight() {
        let x = [1.0, 2.0, 3.0];
        let w = [1.0, 1.0, 1.0];
        let out = run_norm("norm.rmsnorm", &x, &w, &[], 1e-6, 0, 0);
        let ms = x.iter().map(|v| v * v).sum::<f32>() / 3.0;
        let inv = (ms + 1e-6).sqrt().recip();
        assert!(close(&out, &[1.0 * inv, 2.0 * inv, 3.0 * inv]));
    }

    #[test]
    fn rmsnorm_offset_adds_one_to_weight() {
        let x = [1.0, -1.0];
        let w = [0.0, 1.0];
        let plain = run_norm("norm.rmsnorm", &x, &[1.0, 1.0], &[], 1e-5, 0, 0);
        let offset = run_norm("norm.rmsnorm_offset", &x, &w, &[], 1e-5, 0, 0);
        assert!((offset[0] - plain[0]).abs() < 1e-5);
        assert!((offset[1] - 2.0 * plain[1]).abs() < 1e-5);
    }

    #[test]
    fn groupnorm_normalizes_per_group() {
        let x = [1.0, 3.0, 10.0, 12.0];
        let w = [1.0; 4];
        let out = run_norm("norm.groupnorm", &x, &w, &[], 1e-5, 2, 4);
        let g0: f32 = out[..2].iter().sum();
        let g1: f32 = out[2..].iter().sum();
        assert!(g0.abs() < 1e-4);
        assert!(g1.abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one_per_row() {
        let mut x = vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0];
        match super::super::native_op_for("norm.softmax").unwrap() {
            NativeOp::Softmax(f) => f(&mut x, 3),
            other => panic!("{other:?}"),
        }
        assert!((x[..3].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((x[3..].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((x[3] - 1.0 / 3.0).abs() < 1e-5);
    }
}
