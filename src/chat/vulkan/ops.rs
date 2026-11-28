//! Generic Vulkan dispatch for the `xxx_ops` catalog.
//!
//! Translates a registered [`VulkanOp`] (op code + kind + sub-op) and its
//! tensor arguments into the `shared_ops` 5-buffer / 8-push-constant layout,
//! then calls [`cheetah::native::vulkan::run_op_raw`]. Every catalog op has
//! symmetric native and Vulkan paths; the parity tests compare the buffers
//! element-wise.
//!
//! Buffer layout (mirrors `shared_ops.comp`):
//! ```text
//! b0 (readonly)  b1 (readonly)  b2 (write)  b3 (write)  b4 (readonly)
//! ```
//! A slot may be bound as both a source and a destination (recurrent state in
//! `conv1d_step` / `gated_delta_step`); `run_op_raw` preserves the host
//! contents across the round trip.

use std::os::raw::c_void;

use crate::chat::ops::{
    ACT_LEAKY_RELU, CUSTOM_OP_BASE, OP_BIAS_ADD, OP_GROUP_NORM, OP_LAYER_NORM, OP_RMS_NORM,
    OP_RMS_NORM_OFFSET, VulkanOp, custom_op_shader_default,
};

/// Tensor arguments for one catalog op dispatch, shaped to mirror the
/// `NativeOp` variants so native and Vulkan take identical inputs.
pub(crate) enum OpCall<'a> {
    /// `out[i] = f(x[i], param)`.
    Unary { x: &'a [f32], param: f32 },
    /// `out[i] = op(a[i], b[i])` (swiglu / geglu / bias-add).
    InPlaceBinop { a: &'a [f32], b: &'a [f32] },
    /// `out[i] = lhs[i] op rhs[i % period]`; `period == 0` ⇒ `rhs.len()`.
    Binary {
        lhs: &'a [f32],
        rhs: &'a [f32],
        period: usize,
    },
    /// Vector norm over `rows` rows of `width` channels (rms / rms-offset /
    /// layer / group). `bias` may be empty (a zero bias is bound). `groups` is
    /// used only by groupnorm.
    Norm {
        x: &'a [f32],
        weight: &'a [f32],
        bias: &'a [f32],
        eps: f32,
        rows: usize,
        width: usize,
        groups: usize,
    },
    /// Per-head L2 norm: `x` is `heads * head_dim`.
    NormL2 {
        x: &'a [f32],
        heads: usize,
        head_dim: usize,
    },
    /// Row-wise softmax: `x` is `rows * width`.
    Softmax {
        x: &'a [f32],
        rows: usize,
        width: usize,
    },
    /// Rotary embedding over one query span (`kv_heads = 0`, query-only).
    /// The host resolves the per-index frequency table (base theta + any
    /// rope scaling) and the shader reads it through b4.
    Rope {
        x: &'a [f32],
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: usize,
        theta: f32,
        scaling: &'a crate::chat::native::ops::RopeScaling,
    },
    /// Gather one row `row` of `width` from `table`.
    Embedding {
        table: &'a [f32],
        row: usize,
        width: usize,
    },
    /// One causal conv1d step; `state` is updated in place.
    Conv1dStep {
        input: &'a [f32],
        weight: &'a [f32],
        kernel: usize,
        state: &'a mut [f32],
    },
    /// One gated-delta-rule step; `state` is updated in place.
    GatedDeltaStep {
        query: &'a [f32],
        key: &'a [f32],
        value: &'a [f32],
        gate: &'a [f32],
        beta: &'a [f32],
        state: &'a mut [f32],
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
    },
    /// Reduction over one axis of a row-major `[rows, width]` buffer; the
    /// `VulkanOp`'s sub-op picks the reduction.
    Reduce {
        x: &'a [f32],
        rows: usize,
        width: usize,
        axis: u32,
    },
}

/// Output of one dispatch: the primary buffer (b2). Stateful ops mutate their
/// `state` slice in place in addition to returning this buffer.
pub(crate) struct OpOutput {
    pub primary: Vec<f32>,
}

const NULL_SRC: *const c_void = std::ptr::null();
const NULL_DST: *mut c_void = std::ptr::null_mut();

fn bytes(elements: usize) -> usize {
    elements * 4
}

/// Dispatch `op` on the Vulkan backend. Returns the primary (and optional
/// secondary) output buffers; stateful ops also mutate their `state` slice.
pub(crate) fn run_vulkan_op(op: &VulkanOp, call: OpCall<'_>) -> Result<OpOutput, String> {
    let mut params = [0u32; 8];
    let mut sources: [*const c_void; 5] = [NULL_SRC; 5];
    let mut destinations: [*mut c_void; 5] = [NULL_DST; 5];
    let mut sizes = [0usize; 5];
    let invocation_count: usize;
    let mut primary: Vec<f32>;

    // Temporaries whose backing memory must outlive the FFI call.
    let mut bias_zeros: Vec<f32> = Vec::new();
    let mut packed_b4: Vec<f32> = Vec::new();
    let rope_freqs: Vec<f32>;
    let mut token_ids = [0u32; 1];

    params[0] = op.op_code;
    match call {
        OpCall::Unary { x, param } => {
            let n = x.len();
            params[1] = n as u32;
            params[2] = op.sub_op;
            if op.sub_op == ACT_LEAKY_RELU {
                params[3] = param.to_bits();
            }
            sources[0] = x.as_ptr() as *const c_void;
            sizes[0] = bytes(n);
            primary = vec![0.0; n];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(n);
            invocation_count = n;
        }
        OpCall::InPlaceBinop { a, b } => {
            let n = a.len();
            params[1] = n as u32;
            sources[0] = a.as_ptr() as *const c_void;
            sources[1] = b.as_ptr() as *const c_void;
            sizes[0] = bytes(n);
            if op.op_code == OP_BIAS_ADD {
                // run_bias_add: p2 = period, p3 = format (0 = f32).
                params[2] = b.len() as u32;
                sizes[1] = bytes(b.len());
            } else {
                // swiglu (op 2) / geglu (op 6): run_elementwise, b = full-length up.
                sizes[1] = bytes(n);
            }
            primary = vec![0.0; n];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(n);
            invocation_count = n;
        }
        OpCall::Binary { lhs, rhs, period } => {
            let n = lhs.len();
            let p = if period == 0 { rhs.len() } else { period };
            params[1] = n as u32;
            params[2] = p as u32;
            params[3] = op.sub_op;
            sources[0] = lhs.as_ptr() as *const c_void;
            sources[1] = rhs.as_ptr() as *const c_void;
            sizes[0] = bytes(n);
            sizes[1] = bytes(rhs.len());
            primary = vec![0.0; n];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(n);
            invocation_count = n;
        }
        OpCall::Norm {
            x,
            weight,
            bias,
            eps,
            rows,
            width,
            groups,
        } => {
            params[1] = rows as u32;
            params[2] = width as u32;
            params[3] = eps.to_bits();
            params[4] = 0; // weight format 0 = f32
            sources[0] = x.as_ptr() as *const c_void;
            sources[1] = weight.as_ptr() as *const c_void;
            sizes[0] = bytes(x.len());
            sizes[1] = bytes(weight.len());
            primary = vec![0.0; x.len()];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(x.len());
            match op.op_code {
                OP_RMS_NORM => {
                    invocation_count = rows;
                }
                OP_RMS_NORM_OFFSET => {
                    invocation_count = rows * width;
                }
                OP_LAYER_NORM | OP_GROUP_NORM => {
                    if op.op_code == OP_GROUP_NORM {
                        params[5] = groups as u32;
                    }
                    bias_zeros.clear();
                    let b: &[f32] = if bias.is_empty() {
                        bias_zeros = vec![0.0; width];
                        &bias_zeros
                    } else {
                        bias
                    };
                    sources[4] = b.as_ptr() as *const c_void;
                    sizes[4] = bytes(b.len());
                    invocation_count = if op.op_code == OP_GROUP_NORM {
                        rows * width
                    } else {
                        rows
                    };
                }
                other => {
                    return Err(format!("op code {other} is not bound to a Norm dispatch"));
                }
            }
        }
        OpCall::NormL2 { x, heads, head_dim } => {
            params[1] = heads as u32;
            params[2] = head_dim as u32;
            sources[0] = x.as_ptr() as *const c_void;
            sizes[0] = bytes(x.len());
            primary = vec![0.0; x.len()];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(x.len());
            invocation_count = heads * head_dim;
        }
        OpCall::Softmax { x, rows, width } => {
            params[1] = rows as u32;
            params[2] = width as u32;
            sources[0] = x.as_ptr() as *const c_void;
            sizes[0] = bytes(x.len());
            primary = vec![0.0; x.len()];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(x.len());
            invocation_count = rows * width;
        }
        OpCall::Rope {
            x,
            heads,
            head_dim,
            rotary_dim,
            position,
            theta,
            scaling,
        } => {
            let query_count = heads * head_dim;
            params[1] = 1; // sequence = 1
            params[2] = position as u32; // offset
            params[3] = heads as u32; // query_heads
            params[4] = 0; // kv_heads = 0 (query-only span)
            params[5] = head_dim as u32;
            params[6] = theta.to_bits(); // informational; the table in b4 carries the resolved frequencies
            params[7] = rotary_dim as u32;
            rope_freqs = crate::chat::native::ops::scaled_freqs(theta, rotary_dim, scaling);
            sources[0] = x.as_ptr() as *const c_void;
            sources[4] = rope_freqs.as_ptr() as *const c_void;
            sizes[0] = bytes(x.len());
            sizes[4] = bytes(rope_freqs.len());
            primary = vec![0.0; query_count];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(query_count);
            invocation_count = query_count;
        }
        OpCall::Embedding { table, row, width } => {
            token_ids[0] = row as u32;
            params[1] = width as u32; // hidden
            params[2] = 1; // token_count
            params[3] = 0; // format 0 = f32
            params[4] = 0; // no sqrt(hidden) scale
            sources[0] = token_ids.as_ptr() as *const c_void;
            sizes[0] = 4; // one u32 token id
            sources[1] = table.as_ptr() as *const c_void;
            sizes[1] = bytes(table.len());
            primary = vec![0.0; width];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(width);
            invocation_count = width;
        }
        OpCall::Conv1dStep {
            input,
            weight,
            kernel,
            state,
        } => {
            let channels = input.len();
            params[1] = channels as u32;
            params[2] = kernel as u32;
            sources[0] = input.as_ptr() as *const c_void;
            sources[1] = weight.as_ptr() as *const c_void;
            sources[3] = state.as_ptr() as *const c_void;
            sizes[0] = bytes(channels);
            sizes[1] = bytes(weight.len());
            sizes[3] = bytes(state.len());
            primary = vec![0.0; channels];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            destinations[3] = state.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(channels);
            invocation_count = channels;
        }
        OpCall::GatedDeltaStep {
            query,
            key,
            value,
            gate,
            beta,
            state,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        } => {
            params[1] = num_k_heads as u32;
            params[2] = num_v_heads as u32;
            params[3] = head_k_dim as u32;
            params[4] = head_v_dim as u32;
            packed_b4.clear();
            packed_b4.extend_from_slice(value);
            packed_b4.extend_from_slice(gate);
            packed_b4.extend_from_slice(beta);
            sources[0] = query.as_ptr() as *const c_void;
            sources[1] = key.as_ptr() as *const c_void;
            sources[3] = state.as_ptr() as *const c_void;
            sources[4] = packed_b4.as_ptr() as *const c_void;
            sizes[0] = bytes(query.len());
            sizes[1] = bytes(key.len());
            sizes[3] = bytes(state.len());
            sizes[4] = bytes(packed_b4.len());
            let out_len = num_v_heads * head_v_dim;
            primary = vec![0.0; out_len];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            destinations[3] = state.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(out_len);
            invocation_count = num_v_heads;
        }
        OpCall::Reduce {
            x,
            rows,
            width,
            axis,
        } => {
            let out_len = if axis == 0 { rows } else { width };
            params[1] = rows as u32;
            params[2] = width as u32;
            params[3] = op.sub_op;
            params[4] = axis;
            sources[0] = x.as_ptr() as *const c_void;
            sizes[0] = bytes(x.len());
            primary = vec![0.0; out_len];
            destinations[2] = primary.as_mut_ptr() as *mut c_void;
            sizes[2] = bytes(out_len);
            invocation_count = out_len;
        }
    }

    if op.op_code >= CUSTOM_OP_BASE {
        let custom = op.custom_shader.ok_or_else(|| {
            format!(
                "op code {} is a custom-shader op but declares no CustomShader",
                op.op_code
            )
        })?;
        let default_spv = custom_op_shader_default(custom.name).ok_or_else(|| {
            format!(
                "custom op shader '{}' was not compiled; set {} to a SPIR-V path",
                custom.name, custom.env
            )
        })?;
        crate::native::vulkan::run_custom_op_raw(
            custom.env,
            default_spv,
            params,
            sources,
            destinations,
            sizes,
            invocation_count,
        )?;
    } else {
        crate::native::vulkan::run_op_raw(params, sources, destinations, sizes, invocation_count)?;
    }
    Ok(OpOutput { primary })
}
