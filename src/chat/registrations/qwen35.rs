//! Registration for `qwen3_5` — the Qwen3.5 text backbone (Qwen3-Next
//! hybrid architecture).
//!
//! The checkpoint is a VL wrapper (`Qwen3_5ForConditionalGeneration`); under
//! the text-only policy this family registers the language backbone only —
//! the `model.language_model.*` units via the family's `unit_prefix`, with
//! the 312 `visual.*` units ignored.
//!
//! Architecture (per the HF `modeling_qwen3_5` reference, see the 1.4 design
//! notes): 24 layers alternating 3× linear-attention (gated delta net —
//! `causal_conv1d_step` + `gated_delta_step` + `rmsnorm_gated`, the stateful
//! catalog operators) and 1× full attention (qwen3-style per-head q/k norms
//! plus an attention output gate, partial rotary 0.25). No backend is
//! claimed yet: the hybrid native forward is the next 1.4 step, so loading
//! a Qwen3.5 directory reports guidance until then.

use super::templates::qwen;
use super::{ChatTemplate, ModelFamily, ModelRegistration};

/// Architecture names served by this registration: the VL wrapper's
/// architecture plus the bare text model_type.
pub(crate) const ARCHITECTURES: &[&str] = &["Qwen3_5ForConditionalGeneration", "qwen3_5"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "qwen3_5",
        render: qwen,
        header_renderer: None,
    },
    stop_markers: &["<|im_end|>"],
    backends: &[],
    family: Some(qwen35_family()),
};

/// Family constants for the Qwen3.5 text backbone. `unit_prefix` routes every
/// unit through the VL wrapper's language-model nesting.
const fn qwen35_family() -> ModelFamily {
    ModelFamily {
        unit_prefix: "model.language_model",
        rope_theta: 10_000_000.0,
        rms_norm_eps: 1e-6,
        tie_word_embeddings: true,
        profiles: &[(2048, 248320, 8, 2, 256)], // 2B: 8 q heads / 2 kv heads / head_dim 256
        extra_config_json: "",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}
