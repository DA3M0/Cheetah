//! Registration for `mixtral` — declarative capability plus the family's
//! native forward assembly.
//!
//! build.rs collects every file in `src/chat/registrations/` into the
//! generated `REGISTRY`. The declarative half is the lookup source of truth;
//! the runtime half implements `ModelForward` by composing the native
//! OpsToolbox in the llama-shaped order with the sparse feed-forward: router
//! → softmax top-k expert selection (the shared `chat::moe` rule) →
//! route-weighted expert mix. Mixtral specifics: untied lm head, optional
//! attention biases, full-rope rotate-half attention over split
//! `q/k/v_proj`, and the routing-aware per-expert Ant residency (selected
//! experts page in, the rest evict to disk).

use super::native::decoder::DecoderConfig;
use super::native::ops::{self, OpsToolbox, Scratch};
#[cfg(test)]
use super::native::store::MmapWeightStore;
use super::templates::mixtral;
use super::{AntTierControl, BackendLoader, ChatTemplate, ModelFamily, ModelForward, ModelRegistration};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
#[cfg(test)]
use std::sync::Arc;

/// Architecture names served by this registration; the generated
/// `ALL_ARCHITECTURES` index collects them at build time.
pub(crate) const ARCHITECTURES: &[&str] = &["MixtralForCausalLM", "mixtral"];

pub(crate) const REGISTRATION: ModelRegistration = ModelRegistration {
    architectures: ARCHITECTURES,
    chat_template: ChatTemplate {
        id: "mixtral",
        render: mixtral,
        header_renderer: None,
    },
    stop_markers: &["</s>"],
    backends: model_backends(),
    family: Some(mixtral_family()),
};

/// Backends registered for this architecture: the native mmap backend (real
/// Ant tiering; the mixture-of-experts forward routes and mixes experts per
/// token) always, and Vulkan when compiled in.
const fn model_backends() -> &'static [(&'static str, BackendLoader)] {
    #[cfg(feature = "vulkan")]
    {
        &[
            ("native", BackendLoader::Native(mixtral_native_loader)),
            ("vulkan", BackendLoader::Vulkan(mixtral_vulkan_loader)),
        ]
    }
    #[cfg(not(feature = "vulkan"))]
    {
        &[("native", BackendLoader::Native(mixtral_native_loader))]
    }
}

/// Family constants for the Mixtral family: the sparse feed-forward resolves
/// expert counts from the prepared manifest; the MIXTRAL layout names the
/// router (`block_sparse_moe.gate`) and expert projections (`w1/w3/w2`).
const fn mixtral_family() -> ModelFamily {
    ModelFamily {
        rope_theta: 8_000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: false,
        profiles: &[
            (4096, 32768, 32, 8, 128), // Mixtral-8x7B (GQA: 32 q heads / 8 kv heads)
        ],
        extra_config_json: "",
        max_position_embeddings: None,
        moe_layout: super::native::decoder::MoEExpertLayout::MIXTRAL,
        features: super::native::decoder::LLAMA_BASELINE,
    }
}

// ---------------------------------------------------------------------------
// Native forward assembly.
// ---------------------------------------------------------------------------

/// The mixtral family's native runtime: the llama-shaped
/// sequential-residual block with the sparse (routed) feed-forward.
#[derive(Clone)]
pub(crate) struct MixtralNative {
    /// Resolved hyper-parameters (config → units manifest → family constants).
    pub(crate) cfg: DecoderConfig,
    /// Shared operators + inference state (store, caches, Ant plumbing).
    pub(crate) ctx: OpsToolbox,
    /// Reusable activation buffers (preallocated at capacity).
    pub(crate) scratch: Scratch,
}

impl MixtralNative {
    /// Fixture/test constructor around an already-resolved config and store.
    #[cfg(test)]
    pub(crate) fn with_store(cfg: DecoderConfig, store: Arc<MmapWeightStore>) -> Self {
        let ctx = OpsToolbox::with_store(cfg.clone(), store);
        Self {
            scratch: Scratch::new(&cfg),
            cfg,
            ctx,
        }
    }

    /// Loader-path constructor: the toolbox already carries the resolved
    /// config and the installed Ant residency plan.
    fn new(ctx: OpsToolbox) -> Self {
        let cfg = ctx.cfg.clone();
        Self {
            scratch: Scratch::new(&cfg),
            cfg,
            ctx,
        }
    }

    /// Run the full forward for tokens starting at `offset`; returns the
    /// logits of the final position only (matching the engine's sampling
    /// contract).
    fn run_forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.ctx.validate_tokens(tokens)?;
        for (index, &token) in tokens.iter().enumerate() {
            self.forward_position(token as usize, offset + index)?;
        }
        // Mixtral keeps a separate lm_head tensor; the tied branch reads the
        // embedding table instead.
        let head_unit = if self.cfg.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        self.ctx
            .lm_head_into(head_unit, &self.scratch.hidden, &mut self.scratch.logits)?;
        // Ant: layer 0's bucket also holds the embedding / final norm, whose
        // pages lm_head just swept; drop the disk-tier ones now. The pages
        // re-fault (from the OS page cache) on their next access.
        self.ctx.drop_disk_units_of_layer(0);
        Ok(std::mem::take(&mut self.scratch.logits))
    }

    /// One token position: embedding row lookup, then every layer, then the
    /// final norm.
    fn forward_position(&mut self, token: usize, position: usize) -> Result<(), String> {
        self.ctx
            .embedding_row_into(token, &mut self.scratch.hidden)?;
        for layer in 0..self.cfg.num_layers {
            self.forward_layer(layer, position)?;
            // Ant: release this layer's disk-tier pages right away so the
            // transient footprint stays near one layer, not the whole model.
            self.ctx.drop_disk_units_of_layer(layer as i32);
        }
        self.ctx.rmsnorm("model.norm", &mut self.scratch.hidden)?;
        Ok(())
    }

    /// One decoder layer: input norm → attention → residual add →
    /// post-attention norm → routed (or dense) feed-forward → residual add.
    fn forward_layer(&mut self, layer: usize, position: usize) -> Result<(), String> {
        let prefix = format!("model.layers.{layer}.");

        // Input norm: the attention reads it.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx
            .rmsnorm(&format!("{prefix}input_layernorm"), &mut self.scratch.normed)?;

        self.forward_attention(&prefix, layer, position)?;

        // Residual add: hidden += attn_projected.
        for (value, residual) in self
            .scratch
            .attn_projected
            .iter()
            .zip(self.scratch.hidden.iter_mut())
        {
            *residual += *value;
        }

        // Post-attention norm, then the feed-forward block, then its residual.
        self.scratch.normed.clear();
        self.scratch.normed.extend_from_slice(&self.scratch.hidden);
        self.ctx.rmsnorm(
            &format!("{prefix}post_attention_layernorm"),
            &mut self.scratch.normed,
        )?;
        self.forward_mlp(&prefix, layer)?;
        let mlp_out = std::mem::take(&mut self.scratch.mlp_out);
        for (value, residual) in mlp_out.iter().zip(self.scratch.hidden.iter_mut()) {
            *residual += value;
        }
        self.scratch.mlp_out = mlp_out;
        Ok(())
    }

    /// Attention block over `scratch.normed`, written to
    /// `scratch.attn_projected` (the caller owns the residual add). Split
    /// `q/k/v_proj` projections, full-span rotate-half rope, GQA attention,
    /// then `o_proj`. Biases are family-optional — the adds are
    /// unit-optional no-ops.
    fn forward_attention(
        &mut self,
        prefix: &str,
        layer: usize,
        position: usize,
    ) -> Result<(), String> {
        let cfg = &self.cfg;
        let q_dim = cfg.q_heads * cfg.head_dim;
        let kv_dim = cfg.kv_heads * cfg.head_dim;

        self.scratch.q.clear();
        self.scratch.q.resize(q_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.q_proj.weight"),
            q_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.q,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.q, &format!("{prefix}self_attn.q_proj"));

        self.scratch.k.clear();
        self.scratch.k.resize(kv_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.k_proj.weight"),
            kv_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.k,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.k, &format!("{prefix}self_attn.k_proj"));

        self.scratch.v.clear();
        self.scratch.v.resize(kv_dim, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.v_proj.weight"),
            kv_dim,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.v,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.v, &format!("{prefix}self_attn.v_proj"));

        // Rotate-half rope over the full head span of every head.
        ops::rope_heads(
            &mut self.scratch.q,
            cfg.q_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        );
        ops::rope_heads(
            &mut self.scratch.k,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.rotary_dim,
            position,
            cfg.rope_theta,
        );

        // Append K/V to the cache for this position, then GQA attention over
        // the visible cached keys (the resolved sliding window applies).
        self.ctx.cache_append(layer, &self.scratch.k, &self.scratch.v);
        self.scratch.attn_out.clear();
        self.scratch.attn_out.resize(q_dim, 0.0);
        self.ctx
            .attention(layer, position, &self.scratch.q, &mut self.scratch.attn_out);

        // Output projection (plus optional bias).
        self.scratch.attn_projected.clear();
        self.scratch
            .attn_projected
            .resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}self_attn.o_proj.weight"),
            cfg.hidden_size,
            q_dim,
            &self.scratch.attn_out,
            &mut self.scratch.attn_projected,
        )?;
        self.ctx
            .add_bias(&mut self.scratch.attn_projected, &format!("{prefix}self_attn.o_proj"));
        Ok(())
    }

    /// Feed-forward dispatch over `scratch.normed`, written to
    /// `scratch.mlp_out` — the caller owns the residual add. Dense vs MoE
    /// branch: detected by expert counts from config. This runtime check is
    /// negligible overhead and enables clean code paths for both shapes
    /// without compile-time specialization.
    fn forward_mlp(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        if self.cfg.num_local_experts > 0 && self.cfg.num_experts_per_tok > 0 {
            self.forward_moe_block(prefix, layer)
        } else {
            self.forward_dense_mlp(prefix)
        }
    }

    /// Forward pass for MoE blocks: router selection + top-k expert
    /// computation, each selected expert's output accumulated into
    /// `scratch.mlp_out`, weighted by its route weight.
    pub(crate) fn forward_moe_block(&mut self, prefix: &str, layer: usize) -> Result<(), String> {
        let cfg = &self.cfg;

        // Router: compute expert scores from normalized hidden states
        self.scratch.router.clear();
        self.scratch.router.resize(cfg.num_local_experts, 0.0);
        self.ctx.linear(
            &cfg.moe_layout.router_unit(prefix),
            cfg.num_local_experts,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.router,
        )?;

        // Router selection: softmax over the router logits, top-k by score,
        // selected weights renormalized to one — the shared routing rule
        // every backend uses (`chat::moe::select_experts`).
        let num_experts_per_tok = cfg.num_experts_per_tok.min(cfg.num_local_experts);
        let selected = crate::chat::moe::select_experts(&self.scratch.router, num_experts_per_tok)?;
        let picked: Vec<usize> = selected.iter().map(|(expert, _)| *expert).collect();

        // RC2-B: routing-aware per-expert residency management. Evict unused
        // experts, page in newly selected ones. State keyed by (layer,
        // expert) persists across positions.
        self.ctx.update_expert_residency(layer, &picked);

        // Compute selected experts: gate*up activation, then down projection,
        // scaled by the router's mixing weight for this token. The weighted
        // accumulator starts at zero every call (it swaps into `mlp_out` at
        // the end).
        self.scratch.mlp_out_weighted.clear();
        self.scratch
            .mlp_out_weighted
            .resize(cfg.hidden_size, 0.0);
        for (expert, route_weight) in selected {
            let gate_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.gate);
            let up_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.up);
            let down_unit = cfg.moe_layout.expert_unit(prefix, expert, cfg.moe_layout.down);

            self.scratch.gate.clear();
            self.scratch.gate.resize(cfg.intermediate_size, 0.0);
            self.ctx.linear(
                &gate_unit,
                cfg.intermediate_size,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.gate,
            )?;

            self.scratch.up.clear();
            self.scratch.up.resize(cfg.intermediate_size, 0.0);
            self.ctx.linear(
                &up_unit,
                cfg.intermediate_size,
                cfg.hidden_size,
                &self.scratch.normed,
                &mut self.scratch.up,
            )?;

            ops::silu_multiply(&mut self.scratch.gate, &self.scratch.up);

            self.scratch.mlp_out.clear();
            self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
            self.ctx.linear(
                &down_unit,
                cfg.hidden_size,
                cfg.intermediate_size,
                &self.scratch.gate,
                &mut self.scratch.mlp_out,
            )?;

            // Accumulate this expert's output, weighted by the router's
            // probability for the token; the caller owns the residual add.
            for (value, out) in self
                .scratch
                .mlp_out
                .iter()
                .zip(self.scratch.mlp_out_weighted.iter_mut())
            {
                *out += value * route_weight;
            }
        }
        std::mem::swap(
            &mut self.scratch.mlp_out,
            &mut self.scratch.mlp_out_weighted,
        );
        Ok(())
    }

    /// Dense feed-forward over `scratch.normed`, written to `scratch.mlp_out`
    /// (the dense fallback for a mixtral-shaped manifest without routed
    /// experts). SwiGLU: `down(silu(gate·x) ⊙ up·x)`.
    fn forward_dense_mlp(&mut self, prefix: &str) -> Result<(), String> {
        let cfg = &self.cfg;

        self.scratch.gate.clear();
        self.scratch.gate.resize(cfg.intermediate_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.gate_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.gate,
        )?;

        self.scratch.up.clear();
        self.scratch.up.resize(cfg.intermediate_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.up_proj.weight"),
            cfg.intermediate_size,
            cfg.hidden_size,
            &self.scratch.normed,
            &mut self.scratch.up,
        )?;

        ops::silu_multiply(&mut self.scratch.gate, &self.scratch.up);

        self.scratch.mlp_out.clear();
        self.scratch.mlp_out.resize(cfg.hidden_size, 0.0);
        self.ctx.linear(
            &format!("{prefix}mlp.down_proj.weight"),
            cfg.hidden_size,
            cfg.intermediate_size,
            &self.scratch.gate,
            &mut self.scratch.mlp_out,
        )?;
        Ok(())
    }
}

impl ModelForward for MixtralNative {
/// Runs the Mixtral forward pass over `tokens` starting at absolute
/// position `offset` and returns the final-position logits — the
/// backend-agnostic [`ModelForward`] entry delegating to `run_forward`.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, String> {
        self.run_forward(tokens, offset)
    }

/// Clears the toolbox KV caches between generations.
    fn clear_cache(&mut self) {
        self.ctx.clear_kv();
    }

/// Reports the constant backend label `"native-cpu"`.
    fn device_name(&self) -> &'static str {
        "native-cpu"
    }

/// Exposes the family runtime for Ant page-level tier transitions — the
/// native backend owns mmap-backed weights, so its tiers act on real pages.
    fn as_ant_tier_control(&mut self) -> Option<&mut dyn AntTierControl> {
        Some(self)
    }
}

impl AntTierControl for MixtralNative {
/// Ant page-in: faults one unit's pages back into the resident set; the
/// bool reports whether the units manifest knows the unit.
    fn page_in_unit(&mut self, unit_id: &str, _tier: i32) -> bool {
        self.ctx.page_in_unit(unit_id)
    }

/// Ant evict: drops one unit's pages to the disk tier; the bool reports
/// whether the units manifest knows the unit.
    fn evict_unit(&mut self, unit_id: &str) -> bool {
        self.ctx.evict_unit(unit_id)
    }
}

/// Native backend loader for this registration: opens the mmap store,
/// resolves the decoder config through the parameter fallback chain, and
/// installs the Ant residency plan before the first forward.
fn mixtral_native_loader(
    model_dir: &Path,
    _config_bytes: &[u8],
    raw_config: &Value,
    plan: &BTreeMap<String, String>,
    family: ModelFamily,
) -> Result<Box<dyn ModelForward>, String> {
    let ctx = OpsToolbox::open(model_dir, raw_config, plan, family)?;
    Ok(Box::new(MixtralNative::new(ctx)))
}

// ---------------------------------------------------------------------------
// Vulkan executor entry (moved from the former central `vulkan::load`
// dispatch: each registration now owns its executor feature selection).
// ---------------------------------------------------------------------------

/// Vulkan transformer features for the Mixtral family: the sparse
/// mixture-of-experts feed-forward.
#[cfg(feature = "vulkan")]
fn mixtral_vulkan_config(
    raw_config: &Value,
) -> Result<crate::chat::vulkan::transformer::VulkanTransformerConfig, String> {
    use crate::chat::vulkan::required_usize;
    use crate::chat::vulkan::transformer::{
        TransformerFeatures, TransformerFeedForwardFeatures,
    };

    let num_local_experts = required_usize(raw_config, "num_local_experts")?;
    let num_experts_per_tok = required_usize(raw_config, "num_experts_per_tok")?;
    if num_experts_per_tok > num_local_experts {
        return Err("Mixtral expert counts are inconsistent.".into());
    }
    let mut features = TransformerFeatures::standard("Mixtral");
    features.tie_word_embeddings = raw_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    features.feed_forward = TransformerFeedForwardFeatures::MixtureOfExperts {
        num_local_experts,
        num_experts_per_tok,
    };
    crate::chat::vulkan::transformer::VulkanTransformerConfig::from_raw(raw_config, features)
}

/// Vulkan loader for this registration: build the generic transformer
/// executor around the mixtral feature set.
#[cfg(feature = "vulkan")]
fn mixtral_vulkan_loader(
    weight_paths: &[std::path::PathBuf],
    _config_bytes: &[u8],
    raw_config: Value,
    plan: &BTreeMap<String, String>,
) -> Result<Box<dyn ModelForward>, String> {
    let config = mixtral_vulkan_config(&raw_config)?;
    Ok(Box::new(
        crate::chat::vulkan::transformer::VulkanTransformerExecutor::load(
            weight_paths, &config, plan,
        )?,
    ))
}

#[cfg(all(test, feature = "vulkan"))]
mod vulkan_feature_tests {
    use super::*;

    #[test]
    fn mixtral_vulkan_features_configure_moe_tensor_layout() {
        let config = mixtral_vulkan_config(&serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 8,
            "num_experts_per_tok": 2
        }))
        .unwrap();
        assert!(matches!(
            config.features.feed_forward,
            crate::chat::vulkan::transformer::TransformerFeedForwardFeatures::MixtureOfExperts {
                num_local_experts: 8,
                num_experts_per_tok: 2,
            }
        ));
    }

    #[test]
    fn mixtral_vulkan_features_reject_invalid_expert_counts() {
        let raw = serde_json::json!({
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "hidden_act": "silu",
            "num_local_experts": 2,
            "num_experts_per_tok": 3
        });
        assert!(mixtral_vulkan_config(&raw).is_err());
    }
}
