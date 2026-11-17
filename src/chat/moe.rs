//! Mixture-of-experts router selection, shared by every backend.
//!
//! The native decoder and the Vulkan executor both route a token through the
//! layer router and must agree on the same expert order and the same mixing
//! weights for the same logits, so the selection lives in one place: softmax
//! over the router logits, top-k by score, then renormalization of the
//! selected weights to a sum of one.

/// Select `num_experts_per_tok` experts for one token from `logits`.
///
/// Returns `(expert_index, route_weight)` pairs ordered by descending score,
/// with the selected weights normalized to sum to one. Errors when the router
/// produces non-finite scores or the expert count is inconsistent.
pub(crate) fn select_experts(
    logits: &[f32],
    num_experts_per_tok: usize,
) -> Result<Vec<(usize, f32)>, String> {
    if logits.is_empty() || num_experts_per_tok == 0 || num_experts_per_tok > logits.len() {
        return Err("Invalid MoE router logits or expert count.".into());
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err("MoE router produced a non-finite score.".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probabilities = logits
        .iter()
        .map(|logit| (logit - max).exp())
        .collect::<Vec<_>>();
    let denominator: f32 = probabilities.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("MoE router probability normalization failed.".into());
    }
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]));
    let mut selected = order
        .into_iter()
        .take(num_experts_per_tok)
        .map(|index| (index, probabilities[index] / denominator))
        .collect::<Vec<_>>();
    let selected_sum: f32 = selected.iter().map(|(_, weight)| weight).sum();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err("MoE selected expert weights could not be normalized.".into());
    }
    for (_, weight) in &mut selected {
        *weight /= selected_sum;
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_top_experts_and_renormalizes_weights() {
        let selected = select_experts(&[0.0, 1.0, 2.0, -3.0], 2).unwrap();
        assert_eq!(
            selected.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [2, 1]
        );
        let total: f32 = selected.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(selected[0].1 > selected[1].1);
    }

    /// Selecting every expert keeps the plain softmax of the router logits.
    #[test]
    fn full_top_k_equals_softmax_of_the_router_logits() {
        let logits = [0.25, -1.5, 0.75];
        let selected = select_experts(&logits, logits.len()).unwrap();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|logit| (logit - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (index, weight) in &selected {
            assert!((weight - exps[*index] / sum).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_invalid_scores_and_expert_count() {
        assert!(select_experts(&[], 1).is_err());
        assert!(select_experts(&[1.0], 0).is_err());
        assert!(select_experts(&[1.0], 2).is_err());
        assert!(select_experts(&[f32::NAN], 1).is_err());
        assert!(select_experts(&[f32::INFINITY], 1).is_err());
    }
}
