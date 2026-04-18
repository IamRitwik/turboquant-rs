//! Attention simulation harness for TurboQuant Stage 2 quality evaluation.
//!
//! Simulates single-head attention score computation under three strategies:
//! 1. **True attention** — full FP16 vectors, exact dot products
//! 2. **Quantized attention** — PolarQuant compressed keys, decompressed before scoring
//! 3. **QJL-corrected attention** — PolarQuant + QJL residual correction (no decompression)
//!
//! Quality metrics:
//! - **KL divergence** — attention distribution shift from quantization
//! - **Top-k overlap** — fraction of highest-attention positions preserved
//!
//! These measure whether the model would attend to the *same* tokens after
//! quantization — the critical property for LLM output quality.
//!
//! Based on: TurboQuant (arXiv:2504.19874, ICLR 2026), Zandieh et al.

use std::collections::HashSet;

use crate::polarquant::PolarQuant;
use crate::qjl::QjlSketch;

// ---------------------------------------------------------------------------
//  Attention Score Computation
// ---------------------------------------------------------------------------

/// Compute true (unnormalized-then-softmax) single-head attention scores.
///
/// `score[i] = softmax(q · keys[i] / √d)[i]`
///
/// # Arguments
/// * `query` — query vector, length = d
/// * `keys`  — slice of key vectors, each length = d
pub fn true_attention_scores(query: &[f32], keys: &[Vec<f32>]) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt();
    let logits: Vec<f32> = keys
        .iter()
        .map(|k| {
            query
                .iter()
                .zip(k.iter())
                .map(|(q, ki)| q * ki)
                .sum::<f32>()
                / scale
        })
        .collect();
    softmax(&logits)
}

/// Compute attention scores using PolarQuant-compressed keys.
///
/// Keys are decompressed before computing the dot product.
/// This is the baseline: same computation cost as decompression-then-attention.
///
/// # Arguments
/// * `query`           — query vector, length = d
/// * `compressed_keys` — PolarQuant compressed byte payloads
/// * `pq`              — The PolarQuant instance used to compress the keys
pub fn quantized_attention_scores(
    query: &[f32],
    compressed_keys: &[Vec<u8>],
    pq: &PolarQuant,
) -> Vec<f32> {
    let keys: Vec<Vec<f32>> = compressed_keys.iter().map(|c| pq.decompress(c)).collect();
    true_attention_scores(query, &keys)
}

/// Compute attention scores with QJL residual correction (no full decompression needed).
///
/// ```text
/// score[i] = softmax((q · k̂ᵢ + correction(q, sketchᵢ)) / √d)[i]
/// ```
///
/// The QJL correction term corrects the inner product bias introduced by
/// PolarQuant's quantization error, producing statistically unbiased attention scores.
///
/// # Arguments
/// * `query`           — query vector, length = d
/// * `compressed_keys` — PolarQuant compressed byte payloads
/// * `sketches`        — QJL residual sketches for each key
/// * `pq`              — PolarQuant instance (needed for decompression of k̂)
/// * `qjl`             — QjlSketch instance for correction estimation
pub fn qjl_corrected_attention_scores(
    query: &[f32],
    compressed_keys: &[Vec<u8>],
    sketches: &[Vec<i8>],
    pq: &PolarQuant,
    qjl: &QjlSketch,
) -> Vec<f32> {
    let scale = (query.len() as f32).sqrt();
    let logits: Vec<f32> = compressed_keys
        .iter()
        .zip(sketches.iter())
        .map(|(c, s)| {
            let k_approx = pq.decompress(c);
            qjl.estimate_inner_product(query, &k_approx, s) / scale
        })
        .collect();
    softmax(&logits)
}

// ---------------------------------------------------------------------------
//  Quality Metrics
// ---------------------------------------------------------------------------

/// Compute KL divergence D_KL(p ∥ q) between two probability distributions.
///
/// Measures how much the attention distribution shifts due to quantization.
/// Lower is better. Values near 0 mean quantization is transparent.
///
/// Numerically stable: skips pairs where p[i] ≤ 1e-10 or q[i] ≤ 1e-10.
pub fn kl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len(), "kl_divergence: length mismatch");
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 1e-10 && qi > 1e-10)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Compute the top-k overlap: fraction of top-k attention positions preserved.
///
/// Returns a value in [0, 1]. 1.0 = perfect rank preservation.
/// This measures whether the model would attend to the same tokens.
///
/// # Arguments
/// * `true_scores`  — attention scores from full precision
/// * `approx_scores` — attention scores from quantized/corrected method
/// * `k`            — number of top attention positions to compare
pub fn topk_overlap(true_scores: &[f32], approx_scores: &[f32], k: usize) -> f32 {
    assert_eq!(true_scores.len(), approx_scores.len());
    let n = true_scores.len();
    let k = k.min(n);

    let mut true_ranked: Vec<usize> = (0..n).collect();
    true_ranked.sort_by(|&a, &b| {
        true_scores[b]
            .partial_cmp(&true_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut approx_ranked: Vec<usize> = (0..n).collect();
    approx_ranked.sort_by(|&a, &b| {
        approx_scores[b]
            .partial_cmp(&approx_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let true_top: HashSet<usize> = true_ranked[..k].iter().cloned().collect();
    let approx_top: HashSet<usize> = approx_ranked[..k].iter().cloned().collect();
    let overlap = true_top.intersection(&approx_top).count();
    overlap as f32 / k as f32
}

/// Compute mean absolute inner product error across a set of query-key pairs.
///
/// Used to evaluate how far the QJL-estimated inner products are from ground truth.
///
/// # Arguments
/// * `true_ips`  — true (exact) inner products
/// * `est_ips`   — estimated inner products (from QJL)
pub fn mean_abs_ip_error(true_ips: &[f32], est_ips: &[f32]) -> f32 {
    assert_eq!(true_ips.len(), est_ips.len());
    let n = true_ips.len() as f32;
    true_ips
        .iter()
        .zip(est_ips.iter())
        .map(|(t, e)| (t - e).abs())
        .sum::<f32>()
        / n
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------

/// Numerically stable softmax.
pub(crate) fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-12 {
        // Degenerate: return uniform
        return vec![1.0 / v.len() as f32; v.len()];
    }
    exps.iter().map(|&x| x / sum).collect()
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
    }

    #[test]
    fn test_softmax_monotone() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!(probs[0] < probs[1] && probs[1] < probs[2]);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.2_f32, 0.5, 0.3];
        assert!(kl_divergence(&p, &p).abs() < 1e-6, "KL(p||p) == 0");
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = vec![0.7_f32, 0.2, 0.1];
        let q = vec![0.1_f32, 0.2, 0.7];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0, "KL divergence should be > 0 for different dists: {kl}");
    }

    #[test]
    fn test_topk_overlap_perfect() {
        let scores = vec![0.5_f32, 0.3, 0.1, 0.08, 0.02];
        let overlap = topk_overlap(&scores, &scores, 3);
        assert!((overlap - 1.0).abs() < 1e-6, "Perfect overlap should be 1.0");
    }

    #[test]
    fn test_topk_overlap_zero() {
        let true_scores  = vec![1.0_f32, 2.0, 3.0, 4.0];
        let approx_scores = vec![4.0_f32, 3.0, 2.0, 1.0]; // reverse order
        let overlap = topk_overlap(&true_scores, &approx_scores, 1);
        // Top-1 true is index 3 (score=4.0), top-1 approx is index 0 (logit=4.0) → no overlap
        assert!(
            overlap.abs() < 1e-6,
            "Completely reversed should be 0 overlap: {overlap}"
        );
    }

    #[test]
    fn test_true_attention_scores_sums_to_one() {
        let query = vec![1.0_f32, 0.0, 0.0, 0.0];
        let keys: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 0.1, 0.0, 0.0, 0.0])
            .collect();
        let scores = true_attention_scores(&query, &keys);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Attention must sum to 1: {sum}");
    }

    #[test]
    fn test_mean_abs_ip_error_zero() {
        let v = vec![0.1_f32, 0.2, 0.3];
        assert!(mean_abs_ip_error(&v, &v).abs() < 1e-6);
    }
}
