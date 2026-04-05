//! Evaluation metrics for quantization quality.

use half::f16;

/// Compute Mean Squared Error between two f32 slices.
///
/// MSE = (1/n) Σ (original[i] - reconstructed[i])²
#[inline]
pub fn mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f32;
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / n
}

/// Compute Mean Squared Error between two slices (one f16, one f32).
///
/// Useful for comparing f16 baseline with reconstructed f32 outputs without
/// full vector conversion.
#[inline]
pub fn mse_f16(original: &[f16], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f32;
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| {
            let diff = a.to_f32() - b;
            diff * diff
        })
        .sum::<f32>()
        / n
}

/// Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
///
/// PSNR = 10 * log10(signal_power / MSE)
pub fn psnr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let signal_power: f32 = original.iter().map(|x| x * x).sum::<f32>() / original.len() as f32;
    let error = mse(original, reconstructed);
    if error < 1e-12 {
        return f32::INFINITY;
    }
    10.0 * (signal_power / error).log10()
}

/// Compute the cosine similarity between two vectors.
///
/// Measures how well inner-product structure is preserved — critical for
/// attention mechanisms in LLMs.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute the cosine similarity between an f16 vector and an f32 vector.
pub fn cosine_similarity_f16(a: &[f16], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let x_f32 = x.to_f32();
        dot += x_f32 * y;
        norm_a += x_f32 * x_f32;
        norm_b += y * y;
    }

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Compute compression ratio as original_bits / compressed_bits.
///
/// For FP16 input (16 bits/element):
///   - INT8: ratio = 2.0x (nominal)
///   - PolarQuant-4bit: ratio = 4.0x
///   - PolarQuant-3bit: ratio = 5.33x
pub fn compression_ratio(original_bits_per_elem: f32, compressed_bits_per_elem: f32) -> f32 {
    original_bits_per_elem / compressed_bits_per_elem
}

/// Throughput in millions of vectors per second.
pub fn throughput_mvps(num_vectors: usize, elapsed_secs: f64) -> f64 {
    num_vectors as f64 / elapsed_secs / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(mse(&v, &v), 0.0);
    }

    #[test]
    fn test_mse_f16() {
        let a = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let b = vec![1.0, 2.5];
        // error = (1.0-1.0)^2 + (2.0-2.5)^2 = 0.25
        // mse = 0.25 / 2 = 0.125
        assert!((mse_f16(&a, &b) - 0.125).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_f16() {
        let a = vec![f16::from_f32(1.0), f16::from_f32(0.0)];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity_f16(&a, &b) - 1.0).abs() < 1e-6);
    }
}
