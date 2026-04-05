//! Lloyd-Max scalar quantizer for PolarQuant.
//!
//! The Lloyd-Max algorithm finds optimal codebook centroids that minimize MSE
//! for a given distribution. For PolarQuant:
//!   - Radii follow a Rayleigh-like distribution (derived from Chi-2 with 2 DOF)
//!   - Angles follow a Uniform(-π, π) distribution after rotation
//!
//! Because PolarQuant's random rotation makes coordinate distributions
//! data-oblivious (independent of input), these codebooks can be precomputed
//! offline and baked in — no per-vector calibration needed.

use std::f32::consts::PI;
use rand::Rng;

/// A scalar quantizer with a fixed codebook of centroids.
///
/// Given an input value, `quantize()` returns the index of the nearest centroid.
/// `dequantize()` maps an index back to the centroid value.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Sorted centroid values (the codebook).
    pub centroids: Vec<f32>,
    /// Decision boundaries between centroids (len = centroids.len() - 1).
    pub boundaries: Vec<f32>,
}

impl ScalarQuantizer {
    /// Build a scalar quantizer from pre-computed centroids.
    /// Boundaries are set at the midpoints between adjacent centroids.
    pub fn from_centroids(centroids: Vec<f32>) -> Self {
        assert!(centroids.len() >= 2, "Need at least 2 centroids");
        let boundaries = centroids
            .windows(2)
            .map(|w| (w[0] + w[1]) / 2.0)
            .collect();
        Self {
            centroids,
            boundaries,
        }
    }

    /// Encode a scalar value to its nearest centroid index.
    #[inline]
    pub fn quantize(&self, x: f32) -> u8 {
        // Binary search for the partition
        match self.boundaries.binary_search_by(|b| b.partial_cmp(&x).unwrap()) {
            Ok(i) => (i + 1).min(self.centroids.len() - 1) as u8,
            Err(i) => i.min(self.centroids.len() - 1) as u8,
        }
    }

    /// Decode an index back to its centroid value.
    #[inline]
    pub fn dequantize(&self, idx: u8) -> f32 {
        self.centroids[idx as usize]
    }

    /// Number of quantization levels.
    pub fn levels(&self) -> usize {
        self.centroids.len()
    }

    /// Bits required to represent an index.
    pub fn bits(&self) -> u32 {
        (self.centroids.len() as f32).log2().ceil() as u32
    }
}

/// Run the Lloyd-Max algorithm to compute optimal centroids for a distribution
/// sampled via the provided sampling function.
///
/// # Arguments
/// * `levels` – number of quantization levels (must be a power of 2)
/// * `num_samples` – Monte Carlo samples to approximate the distribution
/// * `sample_fn` – closure returning one sample from the target distribution
///
/// # Returns
/// A `ScalarQuantizer` with Lloyd-Max optimal centroids.
pub fn lloyd_max<R, F>(levels: usize, num_samples: usize, mut rng: R, mut sample_fn: F) -> ScalarQuantizer 
where 
    R: Rng,
    F: FnMut(&mut R) -> f32 
{
    // Draw samples to represent the distribution
    let mut samples: Vec<f32> = (0..num_samples).map(|_| sample_fn(&mut rng)).collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Initialize centroids uniformly across quantile range
    let step = num_samples / levels;
    let mut centroids: Vec<f32> = (0..levels)
        .map(|i| samples[(i * step + step / 2).min(num_samples - 1)])
        .collect();

    // Iterate Lloyd-Max until convergence
    // Note: This uses a linear scan O(n·k) per iteration. 
    // Fine for small k (bits <= 8), but would need optimization for large k.
    for _ in 0..200 {
        // E-step: assign each sample to nearest centroid
        let mut sums = vec![0.0f64; levels];
        let mut counts = vec![0usize; levels];

        for &s in &samples {
            let mut best_idx = 0;
            let mut best_dist = (s - centroids[0]).abs();
            for (i, &c) in centroids.iter().enumerate().skip(1) {
                let d = (s - c).abs();
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            sums[best_idx] += s as f64;
            counts[best_idx] += 1;
        }

        // M-step: update centroids to cluster means
        let mut converged = true;
        for i in 0..levels {
            if counts[i] > 0 {
                let new_c = (sums[i] / counts[i] as f64) as f32;
                if (new_c - centroids[i]).abs() > 1e-6 {
                    converged = false;
                }
                centroids[i] = new_c;
            }
        }
        if converged {
            break;
        }
    }

    ScalarQuantizer::from_centroids(centroids)
}

/// Build the optimal radius quantizer for PolarQuant.
///
/// After orthogonal rotation, pairs of coordinates (x, y) satisfy:
///   r = sqrt(x² + y²) ~ Chi distribution with 2 DOF (scaled)
///
/// For a d-dimensional unit-normalized vector:
///   Each component ~ N(0, 1/d), so r ~ Chi(2) * (1/sqrt(d)) * sqrt(d/2)
///   In practice r is in [0, 1] range when vectors are normalized.
///
/// We approximate with Monte Carlo sampling.
pub fn build_radius_quantizer<R: Rng>(levels: usize, dim: usize, mut rng: R) -> ScalarQuantizer {
    use rand_distr::{Normal, Distribution};
    let std = (1.0_f32 / dim as f32).sqrt();
    let normal = Normal::new(0.0_f32, std).unwrap();

    lloyd_max(levels, 100_000, &mut rng, |r| {
        let x: f32 = normal.sample(r);
        let y: f32 = normal.sample(r);
        (x * x + y * y).sqrt()
    })
}

/// Build the optimal angle quantizer for PolarQuant.
///
/// After rotation, angles are uniformly distributed on (-π, π),
/// so the optimal quantizer is simply uniform partitioning.
pub fn build_angle_quantizer(levels: usize) -> ScalarQuantizer {
    // For uniform distribution on [-π, π], Lloyd-Max = uniform partition
    let centroids: Vec<f32> = (0..levels)
        .map(|i| -PI + (2.0 * PI / levels as f32) * (i as f32 + 0.5))
        .collect();
    ScalarQuantizer::from_centroids(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer_roundtrip() {
        let q = ScalarQuantizer::from_centroids(vec![-1.5, -0.5, 0.5, 1.5]);
        for &val in &[-1.4, -0.6, 0.1, 1.2] {
            let idx = q.quantize(val);
            let rec = q.dequantize(idx);
            assert!((val - rec).abs() < 1.1, "val={val} rec={rec}");
        }
    }

    #[test]
    fn test_angle_quantizer_coverage() {
        let q = build_angle_quantizer(8);
        assert_eq!(q.levels(), 8);
        // All angles should round-trip within π/8
        for i in -100..100 {
            let angle = i as f32 * PI / 100.0;
            let idx = q.quantize(angle);
            let rec = q.dequantize(idx);
            assert!((angle - rec).abs() < PI / 8.0 + 0.01, "angle={angle} rec={rec}");
        }
    }
}
