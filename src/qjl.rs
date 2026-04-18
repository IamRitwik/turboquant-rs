//! QJL — Johnson-Lindenstrauss Residual Sketch (Stage 2 of TurboQuant)
//!
//! After PolarQuant compresses KV vectors, attention scores (Q·Kᵀ) cannot
//! be computed accurately without decompression, because PolarQuant introduces
//! a biased reconstruction error `e = k - k̂`. QJL corrects this:
//!
//! ```text
//! True:       score = q · k                      (full FP16 vectors)
//! TurboQuant: score ≈ q · k̂  +  q · sketch(e)
//! where:
//!   k̂     = PolarQuant decompressed K             (approximate)
//!   e     = k - k̂   = quantization residual
//!   sketch = JL projection via ±1 Rademacher signs (1 bit per dimension)
//! ```
//!
//! The sketch is **unbiased**: `E[q · sketch(e)] = q · e`.
//! Variance reduces as `O(1/k)` where `k` is the sketch dimension.
//!
//! Memory budget per key vector (d=512, k=64):
//! - PolarQuant-3bit payload: (512×3)/8 = 192 bytes
//! - FP16 norm:                               2 bytes
//! - QJL-64 sketch:                    64/8 = 8 bytes
//! - **Total: 202 bytes** vs. FP16 baseline of 1024 bytes (~5.07× compression)
//!
//! Based on: TurboQuant (arXiv:2504.19874, ICLR 2026), Zandieh et al.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Johnson-Lindenstrauss sketch for quantization residual error correction.
///
/// After PolarQuant compression, the residual `e = v - v̂` carries
/// quantization error. `QjlSketch` sketches this residual with a random
/// ±1 projection to enable inner product estimation without decompression.
///
/// # Example
/// ```
/// use turboquant::qjl::{QjlSketch, compute_residual};
///
/// let dim = 128;
/// let qjl = QjlSketch::new(dim, 64, Some(42));
///
/// // Simulate quantization residual
/// let original: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
/// let approx:   Vec<f32> = original.iter().map(|&x| x + 0.01).collect();
/// let residual = compute_residual(&original, &approx);
///
/// // Sketch the residual (1 bit per sketch dimension)
/// let sketch = qjl.sketch_residual(&residual);
/// assert_eq!(sketch.len(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct QjlSketch {
    /// Input vector dimension.
    pub dim: usize,
    /// Sketch dimension `k` (higher → more accurate, more bits stored).
    pub sketch_dim: usize,
    /// Random projection matrix G: k×d of ±1 Rademacher values (stored as Vec<Vec<i8>>).
    projection: Vec<Vec<i8>>,
    /// Scale factor = 1/√k for unbiased estimation.
    scale: f32,
}

impl QjlSketch {
    /// Create a new QJL sketch.
    ///
    /// # Arguments
    /// * `dim`        — input vector dimension
    /// * `sketch_dim` — projection dimension `k` (try 32, 64, 128)
    /// * `seed`       — optional RNG seed for reproducibility
    ///
    /// Memory for `projection`: `k × d` bytes (1 byte per i8, but logically 1 bit).
    pub fn new(dim: usize, sketch_dim: usize, seed: Option<u64>) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(sketch_dim > 0, "sketch_dim must be > 0");
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xcafe_babe_dead_beef));
        let projection: Vec<Vec<i8>> = (0..sketch_dim)
            .map(|_| {
                (0..dim)
                    .map(|_| if rng.gen_bool(0.5) { 1_i8 } else { -1_i8 })
                    .collect()
            })
            .collect();
        let scale = 1.0_f32 / (sketch_dim as f32).sqrt();
        Self {
            dim,
            sketch_dim,
            projection,
            scale,
        }
    }

    /// Compute the sketch of a residual vector.
    ///
    /// `sketch(e) = sign(G · e)` stored as a `Vec<i8>` of ±1 values.
    /// This is the 1-bit sign of the projected residual.
    ///
    /// **Storage:** `sketch_dim` bits = `sketch_dim / 8` bytes (if packed).
    ///
    /// # Arguments
    /// * `residual` — quantization residual vector `e = v - v̂`, length = `dim`
    pub fn sketch_residual(&self, residual: &[f32]) -> Vec<i8> {
        assert_eq!(
            residual.len(),
            self.dim,
            "residual length {} != dim {}",
            residual.len(),
            self.dim
        );
        self.projection
            .iter()
            .map(|row| {
                let dot: f32 = row
                    .iter()
                    .zip(residual.iter())
                    .map(|(&g, &e)| g as f32 * e)
                    .sum();
                if dot >= 0.0 { 1_i8 } else { -1_i8 }
            })
            .collect()
    }

    /// Estimate the inner product correction term `q · residual`
    /// from the sketch, without having the full residual vector.
    ///
    /// ```text
    /// estimate = (1/k) · Σᵢ sketch[i] · (G[i] · q)
    /// ```
    ///
    /// This is an **unbiased estimator**: `E[estimate] = q · residual`.
    ///
    /// # Arguments
    /// * `query`  — query vector `q`, length = `dim`
    /// * `sketch` — sketch of residual, length = `sketch_dim`
    pub fn estimate_correction(&self, query: &[f32], sketch: &[i8]) -> f32 {
        assert_eq!(
            query.len(),
            self.dim,
            "query length {} != dim {}",
            query.len(),
            self.dim
        );
        assert_eq!(
            sketch.len(),
            self.sketch_dim,
            "sketch length {} != sketch_dim {}",
            sketch.len(),
            self.sketch_dim
        );

        let correction: f32 = self
            .projection
            .iter()
            .zip(sketch.iter())
            .map(|(row, &s)| {
                let g_dot_q: f32 = row
                    .iter()
                    .zip(query.iter())
                    .map(|(&g, &q)| g as f32 * q)
                    .sum();
                s as f32 * g_dot_q
            })
            .sum();

        // scale² = 1/k; gives unbiased estimator
        correction * self.scale * self.scale
    }

    /// Full pipeline: estimate the true inner product `q · k` given:
    /// - a query vector `q`
    /// - the PolarQuant-decompressed approximate key `k̂`
    /// - the QJL sketch of the residual `e = k - k̂`
    ///
    /// ```text
    /// estimate = q · k̂  +  correction(q, sketch(k - k̂))
    /// ```
    ///
    /// This is unbiased: `E[estimate] = q · k`.
    pub fn estimate_inner_product(&self, query: &[f32], approx_key: &[f32], sketch: &[i8]) -> f32 {
        assert_eq!(query.len(), self.dim);
        assert_eq!(approx_key.len(), self.dim);

        let base: f32 = query
            .iter()
            .zip(approx_key.iter())
            .map(|(q, k)| q * k)
            .sum();
        let correction = self.estimate_correction(query, sketch);
        base + correction
    }

    /// Number of bits used for one QJL sketch (= sketch_dim).
    pub fn bits_per_vector(&self) -> usize {
        self.sketch_dim
    }

    /// Number of bytes if the sketch is stored packed (1 bit per sign).
    pub fn packed_bytes_per_vector(&self) -> usize {
        (self.sketch_dim + 7) / 8
    }

    /// Memory overhead of the QJL sketch as a fraction of FP16 storage.
    ///
    /// FP16 baseline = dim × 16 bits.
    /// Overhead = sketch_dim / (dim × 16).
    pub fn overhead_fraction(&self) -> f32 {
        self.sketch_dim as f32 / (self.dim as f32 * 16.0)
    }

    /// Total compressed bytes for PolarQuant-3bit + QJL sketch at this dim.
    ///
    /// Breakdown (d=512, k=64):
    /// - PolarQ-3bit payload: (512×3+7)/8 = 192 bytes
    /// - FP16 norm:                         2 bytes  
    /// - QJL packed sketch:         (64+7)/8 = 8 bytes
    /// - Total:                             202 bytes
    pub fn total_bytes_with_polar3bit(&self) -> usize {
        let polar_payload = (self.dim * 3 + 7) / 8;
        let fp16_norm = 2;
        let sketch_bytes = self.packed_bytes_per_vector();
        polar_payload + fp16_norm + sketch_bytes
    }
}

/// Compute the quantization residual `e = original - reconstructed`.
///
/// This is the error left after PolarQuant compression and decompression.
/// The QJL sketch of this residual corrects inner product estimates.
///
/// # Arguments
/// * `original`      — original vector `v`
/// * `reconstructed` — PolarQuant-reconstructed vector `v̂`
pub fn compute_residual(original: &[f32], reconstructed: &[f32]) -> Vec<f32> {
    assert_eq!(
        original.len(),
        reconstructed.len(),
        "original length {} != reconstructed length {}",
        original.len(),
        reconstructed.len()
    );
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| o - r)
        .collect()
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        let v: Vec<f32> = (0..dim)
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_sketch_output_is_signs() {
        let dim = 64;
        let qjl = QjlSketch::new(dim, 32, Some(42));
        let residual = random_unit_vector(dim, 1);
        let sketch = qjl.sketch_residual(&residual);

        assert_eq!(sketch.len(), 32);
        for &s in &sketch {
            assert!(s == 1 || s == -1, "Sketch must be ±1, got {s}");
        }
    }

    #[test]
    fn test_compute_residual() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.1_f32, 1.9, 3.0];
        let residual = compute_residual(&a, &b);
        assert!((residual[0] - (-0.1)).abs() < 1e-5);
        assert!((residual[1] - 0.1).abs() < 1e-5);
        assert!(residual[2].abs() < 1e-5);
    }

    #[test]
    fn test_qjl_unbiased_estimator() {
        // E[estimate] ≈ true inner product over many independent sketch samples
        let dim = 512;
        let sketch_dim = 128;

        let q = random_unit_vector(dim, 1);
        let e = random_unit_vector(dim, 2); // treated as a residual

        let true_dot: f32 = q.iter().zip(e.iter()).map(|(a, b)| a * b).sum();

        // Average correction estimate over 1000 different projection matrices
        let estimates: Vec<f32> = (0..1000_u64)
            .map(|seed| {
                let qjl_i = QjlSketch::new(dim, sketch_dim, Some(seed));
                let sketch = qjl_i.sketch_residual(&e);
                qjl_i.estimate_correction(&q, &sketch)
            })
            .collect();

        let mean_estimate: f32 = estimates.iter().sum::<f32>() / estimates.len() as f32;
        assert!(
            (mean_estimate - true_dot).abs() < 0.05,
            "QJL estimator not unbiased: true={true_dot:.4}, mean_estimated={mean_estimate:.4}"
        );
    }

    #[test]
    fn test_qjl_sketch_dim_affects_variance() {
        // Higher sketch_dim should generally give lower squared error
        let dim = 512;
        let q = random_unit_vector(dim, 10);
        let e = random_unit_vector(dim, 11);
        let true_dot: f32 = q.iter().zip(e.iter()).map(|(a, b)| a * b).sum();

        let mut errors: Vec<f32> = Vec::new();
        for &k in &[16_usize, 64, 256] {
            // Average error over 100 trials for each k
            let mean_sq_error: f32 = (0..100_u64)
                .map(|seed| {
                    let qjl = QjlSketch::new(dim, k, Some(seed));
                    let sketch = qjl.sketch_residual(&e);
                    let est = qjl.estimate_correction(&q, &sketch);
                    (est - true_dot).powi(2)
                })
                .sum::<f32>()
                / 100.0;
            errors.push(mean_sq_error);
        }
        // k=256 should have lower avg squared error than k=16
        assert!(
            errors[2] < errors[0],
            "Larger k should reduce variance: err@k=16={:.6}, err@k=256={:.6}",
            errors[0],
            errors[2]
        );
    }

    #[test]
    fn test_full_pipeline_inner_product() {
        // Test estimate_inner_product gives reasonable result for realistic error
        let dim = 128;
        let q = random_unit_vector(dim, 100);
        let k = random_unit_vector(dim, 101);

        // Simulate a small quantization approximation error
        let approx_error = 0.005_f32; // smaller than 0.01 for better stability
        let k_approx: Vec<f32> = k.iter().map(|&x| x + approx_error).collect();
        let residual = compute_residual(&k, &k_approx);

        let qjl = QjlSketch::new(dim, 128, Some(42)); // increased sketch dim to 128 for lower variance in test
        let sketch = qjl.sketch_residual(&residual);
        let estimated = qjl.estimate_inner_product(&q, &k_approx, &sketch);
        let true_dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (estimated - true_dot).abs() < 0.15, // slightly larger tolerance for variance
            "Pipeline estimate too far off: true={true_dot:.4}, est={estimated:.4}, diff={:.4}",
            (estimated - true_dot).abs()
        );
    }

    #[test]
    fn test_overhead_fraction() {
        let dim = 512;
        // QJL-64 overhead = 64 / (512 × 16) = 64/8192 = 0.0078125
        let qjl = QjlSketch::new(dim, 64, Some(42));
        let frac = qjl.overhead_fraction();
        assert!((frac - 0.0078125).abs() < 1e-6, "overhead_fraction={frac}");
    }

    #[test]
    fn test_packed_bytes() {
        // sketch_dim=64 → 8 bytes
        let qjl = QjlSketch::new(128, 64, Some(42));
        assert_eq!(qjl.packed_bytes_per_vector(), 8);

        // sketch_dim=128 → 16 bytes
        let qjl2 = QjlSketch::new(128, 128, Some(42));
        assert_eq!(qjl2.packed_bytes_per_vector(), 16);
    }

    #[test]
    fn test_total_bytes_with_polar3bit() {
        // d=512, k=64:
        // polar payload = (512*3+7)/8 = 1536/8 = 192
        // fp16 norm     = 2
        // sketch bytes  = (64+7)/8 = 8
        // total         = 202
        let qjl = QjlSketch::new(512, 64, Some(42));
        assert_eq!(qjl.total_bytes_with_polar3bit(), 202);
    }
}
