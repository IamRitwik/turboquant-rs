//! FWHT — Fast Walsh-Hadamard Transform (Randomized Hadamard Transform / RHT)
//!
//! Replaces the O(d²) dense QR rotation in PolarQuant with an O(d log d)
//! butterfly structure. The transform is:
//!
//! ```text
//! RHT(v) = (1/√d) · H · D · v
//! ```
//!
//! where:
//! - `H` = Walsh-Hadamard matrix (butterfly structure, no storage needed — computed in place)
//! - `D` = diagonal ±1 Rademacher random signs (precomputed, stored as `Vec<f32>`)
//! - `d` = vector dimension (MUST be a power of 2)
//!
//! **Why this is a valid rotation:** RHT is an orthogonal transform — it preserves
//! L2 norms. This means PolarQuant's codebooks remain valid when FWHT replaces the
//! dense QR rotation; the distribution guarantee holds for RHT.
//!
//! **Memory savings vs dense rotation:**
//! - Dense: d² floats = 1 MB at d=512
//! - FWHT:  d floats  = 2 KB at d=512

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Randomized Fast Walsh-Hadamard Transform.
///
/// Replaces the O(d²) dense matrix rotation used in Stage 1 with an O(d log d)
/// butterfly structure. Mathematically equivalent — the codebooks do NOT change.
///
/// # Example
/// ```
/// use turboquant::fwht::FwhtRotation;
///
/// let fwht = FwhtRotation::new(512, Some(42));
/// let mut v: Vec<f32> = (0..512).map(|i| (i as f32).sin()).collect();
/// let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
/// fwht.apply(&mut v);
/// let new_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((orig_norm - new_norm).abs() < 1e-3);
/// ```
#[derive(Debug, Clone)]
pub struct FwhtRotation {
    /// Dimension — MUST be a power of 2.
    pub dim: usize,
    /// Precomputed Rademacher signs (+1.0 or -1.0), length = dim.
    signs: Vec<f32>,
    /// Normalization factor = 1/√dim.
    scale: f32,
}

impl FwhtRotation {
    /// Create a new FWHT rotation with random Rademacher signs.
    ///
    /// # Arguments
    /// * `dim`  — vector dimension (must be a power of 2: 128, 256, 512, 1024, …)
    /// * `seed` — optional RNG seed for reproducibility
    ///
    /// # Panics
    /// Panics if `dim` is not a power of 2.
    pub fn new(dim: usize, seed: Option<u64>) -> Self {
        assert!(
            dim.is_power_of_two(),
            "FWHT requires power-of-2 dimension, got {dim}"
        );
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xdeadbeef_cafebabe));
        let signs: Vec<f32> = (0..dim)
            .map(|_| if rng.gen_bool(0.5) { 1.0_f32 } else { -1.0_f32 })
            .collect();
        let scale = 1.0_f32 / (dim as f32).sqrt();
        Self { dim, signs, scale }
    }

    /// Apply the Randomized Hadamard Transform **in place**.
    ///
    /// Algorithm: D (Rademacher signs) → WHT butterfly → normalize by 1/√d.
    ///
    /// Complexity: **O(d log d)** — no matrix storage required.
    ///
    /// # Panics
    /// Panics if `v.len() != self.dim`.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(
            v.len(),
            self.dim,
            "FWHT apply: input length {} != dim {}",
            v.len(),
            self.dim
        );

        // Step 1: Apply Rademacher diagonal matrix D
        for (x, &s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }

        // Step 2: In-place Walsh-Hadamard butterfly
        // H at step h: pairs (v[j], v[j+h]) → (v[j]+v[j+h], v[j]-v[j+h])
        let mut h = 1_usize;
        while h < self.dim {
            let step = h * 2;
            let mut i = 0;
            while i < self.dim {
                for j in i..i + h {
                    let a = v[j];
                    let b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
                i += step;
            }
            h *= 2;
        }

        // Step 3: Normalize by 1/√d to make the transform orthogonal
        for x in v.iter_mut() {
            *x *= self.scale;
        }
    }

    /// Apply the inverse Randomized Hadamard Transform **in place**.
    ///
    /// The inverse of `(1/√d) · H · D · x` is `(1/√d) · D · H · y`.
    ///
    /// Derivation: `H² = d · I`, so `H⁻¹ = H/d`.
    /// - `y = (1/√d) · H · D · x`
    /// - `H · y = (1/√d) · H² · D · x = √d · D · x`
    /// - `(1/√d) · D · H · y = D · D · x = x`  ✓
    pub fn apply_inverse(&self, v: &mut [f32]) {
        // Step 1: Apply WHT butterfly (same circuit as forward)
        let mut h = 1_usize;
        while h < self.dim {
            let step = h * 2;
            let mut i = 0;
            while i < self.dim {
                for j in i..i + h {
                    let a = v[j];
                    let b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
                i += step;
            }
            h *= 2;
        }

        // Step 2: Scale by 1/√d  (same factor as in apply — NOT 1/d)
        for x in v.iter_mut() {
            *x *= self.scale;
        }

        // Step 3: Re-apply D signs  (D⁻¹ = D since signs are ±1)
        for (x, &s) in v.iter_mut().zip(self.signs.iter()) {
            *x *= s;
        }
    }

    /// Memory footprint in bytes: just the signs vector (dim × 4 bytes).
    ///
    /// Compare to dense rotation: dim² × 4 bytes.
    /// At d=512: 2 KB (FWHT) vs 1 MB (dense).
    pub fn memory_bytes(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }

    /// Memory savings factor vs. a dense rotation matrix at the same dimension.
    pub fn memory_savings_vs_dense(&self) -> usize {
        self.dim // dim² / dim = dim
    }
}

/// Compare operation counts: dense vs FWHT at a given dimension.
///
/// Returns `(dense_ops, fwht_ops)` where:
/// - `dense_ops` = d² (matrix-vector multiply)
/// - `fwht_ops`  = d · log₂(d) (WHT butterfly)
pub fn ops_comparison(dim: usize) -> (usize, usize) {
    let dense_ops = dim * dim; // O(d²)
    let log2_dim = (dim as f64).log2() as usize;
    let fwht_ops = dim * log2_dim; // O(d log d)
    (dense_ops, fwht_ops)
}

/// Print an ops comparison table for a set of dimensions.
pub fn print_ops_table(dims: &[usize]) {
    println!("\n{:<8} {:>14} {:>14} {:>12}", "Dim", "Dense ops (d²)", "FWHT ops (d·lgd)", "Speedup");
    println!("{}", "─".repeat(52));
    for &d in dims {
        let (dense, fwht) = ops_comparison(d);
        let speedup = dense as f64 / fwht as f64;
        println!("{:<8} {:>14} {:>14} {:>11.1}×", d, dense, fwht, speedup);
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vec(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * 0.1_f32).sin()).collect()
    }

    #[test]
    fn test_fwht_orthogonality() {
        // RHT must preserve vector norms (orthogonal transform property)
        for &dim in &[128_usize, 256, 512, 1024] {
            let fwht = FwhtRotation::new(dim, Some(42));
            let v = make_test_vec(dim);
            let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

            let mut rotated = v.clone();
            fwht.apply(&mut rotated);
            let norm_after: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

            assert!(
                (norm_before - norm_after).abs() < 1e-3,
                "FWHT must preserve L2 norm at d={dim}: before={norm_before:.6} after={norm_after:.6}"
            );
        }
    }

    #[test]
    fn test_fwht_inverse_roundtrip() {
        // apply followed by apply_inverse should recover the original vector
        let dim = 256;
        let fwht = FwhtRotation::new(dim, Some(99));
        let original = make_test_vec(dim);
        let mut v = original.clone();

        fwht.apply(&mut v);
        let norm_after_apply: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("Norm after apply: {norm_after_apply:.6}");

        fwht.apply_inverse(&mut v);
        let norm_after_inverse: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("Norm after inverse: {norm_after_inverse:.6}");

        for (i, (a, b)) in original.iter().zip(v.iter()).enumerate().take(5) {
            println!("Index {i}: original={a:.6}, recovered={b:.6}");
        }

        for (a, b) in original.iter().zip(v.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Inverse roundtrip failed: original={a:.6}, recovered={b:.6}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "FWHT requires power-of-2 dimension")]
    fn test_fwht_power_of_two_only() {
        // Should panic for non-power-of-2 dimensions
        let _ = FwhtRotation::new(300, Some(42));
    }

    #[test]
    fn test_fwht_power_of_two_ok() {
        // Should not panic for valid power-of-2 dimensions
        for &dim in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let _ = FwhtRotation::new(dim, Some(42));
        }
    }

    #[test]
    fn test_ops_reduction_at_512() {
        let (dense, fwht) = ops_comparison(512);
        assert_eq!(dense, 262_144); // 512²
        assert_eq!(fwht, 4_608);   // 512 × 9
        let speedup = dense as f64 / fwht as f64;
        assert!(
            speedup > 50.0,
            "Expected 50x+ ops reduction at d=512, got {speedup:.1}x"
        );
    }

    #[test]
    fn test_fwht_memory_savings() {
        let dim = 512;
        let fwht = FwhtRotation::new(dim, Some(42));
        // Signs array: 512 × 4 bytes = 2048 bytes
        assert_eq!(fwht.memory_bytes(), dim * 4);
        // Savings vs dense: dim× = 512×
        assert_eq!(fwht.memory_savings_vs_dense(), dim);
    }

    #[test]
    fn test_fwht_randomness() {
        // Two different seeds should yield different transforms
        let dim = 64;
        let fwht_a = FwhtRotation::new(dim, Some(1));
        let fwht_b = FwhtRotation::new(dim, Some(2));
        assert_ne!(fwht_a.signs, fwht_b.signs, "Different seeds must yield different transforms");
    }

    #[test]
    fn test_fwht_deterministic() {
        // Same seed should always yield identical transforms (reproducibility)
        let dim = 64;
        let fwht_a = FwhtRotation::new(dim, Some(42));
        let fwht_b = FwhtRotation::new(dim, Some(42));
        assert_eq!(fwht_a.signs, fwht_b.signs);
    }
}
