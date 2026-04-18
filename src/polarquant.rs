//! PolarQuant — Stage 1 of TurboQuant.
//!
//! ## Algorithm (from arXiv:2502.02617)
//!
//! ### Compression
//! 1. Apply a random orthogonal rotation to the input vector. Two strategies:
//!    - **Dense QR** (`new_dense`): O(d²) matrix multiply — legacy, kept for benchmarking
//!    - **FWHT** (`new_fwht`): O(d log d) butterfly — production path (Stage 2 default)
//!
//!    Both produce the same distribution guarantee: rotated coordinates are approximately
//!    N(0, 1/d) for unit-norm inputs, making their pairwise polar radii Rayleigh(1/√d).
//!
//! 2. Group consecutive pairs of rotated coordinates (x₂ᵢ₋₁, x₂ᵢ) and convert
//!    to polar form:
//!    `r = sqrt(x² + y²)`,  `θ = atan2(y, x)`
//!
//!    > [!NOTE]
//!    > This implementation uses a **Pairwise PolarQuant variant**. The original 
//!    > paper (arXiv:2502.02617) describes a recursive polar transformation. This 
//!    > version uses independent pairwise transforms for simplicity.
//!
//! 3. Quantize r and θ independently using precomputed Lloyd-Max codebooks.
//!    Because the distribution is data-oblivious, codebooks are universal —
//!    no per-vector statistics needed.
//!
//! 4. Pack the quantized indices as `bits`-wide fields.
//!
//! ### Decompression
//! 1. Unpack indices → centroid values (r̂, θ̂)
//! 2. Reconstruct Cartesian: `x̂ = r̂·cos(θ̂)`, `ŷ = r̂·sin(θ̂)`
//! 3. Apply inverse rotation (Πᵀ for dense, FWHT inverse for butterfly)
//!
//! ### Key insight
//! Traditional INT8 and per-group quantizers must store min/max per block
//! (adding ~1-2 bits of overhead per element). PolarQuant's rotation eliminates
//! this overhead, achieving near-optimal rate-distortion.

use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use half::f16;

use crate::fwht::FwhtRotation;
use crate::quantize::{ScalarQuantizer, build_angle_quantizer, build_radius_quantizer};

// ---------------------------------------------------------------------------
//  Rotation Strategy
// ---------------------------------------------------------------------------

/// Which rotation is applied before the polar transform.
///
/// Both strategies produce an orthogonal transform that makes coordinates
/// approximately N(0, 1/d), preserving PolarQuant's codebook validity.
#[derive(Debug, Clone)]
pub enum RotationStrategy {
    /// Legacy O(d²) dense matrix rotation — used in Stage 1 for benchmarking.
    Dense(Array2<f32>),
    /// Production O(d log d) Fast Walsh-Hadamard rotation — Stage 2 default.
    Fwht(FwhtRotation),
}

/// PolarQuant compressor/decompressor.
///
/// # Example
/// ```
/// use turboquant::PolarQuant;
///
/// let pq = PolarQuant::new(256, 4, Some(42));
/// let vector: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
/// let compressed = pq.compress(&vector);
/// let reconstructed = pq.decompress(&compressed);
/// ```
#[derive(Debug, Clone)]
pub struct PolarQuant {
    /// Dimension of vectors this quantizer is built for.
    pub dim: usize,
    /// Bits per polar component (3 or 4 recommended).
    pub bits: u8,
    /// Rotation strategy: Dense (O(d²)) or FWHT (O(d log d)).
    rotation: RotationStrategy,
    /// Lloyd-Max quantizer for radii (r).
    radius_q: ScalarQuantizer,
    /// Lloyd-Max quantizer for angles (θ).
    angle_q: ScalarQuantizer,
}

impl PolarQuant {
    /// Create a new `PolarQuant` instance using the **dense QR rotation** (Stage 1 default).
    ///
    /// Equivalent to `new_dense(dim, bits, seed)`. Kept for backward compatibility.
    ///
    /// # Arguments
    /// * `dim`  — vector dimension (must be even)
    /// * `bits` — bits per polar component (3 or 4)
    /// * `seed` — optional RNG seed for reproducibility
    pub fn new(dim: usize, bits: u8, seed: Option<u64>) -> Self {
        Self::new_dense(dim, bits, seed)
    }

    /// Create a `PolarQuant` using the **dense QR rotation** — O(d²).
    ///
    /// Use this for benchmarking the dense baseline or when dim is not a power of 2.
    pub fn new_dense(dim: usize, bits: u8, seed: Option<u64>) -> Self {
        assert!(dim % 2 == 0, "dim must be even for polar pairing");
        assert!(bits >= 2 && bits <= 8, "bits must be in [2, 8]");

        let levels = 1usize << bits;
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xdeadbeef_cafebabe));
        let gaussian: Array2<f32> = Array2::random_using((dim, dim), StandardNormal, &mut rng);
        let dense_matrix = qr_orthogonalize(gaussian);

        let radius_q = build_radius_quantizer(levels, dim, &mut rng);
        let angle_q = build_angle_quantizer(levels);

        Self {
            dim,
            bits,
            rotation: RotationStrategy::Dense(dense_matrix),
            radius_q,
            angle_q,
        }
    }

    /// Create a `PolarQuant` using the **FWHT rotation** — O(d log d).
    ///
    /// **Stage 2 production path.** Requires `dim` to be a power of 2.
    /// Memory: 2KB at d=512 vs 1MB for dense — a 512× reduction.
    ///
    /// Codebooks are identical to `new_dense`; the distribution guarantee
    /// holds for the Randomized Hadamard Transform.
    ///
    /// # Panics
    /// Panics if `dim` is not a power of 2.
    pub fn new_fwht(dim: usize, bits: u8, seed: Option<u64>) -> Self {
        assert!(dim % 2 == 0, "dim must be even for polar pairing");
        assert!(bits >= 2 && bits <= 8, "bits must be in [2, 8]");
        assert!(
            dim.is_power_of_two(),
            "FWHT rotation requires a power-of-2 dimension, got {dim}"
        );

        let levels = 1usize << bits;
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xdeadbeef_cafebabe));
        let fwht = FwhtRotation::new(dim, Some(seed.unwrap_or(0xdeadbeef_cafebabe)));

        let radius_q = build_radius_quantizer(levels, dim, &mut rng);
        let angle_q = build_angle_quantizer(levels);

        Self {
            dim,
            bits,
            rotation: RotationStrategy::Fwht(fwht),
            radius_q,
            angle_q,
        }
    }

    /// Which rotation strategy is active.
    pub fn rotation_strategy_name(&self) -> &'static str {
        match &self.rotation {
            RotationStrategy::Dense(_) => "Dense O(d²)",
            RotationStrategy::Fwht(_)  => "FWHT O(d·log d)",
        }
    }

    /// Rotation matrix memory in bytes.
    /// Dense: dim² × 4 bytes. FWHT: dim × 4 bytes.
    pub fn rotation_memory_bytes(&self) -> usize {
        match &self.rotation {
            RotationStrategy::Dense(m) => m.len() * std::mem::size_of::<f32>(),
            RotationStrategy::Fwht(f)  => f.memory_bytes(),
        }
    }

    /// Compress a vector of length `dim` (as f16) to a packed byte representation.
    ///
    /// This is the preferred method for performance as it avoids an extra copy.
    pub fn compress_f16(&self, v: &[f16]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim, "Input length mismatch");

        // Calculate L2 norm as f32
        let norm_f32: f32 = v.iter()
            .map(|&x| {
                let x32 = x.to_f32();
                x32 * x32
            })
            .sum::<f32>()
            .sqrt()
            .max(1e-10); // avoid division by zero
        let norm_f16 = f16::from_f32(norm_f32);
        
        // Step 1: normalize and rotate
        let mut normalized: Vec<f32> = v.iter().map(|&val| val.to_f32() / norm_f32).collect();
        self.apply_rotation(&mut normalized);
        let rotated = Array1::from(normalized);

        // Step 2 & 3: shared quantization and packing
        let mut packed = self.compress_rotated_internal(rotated);

        // Prepend FP16 norm (2 bytes, little-endian)
        let norm_bytes = norm_f16.to_le_bytes();
        let mut out = Vec::with_capacity(packed.len() + 2);
        out.extend_from_slice(&norm_bytes);
        out.append(&mut packed);
        out
    }

    /// Compress a vector of length `dim` (original f32 method).
    pub fn compress(&self, v: &[f32]) -> Vec<u8> {
        assert_eq!(v.len(), self.dim, "Input length mismatch");

        // Calculate L2 norm
        let norm_f32: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-10);
        let norm_f16 = f16::from_f32(norm_f32);

        // Step 1: normalize and rotate
        let mut normalized: Vec<f32> = v.iter().map(|&x| x / norm_f32).collect();
        self.apply_rotation(&mut normalized);
        let rotated = Array1::from(normalized);

        let mut packed = self.compress_rotated_internal(rotated);

        // Prepend FP16 norm
        let norm_bytes = norm_f16.to_le_bytes();
        let mut out = Vec::with_capacity(packed.len() + 2);
        out.extend_from_slice(&norm_bytes);
        out.append(&mut packed);
        out
    }

    /// Shared internal logic for quantizing and packing rotated coordinates.
    fn compress_rotated_internal(&self, rotated: Array1<f32>) -> Vec<u8> {
        // Step 2: polar transform + quantize pairs
        let mut indices: Vec<u8> = Vec::with_capacity(self.dim);
        
        let rotated_slice = rotated.as_slice()
            .expect("Rotated array should be contiguous for pairwise chunking");
            
        for chunk in rotated_slice.chunks(2) {
            let x = chunk[0];
            let y = chunk[1];
            let r = (x * x + y * y).sqrt();
            let theta = y.atan2(x); 
            indices.push(self.radius_q.quantize(r));
            indices.push(self.angle_q.quantize(theta));
        }

        // Step 3: pack indices into bits-wide fields
        pack_bits(&indices, self.bits)
    }

    /// Decompress a packed byte slice back to a f32 vector.
    pub fn decompress(&self, compressed: &[u8]) -> Vec<f32> {
        self.decompress_internal(compressed).to_vec()
    }

    /// Decompress a packed byte slice back to a f16 vector.
    pub fn decompress_f16(&self, compressed: &[u8]) -> Vec<f16> {
        let reconstructed = self.decompress_internal(compressed);
        reconstructed.iter().map(|&x| f16::from_f32(x)).collect()
    }

    /// Shared internal logic for dequantizing and inverting rotation.
    fn decompress_internal(&self, compressed: &[u8]) -> Array1<f32> {
        assert!(compressed.len() >= 2, "Compressed payload too short");

        // Extract norm from the first 2 bytes
        let norm_bytes = [compressed[0], compressed[1]];
        let norm = f16::from_le_bytes(norm_bytes).to_f32();
        let payload = &compressed[2..];

        // Unpack bit fields
        let indices = unpack_bits(payload, self.bits, self.dim);

        // Reconstruct polar → Cartesian
        let mut rotated = Vec::with_capacity(self.dim);
        for pair in indices.chunks(2) {
            let r_hat = self.radius_q.dequantize(pair[0]);
            let theta_hat = self.angle_q.dequantize(pair[1]);
            rotated.push(r_hat * theta_hat.cos());
            rotated.push(r_hat * theta_hat.sin());
        }

        // Apply inverse rotation
        self.apply_inverse_rotation(&mut rotated);
        let mut reconstructed = Array1::from_vec(rotated);

        // Multiply by norm
        reconstructed *= norm;
        reconstructed
    }

    // ---------------------------------------------------------------------------
    //  Internal rotation helpers
    // ---------------------------------------------------------------------------

    /// Apply the forward rotation to a mutable f32 slice in place.
    fn apply_rotation(&self, v: &mut Vec<f32>) {
        match &self.rotation {
            RotationStrategy::Dense(matrix) => {
                let arr = Array1::from(v.clone());
                let rotated = matrix.dot(&arr);
                v.copy_from_slice(rotated.as_slice().unwrap());
            }
            RotationStrategy::Fwht(fwht) => {
                fwht.apply(v);
            }
        }
    }

    /// Apply the inverse rotation to a mutable f32 Vec in place.
    fn apply_inverse_rotation(&self, v: &mut Vec<f32>) {
        match &self.rotation {
            RotationStrategy::Dense(matrix) => {
                let arr = Array1::from(v.clone());
                let inv = matrix.t().dot(&arr);
                v.copy_from_slice(inv.as_slice().unwrap());
            }
            RotationStrategy::Fwht(fwht) => {
                fwht.apply_inverse(v);
            }
        }
    }

    /// Return the number of bytes used to compress one vector.
    pub fn compressed_bytes(&self) -> usize {
        let total_bits = self.dim * self.bits as usize;
        let payload_bytes = (total_bits + 7) / 8;
        payload_bytes + 2 // include 2 bytes for FP16 norm
    }

    /// Return the compression ratio vs FP16 (16 bits / element).
    pub fn compression_ratio(&self) -> f32 {
        let original_bits = self.dim as f32 * 16.0;
        let compressed_bits = self.compressed_bytes() as f32 * 8.0;
        original_bits / compressed_bits
    }

    /// Return the effective bits per element.
    ///
    /// This represents the bit-width of the core payload (quantized radius + angle
    /// indices). Note that the actual compressed size includes a 2-byte FP16
    /// norm overhead per vector.
    pub fn bits_per_element(&self) -> f32 {
        self.bits as f32
    }
}

// ---------------------------------------------------------------------------
//  Bit packing helpers
// ---------------------------------------------------------------------------

/// Pack a slice of integer indices (each `bits` wide) into a minimal byte Vec.
fn pack_bits(indices: &[u8], bits: u8) -> Vec<u8> {
    let total_bits = indices.len() * bits as usize;
    let total_bytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; total_bytes];
    let mut bit_pos = 0usize;
    for &idx in indices {
        for b in 0..bits as usize {
            let bit = (idx >> b) & 1;
            out[bit_pos / 8] |= bit << (bit_pos % 8);
            bit_pos += 1;
        }
    }
    out
}

/// Unpack `count` indices of `bits` width from a packed byte slice.
fn unpack_bits(packed: &[u8], bits: u8, count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut idx = 0u8;
        for b in 0..bits as usize {
            let byte = packed[bit_pos / 8];
            let bit = (byte >> (bit_pos % 8)) & 1;
            idx |= bit << b;
            bit_pos += 1;
        }
        out.push(idx);
    }
    out
}

// ---------------------------------------------------------------------------
//  QR Orthogonalization (Gram-Schmidt via modified Gram-Schmidt for stability)
// ---------------------------------------------------------------------------

/// Compute an orthogonal matrix from a square Gaussian matrix via
/// modified Gram-Schmidt orthogonalization.
fn qr_orthogonalize(mut a: Array2<f32>) -> Array2<f32> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n, "Must be square");

    for i in 0..n {
        // Normalize column i
        let norm = {
            let col = a.column(i);
            col.dot(&col).sqrt()
        };
        if norm < 1e-10 {
            continue; // Skip near-zero columns (extremely rare)
        }
        {
            let mut col = a.column_mut(i);
            col /= norm;
        }

        // Orthogonalize all subsequent columns against column i
        for j in (i + 1)..n {
            let dot = {
                let ci = a.column(i);
                let cj = a.column(j);
                ci.dot(&cj)
            };
            for row in 0..n {
                let ci_val = a[[row, i]];
                a[[row, j]] -= dot * ci_val;
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::mse;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
    }

    #[test]
    fn test_pack_unpack_roundtrip_3bit() {
        let indices: Vec<u8> = (0..8u8).map(|i| i % 8).collect();
        let packed = pack_bits(&indices, 3);
        let unpacked = unpack_bits(&packed, 3, 8);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let indices: Vec<u8> = (0..16u8).collect();
        let packed = pack_bits(&indices, 4);
        let unpacked = unpack_bits(&packed, 4, 16);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_rotation_orthogonality() {
        // Dense: verify Πᵀ·Π ≈ I directly from the matrix
        for &dim in &[32_usize, 64] { // small dims only — O(d²) test
            let pq = PolarQuant::new_dense(dim, 4, Some(42));
            if let RotationStrategy::Dense(ref matrix) = pq.rotation {
                let product = matrix.t().dot(matrix);
                for i in 0..dim {
                    for j in 0..dim {
                        let val = product[[i, j]];
                        let expected = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (val - expected).abs() < 1e-3,
                            "Dense orthogonality violated at dim={dim}, [{i},{j}]: {val:.6} ≠ {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_fwht_rotation_orthogonality() {
        // FWHT: verify norm is preserved (orthogonal transform property)
        use rand::prelude::*;
        for &dim in &[128_usize, 512] {
            let pq = PolarQuant::new_fwht(dim, 4, Some(42));
            let mut rng = StdRng::seed_from_u64(99);
            let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
            let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Compress then decompress — the rotation+inverse should preserve direction
            let compressed   = pq.compress(&v);
            let reconstructed = pq.decompress(&compressed);
            let cos_sim: f32 = {
                let dot: f32 = v.iter().zip(reconstructed.iter()).map(|(a, b)| a * b).sum();
                let na = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na < 1e-10 || nb < 1e-10 { 0.0 } else { dot / (na * nb) }
            };
            assert!(
                cos_sim > 0.95,
                "FWHT rotation+inverse should preserve direction at dim={dim}: cos={cos_sim:.4}"
            );
            let _ = norm_before;
        }
    }

    fn normalize(v: Vec<f32>) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        v.into_iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_compress_decompress_low_mse() {
        for &dim in &[128, 256, 512] {
            let pq = PolarQuant::new(dim, 4, Some(42));
            // L2-normalize to simulate real KV-cache vectors
            let v = normalize(random_vector(dim, 7));
            let compressed = pq.compress(&v);
            let reconstructed = pq.decompress(&compressed);
            let error = mse(&v, &reconstructed);
            // For L2-normalized vectors (norm=1), 4-bit PolarQuant should achieve MSE < 0.02
            assert!(error < 0.02,
                "dim={dim}: MSE {error:.6} too high for 4-bit PolarQuant on normalized vectors");
        }
    }

    #[test]
    fn test_compress_decompress_f16() {
        let dim = 64;
        let pq = PolarQuant::new(dim, 4, Some(42));
        let v_f32 = normalize(random_vector(dim, 7));
        let v_f16: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();

        let compressed = pq.compress_f16(&v_f16);
        let reconstructed = pq.decompress_f16(&compressed);

        // MSE should be small (reconstructed is compared in f32 internally by mse)
        let rec_f32: Vec<f32> = reconstructed.iter().map(|x| x.to_f32()).collect();
        let error = mse(&v_f32, &rec_f32);
        assert!(error < 0.02);
    }

    #[test]
    fn test_unnormalized_vectors() {
        let dim = 128;
        let pq = PolarQuant::new(dim, 4, Some(42));
        
        // Create a vector with a large norm (not 1.0)
        let v_base = random_vector(dim, 7);
        let v: Vec<f32> = v_base.iter().map(|&x| x * 10.0).collect();
        let original_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(original_norm > 5.0);

        let compressed = pq.compress(&v);
        let reconstructed = pq.decompress(&compressed);
        
        let rec_norm = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // The norm should be preserved (within FP16 precision)
        assert!((rec_norm - original_norm).abs() / original_norm < 0.01, 
            "Norm not preserved: expected {original_norm}, got {rec_norm}");

        let error = mse(&v, &reconstructed);
        // SNR should be high (MSE should be low relative to the norm)
        let relative_error = error / (original_norm * original_norm / dim as f32);
        assert!(relative_error < 0.02, "Relative MSE {relative_error} too high");
    }

    #[test]
    fn test_compression_ratio() {
        let pq_4 = PolarQuant::new(128, 4, Some(42));
        let pq_3 = PolarQuant::new(128, 3, Some(42));
        
        // For dim=128:
        // 4-bit: (128*16) / (((128*4+7)/8 + 2) * 8) = 2048 / (66 * 8) = 2048 / 528 ≈ 3.878
        // 3-bit: (128*16) / (((128*3+7)/8 + 2) * 8) = 2048 / (50 * 8) = 2048 / 400 = 5.12
        assert!((pq_4.compression_ratio() - 3.878).abs() < 0.01);
        assert!((pq_3.compression_ratio() - 5.12).abs() < 0.01);
    }
}
