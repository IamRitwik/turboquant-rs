//! Naive INT8 quantizer — baseline for benchmarking against PolarQuant.
//!
//! ## Algorithm
//! 1. Scan the vector to find min and max values.
//! 2. Compute scale: `(max - min) / 255`
//! 3. Quantize each element: `q = round((v - min) / scale)` clamped to [0, 255]
//! 4. Store as u8 + 2 × f32 metadata (min, scale) per vector.
//!
//! ## Overhead
//! INT8 requires storing per-vector metadata (min + scale = 8 bytes overhead).
//! For a d=256 FP32 vector of 1024 bytes, this adds ~0.78% overhead —
//! but for per-block quantization (common in practice), overhead can reach
//! 1-2 bits per element, reducing effective compression ratio.

use half::f16;

/// Naive INT8 quantizer (per-vector min-max).
#[derive(Debug, Clone)]
pub struct Int8Quant;

/// Compressed INT8 representation.
#[derive(Debug, Clone)]
pub struct Int8Compressed {
    /// Quantized bytes (one per input element).
    pub data: Vec<u8>,
    /// Minimum value of the original vector (for dequantization).
    pub min: f32,
    /// Quantization scale factor.
    pub scale: f32,
}

impl Int8Compressed {
    /// Total bytes used including metadata (data + min + scale).
    pub fn byte_size(&self) -> usize {
        self.data.len() + 8 // 4 bytes each for min and scale
    }
}

impl Int8Quant {
    /// Compress a f16 vector (preferred for performance).
    pub fn compress_f16(&self, v: &[f16]) -> Int8Compressed {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &val in v {
            let val_f32 = val.to_f32();
            if val_f32 < min { min = val_f32; }
            if val_f32 > max { max = val_f32; }
        }
        let range = max - min;
        let scale = if range < 1e-10 { 1.0 } else { range / 255.0 };

        let data = v
            .iter()
            .map(|&x| {
                let q = ((x.to_f32() - min) / scale).round() as i32;
                q.clamp(0, 255) as u8
            })
            .collect();

        Int8Compressed { data, min, scale }
    }

    /// Compress a f32 vector.
    pub fn compress(&self, v: &[f32]) -> Int8Compressed {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let scale = if range < 1e-10 { 1.0 } else { range / 255.0 };

        let data = v
            .iter()
            .map(|&x| {
                let q = ((x - min) / scale).round() as i32;
                q.clamp(0, 255) as u8
            })
            .collect();

        Int8Compressed { data, min, scale }
    }

    /// Decompress an INT8 representation back to f32.
    pub fn decompress(&self, c: &Int8Compressed) -> Vec<f32> {
        c.data.iter().map(|&q| c.min + q as f32 * c.scale).collect()
    }

    /// Decompress an INT8 representation back to f16 (preferred for performance).
    pub fn decompress_f16(&self, c: &Int8Compressed) -> Vec<f16> {
        c.data.iter().map(|&q| f16::from_f32(c.min + q as f32 * c.scale)).collect()
    }

    /// Compression ratio of INT8 vs FP32 (not counting metadata).
    /// True ratio = 32/8 = 4.0x, but effective with metadata is lower.
    pub fn compression_ratio_nominal() -> f32 {
        4.0
    }

    /// Effective compression ratio accounting for min+scale metadata.
    pub fn compression_ratio_effective(dim: usize, original_bits: f32) -> f32 {
        let original_bytes = (dim as f32 * original_bits / 8.0) as usize;
        let compressed_bytes = dim + 8; // 1 byte/elem + 8 bytes metadata
        original_bytes as f32 / compressed_bytes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::mse;

    #[test]
    fn test_int8_roundtrip() {
        let q = Int8Quant;
        let v_f32: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();
        let v_f16: Vec<f16> = v_f32.iter().map(|&x| f16::from_f32(x)).collect();
        
        let c = q.compress_f16(&v_f16);
        let rec = q.decompress_f16(&c);
        
        let rec_f32: Vec<f32> = rec.iter().map(|x| x.to_f32()).collect();
        let err = mse(&v_f32, &rec_f32);
        assert!(err < 0.0001);
    }

    #[test]
    fn test_int8_constant_vector() {
        let q = Int8Quant;
        let v: Vec<f32> = vec![3.14; 64];
        let c = q.compress(&v);
        let rec = q.decompress(&c);
        let err = mse(&v, &rec);
        assert!(err < 1e-10, "Constant vector should reconstruct perfectly");
    }
}
