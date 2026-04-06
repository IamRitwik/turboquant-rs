//! # turboquant-rs
//!
//! A Rust implementation of Google's TurboQuant algorithm for KV-cache vector compression.
//!
//! Based on: *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*
//! (arXiv:2504.19874, ICLR 2026) by Zandieh et al., Google Research.
//!
//! ## Algorithm Overview
//!
//! TurboQuant uses two complementary stages:
//! 1. **PolarQuant** – applies a random orthogonal rotation to make the vector distribution
//!    data-oblivious (concentrated Beta), then quantizes in polar coordinates using
//!    precomputed Lloyd-Max codebooks. No per-block normalization needed.
//! 2. **QJL Stage** – applies a Hadamard-based randomized transform (Fast JL) to the
//!    quantization residual and stores signs to ensure an unbiased inner-product.
//!
//! This crate implements Stage 1 (PolarQuant) and a naive INT8 baseline for benchmarking.

pub mod int8_quant;
pub mod metrics;
pub mod polarquant;
pub mod quantize;

pub use int8_quant::Int8Quant;
pub use metrics::{compression_ratio, cosine_similarity, cosine_similarity_f16, mse, mse_f16};
pub use polarquant::PolarQuant;
