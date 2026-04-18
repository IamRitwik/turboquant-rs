//! # turboquant-rs
//!
//! A Rust implementation of Google's TurboQuant algorithm for KV-cache vector compression.
//!
//! Based on:
//! - *PolarQuant: Quantizing KV Caches with Polar Transformation* (arXiv:2502.02617, AISTATS 2026)
//! - *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv:2504.19874, ICLR 2026)
//!
//! ## Algorithm Overview
//!
//! TurboQuant uses two complementary stages:
//!
//! ### Stage 1 — PolarQuant
//! Applies a random orthogonal rotation (Dense QR or FWHT) to make the vector distribution
//! data-oblivious, then quantizes in polar coordinates using precomputed Lloyd-Max codebooks.
//! No per-block normalization needed.
//!
//! ### Stage 2 — QJL Residual Sketch
//! After PolarQuant compression, a Johnson-Lindenstrauss residual sketch corrects the
//! inner-product bias introduced by quantization. The sketch stores only 1 bit per
//! projected dimension, providing an unbiased inner-product estimator at near-zero
//! additional storage cost.
//!
//! ## Modules
//! - [`polarquant`] — Stage 1 compression (Dense + FWHT rotation variants)
//! - [`fwht`]       — Fast Walsh-Hadamard Transform (O(d log d) rotation)
//! - [`qjl`]        — QJL residual sketch for unbiased inner products
//! - [`attention`]  — Attention score simulation and quality metrics
//! - [`quantize`]   — Lloyd-Max codebook construction
//! - [`int8_quant`] — INT8 baseline
//! - [`metrics`]    — MSE, cosine similarity, compression ratio, throughput

pub mod attention;
pub mod fwht;
pub mod int8_quant;
pub mod metrics;
pub mod polarquant;
pub mod quantize;
pub mod qjl;

pub use fwht::FwhtRotation;
pub use int8_quant::Int8Quant;
pub use metrics::{compression_ratio, cosine_similarity, cosine_similarity_f16, mse, mse_f16};
pub use polarquant::PolarQuant;
pub use qjl::{QjlSketch, compute_residual};
