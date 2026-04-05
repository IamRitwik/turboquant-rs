# TurboQuant-RS

A production-ready Rust implementation of Google's **TurboQuant** algorithm for KV-cache vector compression.

Based on the paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) by Zandieh et al., Google Research.

## Visualizing the Performance

Here's an overview of the performance compared to a naive INT8 quantizer on an **M1 MacBook Air**.

```text
Vectors: 5000 × L2-normalized random (simulated KV cache entries)

Method               Dim  Bits/el    Ratio          MSE   Cosine-Sim       Compress     Decompress
                                   vs FP16                                 (Mvec/s)       (Mvec/s)
──────────────────────────────────────────────────────────────────────────────────────────────────
INT8                 512      8.0     1.97x      ~0.00000     ~1.00000        1.0553         3.3782
PolarQ-4bit          512      4.0     4.00x      0.00003      0.99274         0.0219         0.0044
PolarQ-3bit          512      3.0     5.33x      0.00011      0.97145         0.0238         0.0044
```

### Key Insights

▶ **PolarQuant-4bit vs INT8 at d=512:**
* Compression ratio: 4.00x vs 1.97x (2.0x better)
* Near-perfect Cosine Similarity: 0.99274 vs ~1.00000

▶ **PolarQuant-3bit at d=512:**
* 5.33x compression with low MSE (0.00011) and great cosine similarity (0.97145).

▶ **Memory savings at KV cache scale (LLaMA-3 8B, 128K context):**
* FP16 baseline (32 layers, 8 KV heads, d=128) ≈ 17.1 GB
* PolarQuant-4bit: ≈ 4.3 GB (4x reduction vs FP16)
* PolarQuant-3bit: ≈ 3.2 GB (5.33x reduction vs FP16)

## Algorithm Architecture

TurboQuant uses a mathematically elegant two-stage approach for training-free, data-oblivious quantization. This crate currently implements **Stage 1 (PolarQuant)**, which is the MSE-optimal primary quantization step.

### Stage 1: PolarQuant (The Core Transform)

Traditional quantization (like INT8) requires you to scan the vector, compute a `min` and `max`, and use those to scale the values. This adds 1-2 bits of overhead per element for fine-grained blocks, diminishing your compression ratio.

PolarQuant solves this by transforming the data itself:

1. **Random Orthogonal Rotation**: `v' = Π · v`. We generate a random $d$-dimensional orthogonal matrix $\Pi$ via the QR decomposition of a Gaussian matrix. When we rotate the input vector by this matrix, the coordinates become uniformly distributed, approaching a concentrated Beta distribution. It effectively uniformizes the vector and eliminates any high-variance outliers.
2. **Polar Coordinates**: We pair up the coordinates $(x_1, x_2)$ and convert them into polar form $(r, \theta)$.
3. **Lloyd-Max Quantization**: Because the rotation makes the distribution entirely predictable and independent of the actual input vector, we can use precomputed optimal Lloyd-Max scalar quantizer codebooks for the radius and angle.

*Result: We quantize without storing any per-vector or per-block normalization scales. Pure compression.*

## Project Structure

```text
turboquant-rs/
├── Cargo.toml
├── src/
│   ├── polarquant.rs       # The PolarQuant algorithm (rotation + polar transform)
│   ├── quantize.rs         # Lloyd-Max scalar quantizer codebooks
│   ├── int8_quant.rs       # Baseline INT8 quantizer
│   └── metrics.rs          # Evaluation (MSE, Compression Ratio, Cosine Sim)
└── benches/
    └── compression.rs      # Criterion test harness for throughput/latency
```

## Running the Demo

```bash
git clone https://github.com/IamRitwik/turboquant-rs.git
cd turboquant-rs
cargo run --example demo --release
```

## Benchmark Suite

This repository includes a full benchmark suite using `criterion` testing across multiple vector dimensions spanning real LLM sizes (`d=128, 256, 512, 1024`).

```bash
cargo bench
```

And you can use the included python script to plot the outputs of the benchmark demo. Make sure you've ran `cargo run --example demo --release` first.

```bash
pip install matplotlib numpy
python scripts/plot_results.py
```

