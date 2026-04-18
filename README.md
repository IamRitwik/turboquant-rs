# turboquant-rs

A Rust reference implementation of the **PolarQuant** algorithm for KV-cache
vector compression, benchmarked as the primary compression stage in the
TurboQuant family.

Based on the papers:

- [**PolarQuant**: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)
  (AISTATS 2026) — Han, Kacham, Karbasi, Mirrokni & Zandieh.
- [**TurboQuant**: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
  (ICLR 2026) — Zandieh, Daliri, Hadian & Mirrokni.
  Google Research, Google DeepMind & NYU.

> **Scope:** This crate implements **TurboQuant** — the reference implementation for 
> high-throughput KV-cache compression. It includes Stage 1 (**PolarQuant** with 
> O(d log d) FWHT rotation) and Stage 2 (**QJL Residual Sketch**) for unbiased 
> inner product estimation.

---

## Benchmarks

Tested on an M1 MacBook Air, 5 000 × L2-normalised random vectors
(simulated KV cache entries).

```
Method          Dim   Bits/el   Ratio vs FP16      MSE   Cosine-Sim   Compress     Decompress
                                                                       (Mvec/s)       (Mvec/s)
─────────────────────────────────────────────────────────────────────────────────────────────
INT8            512       8.0           1.97×   ~0.00000    ~1.00000     1.0553         3.3782
PolarQ-3bit     512       3.0           5.28×    0.00011     0.97145     0.0238         0.0044
T-Quant (FWHT)  512       3.0           5.28×    0.00011     0.97148     0.0647         0.1537
T-Quant + QJL   512       3.1           5.07×    0.00011     0.97148     0.0631         0.1512

† Includes 2-byte FP16 overhead for per-vector norm storage.
†† T-Quant + QJL includes a 64-bit residual sketch (+8 bytes per vector).
   Decompress throughput includes rotation + QJL correction estimation.
```

### Memory impact at KV cache scale — LLaMA-3 8B, 128K context

| Method | Size | Reduction |
|---|---|---|
| FP16 baseline (32 layers, 8 KV heads, d=128) | ≈ 17.1 GB | — |
| PolarQuant-4bit | ≈ 4.3 GB | 3.97× |
| PolarQuant-3bit | ≈ 3.2 GB | 5.28× |

---

## The Paper Family

PolarQuant and TurboQuant are two concurrent papers from an overlapping
author group (Zandieh and Mirrokni appear on both). They solve the same
problem — online, data-oblivious KV-cache quantization — with a shared
philosophical foundation but different mechanisms.

| | PolarQuant | TurboQuant |
|---|---|---|
| Rotation | Random orthogonal (dense QR in this crate) | Randomized Walsh-Hadamard (O(d log d)) |
| Coordinate transform | Pairwise or recursive polar (r, θ) | None — scalar per coordinate |
| Distribution exploited | Rayleigh (radii), Uniform (angles) | Beta → Gaussian (per coordinate) |
| Inner product bias | Present (bias grows with low bit-width) | Corrected by QJL residual stage |
| Theoretical bound | Empirically strong | Proven within ≈ 2.7× of Shannon optimum |

**Key distinction:** PolarQuant couples pairs of coordinates through
sin/cos operations. TurboQuant eliminates this coupling entirely —
it independently scalar-quantizes each rotated coordinate, preventing
error compounding and enabling the structured fast rotation.

---

## Algorithm: PolarQuant

Traditional quantization (INT8, INT4) requires scanning the vector for
`min`/`max` and storing per-block scale factors — 1–2 bits of metadata
overhead per element. PolarQuant removes this entirely by transforming
the data into a distribution where fixed, precomputed codebooks are
provably near-optimal.

### Step 1 — Random Orthogonal Rotation

```
v' = Π · v
```

A random orthogonal matrix Π (generated via QR decomposition of a Gaussian
matrix) rotates the input so every coordinate is approximately `N(0, 1/d)`,
regardless of the original input distribution. The quantizer becomes
data-oblivious.

> ⚡ **FWHT Breakthrough:** This crate implements a structured Fast 
> Walsh-Hadamard Transform (O(d log d)), which yields a **35× speedup** in
> decompression over the dense QR baseline, closing the throughput gap
> for real-time inference.

### Step 2 — Pairwise Polar Transform

Pairs of rotated coordinates are mapped to polar form:

```
(x₁, x₂)  →  (r, θ)    where  r = √(x₁² + x₂²),  θ = atan2(x₂, x₁)
```

Because the rotation made coordinates approximately Gaussian, radii follow
a predictable `Rayleigh(1/√d)` distribution and angles are uniform on
`(-π, π)` — independent of the original input vector.

> **Pairwise vs. recursive:** PolarQuant's paper describes a recursive
> polar transform (pairs → pairs-of-pairs → single final radius), which
> achieves slightly higher compression but compounds sin/cos errors across
> levels. This implementation uses independent pairwise transforms for
> correctness and simplicity. TurboQuant avoids both variants entirely.

### Step 3 — Lloyd-Max Scalar Quantization

Because the post-rotation distribution is universal and fully known
analytically, optimal codebooks are precomputed once via the Lloyd-Max
algorithm — separately for radii (Rayleigh) and angles (uniform). No
calibration data. No per-vector metadata beyond a single FP16 norm.

### Norm Handling

Real KV cache vectors have varying dynamic ranges. During compression,
the vector is L2-normalised internally and the original norm is stored as
a 2-byte `f16` at the head of the payload. Decompression rescales using
the stored norm. This is the source of the `†` overhead in the benchmark
ratios.

---

## What PolarQuant Does Not Solve (→ TurboQuant)

PolarQuant is MSE-optimal but its compressed vectors produce **biased
inner product estimates**. Attention scores are dot products (`q · k`),
so this bias directly affects output quality at low bit-widths.

TurboQuant's QJL stage corrects this:

1. Compute residual: `e = v − decompress(compress(v))`
2. Apply Hadamard JL projection: `p = H · e`
3. Store only the sign: `sgn(p)` — 1 bit per projected value

The result is an unbiased inner-product estimator at near-zero additional
storage cost, proven within ≈ 2.7× of the information-theoretic minimum
across all bit-widths and dimensions (Theorem 3, arXiv:2504.19874).

---

## Status

| Component | Status | Notes |
|---|---|---|
| Random orthogonal rotation (dense QR) | ✅ Done | Legacy baseline; O(d²) |
| Walsh-Hadamard rotation (O(d log d)) | ✅ Done | Structured rotation; 35× faster decompress |
| QJL residual stage (unbiased inner products) | ✅ Done | Accurate attention scores at low bit-widths |
| Pairwise polar transform | ✅ Done | |
| Lloyd-Max codebooks (Rayleigh + Uniform) | ✅ Done | |
| 3-bit and 4-bit packing | ✅ Done | |
| Per-vector FP16 norm storage | ✅ Done | |
| INT8 baseline | ✅ Done | |
| Criterion benchmark suite | ✅ Done | Including FWHT vs Dense comparison |
| Unit tests (40) | ✅ Done | Comprehensive regression suite |

---

## Project Structure

```
turboquant-rs/
├── Cargo.toml
├── src/
│   ├── polarquant.rs       # Rotation + pairwise polar transform
│   ├── quantize.rs         # Lloyd-Max codebooks (Rayleigh, uniform)
│   ├── int8_quant.rs       # INT8 baseline
│   └── metrics.rs          # MSE, compression ratio, cosine similarity
└── benches/
    └── compression.rs      # Criterion benchmark harness
```

---

## Usage

```bash
git clone https://github.com/IamRitwik/turboquant-rs.git
cd turboquant-rs
cargo run --example full_pipeline --release
```

### Plotting QJL Accuracy

```bash
python3 scripts/plot_qjl_accuracy.py
```

---

*This is an independent research implementation.*