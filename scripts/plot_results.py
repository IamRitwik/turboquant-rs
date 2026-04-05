#!/usr/bin/env python3
"""
plot_results.py — Generate benchmark visualization charts for LinkedIn carousel.

Usage:
    python3 scripts/plot_results.py

Reads: results/benchmark_results.csv
Outputs:
    results/chart_compression_ratio.png
    results/chart_mse.png
    results/chart_throughput_compress.png
    results/chart_throughput_decompress.png
    results/chart_cosine_sim.png
"""

import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Run: pip3 install matplotlib numpy")
    sys.exit(1)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT_1  = "#58a6ff"   # blue  — PolarQ-4bit
ACCENT_2  = "#3fb950"   # green — PolarQ-3bit
ACCENT_3  = "#f0883e"   # amber — INT8
GRID_CLR  = "#21262d"
TEXT_CLR  = "#e6edf3"
SUBTITLE  = "#8b949e"

METHOD_COLORS = {
    "INT8":        ACCENT_3,
    "PolarQ-4bit": ACCENT_1,
    "PolarQ-3bit": ACCENT_2,
}

PLT_STYLE = {
    "figure.facecolor":   DARK_BG,
    "axes.facecolor":     CARD_BG,
    "axes.edgecolor":     GRID_CLR,
    "axes.labelcolor":    TEXT_CLR,
    "xtick.color":        SUBTITLE,
    "ytick.color":        SUBTITLE,
    "text.color":         TEXT_CLR,
    "grid.color":         GRID_CLR,
    "grid.linewidth":     0.6,
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.facecolor":   CARD_BG,
    "legend.edgecolor":   GRID_CLR,
}

def load_csv(path="results/benchmark_results.csv"):
    results = []
    with open(path) as f:
        for row in csv.DictReader(f):
            results.append({
                "method":          row["method"],
                "dim":             int(row["dim"]),
                "bits":            float(row["bits_per_elem"]),
                "ratio":           float(row["ratio"]),
                "mse":             float(row["mse"]),
                "cosine_sim":      float(row["cosine_sim"]),
                "compress_mvps":   float(row["compress_mvps"]),
                "decompress_mvps": float(row["decompress_mvps"]),
                "latency_us":      float(row["latency_us"]),
            })
    return results

def grouped_bar(ax, dims, method_vals, ylabel, title, fmt="{:.2f}", log_scale=False):
    x = np.arange(len(dims))
    methods = list(method_vals.keys())
    n = len(methods)
    width = 0.25

    bars = []
    for i, method in enumerate(methods):
        offset = (i - n / 2 + 0.5) * width
        vals = [method_vals[method].get(d, 0) for d in dims]
        b = ax.bar(x + offset, vals, width, color=METHOD_COLORS[method],
                   alpha=0.85, label=method, zorder=3, edgecolor=DARK_BG, linewidth=0.5)
        bars.append(b)

        # Value labels on bars
        for bar, val in zip(b, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * (1.04 if not log_scale else 1.3),
                        fmt.format(val),
                        ha="center", va="bottom", fontsize=7.5,
                        color=METHOD_COLORS[method], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in dims], fontsize=10)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def add_watermark(fig):
    fig.text(0.99, 0.01, "turboquant-rs · M1 MacBook Air · arXiv:2504.19874",
             ha="right", va="bottom", fontsize=7, color=SUBTITLE, alpha=0.7)

def save(fig, name):
    os.makedirs("results", exist_ok=True)
    path = f"results/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  ✓  Saved: {path}")

# ── Chart 1: Compression Ratio ─────────────────────────────────────────────────
def chart_compression_ratio(data):
    plt.rcParams.update(PLT_STYLE)
    dims = sorted(set(r["dim"] for r in data))
    methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]

    method_vals = {m: {} for m in methods}
    for r in data:
        if r["method"] in methods:
            method_vals[r["method"]][r["dim"]] = r["ratio"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    grouped_bar(ax, dims, method_vals,
                ylabel="Compression Ratio (vs FP32)",
                title="Compression Ratio — PolarQuant vs INT8",
                fmt="{:.1f}x")
    ax.axhline(1.0, color=SUBTITLE, linewidth=0.8, linestyle="--", alpha=0.5)
    add_watermark(fig)
    save(fig, "chart_compression_ratio.png")
    plt.close(fig)

# ── Chart 2: MSE ───────────────────────────────────────────────────────────────
def chart_mse(data):
    plt.rcParams.update(PLT_STYLE)
    dims = sorted(set(r["dim"] for r in data))
    methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]

    method_vals = {m: {} for m in methods}
    for r in data:
        if r["method"] in methods:
            method_vals[r["method"]][r["dim"]] = r["mse"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    grouped_bar(ax, dims, method_vals,
                ylabel="Mean Squared Error (lower = better)",
                title="Reconstruction Error (MSE) — Lower Is Better",
                fmt="{:.4f}", log_scale=False)
    add_watermark(fig)
    save(fig, "chart_mse.png")
    plt.close(fig)

# ── Chart 3: Compress Throughput ───────────────────────────────────────────────
def chart_compress_throughput(data):
    plt.rcParams.update(PLT_STYLE)
    dims = sorted(set(r["dim"] for r in data))
    methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]

    method_vals = {m: {} for m in methods}
    for r in data:
        if r["method"] in methods:
            method_vals[r["method"]][r["dim"]] = r["compress_mvps"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    grouped_bar(ax, dims, method_vals,
                ylabel="Throughput (Mvec/s — higher = faster)",
                title="Compression Throughput on M1 (Mvec/s)",
                fmt="{:.3f}")
    add_watermark(fig)
    save(fig, "chart_throughput_compress.png")
    plt.close(fig)

# ── Chart 4: Decompress Throughput ─────────────────────────────────────────────
def chart_decompress_throughput(data):
    plt.rcParams.update(PLT_STYLE)
    dims = sorted(set(r["dim"] for r in data))
    methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]

    method_vals = {m: {} for m in methods}
    for r in data:
        if r["method"] in methods:
            method_vals[r["method"]][r["dim"]] = r["decompress_mvps"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    grouped_bar(ax, dims, method_vals,
                ylabel="Throughput (Mvec/s — higher = faster)",
                title="Decompression Throughput on M1 (Mvec/s)",
                fmt="{:.3f}")
    add_watermark(fig)
    save(fig, "chart_throughput_decompress.png")
    plt.close(fig)

# ── Chart 5: Cosine Similarity ─────────────────────────────────────────────────
def chart_cosine_sim(data):
    plt.rcParams.update(PLT_STYLE)
    dims = sorted(set(r["dim"] for r in data))
    methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]

    method_vals = {m: {} for m in methods}
    for r in data:
        if r["method"] in methods:
            method_vals[r["method"]][r["dim"]] = r["cosine_sim"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    grouped_bar(ax, dims, method_vals,
                ylabel="Cosine Similarity (higher = better, 1.0 = perfect)",
                title="Inner-Product Preservation (Cosine Similarity)",
                fmt="{:.4f}")
    ax.axhline(1.0, color="#ff7b72", linewidth=0.8, linestyle="--", alpha=0.6,
               label="Perfect (1.0)")
    ax.set_ylim(0.9, 1.01)
    add_watermark(fig)
    save(fig, "chart_cosine_sim.png")
    plt.close(fig)

# ── Chart 6: Summary Radar-style table comparison for d=512 ──────────────────
def chart_summary_comparison(data):
    plt.rcParams.update(PLT_STYLE)

    d512 = [r for r in data if r["dim"] == 512]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle("d=512 Summary: INT8 vs PolarQuant on M1 MacBook Air",
                 fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.02)

    metrics = [
        ("ratio",        "Compression Ratio", "{:.2f}x", False),
        ("mse",          "MSE (lower=better)", "{:.5f}", False),
        ("compress_mvps","Compress (Mvec/s)",  "{:.4f}", False),
    ]

    for ax, (key, label, fmt, _) in zip(axes, metrics):
        methods = ["INT8", "PolarQ-4bit", "PolarQ-3bit"]
        vals = []
        for m in methods:
            row = next((r for r in d512 if r["method"] == m), None)
            vals.append(row[key] if row else 0)

        colors = [METHOD_COLORS[m] for m in methods]
        bars = ax.bar(methods, vals, color=colors, alpha=0.85,
                      edgecolor=DARK_BG, linewidth=0.5, zorder=3)
        ax.set_title(label, fontweight="bold", pad=8)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.03,
                    fmt.format(val),
                    ha="center", va="bottom", fontsize=9,
                    color=TEXT_CLR, fontweight="bold")

    plt.tight_layout()
    add_watermark(fig)
    save(fig, "chart_summary_d512.png")
    plt.close(fig)


if __name__ == "__main__":
    csv_path = "results/benchmark_results.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run: cargo run --example demo --release")
        sys.exit(1)

    print("\n  turboquant-rs · Generating benchmark charts...")
    data = load_csv(csv_path)

    chart_compression_ratio(data)
    chart_mse(data)
    chart_compress_throughput(data)
    chart_decompress_throughput(data)
    chart_cosine_sim(data)
    chart_summary_comparison(data)

    print("\n  ✓ All charts saved to results/\n")
