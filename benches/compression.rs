//! Criterion benchmark suite for TurboQuant compression methods.
//!
//! Run with: `cargo bench`
//! HTML reports: `target/criterion/report/index.html`
//!
//! Benchmark groups:
//!   - compress/{int8,polarq_3bit,polarq_4bit}/d{128,256,512,1024}
//!   - decompress/{int8,polarq_3bit,polarq_4bit}/d{128,256,512,1024}

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rand::prelude::*;
use rand::rngs::StdRng;
use half::f16;

use turboquant::{Int8Quant, PolarQuant};

const DIMS: &[usize] = &[128, 256, 512, 1024];
const SEED: u64 = 42;

fn random_vector(dim: usize, seed: u64) -> Vec<f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f16> = (0..dim)
        .map(|_| f16::from_f32(rng.gen_range(-1.0_f32..1.0_f32)))
        .collect();
    
    // L2-normalize in f32 for stability
    let v_f32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
    let norm = v_f32.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v_f32.into_iter().map(|x| f16::from_f32(x / norm)).collect()
}

// ---------------------------------------------------------------------------
//  INT8 Compression Benchmarks
// ---------------------------------------------------------------------------

fn bench_int8_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress/int8");
    let q = Int8Quant;

    for &dim in DIMS {
        let v_f16 = random_vector(dim, SEED);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| b.iter(|| q.compress_f16(&v_f16)),
        );
    }
    group.finish();
}

fn bench_int8_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress/int8");
    let q = Int8Quant;

    for &dim in DIMS {
        let v_f16 = random_vector(dim, SEED);
        let compressed = q.compress_f16(&v_f16);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| b.iter(|| q.decompress_f16(&compressed)),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  PolarQuant-4bit Compression Benchmarks
// ---------------------------------------------------------------------------

fn bench_polarq4_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress/polarq_4bit");

    for &dim in DIMS {
        let pq = PolarQuant::new(dim, 4, Some(SEED));
        let v_f16 = random_vector(dim, SEED);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| b.iter(|| pq.compress_f16(&v_f16)),
        );
    }
    group.finish();
}

fn bench_polarq4_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress/polarq_4bit");

    for &dim in DIMS {
        let pq = PolarQuant::new(dim, 4, Some(SEED));
        let v_f16 = random_vector(dim, SEED);
        let compressed = pq.compress_f16(&v_f16);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _: &usize| b.iter(|| pq.decompress_f16(&compressed)),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  PolarQuant-3bit Compression Benchmarks
// ---------------------------------------------------------------------------

fn bench_polarq3_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress/polarq_3bit");

    for &dim in DIMS {
        let pq = PolarQuant::new(dim, 3, Some(SEED));
        let v_f16 = random_vector(dim, SEED);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _| b.iter(|| pq.compress_f16(&v_f16)),
        );
    }
    group.finish();
}

fn bench_polarq3_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress/polarq_3bit");

    for &dim in DIMS {
        let pq = PolarQuant::new(dim, 3, Some(SEED));
        let v_f16 = random_vector(dim, SEED);
        let compressed = pq.compress_f16(&v_f16);
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &dim,
            |b, _: &usize| b.iter(|| pq.decompress_f16(&compressed)),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  End-to-End Pipeline Benchmarks (compress + decompress together)
// ---------------------------------------------------------------------------

fn bench_pipeline_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/end_to_end");

    for &dim in DIMS {
        let q = Int8Quant;
        let pq4 = PolarQuant::new(dim, 4, Some(SEED));
        let pq3 = PolarQuant::new(dim, 3, Some(SEED));
        let v_f16 = random_vector(dim, SEED);

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("int8", dim),
            &dim,
            |b, _| b.iter(|| { let c = q.compress_f16(&v_f16); q.decompress_f16(&c) }),
        );

        group.bench_with_input(
            BenchmarkId::new("polarq_4bit", dim),
            &dim,
            |b, _| b.iter(|| { let c = pq4.compress_f16(&v_f16); pq4.decompress_f16(&c) }),
        );

        group.bench_with_input(
            BenchmarkId::new("polarq_3bit", dim),
            &dim,
            |b, _| b.iter(|| { let c = pq3.compress_f16(&v_f16); pq3.decompress_f16(&c) }),
        );
    }
    group.finish();
}


criterion_group!(
    benches,
    bench_int8_compress,
    bench_int8_decompress,
    bench_polarq4_compress,
    bench_polarq4_decompress,
    bench_polarq3_compress,
    bench_polarq3_decompress,
    bench_pipeline_e2e,
);
criterion_main!(benches);
