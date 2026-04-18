//! Criterion benchmark: QJL sketch_residual throughput across sketch dimensions.
//!
//! Run with: `cargo bench --bench qjl_accuracy`
//! HTML report: `target/criterion/report/index.html`
//!
//! Benchmark groups:
//!   - qjl/sketch_residual/{k=16,32,64,128,256}
//!   - qjl/estimate_correction/{k=16,32,64,128,256}
//!   - qjl/full_pipeline/{k=16,32,64,128,256}

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rand::prelude::*;
use rand::rngs::StdRng;

use turboquant::qjl::{QjlSketch, compute_residual};
use turboquant::PolarQuant;

const DIM: usize = 512;
const SEED: u64 = 42;

fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.into_iter().map(|x| x / norm).collect()
}

// ---------------------------------------------------------------------------
//  sketch_residual throughput
// ---------------------------------------------------------------------------

fn bench_qjl_sketch_residual(c: &mut Criterion) {
    let mut group = c.benchmark_group("qjl/sketch_residual");

    let sketch_dims: &[usize] = &[16, 32, 64, 128, 256];
    let residual = random_unit_vec(DIM, SEED);

    for &k in sketch_dims {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |b, &sk| {
                let qjl = QjlSketch::new(DIM, sk, Some(SEED));
                b.iter(|| qjl.sketch_residual(&residual));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  estimate_correction throughput
// ---------------------------------------------------------------------------

fn bench_qjl_estimate_correction(c: &mut Criterion) {
    let mut group = c.benchmark_group("qjl/estimate_correction");

    let sketch_dims: &[usize] = &[16, 32, 64, 128, 256];
    let query    = random_unit_vec(DIM, 1);
    let residual = random_unit_vec(DIM, 2);

    for &k in sketch_dims {
        let qjl = QjlSketch::new(DIM, k, Some(SEED));
        let sketch = qjl.sketch_residual(&residual);

        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |b, _| {
                b.iter(|| qjl.estimate_correction(&query, &sketch));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  Full pipeline: compress + sketch + estimate_inner_product
// ---------------------------------------------------------------------------

fn bench_qjl_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("qjl/full_pipeline");

    let sketch_dims: &[usize] = &[16, 32, 64, 128, 256];
    let pq = PolarQuant::new_fwht(DIM, 3, Some(SEED));

    let key_orig = random_unit_vec(DIM, 10);
    let query    = random_unit_vec(DIM, 11);

    // Pre-compress key
    let compressed = pq.compress(&key_orig);
    let k_approx   = pq.decompress(&compressed);
    let residual   = compute_residual(&key_orig, &k_approx);

    for &k in sketch_dims {
        let qjl = QjlSketch::new(DIM, k, Some(SEED));
        let sketch = qjl.sketch_residual(&residual);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("k", k),
            &k,
            |b, _| {
                b.iter(|| qjl.estimate_inner_product(&query, &k_approx, &sketch));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_qjl_sketch_residual,
    bench_qjl_estimate_correction,
    bench_qjl_full_pipeline,
);
criterion_main!(benches);
