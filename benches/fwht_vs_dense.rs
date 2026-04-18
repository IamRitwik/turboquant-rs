//! Criterion benchmark: FWHT O(d log d) vs Dense O(d²) rotation throughput.
//!
//! Run with: `cargo bench --bench fwht_vs_dense`
//! HTML report: `target/criterion/report/index.html`
//!
//! Benchmark groups:
//!   - rotation_strategy/dense_O(d²)/{128,256,512,1024}
//!   - rotation_strategy/fwht_O(dlogd)/{128,256,512,1024}
//!
//! This measures the compress throughput difference between the two rotation
//! strategies. FWHT replaces the O(d²) dense matrix-vector multiply with an
//! O(d log d) butterfly, yielding dramatically higher throughput at large d.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rand::prelude::*;
use rand::rngs::StdRng;
use half::f16;

use turboquant::PolarQuant;

const SEED: u64 = 42;

fn random_f16_vector(dim: usize, seed: u64) -> Vec<f16> {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.into_iter().map(|x| f16::from_f32(x / norm)).collect()
}

fn random_f32_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    v.into_iter().map(|x| x / norm).collect()
}

// ---------------------------------------------------------------------------
//  Compress throughput: Dense vs FWHT
// ---------------------------------------------------------------------------

fn bench_rotation_compress_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_strategy/compress_f32");

    // Only power-of-2 dimensions (FWHT requirement)
    let dims: &[usize] = &[128, 256, 512, 1024];

    for &dim in dims {
        let v = random_f32_vector(dim, SEED);
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("dense_O(d²)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_dense(d, 3, Some(SEED));
                b.iter(|| pq.compress(&v));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fwht_O(dlogd)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_fwht(d, 3, Some(SEED));
                b.iter(|| pq.compress(&v));
            },
        );
    }
    group.finish();
}

fn bench_rotation_compress_f16(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_strategy/compress_f16");

    let dims: &[usize] = &[128, 256, 512, 1024];

    for &dim in dims {
        let v = random_f16_vector(dim, SEED);
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("dense_O(d²)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_dense(d, 3, Some(SEED));
                b.iter(|| pq.compress_f16(&v));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fwht_O(dlogd)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_fwht(d, 3, Some(SEED));
                b.iter(|| pq.compress_f16(&v));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  Full round-trip: compress + decompress
// ---------------------------------------------------------------------------

fn bench_rotation_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_strategy/roundtrip");

    let dims: &[usize] = &[128, 256, 512, 1024];

    for &dim in dims {
        let v = random_f32_vector(dim, SEED);
        group.throughput(Throughput::Elements(1));

        group.bench_with_input(
            BenchmarkId::new("dense_O(d²)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_dense(d, 3, Some(SEED));
                b.iter(|| {
                    let c = pq.compress(&v);
                    pq.decompress(&c)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fwht_O(dlogd)", dim),
            &dim,
            |b, &d| {
                let pq = PolarQuant::new_fwht(d, 3, Some(SEED));
                b.iter(|| {
                    let c = pq.compress(&v);
                    pq.decompress(&c)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
//  Raw FWHT kernel: isolate just the transform
// ---------------------------------------------------------------------------

fn bench_raw_fwht_kernel(c: &mut Criterion) {
    use turboquant::FwhtRotation;

    let mut group = c.benchmark_group("fwht_kernel/apply");

    let dims: &[usize] = &[128, 256, 512, 1024];

    for &dim in dims {
        let v = random_f32_vector(dim, SEED);
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(
            BenchmarkId::new("fwht_apply", dim),
            &dim,
            |b, &d| {
                let fwht = FwhtRotation::new(d, Some(SEED));
                b.iter(|| {
                    let mut buf = v.clone();
                    fwht.apply(&mut buf);
                    buf
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rotation_compress_f32,
    bench_rotation_compress_f16,
    bench_rotation_roundtrip,
    bench_raw_fwht_kernel,
);
criterion_main!(benches);
