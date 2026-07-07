use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use drawset::reservoir::{ReservoirSampler, ReservoirSamplerR, WeightedReservoirSampler};
use drawset::{gumbel_max_sample, gumbel_topk_sample};

fn bench_reservoir_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("reservoir");

    // Algorithm L should be fast even for large N
    let sizes = [1_000, 10_000, 100_000];
    let k = 100;

    for &size in &sizes {
        group.bench_function(format!("alg_l_n{}_k{}", size, k), |b| {
            b.iter(|| {
                let mut sampler = ReservoirSampler::new(k);
                for i in 0..size {
                    sampler.add(black_box(i));
                }
                black_box(sampler.samples());
            })
        });
    }

    for &size in &sizes {
        group.bench_function(format!("alg_r_n{}_k{}", size, k), |b| {
            b.iter(|| {
                let mut sampler = ReservoirSamplerR::new(k);
                for i in 0..size {
                    sampler.add(black_box(i));
                }
                black_box(sampler.samples());
            })
        });
    }
    group.finish();
}

fn bench_weighted_reservoir(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_reservoir");

    let sizes = [1_000, 10_000, 100_000];
    let k = 100;

    for &size in &sizes {
        group.bench_function(format!("a_res_n{}_k{}", size, k), |b| {
            b.iter(|| {
                let mut sampler = WeightedReservoirSampler::new(k);
                for i in 0..size {
                    if sampler.add(black_box(i), 1.0).is_err() {
                        return;
                    }
                }
                black_box(sampler.samples());
            })
        });
    }
    group.finish();
}

fn bench_gumbel_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel");
    let sizes = [10, 100, 1000];

    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| i as f32).collect();
        group.bench_function(format!("sample_logits_{}", size), |b| {
            b.iter(|| {
                gumbel_max_sample(black_box(&logits));
            })
        });
    }
    group.finish();
}

fn bench_gumbel_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("gumbel_topk");
    let sizes = [100, 1000];
    let k = 10;

    for &size in &sizes {
        let logits: Vec<f32> = (0..size).map(|i| i as f32).collect();
        group.bench_function(format!("topk{}_logits_{}", k, size), |b| {
            b.iter(|| {
                gumbel_topk_sample(black_box(&logits), black_box(k));
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_reservoir_sampling,
    bench_weighted_reservoir,
    bench_gumbel_max,
    bench_gumbel_topk
);
criterion_main!(benches);
