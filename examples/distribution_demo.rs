//! Visual demonstration of sampling distributions.
//!
//! Runs three samplers and prints ASCII histograms so the distribution shape
//! is visible at a glance.
//!
//! Run: cargo run --example distribution_demo

use kuji::{gumbel_topk_sample_with_rng, ReservoirSampler, WeightedReservoirSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const TRIALS: usize = 10_000;
const N: usize = 100;
const K: usize = 10;
const BUCKETS: usize = 10;

fn histogram(counts: &[u64], label: &str) {
    let max = *counts.iter().max().unwrap_or(&1);
    let bar_max = 40;
    println!("{label}");
    for (i, &c) in counts.iter().enumerate() {
        let lo = i * (N / BUCKETS);
        let hi = lo + (N / BUCKETS) - 1;
        let bar_len = (c as usize * bar_max) / max as usize;
        println!("  [{lo:>2}..{hi:>2}] {c:>5} {}", "#".repeat(bar_len));
    }
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. Uniform reservoir (Algorithm L) ---
    let mut buckets = [0u64; BUCKETS];
    for trial in 0..TRIALS {
        let mut rng = ChaCha8Rng::seed_from_u64(trial as u64);
        let mut sampler = ReservoirSampler::new(K);
        for i in 0..N {
            sampler.add_with_rng(i, &mut rng);
        }
        for &v in sampler.samples() {
            buckets[v / (N / BUCKETS)] += 1;
        }
    }
    histogram(
        &buckets,
        "Reservoir sampling (Algorithm L) -- uniform stream, k=10:",
    );

    // --- 2. Weighted reservoir (A-Res) with power-law weights ---
    let weights: Vec<f64> = (0..N).map(|i| 1.0 / (1.0 + i as f64).powf(1.5)).collect();
    let mut buckets = [0u64; BUCKETS];
    for trial in 0..TRIALS {
        let mut rng = ChaCha8Rng::seed_from_u64(trial as u64);
        let mut sampler = WeightedReservoirSampler::new(K);
        for (i, &w) in weights.iter().enumerate() {
            sampler.add_with_rng(i, w, &mut rng)?;
        }
        for &v in sampler.samples() {
            buckets[v / (N / BUCKETS)] += 1;
        }
    }
    histogram(
        &buckets,
        "Weighted reservoir (A-Res) -- power-law weights w(i)=1/(1+i)^1.5:",
    );

    // --- 3. Gumbel-top-k single draw ---
    let logits: Vec<f32> = weights.iter().map(|&w| (w + 1e-12).ln() as f32).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let selected = gumbel_topk_sample_with_rng(&logits, K, &mut rng);
    println!("Gumbel-top-k single draw (k={K}, seed=42): {selected:?}");

    Ok(())
}
