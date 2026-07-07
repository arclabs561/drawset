//! Weighted candidate selection: Gumbel-top-k vs weighted reservoir.
//!
//! Both are "without replacement", but they induce different distributions for k>1.
//!
//! - Gumbel-top-k samples from the Plackett-Luce distribution: items are drawn
//!   sequentially with probabilities proportional to exp(logit_i) among remaining items.
//! - Weighted reservoir (A-Res, Efraimidis-Spirakis 2006) assigns each item a key
//!   u^(1/w_i) and keeps the top-k keys. For k>1 the resulting joint distribution
//!   differs from Plackett-Luce.
//!
//! This example draws 10,000 samples from each method and prints empirical selection
//! frequencies, making the distributional difference visible.

use drawset::{gumbel_topk_sample_with_rng, WeightedReservoirSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A weight vector shaped like PPR (personalized PageRank) scores:
    // many small weights, few large ones.
    let n = 20;
    let weights: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + (i as f64)).powf(1.3)).collect();

    let eps = 1e-12f64;
    let logits: Vec<f32> = weights.iter().map(|&w| (w + eps).ln() as f32).collect();

    let k = 5usize;
    let n_trials = 10_000usize;

    // -- Single draw (deterministic, for inspection) --
    let mut rng_g = ChaCha8Rng::seed_from_u64(7);
    let pick_g = gumbel_topk_sample_with_rng(&logits, k, &mut rng_g);

    let mut rng_r = ChaCha8Rng::seed_from_u64(7);
    let mut rs = WeightedReservoirSampler::new(k);
    for (i, &w) in weights.iter().enumerate() {
        rs.add_with_rng(i, w, &mut rng_r)?;
    }
    let pick_r: Vec<usize> = rs.samples().to_vec();

    println!("weights (first 10 of {n}):");
    for (i, w) in weights.iter().enumerate().take(10) {
        println!("  i={i:2}  w={w:.6}");
    }
    println!();
    println!("Single draw (seed=7, k={k}):");
    println!("  Gumbel-top-k (Plackett-Luce): {pick_g:?}");
    println!("  Weighted reservoir (A-Res):    {pick_r:?}");

    // -- Frequency table over many draws --
    let mut freq_gumbel = vec![0u64; n];
    let mut freq_reservoir = vec![0u64; n];

    for trial in 0..n_trials {
        let seed = trial as u64;

        // Gumbel-top-k
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let selected = gumbel_topk_sample_with_rng(&logits, k, &mut rng);
        for idx in selected {
            freq_gumbel[idx] += 1;
        }

        // Weighted reservoir
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut sampler = WeightedReservoirSampler::new(k);
        for (i, &w) in weights.iter().enumerate() {
            sampler.add_with_rng(i, w, &mut rng)?;
        }
        for &idx in sampler.samples() {
            freq_reservoir[idx] += 1;
        }
    }

    println!();
    println!("Empirical selection frequency ({n_trials} trials, k={k}, n={n}):");
    println!(
        "{:>5} {:>8} {:>12} {:>12} {:>10}",
        "idx", "weight", "gumbel_freq", "reserv_freq", "delta"
    );
    println!("{}", "-".repeat(51));
    for i in 0..n {
        let gf = freq_gumbel[i] as f64 / n_trials as f64;
        let rf = freq_reservoir[i] as f64 / n_trials as f64;
        let delta = gf - rf;
        println!(
            "{:>5} {:>8.4} {:>12.4} {:>12.4} {:>+10.4}",
            i, weights[i], gf, rf, delta
        );
    }

    println!();
    println!("Observation: for k>1, Gumbel-top-k (Plackett-Luce) and A-Res produce");
    println!("different marginal inclusion probabilities, especially for mid-weight items.");
    println!("For k=1 both reduce to sampling proportional to w_i.");

    Ok(())
}
