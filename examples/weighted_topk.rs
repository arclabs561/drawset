//! Weighted sampling without replacement: Gumbel-top-k and A-Res.
//!
//! With `logits[i] = ln(weights[i])`, Gumbel-top-k and weighted reservoir
//! sampling (A-Res) draw the same distribution over unordered size-k subsets.
//! Choose the API that fits the input you have:
//!
//! - Gumbel-top-k takes a complete slice of logits and returns ranked indices.
//! - A-Res accepts `(item, weight)` pairs one at a time and keeps only `k` items.
//!
//! Their random-number transformations and output order differ, so giving both
//! calls the same RNG seed does not guarantee identical returned arrays.

use drawset::{gumbel_topk_sample_with_rng, WeightedReservoirSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let weights: [f64; 5] = [8.0, 4.0, 2.0, 1.0, 0.5];
    let logits: Vec<f32> = weights.iter().map(|weight| weight.ln() as f32).collect();
    let k = 2;

    let mut gumbel_rng = ChaCha8Rng::seed_from_u64(7);
    let ranked = gumbel_topk_sample_with_rng(&logits, k, &mut gumbel_rng);

    let mut reservoir_rng = ChaCha8Rng::seed_from_u64(7);
    let mut reservoir = WeightedReservoirSampler::new(k);
    for (item, &weight) in weights.iter().enumerate() {
        reservoir.add_with_rng(item, weight, &mut reservoir_rng)?;
    }

    println!("Weights: {weights:?}");
    println!("Gumbel-top-k ranked indices: {ranked:?}");
    println!("A-Res stream sample:          {:?}", reservoir.samples());
    println!();
    println!("Both methods have the same subset law when logits = ln(weights).");
    println!("Use Gumbel-top-k for an in-memory score slice; use A-Res for a stream.");

    Ok(())
}
