//! Weighted candidate selection: Gumbel-top-k vs weighted reservoir.
//!
//! Both are “without replacement”, but they induce different distributions for \(k>1\).
//! For \(k=1\), using logits \(\log w_i\), Gumbel-max samples index \(i\) with probability
//! proportional to \(w_i\).

use kuji::{gumbel_topk_sample_with_rng, WeightedReservoirSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A toy-ish weight vector, but shaped like what you get from PPR:
    // many small weights, few big ones.
    let weights: Vec<f64> = (0..50)
        .map(|i| 1.0 / (1.0 + (i as f64)).powf(1.3))
        .collect();

    // Logits for Gumbel-top-k.
    let eps = 1e-12f64;
    let logits: Vec<f32> = weights.iter().map(|&w| (w + eps).ln() as f32).collect();

    let k = 10usize;

    let mut rng_g = ChaCha8Rng::seed_from_u64(7);
    let pick_g = gumbel_topk_sample_with_rng(&logits, k, &mut rng_g);

    let mut rng_r = ChaCha8Rng::seed_from_u64(7);
    let mut rs = WeightedReservoirSampler::new(k);
    for (i, &w) in weights.iter().enumerate() {
        // A-Res requires w>0.
        rs.add_with_rng(i, w, &mut rng_r)?;
    }
    let pick_r: Vec<usize> = rs.samples().iter().copied().collect();

    println!("weights[0..10]:");
    for i in 0..10 {
        println!("  i={i:2}  w={:.6}", weights[i]);
    }
    println!();
    println!("gumbel-top-k (Plackett–Luce) indices: {pick_g:?}");
    println!("weighted reservoir (A-Res) indices:  {pick_r:?}");

    Ok(())
}
