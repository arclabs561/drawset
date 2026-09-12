//! Numeric Gumbel-Softmax relaxations at several temperatures.
//!
//! With the same seeded RNG for each call, every temperature sees the same
//! Gumbel noise. Lower temperatures then make one relaxed draw approach a hard
//! categorical draw. Across many randomized draws, the low-temperature mean
//! approaches the categorical probabilities, rather than a fixed one-hot vector.
//!
//! `drawset` returns plain `f64` values; it does not build an autodiff graph.
//! Use an autodiff framework's corresponding operation when gradients are needed.

use drawset::gumbel_softmax;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N_SAMPLES: usize = 1_000;

fn main() {
    let logits = [-1.0, -0.5, 0.0, 0.5, 2.0, 1.0];
    let temperatures = [5.0, 1.0, 0.1];

    println!("Logits: {logits:?}");
    println!("Categorical probabilities (softmax logits):");
    print_distribution(&softmax(&logits));

    for temperature in temperatures {
        let mut single_rng = ChaCha8Rng::seed_from_u64(42);
        let single = gumbel_softmax(&logits, temperature, 1.0, &mut single_rng);
        println!("\nOne relaxed draw at temperature {temperature} (same noise seed):");
        print_distribution(&single);

        let mean = average_relaxed_samples(&logits, temperature, N_SAMPLES);
        println!("Mean of {N_SAMPLES} randomized relaxed draws:");
        print_distribution(&mean);
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits.iter().map(|&logit| (logit - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

fn average_relaxed_samples(logits: &[f64], temperature: f64, n: usize) -> Vec<f64> {
    let mut mean = vec![0.0; logits.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..n {
        let sample = gumbel_softmax(logits, temperature, 1.0, &mut rng);
        for (mean, sample) in mean.iter_mut().zip(sample) {
            *mean += sample;
        }
    }

    for value in &mut mean {
        *value /= n as f64;
    }
    mean
}

fn print_distribution(values: &[f64]) {
    for (index, value) in values.iter().enumerate() {
        println!("  item {index}: {value:.4}");
    }
}
