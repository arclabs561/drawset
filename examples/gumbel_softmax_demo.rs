//! Gumbel-Softmax for differentiable subset selection.
//!
//! The Gumbel-Softmax trick (Jang et al. 2017, Maddison et al. 2017) produces
//! soft one-hot vectors that approximate categorical samples while remaining
//! differentiable. Temperature controls the trade-off: high temperature yields
//! near-uniform distributions; low temperature converges to hard one-hot.
//!
//! This example shows three temperature regimes on the same logits and
//! averages over multiple draws to make the convergence visible.

use kuji::gumbel_softmax;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const N_ITEMS: usize = 6;
const N_SAMPLES: usize = 1_000;

fn main() {
    // Logits: item 4 is strongly preferred, item 5 is second.
    let logits: [f64; N_ITEMS] = [-1.0, -0.5, 0.0, 0.5, 2.0, 1.0];

    println!("Logits ({N_ITEMS} candidates):");
    for (i, &l) in logits.iter().enumerate() {
        println!("  item {i}: {l:+.1}");
    }

    // Reference: the true softmax distribution (temperature=1, no noise).
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let z: f64 = exps.iter().sum();
    let true_probs: Vec<f64> = exps.iter().map(|e| e / z).collect();

    println!("\nTrue softmax (for reference):");
    print_distribution(&true_probs);

    // Three temperature regimes.
    let temperatures = [
        (5.0, "high (5.0) -- near-uniform, high entropy"),
        (1.0, "medium (1.0) -- peaked but smooth"),
        (0.1, "low (0.1) -- near-hard one-hot"),
    ];

    for (tau, label) in temperatures {
        let avg = average_gumbel_softmax(&logits, tau, N_SAMPLES);
        println!("\nTemperature {label}:");
        print_distribution(&avg);
    }

    println!();
    println!("As temperature decreases, the average distribution sharpens toward");
    println!("the highest-logit item (item 4). At tau=0.1, it is nearly one-hot.");
    println!();
    println!("In a neural network, each sample is a soft weight vector over candidates.");
    println!("Multiplying these weights by candidate embeddings gives a differentiable");
    println!("\"selection\" -- gradients flow through the softmax back to the logits,");
    println!("allowing gradient-based optimization of which item to select.");
}

/// Draw `n` Gumbel-Softmax samples and return the element-wise mean.
fn average_gumbel_softmax(logits: &[f64], temperature: f64, n: usize) -> Vec<f64> {
    let mut acc = vec![0.0_f64; logits.len()];
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..n {
        let sample = gumbel_softmax(logits, temperature, 1.0, &mut rng);
        for (a, s) in acc.iter_mut().zip(sample.iter()) {
            *a += s;
        }
    }

    for a in &mut acc {
        *a /= n as f64;
    }
    acc
}

/// Print a probability distribution as a labeled bar chart.
fn print_distribution(probs: &[f64]) {
    let bar_width = 40;
    let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
    for (i, &p) in probs.iter().enumerate() {
        let len = if max_p > 0.0 {
            ((p / max_p) * bar_width as f64).round() as usize
        } else {
            0
        };
        let bar: String = "#".repeat(len);
        println!("  item {i}: {p:.4}  {bar}");
    }
}
