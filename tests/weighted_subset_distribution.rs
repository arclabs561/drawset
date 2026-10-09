//! A-Res (Efraimidis & Spirakis 2006) and Gumbel-top-k (Kool et al. 2019)
//! both draw an ordered sample without replacement with probability
//! proportional to weight at each step. Their unordered subsets therefore
//! follow the same distribution. This test checks each sampler against the
//! exact subset probabilities with a chi-square goodness-of-fit test, and
//! checks that seeded runs repeat.

use drawset::{gumbel_topk_sample_with_rng, WeightedReservoirSampler};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const WEIGHTS: [f64; 4] = [8.0, 4.0, 2.0, 1.0];
const DRAWS: usize = 20_000;
// Chi-square critical value for 5 degrees of freedom at p = 0.001.
const CHI2_CRIT_5DF: f64 = 20.515;

/// Index of the unordered pair {i, j}, i < j, among the six pairs of 0..4.
fn pair_index(mut i: usize, mut j: usize) -> usize {
    if i > j {
        std::mem::swap(&mut i, &mut j);
    }
    match (i, j) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (1, 2) => 3,
        (1, 3) => 4,
        (2, 3) => 5,
        _ => unreachable!("not a pair of distinct indices in 0..4"),
    }
}

/// P({i, j}) for successive sampling proportional to weight, k = 2.
fn exact_pair_probabilities() -> [f64; 6] {
    let total: f64 = WEIGHTS.iter().sum();
    let mut p = [0.0; 6];
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                p[pair_index(i, j)] += WEIGHTS[i] / total * WEIGHTS[j] / (total - WEIGHTS[i]);
            }
        }
    }
    p
}

fn chi_square(counts: &[usize; 6]) -> f64 {
    exact_pair_probabilities()
        .iter()
        .zip(counts)
        .map(|(&p, &c)| {
            let expected = p * DRAWS as f64;
            (c as f64 - expected).powi(2) / expected
        })
        .sum()
}

fn a_res_counts(seed: u64) -> [usize; 6] {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut counts = [0usize; 6];
    for _ in 0..DRAWS {
        let mut sampler = WeightedReservoirSampler::new(2);
        for (item, &w) in WEIGHTS.iter().enumerate() {
            sampler.add_with_rng(item, w, &mut rng).unwrap();
        }
        let s = sampler.samples();
        counts[pair_index(s[0], s[1])] += 1;
    }
    counts
}

fn gumbel_counts(seed: u64) -> [usize; 6] {
    let logits: Vec<f32> = WEIGHTS.iter().map(|&w| (w as f32).ln()).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut counts = [0usize; 6];
    for _ in 0..DRAWS {
        let s = gumbel_topk_sample_with_rng(&logits, 2, &mut rng);
        counts[pair_index(s[0], s[1])] += 1;
    }
    counts
}

#[test]
fn a_res_subset_frequencies_match_exact_probabilities() {
    let counts = a_res_counts(7);
    let chi2 = chi_square(&counts);
    assert!(
        chi2 < CHI2_CRIT_5DF,
        "A-Res counts {counts:?}, chi2 {chi2:.2}"
    );
}

#[test]
fn gumbel_topk_subset_frequencies_match_exact_probabilities() {
    let counts = gumbel_counts(11);
    let chi2 = chi_square(&counts);
    assert!(
        chi2 < CHI2_CRIT_5DF,
        "Gumbel counts {counts:?}, chi2 {chi2:.2}"
    );
}

#[test]
fn seeded_weighted_samplers_repeat() {
    assert_eq!(a_res_counts(3), a_res_counts(3));
    assert_eq!(gumbel_counts(3), gumbel_counts(3));
}
