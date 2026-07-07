//! Streaming reservoir sampling with Algorithm L (Li, 1994).
//!
//! Maintains a uniform sample of size k from a stream of unknown (potentially large)
//! length, using O(k) memory regardless of stream size. Algorithm L achieves this
//! with O(k * (1 + log(N/k))) RNG calls instead of the O(N) calls required by
//! the classic Algorithm R.
//!
//! This example:
//! 1. Streams 1,000,000 items through a reservoir of size 100.
//! 2. Prints the reservoir contents and basic statistics.
//! 3. Runs 5,000 independent trials and verifies that each stream position is
//!    selected with approximately equal probability (uniform sampling).
//!
//! Run: cargo run --example streaming_reservoir

use drawset::ReservoirSampler;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let stream_len: usize = 1_000_000;
    let k: usize = 100;

    // --- Part 1: single pass over 1M items ---
    println!("Reservoir sampling (Algorithm L)");
    println!("  stream length: {stream_len}");
    println!("  reservoir size: {k}");
    println!("  memory: O({k}) regardless of stream length");
    println!();

    let mut sampler = ReservoirSampler::new(k);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for i in 0..stream_len {
        sampler.add_with_rng(i, &mut rng);
    }

    assert_eq!(sampler.samples().len(), k);
    assert_eq!(sampler.seen(), stream_len);

    let mut reservoir: Vec<usize> = sampler.samples().to_vec();
    reservoir.sort_unstable();

    println!("Reservoir (sorted, first 20 of {k}):");
    for (pos, &val) in reservoir.iter().enumerate().take(20) {
        println!("  [{pos:>3}] = {val}");
    }
    println!("  ...");
    println!();

    let min = reservoir.iter().min().copied().unwrap_or(0);
    let max = reservoir.iter().max().copied().unwrap_or(0);
    let mean: f64 = reservoir.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
    let expected_mean = (stream_len - 1) as f64 / 2.0;
    println!("Statistics:");
    println!("  min:           {min}");
    println!("  max:           {max}");
    println!("  mean:          {mean:.0}");
    println!("  expected mean: {expected_mean:.0}");
    println!();

    // --- Part 2: uniformity verification ---
    // Bin the stream into 10 buckets and count how often each bucket is represented.
    // Under uniform sampling, each bucket should get ~(k/10) selections per trial.
    let n_trials: usize = 5_000;
    let n_buckets: usize = 10;
    let bucket_size = stream_len / n_buckets;
    let mut bucket_counts = vec![0u64; n_buckets];

    for trial in 0..n_trials {
        let mut s = ReservoirSampler::new(k);
        let mut rng = ChaCha8Rng::seed_from_u64(trial as u64);
        for i in 0..stream_len {
            s.add_with_rng(i, &mut rng);
        }
        for &val in s.samples() {
            let bucket = (val / bucket_size).min(n_buckets - 1);
            bucket_counts[bucket] += 1;
        }
    }

    let expected_per_bucket = n_trials as f64 * k as f64 / n_buckets as f64;
    println!("Uniformity check ({n_trials} trials, {n_buckets} buckets):");
    println!(
        "{:>12} {:>12} {:>12} {:>10}",
        "bucket", "count", "expected", "ratio"
    );
    println!("{}", "-".repeat(50));

    let mut chi2: f64 = 0.0;
    for (b, &bc) in bucket_counts.iter().enumerate().take(n_buckets) {
        let lo = b * bucket_size;
        let hi = lo + bucket_size - 1;
        let count = bc as f64;
        let ratio = count / expected_per_bucket;
        chi2 += (count - expected_per_bucket).powi(2) / expected_per_bucket;
        println!(
            "{:>6}..{:<5} {:>12} {:>12.0} {:>10.4}",
            lo, hi, bc, expected_per_bucket, ratio
        );
    }

    println!();
    println!(
        "Chi-squared statistic: {chi2:.2} (df={}, expect ~{})",
        n_buckets - 1,
        n_buckets - 1
    );
    println!("Ratios near 1.0 confirm uniform sampling across the stream.");
}
