//! Sample a stream of unknown length with bounded memory.
//!
//! This makes one pass over one million items while retaining a uniform sample
//! of 100 items. The reservoir holds O(k) items regardless of stream length.

use drawset::ReservoirSampler;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let stream_len = 1_000_000;
    let sample_size = 100;
    let mut sampler = ReservoirSampler::new(sample_size);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for item in 0..stream_len {
        sampler.add_with_rng(item, &mut rng);
    }

    let mut sample = sampler.samples().to_vec();
    sample.sort_unstable();

    println!("Observed {} stream items.", sampler.seen());
    println!("Retained {} uniformly sampled items.", sample.len());
    println!("First 10 sorted sample items: {:?}", &sample[..10]);
}
