use drawset::gumbel::{gumbel_softmax, relaxed_topk_gumbel};
use drawset::gumbel_topk_sample_with_rng;
use drawset::reservoir::{ReservoirSampler, ReservoirSamplerR, WeightedReservoirSampler};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

proptest! {
    #[test]
    fn prop_reservoir_size_invariant(
        k in 0usize..20,
        items in prop::collection::vec(0u32..1000, 0..50)
    ) {
        let mut s = ReservoirSampler::new(k);
        for &item in &items {
            s.add(item);
        }

        let n = items.len();
        let expected_size = if k == 0 { 0 } else { std::cmp::min(n, k) };

        prop_assert_eq!(s.samples().len(), expected_size);
        prop_assert_eq!(s.seen(), n);
    }

    #[test]
    fn prop_reservoir_r_size_invariant(
        k in 0usize..20,
        items in prop::collection::vec(0u32..1000, 0..50)
    ) {
        let mut s = ReservoirSamplerR::new(k);
        for &item in &items {
            s.add(item);
        }

        let n = items.len();
        let expected_size = if k == 0 { 0 } else { std::cmp::min(n, k) };

        prop_assert_eq!(s.samples().len(), expected_size);
        prop_assert_eq!(s.seen(), n);
    }

    #[test]
    fn prop_weighted_reservoir_size_invariant(
        k in 0usize..20,
        items in prop::collection::vec(0u32..1000, 0..50)
    ) {
        let mut s = WeightedReservoirSampler::new(k);
        for &item in &items {
            s.add(item, 1.0).expect("weight ok");
        }

        let n = items.len();
        let expected_size = if k == 0 { 0 } else { std::cmp::min(n, k) };

        prop_assert_eq!(s.samples().len(), expected_size);
        prop_assert_eq!(s.seen(), n);
    }
}

proptest! {
    #[test]
    fn prop_gumbel_topk_invariants(
        logits in prop::collection::vec(-10.0f32..10.0, 1..20),
        k in 1usize..20
    ) {
        let k = std::cmp::min(k, logits.len());
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let idxs = gumbel_topk_sample_with_rng(&logits, k, &mut rng);

        prop_assert_eq!(idxs.len(), k);

        // Indices are in range and unique.
        let mut seen = std::collections::HashSet::new();
        for &i in &idxs {
            prop_assert!(i < logits.len());
            prop_assert!(seen.insert(i));
        }
    }
}

proptest! {
    #[test]
    fn prop_gumbel_softmax_is_distribution(
        logits in prop::collection::vec(-10.0f64..10.0, 1..30),
        temperature in 1e-3f64..10.0,
        scale in 0.0f64..10.0,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let p = gumbel_softmax(&logits, temperature, scale, &mut rng);

        prop_assert_eq!(p.len(), logits.len());
        prop_assert!(p.iter().all(|x| x.is_finite()));
        prop_assert!(p.iter().all(|&x| x >= 0.0));

        let sum: f64 = p.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-6, "sum was {}", sum);
    }

    #[test]
    fn prop_relaxed_topk_is_k_hot_relaxation(
        scores in prop::collection::vec(-10.0f64..10.0, 2..30),
        k in 1usize..30,
        temperature in 1e-3f64..10.0,
        scale in 0.0f64..10.0,
    ) {
        let k = std::cmp::min(k, scores.len());
        let mut rng = ChaCha8Rng::seed_from_u64(456);
        let m = relaxed_topk_gumbel(&scores, k, temperature, scale, &mut rng);

        prop_assert_eq!(m.len(), scores.len());
        prop_assert!(m.iter().all(|x| x.is_finite()));
        prop_assert!(m.iter().all(|&x| x >= 0.0));

        let sum: f64 = m.iter().sum();
        prop_assert!((sum - k as f64).abs() < 1e-4, "sum was {}", sum);

        // Each iteration contributes a probability vector in [0, 1] summing to 1,
        // so the accumulated mask is in [0, k].
        prop_assert!(m.iter().all(|&x| x <= k as f64 + 1e-6));
    }
}
