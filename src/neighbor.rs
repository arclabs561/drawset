//! Graph neighbor sampling.
//!
//! Provides utilities for sampling neighbors from a graph, useful for
//! GNN training (GraphSAGE) and random walks (Node2Vec).

use rand::prelude::*;
#[cfg(test)]
use std::collections::HashSet;

/// Sampler for graph neighborhoods.
///
/// # Examples
///
/// ```
/// use kuji::NeighborSampler;
///
/// let sampler = NeighborSampler::new().with_seed(42);
/// let neighbors = vec![10, 20, 30, 40, 50];
/// let samples = sampler.sample_uniform_with_replacement(&neighbors, 3);
/// assert_eq!(samples.len(), 3);
/// ```
pub struct NeighborSampler {
    seed: Option<u64>,
}

impl Default for NeighborSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborSampler {
    /// Create a new neighbor sampler.
    ///
    /// # Examples
    ///
    /// ```
    /// use kuji::NeighborSampler;
    ///
    /// let sampler = NeighborSampler::new();
    /// ```
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Set random seed for deterministic sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use kuji::NeighborSampler;
    ///
    /// let sampler = NeighborSampler::new().with_seed(123);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sample `k` neighbors uniformly with replacement.
    ///
    /// # Arguments
    ///
    /// * `neighbors`: Slice of neighbor IDs
    /// * `k`: Number of samples to draw
    ///
    /// # Examples
    ///
    /// ```
    /// use kuji::NeighborSampler;
    ///
    /// let sampler = NeighborSampler::new().with_seed(42);
    /// let neighbors = vec![1, 2, 3];
    /// let samples = sampler.sample_uniform_with_replacement(&neighbors, 5);
    /// assert_eq!(samples.len(), 5);
    /// for s in &samples {
    ///     assert!(neighbors.contains(s));
    /// }
    /// ```
    pub fn sample_uniform_with_replacement<T: Clone>(&self, neighbors: &[T], k: usize) -> Vec<T> {
        if neighbors.is_empty() {
            return Vec::new();
        }

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        (0..k)
            .map(|_| {
                let idx = rng.random_range(0..neighbors.len());
                neighbors[idx].clone()
            })
            .collect()
    }

    /// Sample `k` neighbors uniformly without replacement.
    ///
    /// If `k >= neighbors.len()`, returns all neighbors (shuffled).
    ///
    /// # Examples
    ///
    /// ```
    /// use kuji::NeighborSampler;
    ///
    /// let sampler = NeighborSampler::new().with_seed(42);
    /// let neighbors = vec![1, 2, 3, 4, 5];
    /// let samples = sampler.sample_uniform_without_replacement(&neighbors, 3);
    /// assert_eq!(samples.len(), 3);
    /// // All unique.
    /// let set: std::collections::HashSet<_> = samples.iter().collect();
    /// assert_eq!(set.len(), 3);
    /// ```
    pub fn sample_uniform_without_replacement<T: Clone + Eq + std::hash::Hash>(
        &self,
        neighbors: &[T],
        k: usize,
    ) -> Vec<T> {
        if neighbors.is_empty() {
            return Vec::new();
        }

        if k >= neighbors.len() {
            let mut result = neighbors.to_vec();
            let mut rng: Box<dyn RngCore> = match self.seed {
                Some(s) => Box::new(StdRng::seed_from_u64(s)),
                None => Box::new(rand::rng()),
            };
            result.shuffle(&mut rng);
            return result;
        }

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Reservoir sampling for indices
        let mut indices: Vec<usize> = (0..neighbors.len()).collect();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(k)
            .map(|i| neighbors[i].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_with_replacement() {
        let sampler = NeighborSampler::new().with_seed(42);
        let neighbors = vec![1, 2, 3];
        let samples = sampler.sample_uniform_with_replacement(&neighbors, 10);

        assert_eq!(samples.len(), 10);
        for s in samples {
            assert!(neighbors.contains(&s));
        }
    }

    #[test]
    fn test_sample_without_replacement() {
        let sampler = NeighborSampler::new().with_seed(42);
        let neighbors = vec![1, 2, 3, 4, 5];
        let samples = sampler.sample_uniform_without_replacement(&neighbors, 3);

        assert_eq!(samples.len(), 3);
        // Check uniqueness
        let set: HashSet<_> = samples.iter().collect();
        assert_eq!(set.len(), 3);
        for s in samples {
            assert!(neighbors.contains(&s));
        }
    }

    // --- edge case tests ---

    #[test]
    fn with_replacement_empty_neighbors_returns_empty() {
        let sampler = NeighborSampler::new().with_seed(0);
        let empty: Vec<i32> = vec![];
        let samples = sampler.sample_uniform_with_replacement(&empty, 5);
        assert!(samples.is_empty());
    }

    #[test]
    fn with_replacement_k_zero_returns_empty() {
        let sampler = NeighborSampler::new().with_seed(0);
        let neighbors = vec![1, 2, 3];
        let samples = sampler.sample_uniform_with_replacement(&neighbors, 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn without_replacement_empty_neighbors_returns_empty() {
        let sampler = NeighborSampler::new().with_seed(0);
        let empty: Vec<i32> = vec![];
        let samples = sampler.sample_uniform_without_replacement(&empty, 5);
        assert!(samples.is_empty());
    }

    #[test]
    fn without_replacement_k_ge_len_returns_all() {
        let sampler = NeighborSampler::new().with_seed(0);
        let neighbors = vec![10, 20, 30];
        let samples = sampler.sample_uniform_without_replacement(&neighbors, 5);
        assert_eq!(samples.len(), 3);
        let set: HashSet<_> = samples.iter().collect();
        let expected: HashSet<_> = neighbors.iter().collect();
        assert_eq!(set, expected);
    }

    #[test]
    fn deterministic_with_seed() {
        let sampler1 = NeighborSampler::new().with_seed(99);
        let sampler2 = NeighborSampler::new().with_seed(99);
        let neighbors = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let a = sampler1.sample_uniform_with_replacement(&neighbors, 20);
        let b = sampler2.sample_uniform_with_replacement(&neighbors, 20);
        assert_eq!(a, b);

        let sampler3 = NeighborSampler::new().with_seed(99);
        let sampler4 = NeighborSampler::new().with_seed(99);
        let c = sampler3.sample_uniform_without_replacement(&neighbors, 5);
        let d = sampler4.sample_uniform_without_replacement(&neighbors, 5);
        assert_eq!(c, d);
    }

    #[test]
    fn different_seeds_give_different_results() {
        let neighbors: Vec<i32> = (0..100).collect();
        let sampler1 = NeighborSampler::new().with_seed(1);
        let sampler2 = NeighborSampler::new().with_seed(2);

        let a = sampler1.sample_uniform_with_replacement(&neighbors, 50);
        let b = sampler2.sample_uniform_with_replacement(&neighbors, 50);
        // With 100 items and 50 samples, two different seeds producing identical output
        // is astronomically unlikely.
        assert_ne!(a, b);
    }
}
