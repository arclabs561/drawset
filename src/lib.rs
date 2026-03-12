//! `kuji`: stochastic sampling primitives.
//!
//! This crate is meant to be a low-level “sampling toolbox” that other crates can
//! depend on without pulling in domain-specific machinery.
//!
//! Exposed modules:
//! - `reservoir`: reservoir sampling (Algorithm L/R) + weighted reservoir.
//! - `gumbel`: Gumbel-max / Gumbel-top-k / relaxed k-hot.
//! - `neighbor`: simple neighborhood sampling helpers (useful for graph ML).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod gumbel;
pub mod neighbor;
pub mod reservoir;

pub use gumbel::{
    gumbel_max_sample, gumbel_noise, gumbel_softmax, gumbel_topk_sample,
    gumbel_topk_sample_with_rng, relaxed_topk_gumbel,
};
pub use neighbor::NeighborSampler;
pub use reservoir::{ReservoirSampler, ReservoirSamplerR, WeightedReservoirSampler};
