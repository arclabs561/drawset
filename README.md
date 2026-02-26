# kuji

[![crates.io](https://img.shields.io/crates/v/kuji.svg)](https://crates.io/crates/kuji)
[![Documentation](https://docs.rs/kuji/badge.svg)](https://docs.rs/kuji)
[![CI](https://github.com/arclabs561/kuji/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/kuji/actions/workflows/ci.yml)

Stochastic sampling primitives for unbiased data selection and stream processing.
Implements Gumbel-max top-k, Gumbel-Softmax relaxations, reservoir sampling (Algorithm L/R), and weighted reservoir (A-Res).

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/kuji) | [docs.rs](https://docs.rs/kuji)

## Quickstart

```toml
[dependencies]
kuji = "0.1.3"
```

```rust
use kuji::reservoir::ReservoirSampler;

let mut sampler = ReservoirSampler::new(5);
for i in 0..100 {
    sampler.add(i);
}
let samples = sampler.samples();
assert_eq!(samples.len(), 5);
```

## Operations

| Function / Type | Description |
|----------------|-------------|
| `gumbel_max_sample` | Categorical sample via Gumbel-max trick |
| `gumbel_topk_sample` | Top-k without replacement via Gumbel perturbation |
| `gumbel_softmax` | Differentiable categorical approximation |
| `relaxed_topk_gumbel` | Relaxed k-hot via iterated Gumbel-Softmax |
| `ReservoirSampler` | Algorithm L (Li, 1994) -- O(k(1 + log(N/k))) |
| `ReservoirSamplerR` | Algorithm R (Vitter, 1985) -- O(N) baseline |
| `WeightedReservoirSampler` | A-Res (Efraimidis & Spirakis, 2006) |
| `NeighborSampler` | Graph neighborhood sampling (with/without replacement) |

**vs `rand`**: `rand::seq` provides uniform sampling but not reservoir sampling over streams, weighted reservoir (A-Res), or Gumbel-max top-k.

## Examples

- `cargo run --example distribution_demo`: ASCII histograms showing uniform vs weighted sampling distributions.
- `cargo run --example weighted_topk`: compare Gumbel-top-k (Plackett--Luce) vs weighted reservoir
  (A-Res) on the same weight vector.
- `cargo run --example streaming_reservoir`: stream 1M items through a reservoir of size 100 and verify uniformity.

## Output example

`cargo run --example distribution_demo`:

```text
Reservoir sampling (Algorithm L) -- uniform stream, k=10:
  [ 0.. 9] 10057 #######################################
  [10..19]  9860 ######################################
  [20..29] 10007 #######################################
  [30..39] 10021 #######################################
  [40..49]  9822 ######################################
  [50..59] 10164 ########################################
  [60..69] 10111 #######################################
  [70..79]  9898 ######################################
  [80..89] 10098 #######################################
  [90..99]  9962 #######################################

Weighted reservoir (A-Res) -- power-law weights w(i)=1/(1+i)^1.5:
  [ 0.. 9] 60176 ########################################
  [10..19] 15999 ##########
  [20..29]  7687 #####
  [30..39]  4760 ###
  [40..49]  3124 ##
  [50..59]  2415 #
  [60..69]  1925 #
  [70..79]  1560 #
  [80..89]  1252
  [90..99]  1102

Gumbel-top-k single draw (k=10, seed=42): [1, 0, 12, 82, 3, 11, 2, 7, 8, 49]
```

## Tests

```bash
cargo test -p kuji
```

## See also

- [`innr`](https://crates.io/crates/innr) -- SIMD-accelerated vector similarity primitives
- [`subsume`](https://crates.io/crates/subsume) -- geometric box embeddings (Gumbel boxes use the same distribution family)
- [`anno`](https://crates.io/crates/anno) -- information extraction (NER, coreference)

## References (what these implementations are trying to be faithful to)

- Vitter (1985): reservoir sampling "Algorithm R".
- Li (1994): reservoir sampling "Algorithm L" (skip-based; reduces RNG calls).
- Efraimidis & Spirakis (2006): weighted reservoir sampling (A-Res / A-ExpJ family).
- Gumbel-max trick: classical extreme value sampling identity (often cited via modern ML papers):
  - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
  - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.
