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
kuji = "0.1.2"
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

- `cargo run --example weighted_topk`: compare Gumbel-top-k (Plackett--Luce) vs weighted reservoir
  (A-Res) on the same weight vector.

## Tests

```bash
cargo test -p kuji
```

## References (what these implementations are trying to be faithful to)

- Vitter (1985): reservoir sampling "Algorithm R".
- Li (1994): reservoir sampling "Algorithm L" (skip-based; reduces RNG calls).
- Efraimidis & Spirakis (2006): weighted reservoir sampling (A-Res / A-ExpJ family).
- Gumbel-max trick: classical extreme value sampling identity (often cited via modern ML papers):
  - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
  - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.
