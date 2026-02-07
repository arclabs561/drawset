# kuji

Stochastic sampling primitives for unbiased data selection and stream processing.
Implements reservoir sampling (Algorithm L/R), weighted sampling, and Gumbel-max for top-k.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/kuji) | [docs.rs](https://docs.rs/kuji)

## Quickstart

```toml
[dependencies]
kuji = "0.1.0"
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

## Examples

- `cargo run --example weighted_topk`: compare Gumbel-top-k (Plackett–Luce) vs weighted reservoir
  (A-Res) on the same weight vector.

## References (what these implementations are trying to be faithful to)

- Vitter (1985): reservoir sampling “Algorithm R”.
- Li (1994): reservoir sampling “Algorithm L” (skip-based; reduces RNG calls).
- Efraimidis & Spirakis (2006): weighted reservoir sampling (A-Res / A-ExpJ family).
- Gumbel-max trick: classical extreme value sampling identity (often cited via modern ML papers):
  - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
  - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.
