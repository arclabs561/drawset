# kuji

[![crates.io](https://img.shields.io/crates/v/kuji.svg)](https://crates.io/crates/kuji)
[![Documentation](https://docs.rs/kuji/badge.svg)](https://docs.rs/kuji)
[![CI](https://github.com/arclabs561/kuji/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/kuji/actions/workflows/ci.yml)

A low-level sampling toolbox that other crates can depend on without pulling
in domain-specific machinery. It provides reservoir sampling (Algorithm L/R
and weighted A-Res), the Gumbel-max family (categorical sampling, top-k
without replacement, the Gumbel-Softmax relaxation, relaxed k-hot), graph
neighbor sampling, quasi-Monte Carlo sequences (Halton, Sobol, Owen-scrambled
Sobol), kernel thinning and herding (greedy MMD coreset selection), and
t-norm/t-conorm families for differentiable fuzzy-logic aggregation.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/kuji) | [docs.rs](https://docs.rs/kuji)

## Modules

- `reservoir`: reservoir sampling (Algorithm L/R) and weighted reservoir (A-Res).
- `gumbel`: Gumbel-max, Gumbel-top-k, Gumbel-Softmax, relaxed k-hot.
- `neighbor`: graph neighborhood sampling (with and without replacement).
- `qmc`: quasi-Monte Carlo sequences (Halton, Sobol, Owen-scrambled Sobol).
- `thinning`: kernel thinning and herding (greedy coreset selection via MMD).
- `tconorm`: t-norm and t-conorm families for differentiable aggregation.

## Quickstart

```toml
[dependencies]
kuji = "0.1.10"
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
| `ReservoirSampler` | Algorithm L (Li, 1994): O(k(1 + log(N/k))) |
| `ReservoirSamplerR` | Algorithm R (Vitter, 1985): O(N) baseline |
| `WeightedReservoirSampler` | A-Res (Efraimidis & Spirakis, 2006) |
| `NeighborSampler` | Graph neighborhood sampling (with/without replacement) |
| `halton_sequence` / `sobol_sequence` / `sobol_scrambled` / `SobolGenerator` | Quasi-Monte Carlo low-discrepancy sequences |
| `kernel_thin` / `kernel_herd` / `mmd_sq_from_gram` | Kernel thinning and herding: greedy MMD coreset selection (Dwivedi & Mackey, 2021) |
| `tconorm` / `tnorm` / `tconorm_fold` / `tnorm_fold` | T-norm / t-conorm families (max, probabilistic, Lukasiewicz, Einstein, Hamacher) |

## Examples

- `cargo run --example distribution_demo`: ASCII histograms showing uniform vs weighted sampling distributions.
- `cargo run --example weighted_topk`: compare Gumbel-top-k (Plackett-Luce) vs weighted reservoir
  (A-Res) on the same weight vector.
- `cargo run --example gumbel_softmax_demo`: Gumbel-Softmax (Jang et al. 2017) for differentiable subset selection, the trick that lets discrete sampling sit inside a gradient-trained model.
- `cargo run --example streaming_reservoir`: stream 1M items through a reservoir of size 100 and verify uniformity.

## Tests

```bash
cargo test -p kuji
```

## Performance

![Benchmark throughput](docs/bench_throughput.png)

*Apple Silicon (NEON). Run `cargo bench` to reproduce on your hardware.*

## References (what these implementations are trying to be faithful to)

- Vitter (1985): reservoir sampling "Algorithm R".
- Li (1994): reservoir sampling "Algorithm L" (skip-based; reduces RNG calls).
- Efraimidis & Spirakis (2006): weighted reservoir sampling (A-Res / A-ExpJ family).
- Gumbel-max trick: classical extreme value sampling identity (often cited via modern ML papers):
  - Jang, Gu, Poole (2017): *Categorical Reparameterization with Gumbel-Softmax*.
  - Maddison, Mnih, Teh (2017): *The Concrete Distribution*.

## License

MIT OR Apache-2.0
