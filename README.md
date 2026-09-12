# drawset

[![crates.io](https://img.shields.io/crates/v/drawset.svg)](https://crates.io/crates/drawset)
[![Documentation](https://docs.rs/drawset/badge.svg)](https://docs.rs/drawset)

Sampling and subset-selection primitives.

## Usage

```toml
[dependencies]
drawset = "0.1.1"
```

```rust
use drawset::ReservoirSampler;

// Keep 100 items without collecting the whole stream or knowing its length.
let mut sampler = ReservoirSampler::new(100);
for item in 0..1_000_000 {
    sampler.add(item);
}
println!("Kept {} of {} items", sampler.samples().len(), sampler.seen());
```

```text
Kept 100 of 1000000 items
```

The reservoir stores at most `k` items. Algorithm L skips random draws between
replacements; this item-at-a-time API still visits every input. Use
`add_with_rng` with a seeded RNG for repeatable runs. Algorithm R is also available
as `ReservoirSamplerR`.

## Choosing a sampler

| Input and purpose | API |
|---|---|
| Stream, uniform sample without replacement | `ReservoirSampler` |
| Stream, positive weighted sample without replacement | `WeightedReservoirSampler` |
| Logits, one categorical draw or several distinct indices | `gumbel_max_sample`, `gumbel_topk_sample` |
| Logits, continuous selection weights | `gumbel_softmax`, `relaxed_topk_gumbel` |
| Kernel Gram matrix, deterministic representative indices | `kernel_thin`, `kernel_herd` |

Weighted reservoirs take finite positive weights; Gumbel samplers take logits
(log-weights). A-Res and Gumbel-top-k target the same weighted subset distribution
when those weights correspond. Their returned order need not match. See the
[weighted selection example](examples/weighted_topk.rs).

The relaxations return floating-point vectors; they do not provide automatic
differentiation. The kernel selectors require a dense `n × n` Gram matrix supplied
by the caller. `kernel_thin` greedily minimizes MMD without replacement;
`kernel_herd` can select an index more than once.

The crate also includes `NeighborSampler` for sampling a neighbor slice, plus
quasi-Monte Carlo sequence re-exports from [lowdisc](https://crates.io/crates/lowdisc).
See the [API documentation](https://docs.rs/drawset) for input requirements and
algorithm references.

## Examples and checks

```sh
cargo run --example streaming_reservoir
cargo run --example weighted_topk
cargo run --example gumbel_softmax_demo
cargo test --workspace
```

More examples are listed in [examples/](examples/README.md). For local performance
measurements, run `cargo bench --bench sampling`. Contributor checks are described
in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT OR Apache-2.0
