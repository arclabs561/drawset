# kuji examples

Examples for the `kuji` stochastic sampling crate.

## Running

```sh
cargo run -p kuji --example <name>
```

## Examples

| Example | Description |
|---|---|
| `weighted_topk` | Compares Gumbel-top-k (Plackett-Luce) and weighted reservoir sampling (A-Res). Draws 10,000 samples and prints a frequency table showing the distributional difference for k>1. |
| `streaming_reservoir` | Algorithm L reservoir sampling over a 1M-item stream with reservoir size 100. Demonstrates O(k) memory. Verifies uniform sampling via a bucketed chi-squared check over 5,000 trials. |
| `gumbel_softmax_demo` | Gumbel-Softmax for differentiable subset selection. Shows three temperature regimes (high/medium/low) and how the soft distribution converges to one-hot as temperature decreases. |
