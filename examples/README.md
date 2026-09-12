# drawset examples

Small demonstrations of drawset's sampling APIs.

## Running

```sh
cargo run -p drawset --example <name>
```

## Examples

| Example | Description |
|---|---|
| `weighted_topk` | Weighted selection from a slice of logits or a stream of weights. Both target the same subset distribution, with different interfaces and output ordering. |
| `streaming_reservoir` | Keeps 100 items from a one-million-item stream in one pass. |
| `gumbel_softmax_demo` | Shows how temperature changes individual relaxed samples and their average. Returns numeric weights, without autodiff. |
| `distribution_demo` | Prints histograms of uniform and weighted reservoir samples. |
