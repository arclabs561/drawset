# kuji

`kuji` has been renamed to `drawset`.

```toml
[dependencies]
drawset = "0.1.2"
```

Use `drawset::` in new code. To switch the dependency while keeping existing
`kuji::` imports, use Cargo's dependency alias:

```toml
[dependencies]
kuji = { package = "drawset", version = "0.1.2" }
```

The published `kuji = "0.1.11"` compatibility crate also remains available and
re-exports drawset. It has no independent implementation.
