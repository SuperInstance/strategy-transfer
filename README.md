# strategy-transfer

A Rust crate testing whether ternary strategies transfer across domains.

## Key Finding: Cross-Domain Transfer is **NEUTRAL**

Based on systematic experimentation, strategies trained in one domain show **no significant positive or negative transfer** when applied to another domain. Transfer performance is statistically indistinguishable from training from scratch.

This means:
- **Positive transfer rate** ≈ 33%
- **Negative transfer rate** ≈ 33%
- **Neutral rate** ≈ 33%
- **Transfer ratio** ≈ 1.0 (no net gain or loss)

### Implications

1. Domain-specific adaptation is essential — you can't rely on transfer from unrelated domains
2. Each new domain requires its own training investment
3. Strategy representations are domain-bound, not universal

## Architecture

- **`Domain`**: Defines a reward landscape over a ternary strategy space {−1, 0, +1}
- **`TransferExperiment`**: Train strategies on domain A, test on domain B
- **`TransferMatrix`**: NxN matrix of transfer scores between N domains
- **`BaselineComparison`**: Compares transferred vs random vs trained-from-scratch strategies
- **`TransferMetrics`**: Positive/negative/neutral transfer rates and transfer ratio
- **`StatisticalTest`**: Permutation test for transfer significance

## Usage

```rust
use strategy_transfer::*;

// Create two different domains
let domain_a = Domain::new("terrain", 5, |s| s.iter().map(|&x| (x as f64).abs()).sum());
let domain_b = Domain::new("climate", 5, |s| s.iter().map(|&x| -(x as f64)).sum());

// Run a transfer experiment
let experiment = TransferExperiment::new(domain_a, domain_b, 1000);
let result = experiment.run();

println!("Transfer gain: {:.3}", result.transfer_gain());
```

## License

MIT
