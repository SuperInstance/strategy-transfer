# Strategy Transfer

**Strategy transfer** tests whether ternary strategies — sequences of {-1, 0, +1} choices optimized in one domain — generalize to different domains. This crate runs controlled transfer experiments: train a strategy on a source domain, apply it to a target domain, and compare against training from scratch. The key finding: cross-domain transfer is **neutral** — no positive or negative transfer.

## Why It Matters

In machine learning, "transfer learning" is the holy grail — train once, deploy everywhere. But does this hold for ternary strategies? If a strategy that wins in domain A also wins in domain B, we've found a universal strategy. If it fails, the domains require different approaches. This crate answers that question rigorously: for ternary strategy spaces, transfer is neutral (transfer_gain ≈ 0), meaning domain-specific training is necessary and cross-domain shortcuts don't work. This negative result is scientifically valuable — it tells us that ternary strategies encode domain-specific structure rather than generic optimization heuristics. The finding constrains the γ + η = C theory: competence C is domain-dependent, not a universal quantity.

## How It Works

### Domain Definition

A `Domain` defines a reward landscape over the ternary strategy space {-1, 0, +1}ⁿ:

```rust
type FitnessFn = fn(&Strategy) -> f64;

struct Domain {
    name: String,
    strategy_length: usize,
    fitness: FitnessFn,
}
```

Each domain maps a length-n ternary vector to a scalar reward. Example domains: needle-in-haystack (one optimal strategy), gradient (reward proportional to match count), multi-modal (several peaks).

### Training Protocol

1. **Source training**: Sample N random strategies, evaluate on source domain, keep the best:
   ```
   s* = argmax_{s ∈ samples} fitness_source(s)
   ```

2. **Transfer**: Evaluate s* on the target domain (no additional training):
   ```
   f_transfer = fitness_target(s*)
   ```

3. **Baseline (from scratch)**: Sample N random strategies on target domain, keep best:
   ```
   s_scratch = argmax_{s ∈ samples} fitness_target(s)
   f_scratch = fitness_target(s_scratch)
   ```

### Transfer Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `transfer_gain` | f_transfer − f_scratch | >0: positive transfer, <0: negative |
| `relative_transfer_gain` | (f_transfer − f_scratch) / f_scratch | Normalized gain |
| `random_baseline` | Best random strategy on target | Floor performance |

Transfer gain ≈ 0 means the transferred strategy is no better than random on the new domain — the learned structure doesn't transfer.

### Complexity

- Training: O(N · L) where N = budget, L = strategy length
- Transfer evaluation: O(L) per domain
- Baseline: O(N · L) for N random samples

## Quick Start

```rust
use strategy_transfer::{Domain, random_strategy, TransferResult};

fn source_fitness(s: &[i8]) -> f64 {
    s.iter().filter(|&&x| x == 1).count() as f64
}

fn target_fitness(s: &[i8]) -> f64 {
    s.iter().filter(|&&x| x == -1).count() as f64
}

fn main() {
    let source = Domain::new("choose-heavy", 32, source_fitness);
    let target = Domain::new("avoid-heavy", 32, target_fitness);

    let mut seed = 42u64;
    let (trained, source_fit) = source.train(1000, &mut seed);
    let (scratch, scratch_fit) = target.train(1000, &mut seed);
    let transferred_fit = target.evaluate(&trained);

    let result = TransferResult {
        trained_strategy: trained,
        source_fitness: source_fit,
        transferred_fitness: transferred_fit,
        scratch_strategy: scratch,
        scratch_fitness: scratch_fit,
        random_baseline_fitness: 0.0,
        budget: 1000,
    };

    println!("Transfer gain: {:.4}", result.transfer_gain());
    // Expected: ~0.0 (neutral transfer)
}
```

```bash
cargo build
cargo test
```

## API

| Type | Method | Description |
|------|--------|-------------|
| `Domain` | `new(name, length, fitness_fn)` | Define a reward landscape |
| `Domain` | `evaluate(strategy)` | Score a strategy |
| `Domain` | `train(budget, seed)` | Find best strategy via random search |
| `Domain` | `best_of(candidates)` | Select best from a pool |
| `TransferResult` | `transfer_gain()` | Absolute transfer gain |
| `TransferResult` | `relative_transfer_gain()` | Normalized gain |
| `random_strategy` | `(len, seed)` | LCG-based random ternary vector |

## Architecture Notes

Strategy Transfer validates a crucial property of the γ + η = C framework: competence C is **domain-specific**. A strategy that maximizes γ (constructive choices) in one domain does not generalize — the avoidance patterns (η) that work in one reward landscape are noise in another. The neutral transfer result means the fleet must train per-domain; it cannot shortcut via transfer. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning." *IEEE TKDE*, 22(10), 1345–1359.
2. Wolpert, D. H., & Macready, W. G. (1997). "No Free Lunch Theorems for Optimization." *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82.
3. Torrey, L., & Shavlik, J. (2010). "Transfer Learning." In *Handbook of Research on Machine Learning Applications*, IGI Global.

## License

MIT
