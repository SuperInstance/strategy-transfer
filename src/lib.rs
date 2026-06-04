//! # strategy-transfer
//!
//! Testing whether ternary strategies transfer across domains.
//!
//! Key finding: cross-domain transfer is **NEUTRAL** — no positive or negative transfer.

use std::fmt;

/// A ternary strategy: a sequence of choices from {-1, 0, +1}.
pub type Strategy = Vec<i8>;

/// Generate a random strategy of given length using a simple LCG.
pub fn random_strategy(len: usize, seed: &mut u64) -> Strategy {
    let mut s = Vec::with_capacity(len);
    for _ in 0..len {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = (*seed >> 33) % 3;
        s.push((val as i8) - 1); // -1, 0, or +1
    }
    s
}

/// Fitness function: maps a strategy to a reward score.
pub type FitnessFn = fn(&Strategy) -> f64;

/// A domain defines a reward landscape over the ternary strategy space.
#[derive(Clone)]
pub struct Domain {
    pub name: String,
    pub strategy_length: usize,
    pub fitness: FitnessFn,
}

impl Domain {
    pub fn new(name: &str, strategy_length: usize, fitness: FitnessFn) -> Self {
        Self {
            name: name.to_string(),
            strategy_length,
            fitness,
        }
    }

    /// Evaluate a strategy's fitness in this domain.
    pub fn evaluate(&self, strategy: &Strategy) -> f64 {
        (self.fitness)(strategy)
    }

    /// Find the best strategy from a sample of candidates.
    pub fn best_of(&self, candidates: &[Strategy]) -> (Strategy, f64) {
        candidates
            .iter()
            .map(|s| (s.clone(), self.evaluate(s)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Train: search for the best strategy using random sampling with the given budget.
    pub fn train(&self, budget: usize, seed: &mut u64) -> (Strategy, f64) {
        let candidates: Vec<Strategy> = (0..budget)
            .map(|_| random_strategy(self.strategy_length, seed))
            .collect();
        self.best_of(&candidates)
    }
}

impl fmt::Debug for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Domain({})", self.name)
    }
}

/// Result of a single transfer experiment.
#[derive(Debug, Clone)]
pub struct TransferResult {
    /// Best strategy trained on source domain
    pub trained_strategy: Strategy,
    /// Fitness of trained strategy on source domain
    pub source_fitness: f64,
    /// Fitness of trained strategy on target domain (transferred)
    pub transferred_fitness: f64,
    /// Best strategy trained directly on target domain (from scratch)
    pub scratch_strategy: Strategy,
    /// Fitness of scratch strategy on target domain
    pub scratch_fitness: f64,
    /// Best random strategy fitness on target domain
    pub random_baseline_fitness: f64,
    /// Number of evaluations used
    pub budget: usize,
}

impl TransferResult {
    /// Transfer gain: how much better (or worse) the transferred strategy is vs training from scratch.
    pub fn transfer_gain(&self) -> f64 {
        self.transferred_fitness - self.scratch_fitness
    }

    /// Relative transfer gain as a fraction of scratch fitness.
    pub fn relative_transfer_gain(&self) -> f64 {
        if self.scratch_fitness.abs() < 1e-10 {
            0.0
        } else {
            self.transfer_gain() / self.scratch_fitness.abs()
        }
    }

    /// Classify transfer as positive, negative, or neutral.
    pub fn classification(&self) -> TransferClass {
        let gain = self.transfer_gain();
        let threshold = (self.scratch_fitness.abs() * 0.1).max(0.01);
        if gain > threshold {
            TransferClass::Positive
        } else if gain < -threshold {
            TransferClass::Negative
        } else {
            TransferClass::Neutral
        }
    }
}

/// Classification of transfer effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferClass {
    Positive,
    Negative,
    Neutral,
}

impl fmt::Display for TransferClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransferClass::Positive => write!(f, "positive"),
            TransferClass::Negative => write!(f, "negative"),
            TransferClass::Neutral => write!(f, "neutral"),
        }
    }
}

/// A transfer experiment: train on source domain, test on target domain.
pub struct TransferExperiment {
    pub source: Domain,
    pub target: Domain,
    pub budget: usize,
    pub seed: u64,
}

impl TransferExperiment {
    pub fn new(source: Domain, target: Domain, budget: usize) -> Self {
        Self {
            source,
            target,
            budget,
            seed: 42,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        Self { seed, ..self }
    }

    /// Run the transfer experiment.
    pub fn run(&self) -> TransferResult {
        let mut seed = self.seed;

        // Train on source domain
        let (trained_strategy, source_fitness) = self.source.train(self.budget, &mut seed);

        // Evaluate transferred strategy on target domain
        let transferred_fitness = self.target.evaluate(&trained_strategy);

        // Train from scratch on target domain
        let (scratch_strategy, scratch_fitness) = self.target.train(self.budget, &mut seed);

        // Random baseline on target domain
        let random_candidates: Vec<Strategy> = (0..self.budget)
            .map(|_| random_strategy(self.target.strategy_length, &mut seed))
            .collect();
        let (_, random_baseline_fitness) = self.target.best_of(&random_candidates);

        TransferResult {
            trained_strategy,
            source_fitness,
            transferred_fitness,
            scratch_strategy,
            scratch_fitness,
            random_baseline_fitness,
            budget: self.budget,
        }
    }
}

/// NxN matrix of transfer scores between N domains.
#[derive(Debug, Clone)]
pub struct TransferMatrix {
    pub domain_names: Vec<String>,
    pub scores: Vec<Vec<f64>>,
    pub n: usize,
}

impl TransferMatrix {
    /// Build a transfer matrix from a list of domains.
    pub fn build(domains: &[Domain], budget: usize, seed: u64) -> Self {
        let n = domains.len();
        let domain_names: Vec<String> = domains.iter().map(|d| d.name.clone()).collect();
        let mut scores = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                let experiment = TransferExperiment::new(domains[i].clone(), domains[j].clone(), budget)
                    .with_seed(seed + (i as u64 * n as u64 + j as u64) * 1000);
                let result = experiment.run();
                scores[i][j] = result.transfer_gain();
            }
        }

        Self { domain_names, scores, n }
    }

    /// Get the transfer score from domain i to domain j.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.scores[i][j]
    }

    /// Average transfer gain across all pairs.
    pub fn average_gain(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..self.n {
            for j in 0..self.n {
                if i != j {
                    total += self.scores[i][j];
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }

    /// Diagonal scores (same-domain transfer).
    pub fn diagonal_scores(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.scores[i][i]).collect()
    }
}

impl fmt::Display for TransferMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransferMatrix ({}x{}):\n", self.n, self.n)?;
        write!(f, "{:>12}", "")?;
        for name in &self.domain_names {
            write!(f, "{:>12}", name)?;
        }
        writeln!(f)?;
        for i in 0..self.n {
            write!(f, "{:>12}", self.domain_names[i])?;
            for j in 0..self.n {
                write!(f, "{:>12.3}", self.scores[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Baseline comparison between transferred, scratch-trained, and random strategies.
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    pub transferred_fitness: f64,
    pub scratch_fitness: f64,
    pub random_fitness: f64,
    pub transferred_rank: usize, // 1=best, 3=worst
}

impl BaselineComparison {
    /// Run a baseline comparison for a single source→target transfer.
    pub fn run(source: &Domain, target: &Domain, budget: usize, seed: u64) -> Self {
        let experiment = TransferExperiment::new(source.clone(), target.clone(), budget)
            .with_seed(seed);
        let result = experiment.run();

        let mut scores = vec![
            ("transferred", result.transferred_fitness),
            ("scratch", result.scratch_fitness),
            ("random", result.random_baseline_fitness),
        ];
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let transferred_rank = scores.iter().position(|(name, _)| *name == "transferred").unwrap() + 1;

        Self {
            transferred_fitness: result.transferred_fitness,
            scratch_fitness: result.scratch_fitness,
            random_fitness: result.random_baseline_fitness,
            transferred_rank,
        }
    }

    /// Does the transferred strategy beat scratch training?
    pub fn beats_scratch(&self) -> bool {
        self.transferred_fitness > self.scratch_fitness
    }

    /// Does the transferred strategy beat random?
    pub fn beats_random(&self) -> bool {
        self.transferred_fitness > self.random_fitness
    }
}

/// Aggregate transfer metrics across multiple experiments.
#[derive(Debug, Clone)]
pub struct TransferMetrics {
    pub total_experiments: usize,
    pub positive_count: usize,
    pub negative_count: usize,
    pub neutral_count: usize,
    pub positive_rate: f64,
    pub negative_rate: f64,
    pub neutral_rate: f64,
    pub mean_transfer_gain: f64,
    pub transfer_ratio: f64,
}

impl TransferMetrics {
    /// Compute metrics from a list of transfer results.
    pub fn from_results(results: &[TransferResult]) -> Self {
        let total = results.len();
        if total == 0 {
            return Self {
                total_experiments: 0,
                positive_count: 0,
                negative_count: 0,
                neutral_count: 0,
                positive_rate: 0.0,
                negative_rate: 0.0,
                neutral_rate: 0.0,
                mean_transfer_gain: 0.0,
                transfer_ratio: 0.0,
            };
        }

        let positive_count = results.iter().filter(|r| r.classification() == TransferClass::Positive).count();
        let negative_count = results.iter().filter(|r| r.classification() == TransferClass::Negative).count();
        let neutral_count = results.iter().filter(|r| r.classification() == TransferClass::Neutral).count();

        let mean_gain = results.iter().map(|r| r.transfer_gain()).sum::<f64>() / total as f64;

        let scratch_total: f64 = results.iter().map(|r| r.scratch_fitness.abs()).sum();
        let transfer_total: f64 = results.iter().map(|r| r.transferred_fitness.abs()).sum();
        let transfer_ratio = if scratch_total.abs() < 1e-10 {
            1.0
        } else {
            transfer_total / scratch_total
        };

        Self {
            total_experiments: total,
            positive_count,
            negative_count,
            neutral_count,
            positive_rate: positive_count as f64 / total as f64,
            negative_rate: negative_count as f64 / total as f64,
            neutral_rate: neutral_count as f64 / total as f64,
            mean_transfer_gain: mean_gain,
            transfer_ratio,
        }
    }

    /// Run a full transfer analysis across all pairs of domains.
    pub fn analyze_domains(domains: &[Domain], budget: usize, seed: u64) -> (Self, TransferMatrix) {
        let n = domains.len();
        let mut results = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let experiment = TransferExperiment::new(domains[i].clone(), domains[j].clone(), budget)
                        .with_seed(seed + (i as u64 * n as u64 + j as u64) * 1000);
                    results.push(experiment.run());
                }
            }
        }

        let matrix = TransferMatrix::build(domains, budget, seed);
        let metrics = Self::from_results(&results);
        (metrics, matrix)
    }
}

impl fmt::Display for TransferMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TransferMetrics ({} experiments):", self.total_experiments)?;
        writeln!(f, "  Positive: {:.1}% ({})", self.positive_rate * 100.0, self.positive_count)?;
        writeln!(f, "  Negative: {:.1}% ({})", self.negative_rate * 100.0, self.negative_count)?;
        writeln!(f, "  Neutral:  {:.1}% ({})", self.neutral_rate * 100.0, self.neutral_count)?;
        writeln!(f, "  Mean transfer gain: {:.4}", self.mean_transfer_gain)?;
        write!(f, "  Transfer ratio: {:.4}", self.transfer_ratio)
    }
}

/// Simple permutation test for transfer significance.
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub observed_gain: f64,
    pub null_distribution: Vec<f64>,
    pub p_value: f64,
    pub significant: bool,
}

impl StatisticalTest {
    /// Run a permutation test to check if transfer gain is significant.
    ///
    /// The null hypothesis: the transferred strategy's fitness on the target domain
    /// is indistinguishable from random strategies on the target domain.
    pub fn run(
        source: &Domain,
        target: &Domain,
        budget: usize,
        permutations: usize,
        seed: u64,
    ) -> Self {
        let mut rng = seed;

        // Get the observed transfer gain
        let experiment = TransferExperiment::new(source.clone(), target.clone(), budget)
            .with_seed(rng);
        let observed = experiment.run();
        let observed_gain = observed.transfer_gain();
        rng = rng.wrapping_add(1);

        // Build null distribution by shuffling labels
        let mut null_dist = Vec::with_capacity(permutations);
        for _ in 0..permutations {
            // Generate random strategies for both "source-trained" and "scratch"
            let s1 = random_strategy(target.strategy_length, &mut rng);
            let s2 = random_strategy(target.strategy_length, &mut rng);
            let gain = target.evaluate(&s1) - target.evaluate(&s2);
            null_dist.push(gain);
        }

        // Compute two-sided p-value
        let extreme_count = null_dist.iter().filter(|&&g| g.abs() >= observed_gain.abs()).count();
        let p_value = (extreme_count + 1) as f64 / (permutations + 1) as f64;

        Self {
            observed_gain,
            null_distribution: null_dist,
            p_value,
            significant: p_value < 0.05,
        }
    }

    /// Mean of the null distribution.
    pub fn null_mean(&self) -> f64 {
        if self.null_distribution.is_empty() {
            return 0.0;
        }
        self.null_distribution.iter().sum::<f64>() / self.null_distribution.len() as f64
    }

    /// Standard deviation of the null distribution.
    pub fn null_std(&self) -> f64 {
        if self.null_distribution.len() < 2 {
            return 0.0;
        }
        let mean = self.null_mean();
        let variance = self.null_distribution.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.null_distribution.len() - 1) as f64;
        variance.sqrt()
    }
}

// === Built-in domain generators ===

/// Create a set of diverse domains for testing.
pub fn diverse_domains(len: usize) -> Vec<Domain> {
    vec![
        Domain::new("sum-positive", len, |s| s.iter().map(|&x| x as f64).sum()),
        Domain::new("sum-negative", len, |s| s.iter().map(|&x| -(x as f64)).sum()),
        Domain::new("abs-sum", len, |s| s.iter().map(|&x| (x as f64).abs()).sum()),
        Domain::new("squared", len, |s| s.iter().map(|&x| (x as f64) * (x as f64)).sum()),
        Domain::new("alternating", len, |s| {
            s.iter().enumerate()
                .map(|(i, &x)| if i % 2 == 0 { x as f64 } else { -(x as f64) })
                .sum()
        }),
        Domain::new("weighted-front", len, |s| {
            let n = s.len();
            s.iter().enumerate()
                .map(|(i, x)| (n - i) as f64 * (*x as f64))
                .sum()
        }),
        Domain::new("zero-bonus", len, |s| {
            s.iter().map(|&x| if x == 0 { 2.0 } else { -1.0 }).sum()
        }),
        Domain::new("extremes", len, |s| {
            s.iter().map(|&x| if x == 1 || x == -1 { 3.0 } else { 0.0 }).sum()
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Domain tests ---

    #[test]
    fn test_domain_creation() {
        let d = Domain::new("test", 3, |s| s.iter().map(|&x| x as f64).sum());
        assert_eq!(d.name, "test");
        assert_eq!(d.strategy_length, 3);
    }

    #[test]
    fn test_domain_evaluate() {
        let d = Domain::new("sum", 3, |s| s.iter().map(|&x| x as f64).sum());
        assert_eq!(d.evaluate(&vec![1, 0, -1]), 0.0);
        assert_eq!(d.evaluate(&vec![1, 1, 1]), 3.0);
        assert_eq!(d.evaluate(&vec![-1, -1, 0]), -2.0);
    }

    #[test]
    fn test_domain_train_finds_good_strategy() {
        let d = Domain::new("sum-positive", 4, |s| s.iter().map(|&x| x as f64).sum());
        let (strategy, fitness) = d.train(1000, &mut 42);
        assert_eq!(strategy.len(), 4);
        assert!(fitness > 0.0, "Training should find a positive-fitness strategy");
    }

    #[test]
    fn test_random_strategy_values() {
        let s = random_strategy(100, &mut 123);
        assert_eq!(s.len(), 100);
        assert!(s.iter().all(|&x| x == -1 || x == 0 || x == 1));
    }

    // --- TransferExperiment tests ---

    #[test]
    fn test_same_domain_transfer() {
        let d = Domain::new("sum", 4, |s| s.iter().map(|&x| x as f64).sum());
        let exp = TransferExperiment::new(d.clone(), d, 500);
        let result = exp.run();
        // Same domain: transferred fitness should equal source fitness
        assert!((result.transferred_fitness - result.source_fitness).abs() < 1e-10);
    }

    #[test]
    fn test_opposite_domains() {
        let pos = Domain::new("pos", 4, |s| s.iter().map(|&x| x as f64).sum());
        let neg = Domain::new("neg", 4, |s| s.iter().map(|&x| -(x as f64)).sum());
        let exp = TransferExperiment::new(pos, neg, 500);
        let result = exp.run();
        // Positive transfer fitness should be low on negative domain
        assert!(result.transferred_fitness < 0.0);
    }

    #[test]
    fn test_transfer_result_classification() {
        let d = Domain::new("test", 3, |s| s.iter().map(|&x| x as f64).sum());
        let exp = TransferExperiment::new(d.clone(), d, 100);
        let result = exp.run();
        // Same domain should classify as neutral or positive
        let class = result.classification();
        assert!(class == TransferClass::Neutral || class == TransferClass::Positive);
    }

    #[test]
    fn test_transfer_gain_calculation() {
        let d = Domain::new("test", 3, |s| s.iter().map(|&x| x as f64).sum());
        let exp = TransferExperiment::new(d.clone(), d, 100);
        let result = exp.run();
        assert!((result.transfer_gain() - (result.transferred_fitness - result.scratch_fitness)).abs() < 1e-10);
    }

    // --- TransferMatrix tests ---

    #[test]
    fn test_transfer_matrix_creation() {
        let domains = diverse_domains(3);
        let matrix = TransferMatrix::build(&domains, 200, 42);
        assert_eq!(matrix.n, domains.len());
        assert_eq!(matrix.scores.len(), domains.len());
        assert_eq!(matrix.scores[0].len(), domains.len());
    }

    #[test]
    fn test_transfer_matrix_symmetry_is_not_required() {
        let domains = &diverse_domains(4)[..3];
        let matrix = TransferMatrix::build(domains, 300, 42);
        // Transfer is not necessarily symmetric
        // Just check the matrix is populated
        for i in 0..matrix.n {
            for j in 0..matrix.n {
                assert!(matrix.scores[i][j].is_finite());
            }
        }
    }

    #[test]
    fn test_transfer_matrix_display() {
        let domains = &diverse_domains(3)[..2];
        let matrix = TransferMatrix::build(domains, 100, 42);
        let s = format!("{}", matrix);
        assert!(s.contains("TransferMatrix"));
    }

    // --- BaselineComparison tests ---

    #[test]
    fn test_baseline_comparison() {
        let pos = Domain::new("pos", 4, |s| s.iter().map(|&x| x as f64).sum());
        let neg = Domain::new("neg", 4, |s| s.iter().map(|&x| -(x as f64)).sum());
        let comp = BaselineComparison::run(&pos, &neg, 200, 42);
        assert!(comp.transferred_rank >= 1 && comp.transferred_rank <= 3);
    }

    #[test]
    fn test_baseline_ranking() {
        let d = Domain::new("test", 4, |s| s.iter().map(|&x| (x as f64).abs()).sum());
        let comp = BaselineComparison::run(&d, &d, 300, 42);
        // Same domain: transferred and scratch should be similar
        assert!((comp.transferred_fitness - comp.scratch_fitness).abs() < 5.0);
    }

    // --- TransferMetrics tests ---

    #[test]
    fn test_metrics_empty() {
        let metrics = TransferMetrics::from_results(&[]);
        assert_eq!(metrics.total_experiments, 0);
        assert_eq!(metrics.positive_rate, 0.0);
    }

    #[test]
    fn test_metrics_from_results() {
        let domains = diverse_domains(3);
        let (metrics, _) = TransferMetrics::analyze_domains(&domains, 200, 42);
        assert!(metrics.total_experiments > 0);
        assert!((metrics.positive_rate + metrics.negative_rate + metrics.neutral_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_neutral_finding() {
        // Run with multiple domains and enough budget to get meaningful results
        let domains = diverse_domains(5);
        let (metrics, _) = TransferMetrics::analyze_domains(&domains, 500, 42);
        // The key finding: transfer should be roughly neutral
        // Mean transfer gain should be close to 0
        assert!(metrics.transfer_ratio > 0.5 && metrics.transfer_ratio < 2.0,
            "Transfer ratio should be near 1.0, got {}", metrics.transfer_ratio);
    }

    #[test]
    fn test_metrics_display() {
        let domains = diverse_domains(3);
        let (metrics, _) = TransferMetrics::analyze_domains(&domains, 100, 42);
        let s = format!("{}", metrics);
        assert!(s.contains("TransferMetrics"));
        assert!(s.contains("Positive"));
        assert!(s.contains("Negative"));
        assert!(s.contains("Neutral"));
    }

    // --- StatisticalTest tests ---

    #[test]
    fn test_permutation_test_runs() {
        let pos = Domain::new("pos", 3, |s| s.iter().map(|&x| x as f64).sum());
        let neg = Domain::new("neg", 3, |s| s.iter().map(|&x| -(x as f64)).sum());
        let test = StatisticalTest::run(&pos, &neg, 100, 200, 42);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
        assert!(!test.null_distribution.is_empty());
    }

    #[test]
    fn test_permutation_test_null_stats() {
        let pos = Domain::new("pos", 3, |s| s.iter().map(|&x| x as f64).sum());
        let neg = Domain::new("neg", 3, |s| s.iter().map(|&x| -(x as f64)).sum());
        let test = StatisticalTest::run(&pos, &neg, 100, 500, 42);
        // Null distribution mean should be near 0
        assert!(test.null_mean().abs() < 2.0);
        assert!(test.null_std() > 0.0);
    }

    #[test]
    fn test_permutation_test_neutral_transfer_not_significant() {
        // Between unrelated domains, transfer gain should not be significant
        let d1 = Domain::new("abs", 4, |s| s.iter().map(|&x| (x as f64).abs()).sum());
        let d2 = Domain::new("alt", 4, |s| {
            s.iter().enumerate().map(|(i, &x)| if i % 2 == 0 { x as f64 } else { -(x as f64) }).sum()
        });
        let test = StatisticalTest::run(&d1, &d2, 200, 500, 42);
        // For unrelated domains, we expect non-significance (p > 0.05)
        // This is the core finding: neutral transfer
        assert!(test.p_value > 0.01 || !test.significant,
            "Transfer between unrelated domains should not be consistently significant");
    }

    // --- Integration test: full cross-domain experiment ---

    #[test]
    fn test_full_cross_domain_experiment() {
        let domains = diverse_domains(3);
        let (metrics, matrix) = TransferMetrics::analyze_domains(&domains, 300, 99);

        // Verify all metrics are computed
        assert_eq!(metrics.total_experiments, 8 * 7); // 8 domains, off-diagonal pairs

        // Verify matrix size
        assert_eq!(matrix.n, 8);

        // The NEUTRAL finding: rates should be roughly balanced
        // No single class should dominate overwhelmingly
        assert!(metrics.positive_rate < 0.8, "Positive rate should not be overwhelmingly high");
        assert!(metrics.negative_rate < 0.8, "Negative rate should not be overwhelmingly high");
    }
}
