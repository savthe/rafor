use super::{ensemble_predictor, ensemble_trainer};
use crate::{
    config::*,
    config_builders::*,
    decision_tree::{Trainset, TreeRegressorImpl},
    Dataset, DatasetView, FloatTarget
};
use serde::{Deserialize, Serialize};

/// A random forest regressor.
/// # Example
/// ```
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = rf::Regressor::fit(&dataset, &targets, &rf::Regressor::default_config());
/// let predictions = predictor.predict(&dataset, 1);
/// println!("{:?}", predictions);
/// ```
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regressor {
    ensemble: Vec<TreeRegressorImpl>,
}

/// Configuration for ensemble regressor. Default values:
/// ```
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
/// seed: 42,
/// metric: Metric::MSE,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
/// num_trees: 100,
/// num_threads: 1,
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressorConfig {
    pub train_config: TrainConfig,
    pub ensemble_config: EnsembleConfig,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            train_config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                seed: 42,
                metric: Metric::MSE,
                min_samples_leaf: 1,
                min_samples_split: 2
            },
            ensemble_config: EnsembleConfig {
                num_trees: 100,
                num_threads: 1,
            },
        }
    }
}

#[derive(Clone)]
struct Trainee {
    tree: TreeRegressorImpl,
    train_config: TrainConfig,
}

impl Trainee {
    fn new(train_config: TrainConfig) -> Self {
        Self {
            tree: TreeRegressorImpl::default(),
            train_config,
        }
    }
}

impl ensemble_trainer::Trainable<FloatTarget> for Trainee {
    fn fit(&mut self, ts: Trainset<FloatTarget>, seed: u64) {
        self.train_config.seed = seed;
        self.tree = TreeRegressorImpl::fit(ts, &self.train_config);
    }
}

impl ensemble_predictor::Predictor for TreeRegressorImpl {
    fn predict(&self, dataset: &DatasetView) -> Vec<FloatTarget> {
        self.predict(dataset)
    }
}

impl Regressor {
    /// Predicts regression values for a set of samples using `num_threads` threads.
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<FloatTarget> {
        let view = DatasetView::new(dataset, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, num_threads)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        let view = DatasetView::new(sample, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1)[0]
    }

    /// Trains a random forest regressor with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn fit(data: &[f32], targets: &[FloatTarget], config: &RegressorConfig) -> Regressor {
        let ds = Dataset::with_transposed(data, targets.len());
        let trainee = Trainee::new(config.train_config.clone());
        let ens = ensemble_trainer::fit(
            trainee,
            ds.as_view(),
            targets,
            &config.ensemble_config,
            config.train_config.seed,
        );

        Regressor {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
        }
    }

    /// Returns a number of features for a trained forest.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    /// Returns training config filled with default values.
    pub fn default_config() -> RegressorConfig {
        RegressorConfig::default()
    }
}

impl TrainConfigProvider for RegressorConfig {
    fn train_config(&mut self) -> &mut TrainConfig {
        &mut self.train_config
    }
}

impl EnsembleConfigProvider for RegressorConfig {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.ensemble_config
    }
}

impl CommonConfigBuilder for RegressorConfig {}
impl EnsembleConfigBuilder for RegressorConfig {}
