use super::{ensemble_predictor, ensemble_trainer};
use crate::{
    config::*, config_builders::*, decision_tree::{Trainset, TreeRegressorImpl}, Dataset, DatasetView
};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub struct Regressor {
    ensemble: Vec<TreeRegressorImpl>,
}

#[derive(Clone)]
pub struct RegressorConfig {
    tree_config: TreeConfig,
    ensemble_config: EnsembleConfig,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            tree_config: TreeConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::SQRT,
                seed: 42,
                metric: Metric::MSE,
            },
            ensemble_config: EnsembleConfig {
                num_trees: 100,
                num_threads: 1,
                seed: 42,
            },
        }
    }
}

#[derive(Clone)]
struct Trainee {
    tree: TreeRegressorImpl,
    tree_config: TreeConfig,
}

impl Trainee {
    fn new(tree_config: TreeConfig) -> Self {
        Self {
            tree: TreeRegressorImpl::default(),
            tree_config,
        }
    }
}

impl ensemble_trainer::Trainable<f32> for Trainee {
    fn fit(&mut self, ts: Trainset<f32>, seed: u64) {
        self.tree_config.seed = seed;
        self.tree = TreeRegressorImpl::fit(ts, &self.tree_config);
    }
}

impl ensemble_predictor::Predictor for TreeRegressorImpl {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
    }
}

/// A random forest regressor.
/// # Examples
///
/// ```
/// // Note that we have two samples (0.7, 0.0) pointing to different values: [1.0, 0.2].
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = rf::Regressor::fit(&dataset, &targets, &rf::Regressor::default_config());
/// let predictions = predictor.predict(&dataset, 1);
/// println!("Predictions: {:?}", predictions);
/// ```
impl Regressor {
    /// Predicts regression values for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<f32> {
        let view = DatasetView::new(dataset, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1, num_threads)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        let view = DatasetView::new(sample, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1, 1)[0]
    }

    /// Trains a random forest regressor with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn fit(data: &[f32], targets: &[f32], config: &RegressorConfig) -> Regressor {
        let ds = Dataset::with_transposed(data, targets.len());
        let trainee = Trainee::new(config.tree_config.clone());
        let ens = ensemble_trainer::fit(trainee, ds.as_view(), targets, &config.ensemble_config);

        Regressor {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
        }
    }

    // Returns a number of features for a trained forest.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    /// Returns training config filled with default values.
    pub fn default_config() -> RegressorConfig {
        RegressorConfig::default()
    }
}

impl TreeConfigProvider for RegressorConfig {
    fn tree_config(&mut self) -> &mut TreeConfig {
        &mut self.tree_config
    }
}

impl EnsembleConfigProvider for RegressorConfig {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.ensemble_config
    }
}

impl CommonConfigBuilder for RegressorConfig {}
impl EnsembleConfigBuilder for RegressorConfig {}
