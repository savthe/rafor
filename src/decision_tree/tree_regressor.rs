use super::Trainset;
use super::TreeRegressorImpl;
use crate::config::{Metric, NumFeatures, TrainConfig};
use crate::config_builders::*;
use crate::{Dataset, DatasetView, FloatTarget};

use serde::{Deserialize, Serialize};

/// A regression tree.
/// # Example
/// ```
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = dt::Regressor::fit(&dataset, &targets, &dt::Regressor::default_config());
/// let predictions = predictor.predict(&dataset);
/// println!("Predictions: {:?}", predictions);
/// ```
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Regressor {
    regressor: TreeRegressorImpl,
}

/// Configuration for tree regressor. Default values:
/// ```
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
/// metric: Metric::MSE,
/// seed: 42,
/// min_samples_leaf: 1,
/// min_samples_split: 2
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressorConfig {
    pub config: TrainConfig,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                metric: Metric::MSE,
                seed: 42,
                min_samples_leaf: 1,
                min_samples_split: 2
            },
        }
    }
}

impl TrainConfigProvider for RegressorConfig {
    fn train_config(&mut self) -> &mut TrainConfig {
        &mut self.config
    }
}

impl CommonConfigBuilder for RegressorConfig {}
impl RegressorConfigBuilder for RegressorConfig {}

impl Regressor {
    /// Predicts regression values for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32]) -> Vec<FloatTarget> {
        let view = DatasetView::new(dataset, self.regressor.num_features());
        self.regressor.predict(&view)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        self.regressor.predict_one(sample)
    }

    /// Trains a regression tree with dataset given by a slice of length divisible by targets.len().
    pub fn fit(dataset: &[f32], targets: &[FloatTarget], config: &RegressorConfig) -> Self {
        let ds = Dataset::with_transposed(dataset, targets.len());
        let trainset = Trainset::from_dataset(ds.as_view(), targets);

        Regressor {
            regressor: TreeRegressorImpl::fit(trainset, &config.config),
        }
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.regressor.num_features()
    }

    /// Returns training config filled with default values.
    pub fn default_config() -> RegressorConfig {
        RegressorConfig::default()
    }
}
