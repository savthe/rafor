use super::Trainset;
use super::TreeRegressorImpl;
use crate::config::{Metric, NumFeatures, TreeConfig};
use crate::config_builders::*;
use crate::{Dataset, DatasetView};

#[derive(Default)]
pub struct Regressor {
    regressor: TreeRegressorImpl,
}

pub struct RegressorConfig {
    config: TreeConfig,
}

impl Default for RegressorConfig {
    fn default() -> Self {
        Self {
            config: TreeConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                metric: Metric::MSE,
                seed: 42,
            },
        }
    }
}

impl TreeConfigProvider for RegressorConfig {
    fn tree_config(&mut self) -> &mut TreeConfig {
        &mut self.config
    }
}

impl CommonConfigBuilder for RegressorConfig {}
impl RegressorConfigBuilder for RegressorConfig {}

/// A regression tree.
/// # Examples
///
/// ```
/// // Note that we have two samples (0.7, 0.0) pointing to different values: [1.0, 0.2].
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = dt::Regressor::fit(&dataset, &targets, &dt::Regressor::default_config());
/// let predictions = predictor.predict(&dataset);
/// println!("Predictions: {:?}", predictions);
/// ```
impl Regressor {
    /// Predicts regression values for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        let view = DatasetView::new(dataset, self.regressor.num_features());
        self.regressor.predict(&view)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        self.regressor.predict_one(sample)
    }

    /// Trains a regression tree with dataset given by a slice of length divisible by targets.len().
    pub fn fit(dataset: &[f32], targets: &[f32], config: &RegressorConfig) -> Self {
        let ds = Dataset::with_transposed(dataset, targets.len());
        let trainset = Trainset::from_dataset(ds.as_view(), targets);

        Regressor {
            regressor: TreeRegressorImpl::fit(trainset, &config.config),
        }
    }

    // Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.regressor.num_features()
    }

    /// Returns training config filled with default values.
    pub fn default_config() -> RegressorConfig {
        RegressorConfig::default()
    }
}
