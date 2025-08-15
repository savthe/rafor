use super::Trainset;
use super::TreeRegressorImpl;
use crate::options::{
    Metric, NumFeatures, RegressorOptionsBuilder, TreeOptions, TreeOptionsBuilder,
    TreeOptionsProvider,
};
use crate::{Dataset, DatasetView};

#[derive(Default)]
pub struct Regressor {
    regressor: TreeRegressorImpl,
}

pub struct TrainOptions {
    opts: TreeOptions,
}

impl TreeOptionsProvider for TrainOptions {
    fn tree_options(&mut self) -> &mut TreeOptions {
        &mut self.opts
    }
}

impl TreeOptionsBuilder for TrainOptions {}
impl RegressorOptionsBuilder for TrainOptions {}

/// A regression tree.
/// # Examples
///
/// ```
/// // Note that we have two samples (0.7, 0.0) pointing to different values: [1.0, 0.2].
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = dt::Regressor::fit(&dataset, &targets, &dt::Regressor::train_defaults());
/// let predictions = predictor.predict(&dataset);
/// let epsilon = 0.05;
/// assert!(0.6 - epsilon <= predictions[0] && predictions[0] <= 0.6 + epsilon);
/// assert!(0.5 - epsilon <= predictions[1] && predictions[1] <= 0.5 + epsilon);
/// assert!(0.6 - epsilon <= predictions[2] && predictions[2] <= 0.6 + epsilon);
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
    pub fn fit(dataset: &[f32], targets: &[f32], config: &TrainOptions) -> Self {
        let ds = Dataset::with_transposed(dataset, targets.len());
        let trainset = Trainset::from_dataset(ds.as_view(), targets);

        Regressor {
            regressor: TreeRegressorImpl::fit(trainset, &config.opts),
        }
    }

    // Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.regressor.num_features()
    }

    // Returns TrainOptions object filled with default values for training.
    pub fn train_defaults() -> TrainOptions {
        TrainOptions {
            opts: TreeOptions {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                metric: Metric::MSE,
                seed: 42,
            },
        }
    }
}
