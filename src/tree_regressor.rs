use super::{decision_tree::RegressorModel, Trainset};
use crate::decision_tree;
use crate::trainer_builders::*;
use crate::transposed;
use crate::FloatTarget;

use serde::{Deserialize, Serialize};

/// A regression tree.
/// # Training
/// The [Trainer] implements [CommonTrainerBuilder]. Default training parameters:
/// ```text
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
/// seed: 42,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
///```
///
/// # Example
/// ```
/// use rafor::dt;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = dt::Regressor::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict(&dataset);
/// println!("Predictions: {:?}", predictions);
/// ```
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Regressor {
    regressor: RegressorModel,
}

/// Trainer for tree regressor.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trainer {
    pub config: decision_tree::TrainConfig,
}

impl TrainConfigProvider for Trainer {
    fn train_config(&mut self) -> &mut decision_tree::TrainConfig {
        &mut self.config
    }
}

impl CommonTrainerBuilder for Trainer {}
//impl RegressorConfigBuilder for Trainer {}

impl Trainer {
    /// Trains a regression tree with dataset given by a slice of length divisible by targets.len().
    pub fn train(&self, data: &[f32], targets: &[FloatTarget]) -> Regressor {
        let dataset = transposed(data, targets.len());
        let trainset = Trainset::new(&dataset, &targets);

        Regressor {
            regressor: RegressorModel::train(trainset, &self.config),
        }
    }
}

impl Regressor {
    /// Predicts regression values for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32]) -> Vec<FloatTarget> {
        //let view = DatasetView::new(dataset, self.regressor.num_features());
        self.regressor.predict(dataset)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        self.regressor.predict_one(sample)
    }

    /// Provides trainer for training a regressor tree.
    pub fn trainer() -> Trainer {
        Trainer::default()
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.regressor.num_features()
    }
}
