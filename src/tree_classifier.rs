use super::{decision_tree::ClassifierModel, TrainView};
use crate::{
    classify,
    trainer_builders::*,
    ClassDecode, ClassesMapping, Dataset, DatasetView,
    decision_tree
};
use crate::MaxFeaturesPolicy;
use argminmax::ArgMinMax;
use serde::{Deserialize, Serialize};

/// A classifier tree.
/// # Training
/// The [Trainer] implements [CommonTrainerBuilder]. Default training
/// parameters:
/// ```text
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
/// seed: 42,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
///```
///
/// # Examples
/// ```
/// use rafor::dt;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = dt::Classifier::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict(&dataset);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Classifier {
    classifier: ClassifierModel,
    classes_map: ClassesMapping,
}

/// A trainer for tree classifier.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Trainer {
    config: decision_tree::trainer::Config
}

impl Default for Trainer {
    fn default() -> Self {
        Self {
            config: decision_tree::trainer::Config {
                max_depth: usize::MAX,
                max_features: MaxFeaturesPolicy::NUMBER(usize::MAX),
                seed: 42,
                min_samples_leaf: 1,
                min_samples_split: 2,
            },
        }
    }
}

impl TrainConfigProvider for Trainer {
    fn train_config(&mut self) -> &mut decision_tree::trainer::Config {
        &mut self.config
    }
}

impl CommonTrainerBuilder for Trainer {}
//impl ClassifierConfigBuilder for Trainer {}

impl Trainer {
    /// Trains a classifier tree with dataset given by a slice of length divisible by targets.len().
    pub fn train(&self, raw_dataset: &[f32], labels: &[i64]) -> Classifier {
        let dataset = Dataset::with_transposed(raw_dataset, labels.len());
        let (classes_map, encoded_labels) = ClassesMapping::with_encode(labels);
        let weights = vec![1.; labels.len()];
        let tv = TrainView::new(dataset.as_view(), &encoded_labels, &weights);
        Classifier {
            classifier: ClassifierModel::train(tv, classes_map.num_classes(), &self.config),
            classes_map,
        }
    }
}

impl Classifier {
    /// Predicts classes for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32]) -> Vec<i64> {
        classify(&self.proba(dataset), &self.classes_map)
    }

    /// Predicts class for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> i64 {
        self.classes_map.decode(self.proba(sample).argmax())
    }

    /// Predicts classes probabilities for each sample. The length of result vector is
    /// number_of_samples * num_classes().
    pub fn proba(&self, dataset: &[f32]) -> Vec<f32> {
        let view = DatasetView::new(dataset, self.classifier.num_features());
        self.classifier.predict(&view)
    }

    /// Provides trainer for training a classifier tree.
    pub fn trainer() -> Trainer {
        Trainer::default()
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.classifier.num_features()
    }
}

impl ClassDecode for Classifier {
    fn get_decode_table(&self) -> &[i64] {
        self.classes_map.get_decode_table()
    }
}

