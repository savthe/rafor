use crate::{
    classify,
    decision_tree::{self, BlockTree, ClassifierModel, Predictor},
    trainer_builders::*,
    ClassDecode, ClassesMapping, Trainset,
};
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
/// sample_weights: empty (1.0 for each sample)
///```
///
/// # Examples
/// ```
/// use rafor::dt;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = <dt::Classifier>::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict_batch(&dataset);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Classifier<P: Predictor = BlockTree> {
    classifier: ClassifierModel<P>,
    classes_map: ClassesMapping,
}

/// A trainer for tree classifier.
#[derive(Clone, PartialEq, Debug)]
pub struct Trainer<P: Predictor> {
    config: decision_tree::TrainConfig,
    _marker: std::marker::PhantomData<P>,
}

impl<P: Predictor> TrainConfigProvider for Trainer<P> {
    fn train_config(&mut self) -> &mut decision_tree::TrainConfig {
        &mut self.config
    }
}

impl<P: Predictor> CommonTrainerBuilder for Trainer<P> {}

impl<P: Predictor> Trainer<P> {
    /// Trains a classifier tree with dataset given by a slice of length divisible by targets.len().
    pub fn train(&self, data: &[f32], labels: &[i64]) -> Classifier<P> {
        let (classes_map, encoded_labels) = ClassesMapping::with_encode(labels);
        let ts = Trainset::with_transposed(data, &encoded_labels);
        Classifier {
            classifier: ClassifierModel::train(&ts, classes_map.num_classes(), &self.config),
            classes_map,
        }
    }
}

impl<P: Predictor> Classifier<P> {
    /// Predicts classes for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict_batch(&self, dataset: &[f32]) -> Vec<i64> {
        classify(&self.proba(dataset), &self.classes_map)
    }

    /// Predicts class for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> i64 {
        self.classes_map.decode(self.proba(sample).argmax())
    }

    /// Predicts classes probabilities for each sample. The length of result vector is
    /// number_of_samples * num_classes().
    pub fn proba(&self, dataset: &[f32]) -> Vec<f32> {
        self.classifier.predict(dataset)
    }

    /// Provides trainer for training a classifier tree.
    pub fn trainer() -> Trainer<P> {
        Trainer {
            config: decision_tree::TrainConfig::default(),
            _marker: std::marker::PhantomData::default(),
        }
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
