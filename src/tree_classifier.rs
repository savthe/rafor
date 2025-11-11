use super::{decision_tree::ClassifierModel, TrainView};
use crate::{
    classify,
    config::{Metric, NumFeatures, TrainConfig},
    config_builders::*,
    ClassDecode, ClassesMapping, Dataset, DatasetView,
};
use argminmax::ArgMinMax;
use serde::{Deserialize, Serialize};

/// A classifier tree.
/// # Examples
/// ```
/// use rafor::dt;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = dt::Classifier::fit(&dataset, &targets, &dt::Classifier::default_config());
/// let predictions = predictor.predict(&dataset);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Classifier {
    classifier: ClassifierModel,
    classes_map: ClassesMapping,
}

/// A training configuration for tree classifier. Default values:
/// ```ignore
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
/// seed: 42,
/// metric: Metric::GINI,
/// min_samples_leaf: 1,
/// min_samples_split: 2
/// ```
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub config: TrainConfig,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                seed: 42,
                metric: Metric::GINI,
                min_samples_leaf: 1,
                min_samples_split: 2,
            },
        }
    }
}

impl TrainConfigProvider for ClassifierConfig {
    fn train_config(&mut self) -> &mut TrainConfig {
        &mut self.config
    }
}

impl CommonConfigBuilder for ClassifierConfig {}
impl ClassifierConfigBuilder for ClassifierConfig {}

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

    /// Trains a classifier tree with dataset given by a slice of length divisible by targets.len().
    pub fn fit(raw_dataset: &[f32], labels: &[i64], config: &ClassifierConfig) -> Self {
        let dataset = Dataset::with_transposed(raw_dataset, labels.len());
        let (classes_map, encoded_labels) = ClassesMapping::with_encode(labels);
        let weights = vec![1; labels.len()];
        let tv = TrainView::new(dataset.as_view(), &encoded_labels, &weights);
        Classifier {
            classifier: ClassifierModel::fit(tv,
                classes_map.num_classes(),
                &config.config,
            ),
            classes_map,
        }
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.classifier.num_features()
    }

    /// Returns training config filled with default values.
    pub fn default_config() -> ClassifierConfig {
        ClassifierConfig::default()
    }
}

impl ClassDecode for Classifier {
    fn get_decode_table(&self) -> &[i64] {
        self.classes_map.get_decode_table()
    }
}

// #[rustfmt::skip]
// #[test]
// fn overfit() {
//     let dataset = [
//         0.6, 1.0, 
//         0.7, 0.0,
//         0.8, 1.0, 
//         0.4, -1.0, 
//         0.4, -2.0, 
//         0.4, 1.0, 
//         0.4, 2.0 
//     ];
//
//     let targets = [1, 1, 1, 0, 0, 1, 1];
//     let predictor = Classifier::fit(&dataset, &targets, &Classifier::default_config());
//     let predictions = predictor.predict(&dataset);
//     assert_eq!(&predictions, &[1, 1, 1, 0, 0, 1, 1]);
// }
//
// #[rustfmt::skip]
// #[test]
// fn proba() {
//     let dataset = [
//         0.1, 0.1, 
//         0.2, 0.2, 
//         0.1, 0.1,
//         0.1, 0.1, 
//         0.1, 0.1, 
//     ];
//
//     let targets = [0, 2, 0, 0, 1];
//     let predictor = Classifier::fit(&dataset, &targets, &Classifier::default_config());
//
//     assert_eq!(predictor.proba(&[0.1, 0.1]), &[0.75, 0.25, 0.0]);
//     assert_eq!(predictor.proba(&[0.2, 0.2]), &[0.0, 0.0, 1.0]);
// }
