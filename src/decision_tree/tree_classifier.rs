use super::{classify, ClassDecode, ClassesMapping, Trainset, TreeClassifierImpl};
use crate::{
    config::{Metric, NumFeatures, TrainConfig},
    config_builders::*,
    Dataset, DatasetView,
};
use serde::{Deserialize, Serialize};

/// A classifier tree.
/// # Examples
/// ```
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = dt::Classifier::fit(&dataset, &targets, &dt::Classifier::default_config());
/// let predictions = predictor.predict(&dataset);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Serialize, Deserialize)]
pub struct Classifier {
    classifier: TreeClassifierImpl,
    classes_map: ClassesMapping,
}

pub struct ClassifierConfig {
    config: TrainConfig,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::NUMBER(usize::MAX),
                seed: 42,
                metric: Metric::GINI,
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
        classify(&self.proba(sample), &self.classes_map)[0]
    }

    /// Predicts classes probabilities for each sample. The length of result vector is
    /// number_of_samples * num_classes().
    pub fn proba(&self, dataset: &[f32]) -> Vec<f32> {
        let view = DatasetView::new(dataset, self.classifier.num_features());
        self.classifier.predict(&view)
    }

    /// Trains a classifier tree with dataset given by a slice of length divisible by targets.len().
    pub fn fit(data: &[f32], labels: &[i64], config: &ClassifierConfig) -> Self {
        let ds = Dataset::with_transposed(data, labels.len());

        let mut classes_map = ClassesMapping::default();
        let encoded_labels = classes_map.encode(labels);

        let trainset = Trainset::from_dataset(ds.as_view(), &encoded_labels);

        Classifier {
            classifier: TreeClassifierImpl::fit(
                trainset,
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
