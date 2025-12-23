use crate::{
    classify, decision_tree,
    decision_tree::ClassifierModel,
    ensemble_predictor,
    ensemble_trainer::{self, EnsembleConfig},
    trainer_builders::*,
    ClassDecode, ClassTarget, ClassesMapping, MaxFeaturesPolicy, Trainset,
};
use serde::{Deserialize, Serialize};
/// A random forest classifier.
/// # Training
/// The [Trainer] implements [CommonTrainerBuilder] and [EnsembleTrainerBuilder]. Default training
/// parameters:
/// ```text
/// max_depth: usize::MAX,
/// max_features: NumFeatures::SQRT,
/// seed: 42,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
/// sample_weights: empty (1.0 for each sample)
/// num_trees: 100,
/// num_threads: 1,
///```
/// # Example
/// ```
/// use rafor::rf;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = rf::Classifier::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict(&dataset, 1);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
///
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Classifier {
    ensemble: Vec<ClassifierModel>,
    classes_map: ClassesMapping,
}

/// Trainer for ensemble classifier.
#[derive(Clone, PartialEq, Debug)]
pub struct Trainer {
    pub config: EnsembleConfig,
}

impl Default for Trainer {
    fn default() -> Self {
        let mut config = EnsembleConfig::default();
        config.tree_config_proto.max_features = MaxFeaturesPolicy::SQRT;
        Self { config }
    }
}

#[derive(Clone)]
struct Trainee {
    tree: ClassifierModel,
    num_classes: usize,
}

impl ensemble_trainer::Trainable<ClassTarget> for Trainee {
    fn fit(&mut self, ts: &Trainset<ClassTarget>, config: decision_tree::TrainConfig) {
        self.tree = ClassifierModel::train(ts, self.num_classes, &config);
    }
}

impl ensemble_predictor::Predictor for ClassifierModel {
    fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        self.predict(dataset)
    }
}

impl Trainer {
    /// Trains a classifier random forest with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn train(&self, data: &[f32], labels: &[i64]) -> Classifier {
        let (classes_map, labels_enc) = ClassesMapping::with_encode(labels);

        let proto = Trainee {
            tree: ClassifierModel::default(),
            num_classes: classes_map.num_classes(),
        };
        let trainset = Trainset::with_transposed(data, &labels_enc);

        let ens = ensemble_trainer::fit(proto, &trainset, &self.config);

        Classifier {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
            classes_map,
        }
    }
}

impl Classifier {
    /// Predicts classes for a set of samples using `num_threads` threads.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<i64> {
        classify(&self.proba(dataset, num_threads), &self.classes_map)
    }

    /// Predicts class for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> i64 {
        classify(&self.proba(sample, 1), &self.classes_map)[0]
    }

    /// Predicts classes probabilities for each sample using `num_threads` threads. The length of
    /// result vector is number_of_samples * num_classes().
    pub fn proba(&self, dataset: &[f32], num_threads: usize) -> Vec<f32> {
        ensemble_predictor::predict(&self.ensemble, dataset, num_threads)
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    /// Provides trainer for training a random forest classifier.
    pub fn trainer() -> Trainer {
        Trainer::default()
    }
}

impl ClassDecode for Classifier {
    fn get_decode_table(&self) -> &[i64] {
        self.classes_map.get_decode_table()
    }
}

impl TrainConfigProvider for Trainer {
    fn train_config(&mut self) -> &mut decision_tree::TrainConfig {
        &mut self.config.tree_config_proto
    }
}

impl EnsembleConfigProvider for Trainer {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.config
    }
}

impl CommonTrainerBuilder for Trainer {}
impl EnsembleTrainerBuilder for Trainer {}
