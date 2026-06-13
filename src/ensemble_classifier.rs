use crate::{
    classify, decision_tree,
    decision_tree::{BlockTree, ClassifierModel, Predictor},
    ensemble_predictor,
    ensemble_trainer::{self, EnsembleConfig},
    trainer_builders::*,
    BatchPredictor, ClassDecode, ClassTarget, ClassesMapping, MaxFeaturesPolicy, Trainset,
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
/// let predictor: rf::Classifier = rf::Classifier::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict_batch(&dataset, 1);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
///
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Classifier<P: Predictor = BlockTree> {
    ensemble: Vec<ClassifierModel<P>>,
    classes_map: ClassesMapping,
}

/// Trainer for ensemble classifier.
#[derive(Clone, PartialEq, Debug)]
pub struct Trainer<P: Predictor> {
    pub config: EnsembleConfig,
    _marker: std::marker::PhantomData<P>,
}

impl<P: Predictor> Default for Trainer<P> {
    fn default() -> Self {
        let mut config = EnsembleConfig::default();
        config.tree_config_proto.max_features = MaxFeaturesPolicy::SQRT;
        Self {
            config,
            _marker: std::marker::PhantomData::default(),
        }
    }
}

#[derive(Clone)]
struct Trainee<P: Predictor> {
    tree: ClassifierModel<P>,
    num_classes: usize,
}

impl<P: Predictor> ensemble_trainer::Trainable<ClassTarget> for Trainee<P> {
    fn fit(&mut self, ts: &Trainset<ClassTarget>, config: decision_tree::TrainConfig) {
        self.tree = ClassifierModel::train(ts, self.num_classes, &config);
    }
}

impl<P: Predictor> BatchPredictor for ClassifierModel<P> {
    fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        Self::predict(self, dataset)
    }
}

impl<P: Predictor + Default + Clone + Sync + Send> Trainer<P> {
    /// Trains a classifier random forest with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn train(&self, data: &[f32], labels: &[i64]) -> Classifier<P> {
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

impl<P: Predictor + Send + Sync> Classifier<P> {
    /// Predicts classes for a set of samples using `num_threads` threads.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict_batch(&self, dataset: &[f32], num_threads: usize) -> Vec<i64> {
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
    pub fn trainer() -> Trainer<P> {
        Trainer::default()
    }
}

impl<P: Predictor> ClassDecode for Classifier<P> {
    fn get_decode_table(&self) -> &[i64] {
        self.classes_map.get_decode_table()
    }
}

impl<P: Predictor> TrainConfigProvider for Trainer<P> {
    fn train_config(&mut self) -> &mut decision_tree::TrainConfig {
        &mut self.config.tree_config_proto
    }
}

impl<P: Predictor> EnsembleConfigProvider for Trainer<P> {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.config
    }
}

impl<P: Predictor> CommonTrainerBuilder for Trainer<P> {}
impl<P: Predictor> EnsembleTrainerBuilder for Trainer<P> {}
