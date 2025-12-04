use crate::{
    classify, config::*, config_builders::*, decision_tree::ClassifierModel, ensemble_predictor,
    ensemble_trainer, ClassDecode, ClassTarget, ClassesMapping, Dataset, DatasetView, TrainView,
};
use serde::{Deserialize, Serialize};

/// A random forest classifier.
/// # Example
/// ```
/// use rafor::rf;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = rf::Classifier::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict(&dataset, 1);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Classifier {
    ensemble: Vec<ClassifierModel>,
    classes_map: ClassesMapping,
}

/// Trainer for ensemble classifier.
/// # Default values:
/// ```ignore
/// max_depth: usize::MAX,
/// max_features: NumFeatures::SQRT,
/// seed: 42,
/// metric: Metric::GINI,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
/// num_trees: 100,
/// num_threads: 1,
///```
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Trainer {
    pub train_config: TrainConfig,
    pub ensemble_config: EnsembleConfig,
}

impl Default for Trainer {
    fn default() -> Self {
        Self {
            train_config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::SQRT,
                seed: 42,
                metric: Metric::GINI,
                min_samples_leaf: 1,
                min_samples_split: 2,
            },
            ensemble_config: EnsembleConfig {
                num_trees: 100,
                num_threads: 1,
            },
        }
    }
}

#[derive(Clone)]
struct Trainee {
    tree: ClassifierModel,
    num_classes: usize,
    conf: TrainConfig,
}

impl ensemble_trainer::Trainable<ClassTarget> for Trainee {
    fn fit(&mut self, tv: TrainView<ClassTarget>, seed: u64) {
        self.conf.seed = seed;
        self.tree = ClassifierModel::train(tv, self.num_classes, &self.conf);
    }
}

impl ensemble_predictor::Predictor for ClassifierModel {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
    }
}

impl Trainer {
    /// Trains a classifier random forest with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn train(&self, data: &[f32], labels: &[i64]) -> Classifier {
        let ds = Dataset::with_transposed(data, labels.len());

        let (classes_map, labels_enc) = ClassesMapping::with_encode(labels);

        let proto = Trainee {
            tree: ClassifierModel::default(),
            num_classes: classes_map.num_classes(),
            conf: self.train_config.clone(),
        };

        // TODO config by ref or copy
        let ens = ensemble_trainer::fit(
            proto,
            ds.as_view(),
            &labels_enc,
            &self.ensemble_config,
            self.train_config.seed,
        );

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
        let dataset = DatasetView::new(dataset, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &dataset, num_threads)
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
    fn train_config(&mut self) -> &mut TrainConfig {
        &mut self.train_config
    }
}

impl EnsembleConfigProvider for Trainer {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.ensemble_config
    }
}

impl CommonConfigBuilder for Trainer {}
impl EnsembleConfigBuilder for Trainer {}
