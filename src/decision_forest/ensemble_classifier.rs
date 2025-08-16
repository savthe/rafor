use super::{ensemble_predictor, ensemble_trainer};
use crate::{
    config::*,
    config_builders::*,
    decision_tree::{classify, ClassDecode, ClassesMapping, Trainset, TreeClassifierImpl},
    ClassLabel, Dataset, DatasetView,
};
use serde::{Deserialize, Serialize};

/// A random forest classifier.
/// # Example
/// ```
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = rf::Classifier::fit(&dataset, &targets, &rf::Classifier::default_config());
/// let predictions = predictor.predict(&dataset, 1);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
#[derive(Default, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Classifier {
    ensemble: Vec<TreeClassifierImpl>,
    classes_map: ClassesMapping,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ClassifierConfig {
    train_config: TrainConfig,
    ensemble_config: EnsembleConfig,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            train_config: TrainConfig {
                max_depth: usize::MAX,
                max_features: NumFeatures::SQRT,
                seed: 42,
                metric: Metric::GINI,
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
    tree: TreeClassifierImpl,
    num_classes: usize,
    conf: TrainConfig,
}

impl ensemble_trainer::Trainable<ClassLabel> for Trainee {
    fn fit(&mut self, ts: Trainset<ClassLabel>, seed: u64) {
        self.conf.seed = seed;
        self.tree = TreeClassifierImpl::fit(ts, self.num_classes, &self.conf);
    }
}

impl ensemble_predictor::Predictor for TreeClassifierImpl {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
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
        let num_classes = self.classes_map.num_classes();
        ensemble_predictor::predict(&self.ensemble, &dataset, num_classes, num_threads)
    }

    /// Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    /// Trains a classifier random forest with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn fit(data: &[f32], labels: &[i64], conf: &ClassifierConfig) -> Classifier {
        let ds = Dataset::with_transposed(data, labels.len());

        let mut classes_map = ClassesMapping::default();
        let labels_enc = classes_map.encode(labels);

        let proto = Trainee {
            tree: TreeClassifierImpl::default(),
            num_classes: classes_map.num_classes(),
            conf: conf.train_config.clone(),
        };

        let ens = ensemble_trainer::fit(
            proto,
            ds.as_view(),
            &labels_enc,
            &conf.ensemble_config,
            conf.train_config.seed,
        );

        Classifier {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
            classes_map,
        }
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

impl TrainConfigProvider for ClassifierConfig {
    fn train_config(&mut self) -> &mut TrainConfig {
        &mut self.train_config
    }
}

impl EnsembleConfigProvider for ClassifierConfig {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.ensemble_config
    }
}

impl CommonConfigBuilder for ClassifierConfig {}
impl EnsembleConfigBuilder for ClassifierConfig {}
