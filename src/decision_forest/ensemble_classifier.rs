use super::{ensemble_predictor, ensemble_trainer};
use crate::{
    decision_tree::{classify, ClassDecode, ClassesMapping, Trainset, TreeClassifierImpl},
    options::*,
    ClassLabel, Dataset, DatasetView,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Classifier {
    ensemble: Vec<TreeClassifierImpl>,
    classes_map: ClassesMapping,
}

#[derive(Clone)]
pub struct TrainOptions {
    tree_opts: TreeOptions,
    ensemble_opts: EnsembleOptions,
}

#[derive(Clone)]
struct Trainee {
    tree: TreeClassifierImpl,
    num_classes: usize,
    tree_opts: TreeOptions,
}

impl ensemble_trainer::Trainable<ClassLabel> for Trainee {
    fn fit(&mut self, ts: Trainset<ClassLabel>, seed: u64) {
        self.tree_opts.seed = seed;
        self.tree = TreeClassifierImpl::fit(ts, self.num_classes, &self.tree_opts);
    }
}

impl ensemble_predictor::Predictor for TreeClassifierImpl {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
    }
}

/// A random forest classifier.
/// # Examples
///
/// ```
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1, 5, 1];
/// let predictor = rf::Classifier::fit(&dataset, &targets, &rf::Classifier::train_defaults());
/// let predictions = predictor.predict(&dataset, 1);
/// assert_eq!(&predictions, &[1, 5, 1]);
/// ```
impl Classifier {
    /// Predicts classes for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<i64> {
        classify(&self.proba(dataset, num_threads), &self.classes_map)
    }

    /// Predicts class for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> i64 {
        classify(&self.proba(sample, 1), &self.classes_map)[0]
    }

    /// Predicts classes probabilities for each sample. The length of result vector is
    /// number_of_samples * num_classes().
    pub fn proba(&self, dataset: &[f32], num_threads: usize) -> Vec<f32> {
        let dataset = DatasetView::new(dataset, self.ensemble[0].num_features());
        let num_classes = self.classes_map.num_classes();
        ensemble_predictor::predict(&self.ensemble, &dataset, num_classes, num_threads)
    }

    // Returns a number of features for a trained tree.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    /// Trains a classifier random forest with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn fit(data: &[f32], labels: &[i64], opts: &TrainOptions) -> Classifier {
        let ds = Dataset::with_transposed(data, labels.len());

        let mut classes_map = ClassesMapping::default();
        let labels_enc = classes_map.encode(labels);

        let proto = Trainee {
            tree: TreeClassifierImpl::default(),
            num_classes: classes_map.num_classes(),
            tree_opts: opts.tree_opts.clone(),
        };

        let ens = ensemble_trainer::fit(proto, ds.as_view(), &labels_enc, &opts.ensemble_opts);

        Classifier {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
            classes_map,
        }
    }

    /// Returns TrainOptions object filled with default values for training.
    pub fn train_defaults() -> TrainOptions {
        TrainOptions {
            tree_opts: TreeOptions {
                max_depth: usize::MAX,
                max_features: NumFeatures::SQRT,
                seed: 42,
                metric: Metric::GINI,
            },
            ensemble_opts: EnsembleOptions {
                num_trees: 100,
                num_threads: 1,
                seed: 42,
            },
        }
    }
}

impl ClassDecode for Classifier {
    fn get_decode_table(&self) -> &[i64] {
        self.classes_map.get_decode_table()
    }
}

impl TreeOptionsProvider for TrainOptions {
    fn tree_options(&mut self) -> &mut TreeOptions {
        &mut self.tree_opts
    }
}

impl EnsembleOptionsProvider for TrainOptions {
    fn ensemble_options(&mut self) -> &mut EnsembleOptions {
        &mut self.ensemble_opts
    }
}

impl TreeOptionsBuilder for TrainOptions {}
impl EnsembleOptionsBuilder for TrainOptions {}
