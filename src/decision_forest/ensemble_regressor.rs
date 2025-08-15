use super::{ensemble_predictor, ensemble_trainer};
use crate::{
    decision_tree::{Trainset, TreeRegressorImpl},
    options::*,
    Dataset, DatasetView,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
pub struct Regressor {
    ensemble: Vec<TreeRegressorImpl>,
}

#[derive(Clone)]
pub struct TrainOptions {
    tree_opts: TreeOptions,
    ensemble_opts: EnsembleOptions,
}

#[derive(Clone)]
struct Trainee {
    tree: TreeRegressorImpl,
    tree_opts: TreeOptions,
}

impl Trainee {
    fn new(tree_opts: TreeOptions) -> Self {
        Self {
            tree: TreeRegressorImpl::default(),
            tree_opts,
        }
    }
}

impl ensemble_trainer::Trainable<f32> for Trainee {
    fn fit(&mut self, ts: Trainset<f32>, seed: u64) {
        self.tree_opts.seed = seed;
        self.tree = TreeRegressorImpl::fit(ts, &self.tree_opts);
    }
}

impl ensemble_predictor::Predictor for TreeRegressorImpl {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
    }
}

/// A random forest regressor.
/// # Examples
///
/// ```
/// // Note that we have two samples (0.7, 0.0) pointing to different values: [1.0, 0.2].
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = rf::Regressor::fit(&dataset, &targets, &rf::Regressor::train_defaults());
/// let predictions = predictor.predict(&dataset, 1);
/// let epsilon = 0.05;
/// assert!(0.6 - epsilon <= predictions[0] && predictions[0] <= 0.6 + epsilon);
/// assert!(0.5 - epsilon <= predictions[1] && predictions[1] <= 0.5 + epsilon);
/// assert!(0.6 - epsilon <= predictions[2] && predictions[2] <= 0.6 + epsilon);
/// ```
impl Regressor {
    /// Predicts regression values for a set of samples.
    /// Dataset is a vector of floats with length multiple of num_features().
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<f32> {
        let view = DatasetView::new(dataset, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1, num_threads)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        let view = DatasetView::new(sample, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1, 1)[0]
    }

    /// Trains a random forest regressor with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn fit(data: &[f32], targets: &[f32], opts: &TrainOptions) -> Regressor {
        let ds = Dataset::with_transposed(data, targets.len());
        let trainee = Trainee::new(opts.tree_opts.clone());
        let ens = ensemble_trainer::fit(trainee, ds.as_view(), targets, &opts.ensemble_opts);

        Regressor {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
        }
    }

    // Returns a number of features for a trained forest.
    pub fn num_features(&self) -> usize {
        self.ensemble[0].num_features()
    }

    // Returns TrainOptions object filled with default values for training.
    pub fn train_defaults() -> TrainOptions {
        TrainOptions {
            tree_opts: TreeOptions {
                max_depth: usize::MAX,
                max_features: NumFeatures::SQRT,
                seed: 42,
                metric: Metric::MSE,
            },
            ensemble_opts: EnsembleOptions {
                num_trees: 100,
                num_threads: 1,
                seed: 42,
            },
        }
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
