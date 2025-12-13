use crate::{
    trainer_builders::*, decision_tree::RegressorModel, ensemble_predictor,
    ensemble_trainer, Dataset, DatasetView, FloatTarget, TrainView,
    decision_tree
};
use crate::MaxFeaturesPolicy;
use serde::{Deserialize, Serialize};
use ensemble_trainer::EnsembleConfig;

/// A random forest regressor.
/// # Training
/// The [Trainer] implements [CommonTrainerBuilder] and [EnsembleTrainerBuilder]. Default training
/// parameters:
/// ```text
/// max_depth: usize::MAX,
/// max_features: NumFeatures::SQRT,
/// seed: 42,
/// min_samples_leaf: 1,
/// min_samples_split: 2,
/// num_trees: 100,
/// num_threads: 1,
///```
/// # Example
/// ```
/// use rafor::rf;
/// let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
/// let targets = [1.0, 0.5, 0.2];
/// let predictor = rf::Regressor::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict(&dataset, 1);
/// println!("{:?}", predictions);
/// ```
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regressor {
    ensemble: Vec<RegressorModel>,
}

/// Trainer for ensemble regressor.
#[derive(Clone, Debug, PartialEq)]
pub struct Trainer {
    pub train_config: decision_tree::trainer::Config,
    pub ensemble_config: EnsembleConfig,
}

impl Default for Trainer {
    fn default() -> Self {
        Self {
            train_config: decision_tree::trainer::Config {
                max_depth: usize::MAX,
                max_features: MaxFeaturesPolicy::NUMBER(usize::MAX),
                seed: 42,
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
    tree: RegressorModel,
    train_config: decision_tree::trainer::Config,
}

impl Trainee {
    fn new(train_config: decision_tree::trainer::Config) -> Self {
        Self {
            tree: RegressorModel::default(),
            train_config,
        }
    }
}

impl ensemble_trainer::Trainable<FloatTarget> for Trainee {
    fn fit(&mut self, tv: TrainView<FloatTarget>, seed: u64) {
        self.train_config.seed = seed;
        self.tree = RegressorModel::train(tv, &self.train_config);
    }
}

impl ensemble_predictor::Predictor for RegressorModel {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        self.predict(dataset)
    }
}

impl Trainer {
    /// Trains a random forest regressor with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn train(&self, data: &[f32], targets: &[FloatTarget]) -> Regressor {
        let ds = Dataset::with_transposed(data, targets.len());
        let trainee = Trainee::new(self.train_config.clone());
        let ens = ensemble_trainer::fit(
            trainee,
            ds.as_view(),
            targets,
            &self.ensemble_config,
            self.train_config.seed,
        );

        Regressor {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
        }
    }
}
impl Regressor {
    /// Predicts regression values for a set of samples using `num_threads` threads.
    pub fn predict(&self, dataset: &[f32], num_threads: usize) -> Vec<FloatTarget> {
        let view = DatasetView::new(dataset, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, num_threads)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        let view = DatasetView::new(sample, self.ensemble[0].num_features());
        ensemble_predictor::predict(&self.ensemble, &view, 1)[0]
    }

    /// Provides trainer for training a random forest regressor.
    pub fn trainer() -> Trainer {
        Trainer::default() 
    }
}

impl TrainConfigProvider for Trainer {
    fn train_config(&mut self) -> &mut decision_tree::trainer::Config {
        &mut self.train_config
    }
}

impl EnsembleConfigProvider for Trainer {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig {
        &mut self.ensemble_config
    }
}

impl CommonTrainerBuilder for Trainer {}
impl EnsembleTrainerBuilder for Trainer {}
