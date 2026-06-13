use crate::{
    decision_tree::{self, BlockTree, Predictor, RegressorModel},
    ensemble_predictor,
    ensemble_trainer::{self, EnsembleConfig},
    trainer_builders::*,
    BatchPredictor, FloatTarget, Trainset,
};
use serde::{Deserialize, Serialize};

/// A random forest regressor.
/// # Training
/// The [Trainer] implements [CommonTrainerBuilder] and [EnsembleTrainerBuilder]. Default training
/// parameters:
/// ```text
/// max_depth: usize::MAX,
/// max_features: NumFeatures::NUMBER(usize::MAX),
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
/// let targets = [1.0, 0.5, 0.2];
/// let predictor: rf::Regressor = rf::Regressor::trainer().train(&dataset, &targets);
/// let predictions = predictor.predict_batch(&dataset, 1);
/// println!("{:?}", predictions);
/// ```
#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regressor<P: Predictor = BlockTree> {
    ensemble: Vec<RegressorModel<P>>,
}

/// Trainer for ensemble regressor.
#[derive(Clone, Debug, PartialEq)]
pub struct Trainer<P: Predictor> {
    config: EnsembleConfig,
    _marker: std::marker::PhantomData<P>,
}

impl<P: Predictor> Default for Trainer<P> {
    fn default() -> Self {
        Self {
            config: EnsembleConfig::default(),
            _marker: std::marker::PhantomData::default(),
        }
    }
}

#[derive(Clone, Default)]
struct Trainee<P: Predictor> {
    tree: RegressorModel<P>,
}

impl<P: Predictor> ensemble_trainer::Trainable<FloatTarget> for Trainee<P> {
    fn fit(&mut self, ts: &Trainset<FloatTarget>, config: decision_tree::TrainConfig) {
        self.tree = RegressorModel::train(ts, &config);
    }
}

impl<P: Predictor> BatchPredictor for RegressorModel<P> {
    fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        //self.predict(dataset)
        Self::predict(&self, dataset)
    }
}

impl<P: Predictor + Default + Sync + Send + Clone> Trainer<P> {
    /// Trains a random forest regressor with dataset given by a slice of length divisible by
    /// targets.len().
    pub fn train(&self, data: &[f32], targets: &[FloatTarget]) -> Regressor<P> {
        let trainset = Trainset::with_transposed(data, targets);
        let trainee: Trainee<P> = Trainee::default();
        let ens = ensemble_trainer::fit(trainee, &trainset, &self.config);

        Regressor {
            ensemble: ens.into_iter().map(|t| t.tree).collect(),
        }
    }
}

impl<P: Predictor + Sync + Send> Regressor<P> {
    /// Predicts regression values for a set of samples using `num_threads` threads.
    pub fn predict_batch(&self, dataset: &[f32], num_threads: usize) -> Vec<FloatTarget> {
        ensemble_predictor::predict(&self.ensemble, dataset, num_threads)
    }

    /// Predicts regression value for a single sample given by a slice of length num_features().
    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        ensemble_predictor::predict(&self.ensemble, sample, 1)[0]
    }

    /// Provides trainer for training a random forest regressor.
    pub fn trainer() -> Trainer<P> {
        Trainer::default()
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
