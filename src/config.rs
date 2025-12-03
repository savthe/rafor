//! Configurations for training.
use serde::{Deserialize, Serialize};

/// Defines a split impurit metric. Currently only 2 metrics supported: Gini index for
/// classification trees and MSE for regression trees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    GINI,
    MSE,
}

/// Configuration for training a decision tree.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Max depth of a tree.
    pub max_depth: usize,

    /// Seed for randomizing feature sets in splits if `max_features < num_features`. When used in
    /// ensemble config, it sets the seed for random numbers in bootstrapping and feature sets in
    /// splits.
    pub seed: u64,

    /// Metric of split impurity.
    pub metric: Metric,

    /// Maximum number of features to use in each split. If `max_features` is less than total
    /// amount of features, at each split will be used a random subset of features with size at
    /// least `num_features`.
    ///
    /// **Note**. If trainer is unable find a splitting value in `num_features` features, it will
    /// consider additional features.
    pub max_features: NumFeatures,

    /// Minimal number of samples in the node that can be splitted.
    pub min_samples_split: usize,

    /// Forces leaves to have at least min_samples_leaf samples.
    pub min_samples_leaf: usize,
}

/// Configuration for training the ensembles of trees.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of decision trees in ensemble.
    pub num_trees: usize,

    /// Number of threads to use. Please note that there is no specific value for "use all cores".
    /// Maximun number of theads can be obtained useing, for instance, crate `num_cpus`.
    pub num_threads: usize,
}

/// Defines the limiting strategy for a number of features that are selected at each split.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum NumFeatures {
    /// Takes `sqrt(total_features)`.
    SQRT,
    /// Takes `log2(total_features)`.
    LOG,
    /// Sets the exact number.
    NUMBER(usize),
}
