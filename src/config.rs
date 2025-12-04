//! Configurations for training.
use serde::{Deserialize, Serialize};

//TODO move config to decision tree trainer.

/// Defines a split impurit metric. Currently only 2 metrics supported: Gini index for
/// classification trees and MSE for regression trees.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    GINI,
    MSE,
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

