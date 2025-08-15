mod dataset;
mod decision_forest;
mod decision_tree;
mod metrics;
pub mod options;
mod utils;
mod weight;
use dataset::{Dataset, DatasetView};
use weight::Weightable;

type IndexRange = std::ops::Range<usize>;
type ClassLabel = u32;
type LabelWeight = u32;

pub use decision_forest::ensemble_classifier::Classifier;
pub use decision_forest::ensemble_regressor::Regressor;

pub mod dt {
    pub use super::decision_tree::tree_classifier::Classifier;
    pub use super::decision_tree::tree_regressor::Regressor;
}

pub mod builders {
    pub use super::options::{
        ClassifierOptionsBuilder, EnsembleOptionsBuilder, RegressorOptionsBuilder,
        TreeOptionsBuilder,
    };
}
