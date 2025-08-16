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

pub mod prelude {
    pub use super::options::{
        ClassifierOptionsBuilder, EnsembleOptionsBuilder, RegressorOptionsBuilder,
        TreeOptionsBuilder,
    };
    pub use crate::decision_tree::ClassDecode;
}

pub mod dt {
    pub use super::decision_tree::tree_classifier::Classifier;
    pub use super::decision_tree::tree_regressor::Regressor;
}

pub mod rf {
    pub use crate::decision_forest::ensemble_classifier::Classifier;
    pub use crate::decision_forest::ensemble_regressor::Regressor;
}
