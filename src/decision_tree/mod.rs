use argminmax::ArgMinMax;

pub mod tree_classifier_impl;
pub mod tree_regressor_impl;

pub mod tree_classifier;
pub mod tree_regressor;

mod decision_tree;
mod trainset;
mod tree_trainer;
use decision_tree::DecisionTree;

pub use trainset::Trainset;
pub use tree_classifier_impl::TreeClassifierImpl;
pub use tree_regressor_impl::TreeRegressorImpl;
