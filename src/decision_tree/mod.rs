pub mod tree_classifier_impl;
pub mod tree_regressor_impl;

pub mod tree_classifier;
pub mod tree_regressor;

mod decision_tree;
mod tree_trainer;
pub use tree_trainer::Trainset;

use decision_tree::DecisionTree;

pub use tree_classifier_impl::TreeClassifierImpl;
pub use tree_regressor_impl::TreeRegressorImpl;
