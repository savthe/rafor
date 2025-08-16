use argminmax::ArgMinMax;

pub mod tree_classifier_impl;
pub mod tree_regressor_impl;

pub mod tree_classifier;
pub mod tree_regressor;

mod classes_mapping;
mod decision_tree;
mod trainset;
mod tree_trainer;
use decision_tree::DecisionTree;

pub use classes_mapping::{ClassDecode, ClassesMapping};
pub use trainset::Trainset;
pub use tree_classifier_impl::TreeClassifierImpl;
pub use tree_regressor_impl::TreeRegressorImpl;

pub fn classify(proba: &[f32], mapping: &ClassesMapping) -> Vec<i64> {
    assert!(proba.len() % mapping.num_classes() == 0);
    proba
        .chunks(mapping.num_classes())
        .map(|c| mapping.decode(c.argmax()))
        .collect()
}
