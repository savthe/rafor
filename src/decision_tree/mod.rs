mod classifier_model;
mod decision_tree;
mod metrics;
mod regressor_model;
mod splitter;
pub mod trainer;
pub use classifier_model::ClassifierModel;
use decision_tree::DecisionTree;
pub use regressor_model::RegressorModel;

use crate::{DatasetView, SampleWeight};

pub struct TrainView<'a, T> {
    pub dataview: DatasetView<'a>,
    pub targets: &'a [T],
    pub weights: Vec<SampleWeight>,
}

impl<'a, T> TrainView<'a, T> {
    pub fn new(dataview: DatasetView<'a>, targets: &'a [T], weights: &[SampleWeight]) -> Self {
        let weights = if weights.is_empty() {
            vec![1.; targets.len()]
        } else {
            weights.to_vec()
        };

        assert!(weights.len() == targets.len());
        
        Self {
            dataview,
            targets,
            weights,
        }
    }
}
