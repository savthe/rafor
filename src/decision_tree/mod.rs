mod metrics;
mod decision_tree;
mod splitter;
mod classifier_model;
mod regressor_model;
pub mod trainer;
pub use classifier_model::ClassifierModel;
pub use regressor_model::RegressorModel;
use decision_tree::DecisionTree;

use crate::DatasetView;
use crate::labels::SampleWeight;

pub struct TrainView<'a, T> {
    pub dataview: DatasetView<'a>,
    pub targets: &'a [T],
    pub weights: &'a [SampleWeight],
}

impl<'a, T> TrainView<'a, T> {
    pub fn new(dataview: DatasetView<'a>, targets: &'a [T], weights: &'a [SampleWeight]) -> Self {
        Self {
            dataview,
            targets,
            weights,
        }
    }
}

