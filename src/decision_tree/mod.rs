mod classifier_model;
mod regressor_model;
pub use classifier_model::ClassifierModel;
pub use regressor_model::RegressorModel;
mod decision_tree;
mod trainer;

pub use trainer::Trainset;

use decision_tree::DecisionTree;
use decision_tree::Splittable;
