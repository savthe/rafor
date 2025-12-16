//! A decision trees and random forest implementation focused on delivering good performance.
//!
//! # Classification
//! Rafor provide a decision tree (DT) classifier [`dt::Classifier`] and a random forest (RF) classifier
//! [`rf::Classifier`]. The class label is `i64` value. Classifiers use Gini index for
//! evaluating the split impurity.
//!
//! Classifiers provide method `predict` for predicting a batch of samples, it returns `Vec<i64>`
//! with predicted class labels. Method `predict_one` returns `i64` -- a predicted class for a
//! single sample.
//!
//! To get probabilities distribution, there is a method `proba` which returns a `Vec<f32>` of
//! length `num_samples * num_classes` where `i`-th chunk of length `num_classes` contains the
//! probabilities of classes for `i`-th sample. The classes are ordered by their values.
//!
//! # Regression
//! Regression models are decision tree regressor [`dt::Regressor`] and random forest regressor
//! [`rf::Regressor`]. The targets are `f32` values. By default regressors use MSE score for evaluating
//! the split impurity.
//!
//! # Dataset
//! Multiple samples for inference or training are provided as a single `f32` slice, where each chunk of
//! the size of feature space (`num_features`) is treated as a feature vector of a single sample.
//! During training, `num_features` is derieved as a length of the `f32` input vector of samples
//! deviced by the number of proviced targets.
//!
//! # Model training
//! All models provide method `trainer()` which returns a `Trainer` object for particular model. The
//! `Trainer` incorporates builder interface (`use rafor::prelude::*`) for setting optional
//! train parameters and a method `train` for feeding dataset and targets.
//!
//! Currently supported training parameters are given below. Please see default values in concrete
//! models.
//! ## Common parameters
//! The following parameters are common for decision trees and forests.
//!
//! `max_depth: usize` defines the maximal tree depth.
//!
//! `max_features`: [MaxFeaturesPolicy], the maximal number of features that are considered when finding
//! best split value for decision tree node. Note that if no split value found, additional features
//! will be considered until split is found or all features used.
//!
//! `seed: u64`, defines the seed for random number generator. For trees the random numbers are
//! used for generating the feature sequence when finding split when `max_features` is less than the
//! number of all features of training dataset. In RF, the datasets are generated using random sampling,
//! also the seeds for individual trees are randomly generated, because in RF by default `max_features`
//! is less than the total number of features.
//!
//! `min_samples_leaf: usize`, guarantees that each leaf has at least `min_samples_leaf` nodes.
//!  Default: `1`.
//!
//! `min_samples_split: usize`, the minimal samples in node to consider splitting it.
//!
//! ## Ensemble parameters
//! `num_trees: usize` defines the number of individual trees in ensemble.
//!
//! `num_threads: usize` defines the number of CPU threads to use for training.
//!
//! # Model serialization and deserialization
//! All models support [serde](https://docs.rs/serde/latest/serde/), so any lib that supports `serde`
//! can be used for serialization and deserialization.
//!
//! # Example
//! ```
//! use rafor::prelude::*; // Required for .with_option builders and .num_classes().
//! use rafor::rf::Classifier;
//! use num_cpus; // Requires num_cpus dependency in Cargo.toml
//!
//! fn main() {
//!     // Dataset for 5 samples (number of samples is defined by the number of targets).
//!     let dataset = [
//!         0.7, 0.0,
//!         0.8, 1.0,
//!         0.3, 0.0,
//!         1.0, 1.3,
//!         0.4, 2.1
//!     ];
//!
//!     // Target classes.
//!     let targets = [1, 5, 1, -15, 5];
//!
//!     let predictor = Classifier::trainer()
//!         .with_max_depth(15)
//!         .with_trees(40)
//!         .with_threads(num_cpus::get())
//!         .with_seed(42)
//!         .train(&dataset, &targets);
//!
//!     // Get predictions for same dataset.
//!     let predictions = predictor.predict(&dataset, num_cpus::get());
//!     println!("Predictions: {:?}", predictions);
//!
//!     // Now let's get probability distributions for each class. Use all CPU cores.
//!     let proba = predictor.proba(&dataset, num_cpus::get());
//!     println!("Probability distributions:");
//!     for p in proba.chunks(predictor.num_classes()) {
//!         println!("{:?}", p);
//!     }
//! }
//! ```
//!
//! # Space / performance considerations
//! Rafor utilizes compact trees representation under the following restrictions:
//! 1. split threshold is `f32`;
//! 2. feature index is `u16`, up to 2^16 = 65,536 features allowed;
//! 3. in regression tasks, the target type is `f32`;
//! 4. in classification tasks, the class is represented by `u32` (the input `i64` labels are mapped
//! into `u32` internally, and restored during prediction);
//! 5. child node index is `u32`, up to 2^32 = 4,294,967,296 nodes allowed.
//!
//! The decision tree is represented by a vector of internal (parent) nodes. The leaf value
//! (`f32` for regression trees, `u32` index pointing to the class probabilities for classification
//! trees) is bit-packed into parent's `u32` child node index.
mod classes_mapping;
mod decision_tree;
pub mod ensemble_classifier;
mod ensemble_predictor;
pub mod ensemble_regressor;
mod ensemble_trainer;
pub mod trainer_builders;
pub mod tree_classifier;
pub mod tree_regressor;
use argminmax::ArgMinMax;
use classes_mapping::{ClassDecode, ClassesMapping};
pub use decision_tree::MaxFeaturesPolicy;

type ClassTarget = u32;
type FloatTarget = f32;
type SampleWeight = f32;

type IndexRange = std::ops::Range<usize>;

pub mod prelude {
    // TODO pub use from lib.
    pub use crate::classes_mapping::ClassDecode;
    pub use crate::trainer_builders::{CommonTrainerBuilder, EnsembleTrainerBuilder};
    pub use crate::MaxFeaturesPolicy;
}

pub mod dt {
    //! Decision Tree implementation.
    pub use super::tree_classifier::Classifier;
    pub use super::tree_regressor::Regressor;
}

pub mod rf {
    //! Random Forest implementation.
    pub use crate::ensemble_classifier::Classifier;
    pub use crate::ensemble_regressor::Regressor;
}

fn classify(proba: &[f32], mapping: &ClassesMapping) -> Vec<i64> {
    assert!(proba.len() % mapping.num_classes() == 0);
    proba
        .chunks(mapping.num_classes())
        .map(|c| mapping.decode(c.argmax()))
        .collect()
}

pub fn transposed(data: &[f32], num_samples: usize) -> Vec<f32> {
    assert!(data.len() % num_samples == 0);
    let num_features = data.len() / num_samples;

    let mut res: Vec<f32> = Vec::with_capacity(data.len());
    for feature in 0..num_features {
        res.extend(data.iter().skip(feature).step_by(num_features));
    }

    res
}

#[derive(Clone, PartialEq, Debug)]
struct Trainset<'a, T> {
    pub data: &'a [f32],
    pub targets: &'a [T],
    //    pub weights: Vec<SampleWeight>,
}

impl<'a, T> Trainset<'a, T> {
    pub fn new(data: &'a [f32], targets: &'a [T]) -> Self {
        Self {
            data,
            targets,
            //           weights: vec![1.0; targets.len()],
        }
    }

    // pub fn scale_weights(&mut self, scalars: &[SampleWeight]) {
    //     assert!(scalars.is_empty() || scalars.len() == self.weights.len());
    //
    //     self.weights
    //         .iter_mut()
    //         .zip(scalars)
    //         .for_each(|(w, s)| *w *= s);
    // }
    //
    pub fn size(&self) -> usize {
        self.targets.len()
    }
}

#[cfg(test)]
mod tests;
