//! A crate with Random Forest implementation for training and inference focusing on delivering
//! good performance. It also provides single decision trees.
//!
//! # Dataset
//! The dataset is a single `f32` slice which is processed in chunks of `num_features` elements,
//! each chunk is a single sample. During training, `num_features` is defined as
//! `dataset.len() / targets.len()`.
//! Train will panic if `dataset.len()` is not divisible by `targets.len()`.
//! # Classification
//! Decision tree classifier (`dt::Classifier`) and random forest classifier
//! (`rf::Classifier`) expect the labels to be `i64`. Classifiers use Gini index for
//! evaluating the split impurity.
//!
//! Classifiers provide method `predict` for predicting a batch of samples, it returns `Vec<i64>`
//! with predicted class labels. Method `predict_one` returns `i64` -- a predicted class for a
//! single sample.
//!
//! To get probabilities distribution, there is a method `proba` which returns a `Vec<f32>` of
//! length `num_samples * num_classes` where `i`-th chunk of length `num_classes` elements contains the
//! probabilities of classes for `i`-th sample. During training, the `i64` class labels are mapped into
//! numbers `0, 1, ...` preserving values ordering (initial classes `5, -1, 8` mapped to `1, 0, 2`).
//! To decode classes, `Classifier` provides method `get_decode_table`, which returns
//! `&[i64]` - a map where index is an internal representation, and a value - `i64` class. Also there
//! is `decode` method which decodes single `usize` value into `i64` class.
//!
//! # Regression
//! Decision tree regressor (`rafor::dt::Regressor`) and random forest regressor (`rafor::Regressor`)
//! expect the targets to be `f32`. By default regressors use MSE score for evaluating the split
//! impurity.
//!
//! Regressor interface is mostly similar to `Classifier`, please see examples folder.
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
//!     // We have 5 samples with 3 classes.
//!     let dataset = [
//!         0.7, 0.0,
//!         0.8, 1.0,
//!         0.3, 0.0,
//!         1.0, 1.3,
//!         0.4, 2.1
//!     ];
//!     let targets = [1, 5, 1, -15, 5];
//!     let conf = Classifier::default_config()
//!         .with_max_depth(15)
//!         .with_trees(40)
//!         .with_threads(num_cpus::get())
//!         .with_seed(42)
//!         .clone();
//!     let predictor = Classifier::fit(&dataset, &targets, &conf);
//!
//!     // Get predictions for same dataset.
//!     let predictions = predictor.predict(&dataset, num_cpus::get());
//!     println!("Predictions: {:?}", predictions);
//!
//!     // Now let's get probability distributions for each class.
//!     let proba = predictor.proba(&dataset, num_cpus::get());
//!     println!("Probability distributions:");
//!     for p in proba.chunks(predictor.num_classes()) {
//!         println!("{:?}", p);
//!     }
//! }
//! ```
pub mod config;
use argminmax::ArgMinMax;
mod classes_mapping;
mod dataset;
mod decision_forest;
mod decision_tree;
mod metrics;
use classes_mapping::{ClassDecode, ClassesMapping};
mod weightable;

use dataset::{Dataset, DatasetView};
use weightable::{LabelWeight, Weightable, WEIGHT_MASK};
mod config_builders;

type IndexRange = std::ops::Range<usize>;
type ClassLabel = u32;

pub mod prelude {
    pub use crate::classes_mapping::ClassDecode;
    pub use crate::config_builders::{
        ClassifierConfigBuilder, CommonConfigBuilder, EnsembleConfigBuilder, RegressorConfigBuilder,
    };
}

pub mod dt {
    //! Decision Tree implementation.
    pub use super::decision_tree::tree_classifier::Classifier;
    pub use super::decision_tree::tree_regressor::Regressor;
}

pub mod rf {
    //! Random Forest implementation.
    pub use crate::decision_forest::ensemble_classifier::Classifier;
    pub use crate::decision_forest::ensemble_regressor::Regressor;
}

fn classify(proba: &[f32], mapping: &ClassesMapping) -> Vec<i64> {
    assert!(proba.len() % mapping.num_classes() == 0);
    proba
        .chunks(mapping.num_classes())
        .map(|c| mapping.decode(c.argmax()))
        .collect()
}
