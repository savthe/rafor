# Overview
Rafor is a performance-oriented decision trees and random forest library.

The dataset is a single `f32` slice which is processed in chunks of `num_features` elements,
each chunk is a single sample. During training, `num_features` is defined as 
`dataset.len() / targets.len()`.
Train will panic if `dataset.len()` is not divisible by `targets.len()`.

# Classification
Decision tree classifier (`rafor::dt::Classifier`) and random forest classifier
(`rafor::Classifier`) expect the labels to be `i64`. By default classifiers use Gini index for
evaluating the split impurity.

Classifiers provide method `predict` for predicting a batch of samples, it returns `Vec<i64>` with
predicted class labels. Method `predict_one` returns `i64` -- a predicted class for a single sample.

To get probabilities distribution, there is a method `proba` which returns a `Vec<f32>` of length
$NumSamples \cdot NumClasses$ where each chunk of $NumClasses$ elements contains the probabilities
of classes for a sample. Internally the `i64` class labels are mapped into numbers `0, 1, ...` of
type `u32`. To decode classes, `Classifier` provides method `get_decode_table`, which returns 
`&[i64]` - a map where index is an internal representation, and a value - `i64` class. Also there
is `decode` method which receives `u32` internal label and returns `i64` value.

```Rust
use rafor::builders::*; // Required to use .with_option builders.
use rafor::Classifier;
use num_cpus; // Requires num_cpus dependency in Cargo.toml

fn main() {
    // We have 5 samples with 3 classes.
    let dataset = [
        0.7, 0.0, 
        0.8, 1.0, 
        0.3, 0.0, 
        1.0, 1.3, 
        0.4, 2.1
    ];
    let targets = [1, 5, 1, -15, 5];
    let options = Classifier::train_defaults()
        .with_max_depth(15)
        .with_trees(40)
        .with_threads(num_cpus::get())
        .with_seed(42)
        .clone();
    let predictor = Classifier::fit(&dataset, &targets, &options);

    // Get predictions for same dataset.
    let predictions = predictor.predict(&dataset, num_cpus::get());
    println!("Predictions: {:?}", predictions);

    let proba = predictor.proba(&dataset, num_cpus::get());
    println!("Probability distributions:");
    for p in proba.chunks(predictor.num_classes()) {
        println!("{:?}", p);
    }
}
```

# Regression
Decision tree regressor (`rafor::dt::Regressor`) and random forest regressor (`rafor::Regressor`)
expect the targets to be `f32`. By default regressors use Mse score for evaluating the split
impurity.

Regressor interface is mostly similar to `Classifier`, please see examples folder.

# Model serialization and deserialization
All models support [serde](https://docs.rs/serde/latest/serde/), so any lib that supports `serde`
can be used for serialization and deserialization. 

Below is an exemple of using [bincode](https://docs.rs/bincode/latest/bincode/). 
```Rust
use std::fs::File;
use rafor::Classifier;

fn main() {
    let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
    let targets = [1, 5, 1];
    let predictor = Classifier::fit(&dataset, &targets, &Classifier::train_defaults());

    // Storing model.
    let mut fout = File::create("model.bin").unwrap();
    let config = bincode::config::standard();
    // Requires dependency bincode with 'serde' feature in Cargo.toml:
    // bincode = { version = "2.0", features = ["serde"] }
    let _ = bincode::serde::encode_into_std_write(&predictor, &mut fout, config);

    // Loading stored model.
    let mut fin = File::open("model.bin").unwrap();
    let config = bincode::config::standard();
    let predictor: Classifier = bincode::serde::decode_from_std_read(&mut fin, config).unwrap();

    let predictions = predictor.predict(&dataset, 1);
    assert_eq!(&predictions, &[1, 5, 1]);
}
```

## License
Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
rafor by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
