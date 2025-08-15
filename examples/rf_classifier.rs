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
