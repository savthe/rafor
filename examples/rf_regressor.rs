use rafor::rf::Regressor;
use rafor::prelude::*; // Required for .with_option builders.
use num_cpus; // Requires num_cpus dependency in Cargo.toml

fn main() {
    // 5 samples with 2 features each.
    let dataset = [
        0.7, 0.0, 
        0.8, 1.0, 
        0.3, 0.0,
        1.0, 1.3,
        0.4, 2.1
    ];
    let targets = [0.8, 0.2, 0.7, 0.3, 1.2];
    let predictor = Regressor::fit(
        &dataset,
        &targets,
        Regressor::default_config()
            .with_max_depth(15)
            .with_trees(40)
            .with_threads(num_cpus::get())
            .with_seed(42),
    );

    // Get predictions for same dataset.
    let predictions = predictor.predict(&dataset, num_cpus::get());
    println!("Predictions: {:?}", predictions);
}
