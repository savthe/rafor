use rafor::dt::Regressor;
use rafor::prelude::*; // Required to use .with_option builders.

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
            .with_max_depth(2)
    );

    // Get predictions for same dataset.
    let predictions = predictor.predict(&dataset);
    println!("Predictions: {:?}", predictions);
}
