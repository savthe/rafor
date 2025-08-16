use rafor::dt::Classifier;
use rafor::prelude::*; // Required for .with_option builders.

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
    let predictor = Classifier::fit(
        &dataset,
        &targets,
        Classifier::default_config()
            .with_max_depth(15)
    );

    // Get predictions for same dataset.
    let predictions = predictor.predict(&dataset);
    println!("Predictions: {:?}", predictions);

    let proba = predictor.proba(&dataset);
    println!("Probability distributions:");
    for p in proba.chunks(predictor.num_classes()) {
        println!("{:?}", p);
    }
}
