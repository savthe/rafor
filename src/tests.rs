use std::fs::read_to_string;
use crate::prelude::*;
use crate::dt;

#[test]
fn test_wine_classifier_overfit() {
    let (samples, targets) = load_wine_dataset();
    let predictor = dt::Classifier::fit(
        &samples,
        &targets,
        dt::Classifier::default_config().with_max_depth(500),
    );

    let predicts = predictor.predict(&samples);
    let correct = targets.iter().zip(predicts.iter()).filter(|(x, y)| *x == *y).count();
    assert_eq!(correct, targets.len());
}

fn load_wine_dataset() -> (Vec<f32>, Vec<i64>) {
    let lines: Vec<String> = read_to_string("datasets/winequality-red.csv")
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();
    let mut samples: Vec<f32> = Vec::new();
    let mut targets: Vec<i64> = Vec::new();
    const NUM_FEATURES: usize = 11;

    for line in lines.iter().skip(1) {
        let w: Vec<_> = line.split(";").collect();

        samples.extend(
            w.iter()
                .take(NUM_FEATURES)
                .map(|s| s.parse::<f32>().unwrap()),
        );
        targets.push(
            w.last()
                .unwrap()
                .parse::<i64>()
                .expect("Couldn't parse target value"),
        );
    }
    (samples, targets)
}
