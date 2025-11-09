use crate::dt;
use crate::prelude::*;
use crate::rf;
use std::fs::read_to_string;

#[test]
fn classifier_tree_overfit_self() {
    let (samples, targets) = load_wine_dataset();
    let predictor = dt::Classifier::fit(
        &samples,
        &targets,
        dt::Classifier::default_config().with_max_depth(500),
    );

    let y_pred = predictor.predict(&samples);
    let acc = classifier_accuracy(&y_pred, &targets);
    assert!(acc == 1.0);
}

#[test]
fn classifier_tree_depth10() {
    let (samples, targets) = load_wine_dataset();
    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = dt::Classifier::fit(
        &x_train,
        &y_train,
        dt::Classifier::default_config().with_max_depth(10),
    );

    let y_pred = predictor.predict(&x_pred);
    let acc = classifier_accuracy(&y_pred, &y_ref);
    assert!(acc >= 0.61);
}

#[test]
fn random_forest_classifier_overfit_self() {
    let (samples, targets) = load_wine_dataset();
    let predictor = rf::Classifier::fit(
        &samples,
        &targets,
        rf::Classifier::default_config()
            .with_max_depth(500)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&samples, 8);
    let acc = classifier_accuracy(&y_pred, &targets);
    assert!(acc == 1.0);
}

#[test]
fn random_forest_classifier_depth10() {
    let (samples, targets) = load_wine_dataset();
    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = rf::Classifier::fit(
        &x_train,
        &y_train,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&x_pred, 8);
    let acc = classifier_accuracy(&y_pred, &y_ref);
    assert!(acc >= 0.67);
}

#[test]
fn random_forest_classifier_depth10_self() {
    let (samples, targets) = load_wine_dataset();
    let predictor = rf::Classifier::fit(
        &samples,
        &targets,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&samples, 8);
    let acc = classifier_accuracy(&y_pred, &targets);
    assert!(acc >= 0.93);
}

#[test]
fn random_forest_binary_classifier() {
    const NUM_FEATURES: usize = 10;

    let lines: Vec<String> = read_to_string("datasets/magic04.data")
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();

    let mut samples: Vec<f32> = Vec::new();
    let mut targets: Vec<i64> = Vec::new();

    for line in lines.iter().skip(1) {
        let w: Vec<_> = line.split(",").collect();

        samples.extend(
            w.iter()
                .take(NUM_FEATURES)
                .map(|s| s.parse::<f32>().unwrap()),
        );
        targets.push((w.last().unwrap() == &"h") as i64);
    }

    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = rf::Classifier::fit(
        &x_train,
        &y_train,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&x_pred, 8);
    let f1 = f1score(&y_pred, &y_ref);
    assert!(f1 >= 0.80);
}

fn f1score(pred: &[i64], target: &[i64]) -> f64 {
    let tp = pred.iter().zip(target.iter()).filter(|&(p, t)| *p == 1 && *t == 1).count();
    let fp = pred.iter().zip(target.iter()).filter(|&(p, t)| *p == 1 && *t == 0).count();
    let fnn = pred.iter().zip(target.iter()).filter(|&(p, t)| *p == 0 && *t == 1).count();
    let precision = tp as f64 / (tp as f64 + fp as f64);
    let recall = tp as f64 / (tp as f64 + fnn as f64);
    let f1 = 2. * precision * recall / (precision + recall);
    f1
}

fn split_dataset<T: Copy>(x: &[f32], y: &[T]) -> (Vec<f32>, Vec<T>, Vec<f32>, Vec<T>) {
    assert!(x.len() % y.len() == 0);
    let features = x.len() / y.len();

    let x_train: Vec<f32> = x
        .chunks(features)
        .enumerate()
        .filter(|(i, _)| i % 5 > 0)
        .flat_map(|(_, chunk)| chunk.iter().cloned())
        .collect();

    let y_train: Vec<T> = y
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 5 > 0)
        .map(|(_, v)| *v)
        .collect();

    let x_pred: Vec<f32> = x
        .chunks(features)
        .enumerate()
        .filter(|(i, _)| i % 5 == 0)
        .flat_map(|(_, chunk)| chunk.iter().cloned())
        .collect();

    let y_ref: Vec<T> = y
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 5 == 0)
        .map(|(_, v)| *v)
        .collect();

    (x_train, y_train, x_pred, y_ref)
}

fn classifier_accuracy(v: &[i64], u: &[i64]) -> f64 {
    assert!(v.len() == u.len());
    v.iter().zip(u.iter()).filter(|(x, y)| x == y).count() as f64 / u.len() as f64
}

fn load_wine_dataset() -> (Vec<f32>, Vec<i64>) {
    const NUM_FEATURES: usize = 11;

    let lines: Vec<String> = read_to_string("datasets/winequality-red.csv")
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();

    let mut samples: Vec<f32> = Vec::new();
    let mut targets: Vec<i64> = Vec::new();

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
