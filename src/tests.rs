use crate::dt;
use crate::prelude::*;
use crate::rf;
use std::fs::read_to_string;
use std::str::FromStr;

const MAX_THREADS: usize = 8;

#[test]
fn classifier_tree_overfit_self() {
    let (samples, targets) = load_dataset::<i64>("datasets/winequality-red.csv", ";", true);
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
    let (samples, targets) = load_dataset::<i64>("datasets/winequality-red.csv", ";", true);
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
    let (samples, targets) = load_dataset::<i64>("datasets/winequality-red.csv", ";", true);
    let predictor = rf::Classifier::fit(
        &samples,
        &targets,
        rf::Classifier::default_config()
            .with_max_depth(500)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&samples, MAX_THREADS);
    let acc = classifier_accuracy(&y_pred, &targets);
    assert!(acc == 1.0);
}

#[test]
fn random_forest_classifier_depth10() {
    let (samples, targets) = load_dataset::<i64>("datasets/winequality-red.csv", ";", true);
    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = rf::Classifier::fit(
        &x_train,
        &y_train,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&x_pred, MAX_THREADS);
    let acc = classifier_accuracy(&y_pred, &y_ref);
    assert!(acc >= 0.67);
}

#[test]
fn random_forest_classifier_depth10_self() {
    let (samples, targets) = load_dataset::<i64>("datasets/winequality-red.csv", ";", true);
    let predictor = rf::Classifier::fit(
        &samples,
        &targets,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&samples, MAX_THREADS);
    let acc = classifier_accuracy(&y_pred, &targets);
    assert!(acc >= 0.93);
}

#[test]
fn random_forest_binary_classifier() {
    let (samples, targets) = load_dataset::<String>("datasets/magic04.data", ",", false);
    let targets: Vec<i64> = targets.iter().map(|t| (t == "h") as i64).collect();

    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = rf::Classifier::fit(
        &x_train,
        &y_train,
        rf::Classifier::default_config()
            .with_max_depth(10)
            .with_trees(100),
    );

    let y_pred = predictor.predict(&x_pred, MAX_THREADS);
    let f1 = f1score(&y_pred, &y_ref);
    assert!(f1 >= 0.79);
}

fn f1score(pred: &[i64], target: &[i64]) -> f64 {
    let tp = pred
        .iter()
        .zip(target.iter())
        .filter(|&(p, t)| *p == 1 && *t == 1)
        .count();
    let fp = pred
        .iter()
        .zip(target.iter())
        .filter(|&(p, t)| *p == 1 && *t == 0)
        .count();
    let fnn = pred
        .iter()
        .zip(target.iter())
        .filter(|&(p, t)| *p == 0 && *t == 1)
        .count();
    let precision = tp as f64 / (tp as f64 + fp as f64);
    let recall = tp as f64 / (tp as f64 + fnn as f64);
    let f1 = 2. * precision * recall / (precision + recall);
    f1
}

#[test]
fn decision_tree_regressor_overfit_self() {
    let (samples, targets) = load_dataset::<f32>("datasets/Folds5x2_pp.csv", ",", true);
    let predictor = dt::Regressor::fit(
        &samples,
        &targets,
        dt::Regressor::default_config().with_max_depth(500),
    );

    let y_pred = predictor.predict(&samples);
    let mse = mean_squared_error(&y_pred, &targets);
    assert!(mse < 0.000001);
}

#[test]
fn decision_tree_regressor() {
    let (samples, targets) = load_dataset::<f32>("datasets/Folds5x2_pp.csv", ",", true);
    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = dt::Regressor::fit(
        &x_train,
        &y_train,
        dt::Regressor::default_config().with_max_depth(500),
    );

    let y_pred = predictor.predict(&x_pred);
    let mse = mean_squared_error(&y_pred, &y_ref);
    assert!(mse < 23.0);
}

#[test]
fn random_forest_regressor_overfit_self() {
    let (samples, targets) = load_dataset::<f32>("datasets/Folds5x2_pp.csv", ",", true);
    let predictor = rf::Regressor::fit(
        &samples,
        &targets,
        rf::Regressor::default_config()
            .with_max_depth(1000)
            .with_trees(25),
    );

    let y_pred = predictor.predict(&samples, MAX_THREADS);
    let mse = mean_squared_error(&y_pred, &targets);
    assert!(mse < 1.7);
}

fn random_forest_regressor(max_depth: usize) -> f64 {
    let (samples, targets) = load_dataset::<f32>("datasets/Folds5x2_pp.csv", ",", true);
    let (x_train, y_train, x_pred, y_ref) = split_dataset(&samples, &targets);
    let predictor = rf::Regressor::fit(
        &x_train,
        &y_train,
        rf::Regressor::default_config().with_max_depth(max_depth),
    );

    let y_pred = predictor.predict(&x_pred, MAX_THREADS);
    mean_squared_error(&y_pred, &y_ref)
}

#[test]
fn random_forest_regressor_depth5() {
    let mse = random_forest_regressor(5);
    assert!(mse < 19.5);
}

#[test]
fn random_forest_regressor_depth10() {
    let mse = random_forest_regressor(10);
    assert!(mse < 15.0);
}

fn mean_squared_error(v: &[f32], u: &[f32]) -> f64 {
    assert!(v.len() == u.len());
    let mse = v
        .iter()
        .zip(u.iter())
        .map(|(&x, &y)| (x - y) as f64 * (x - y) as f64)
        .sum::<f64>()
        / v.len() as f64;
    println!("MSE: {}", mse);
    mse
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

fn load_dataset<T>(fname: &str, delimeter: &str, skip_first_line: bool) -> (Vec<f32>, Vec<T>)
where
    T: FromStr,
    T::Err: std::fmt::Debug,
{
    let lines: Vec<String> = read_to_string(fname)
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();
    let mut samples: Vec<f32> = Vec::new();
    let mut targets: Vec<T> = Vec::new();
    let mut tokens_per_line = 0;
    for line in lines.iter().skip(skip_first_line as usize) {
        let w: Vec<_> = line.split(delimeter).collect();
        if tokens_per_line == 0 {
            tokens_per_line = w.len();
        }
        assert!(
            tokens_per_line == w.len(),
            "Lines in dataset has different sizes."
        );

        samples.extend(
            w.iter()
                .take(tokens_per_line - 1)
                .map(|s| s.parse::<f32>().unwrap()),
        );
        targets.push(
            w.last()
                .unwrap()
                .parse::<T>()
                .expect("Couldn't parse target value"),
        );
    }
    (samples, targets)
}
