use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;

use crate::DatasetView;

pub trait Predictor {
    fn predict(&self, dataset: &DatasetView) -> Vec<f32>;
}

pub fn predict<P: Predictor + Sync + Send>(
    predictors: &Vec<P>,
    dataset: &DatasetView,
    num_threads: usize,
) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();
    if num_threads == 1 {
        for p in predictors.iter() {
            result.aggregate(&p.predict(&dataset));
        }
    } else {
        let task_id = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..num_threads {
                let handle = s.spawn(|| {
                    let mut thread_result: Vec<f32> = Vec::new();
                    loop {
                        let id = task_id.fetch_add(1, Ordering::Relaxed);
                        if id < predictors.len() {
                            thread_result.aggregate(&predictors[id].predict(&dataset));
                        } else {
                            break;
                        }
                    }
                    thread_result
                });

                handles.push(handle);
            }
            for handle in handles {
                result.aggregate(&handle.join().unwrap());
            }
        });
    }

    for x in result.iter_mut() {
        *x /= predictors.len() as f32;
    }

    result
}

pub trait Aggregate {
    fn aggregate(&mut self, other: &Vec<f32>);
}

impl Aggregate for Vec<f32> {
    fn aggregate(&mut self, other: &Vec<f32>) {
        if self.is_empty() {
            *self = other.clone();
        }
        else {
            assert!(other.is_empty() || self.len() == other.len());
            for (s, x) in self.iter_mut().zip(other.iter()) {
                *s += *x;
            }
        }
    }
}
