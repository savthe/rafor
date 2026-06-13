use super::{splitter::MseSplitter, trainer, Predictor, TrainConfig};

use crate::{FloatTarget, SampleWeight, Trainset};

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressorModel<P: Predictor> {
    predictor: P,
    num_features: usize,
}

#[derive(Default)]
struct Aggregator {}

impl<P: Predictor> RegressorModel<P> {
    pub fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        assert!(dataset.len() % self.num_features == 0);
        dataset
            .chunks_exact(self.num_features)
            .map(|s| self.predict_one(s))
            .collect()
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    #[inline(always)]
    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        f32::from_bits(self.predictor.resolve(sample))
    }

    pub fn train(ts: &Trainset<FloatTarget>, config: &TrainConfig) -> RegressorModel<P> {
        let mut aggregator = Aggregator::default();
        let tree = trainer::train(
            ts,
            config.clone(),
            MseSplitter::new(config.min_samples_leaf),
            &mut aggregator,
        );

        RegressorModel {
            predictor: tree,
            num_features: ts.num_features,
        }
    }
}

impl trainer::Aggregator<FloatTarget> for Aggregator {
    fn aggregate(&mut self, leaf_items: &[(FloatTarget, SampleWeight)]) -> u32 {
        let mut s: f64 = 0.;
        let mut total_weight: f64 = 0.;
        for &(x, w) in leaf_items.iter() {
            s += (x * w) as f64;
            total_weight += w as f64;
        }
        let value = (s / total_weight) as f32;
        value.to_bits()
    }
}
