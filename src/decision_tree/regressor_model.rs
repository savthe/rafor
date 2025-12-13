use super::DecisionTree;

use super::splitter::MseSplitter;

use super::trainer;
use super::TrainView;
use crate::{DatasetView, FloatTarget, SampleWeight};

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressorModel {
    tree: DecisionTree,
}

#[derive(Default)]
struct Aggregator {}

impl RegressorModel {
    pub fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        dataset.samples().map(|s| self.predict_one(s)).collect()
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.tree.num_features()
    }

    #[inline(always)]
    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        f32::from_bits(self.tree.predict(sample))
    }

    pub fn train(tv: TrainView<FloatTarget>, config: &trainer::Config) -> RegressorModel {
        let mut aggregator = Aggregator::default();
        let tree = trainer::train(
            tv,
            config.clone(),
            MseSplitter::new(config.min_samples_leaf),
            &mut aggregator,
        );

        RegressorModel { tree }
    }
}

impl trainer::Aggregator<FloatTarget> for Aggregator {
    fn aggregate(&mut self, leaf_items: &[(FloatTarget, SampleWeight)]) -> u32 {
        // TODO idiomatic
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
