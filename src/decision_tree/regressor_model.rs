use super::DecisionTree;

use super::splitter::MseSplitter;

use super::trainer;
use super::TrainView;
use crate::{
    config::{Metric, TrainConfig},
    DatasetView, FloatTarget, SampleWeight,
};

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

    pub fn train(tv: TrainView<FloatTarget>, config: &TrainConfig) -> RegressorModel {
        let mut aggregator = Aggregator::default();
        let tree = match config.metric {
            Metric::MSE => trainer::train(
                tv,
                config.clone(),
                MseSplitter::new(config.min_samples_leaf),
                &mut aggregator,
            ),
            _ => panic!("Metric is not supported for regressor tree"),
        };

        RegressorModel { tree }
    }
}

impl trainer::Aggregator<FloatTarget> for Aggregator {
    fn aggregate(&mut self, leaf_items: &[(FloatTarget, SampleWeight)]) -> u32 {
        // TODO idiomatic
        let mut s: f32 = 0.;
        let mut n = 0;
        for &(x, w) in leaf_items.iter() {
            s += x * w as f32;
            n += w;
        }
        let value = s / n as f32;
        value.to_bits()
    }
}
