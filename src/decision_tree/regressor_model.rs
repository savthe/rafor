use super::DecisionTree;

use super::splitter::MseSplitter;

use super::trainer;
use super::TrainView;
use crate::{
    config::{Metric, TrainConfig},
    labels::*,
    DatasetView,
};
use trainer::TrainSpace;

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressorModel {
    tree: DecisionTree,
}

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

    pub fn fit(tv: TrainView<FloatTarget>, config: &TrainConfig) -> RegressorModel {
        let mut tr = RegressorModel {
            tree: DecisionTree::new(tv.dataview.num_features() as u16),
        };
        let mut space = TrainSpace::<(f32, SampleWeight)>::new(tv);
        let (tree, ranges) = match config.metric {
            Metric::MSE => trainer::fit(
                &mut space,
                config.clone(),
                MseSplitter::new(config.min_samples_leaf),
            ),
            _ => panic!("Metric is not supported for regressor tree"),
        };

        tr.tree = tree;
        for (node, range) in ranges.iter() {
            let targets = &space.targets(&range);
            let mut s: f32 = 0.;
            let mut n = 0;
            for &(x, w) in targets.iter() {
                s += x * w as f32;
                n += w;
            }
            let value = s / n as f32;
            tr.tree.set_leaf_value(&node, value.to_bits());
        }

        tr
    }
}
