use super::{tree_builder, DecisionTree, Trainset};
use crate::{
    config::{Metric, TrainConfig},
    metrics::Mse,
    DatasetView, FloatTarget, Weightable,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TreeRegressorImpl {
    tree: DecisionTree<()>,
}

impl TreeRegressorImpl {
    pub fn predict(&self, dataset: &DatasetView) -> Vec<FloatTarget> {
        dataset.samples().map(|s| self.predict_one(s)).collect()
    }

    pub fn num_features(&self) -> usize {
        self.tree.num_features()
    }

    pub fn predict_one(&self, sample: &[f32]) -> FloatTarget {
        self.tree.predict(sample).0
    }

    pub fn fit(trainset: Trainset<FloatTarget>, config: &TrainConfig) -> TreeRegressorImpl {
        let mut tr = TreeRegressorImpl {
            tree: DecisionTree::new(trainset.num_features() as u16),
        };

        let (ranges, targets) = match config.metric {
            Metric::MSE => {
                tree_builder::build(trainset, &mut tr.tree, config.clone(), Mse::default())
            }
            _ => panic!("Metric is not supported for regressor tree"),
        };

        for (node, range) in ranges.iter() {
            let targets = &targets[range.clone()];
            let mut s: f32 = 0.;
            let mut n = 0;
            for (x, w) in targets.iter().map(|t| FloatTarget::unweight(t)) {
                s += x * w as f32;
                n += w;
            }
            tr.tree.set_node_threshold(*node, s / n as f32);
        }

        tr
    }
}
