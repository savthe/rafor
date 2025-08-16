use super::{
    tree_trainer::{Trainable, Trainer},
    DecisionTree, Trainset,
};
use crate::{
    metrics::Mse,
    config::{Metric, TreeConfig},
    DatasetView, LabelWeight,
};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TreeRegressorImpl {
    tree: DecisionTree<()>,
}

impl Trainable<f32> for TreeRegressorImpl {
    fn split_node(&mut self, node: usize, feature: u16, threshold: f32) -> (usize, usize) {
        self.tree.set_node(node, feature, threshold, ());
        self.tree.split_node(node)
    }

    fn handle_leaf(&mut self, node: usize, targets: &[(f32, LabelWeight)]) {
        let mut s: f32 = 0.;
        let mut n = 0;
        for &(x, w) in targets.iter() {
            s += x * w as f32;
            n += w;
        }

        self.tree.set_node(node, 0, s / n as f32, ());
    }
}

impl TreeRegressorImpl {
    pub fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        dataset.samples().map(|s| self.predict_one(s)).collect()
    }

    pub fn num_features(&self) -> usize {
        self.tree.num_features()
    }

    pub fn predict_one(&self, sample: &[f32]) -> f32 {
        self.tree.predict(sample).0
    }

    pub fn fit(trainset: Trainset<f32>, config: &TreeConfig) -> TreeRegressorImpl {
        let mut tr = TreeRegressorImpl {
            tree: DecisionTree::new(trainset.num_features() as u16),
        };

        match config.metric {
            Metric::MSE => Trainer::fit(trainset, config.clone(), &mut tr, Mse::default()),
            _ => panic!("Metric is not supported for regressor tree"),
        };

        tr
    }
}
