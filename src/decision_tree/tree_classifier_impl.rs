use serde::{Deserialize, Serialize};

use super::{
    tree_trainer::{Trainable, Trainer},
    DecisionTree, Trainset,
};
use crate::{
    metrics::{self, WithClasses},
    config::{Metric, TreeConfig},
    ClassLabel, DatasetView, Weightable,
};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TreeClassifierImpl {
    proba: Vec<f32>,
    num_classes: usize,
    tree: DecisionTree<ClassLabel>,
}

impl Trainable<ClassLabel> for TreeClassifierImpl {
    fn split_node(&mut self, node: usize, feature: u16, threshold: f32) -> (usize, usize) {
        self.tree.set_node(node, feature, threshold, 0);
        self.tree.split_node(node)
    }

    //TODO(sav). Weightables are not fully implemented. Consider hiding them into tree trainer.
    //TODO(sav). Actually we dotn't need trainable trait. Tree trainer can return subsets
    //of targets for created nodes. Caller will process them.
    fn handle_leaf(&mut self, node: usize, targets: &[u32]) {
        let offset = self.proba.len();
        self.proba.resize(offset + self.num_classes, 0.);
        let bins = &mut self.proba[offset..];

        let mut count = 0;
        for t in targets.iter() {
            let (x, w) = u32::unweight(t);
            bins[x as usize] += w as f32;
            count += w;
        }

        for x in bins.iter_mut() {
            *x /= count as f32;
        }

        self.tree.set_node(node, 0, 0., offset as ClassLabel);
    }
}

impl TreeClassifierImpl {
    pub fn predict(&self, dataset: &DatasetView) -> Vec<f32> {
        let mut result = vec![0.; dataset.size() * self.num_classes];

        for (r, sample) in result
            .chunks_exact_mut(self.num_classes)
            .zip(dataset.samples())
        {
            r.copy_from_slice(self.predict_one(sample));
        }
        result
    }

    pub fn num_features(&self) -> usize {
        self.tree.num_features()
    }

    pub fn predict_one(&self, sample: &[f32]) -> &[f32] {
        let i = self.tree.predict(sample).1 as usize;
        &self.proba[i..i + self.num_classes]
    }

    pub fn fit(
        ts: Trainset<ClassLabel>,
        num_classes: usize,
        config: &TreeConfig,
    ) -> TreeClassifierImpl {
        let mut tr = TreeClassifierImpl {
            proba: Vec::new(),
            num_classes,
            tree: DecisionTree::new(ts.num_features() as u16),
        };

        match config.metric {
            Metric::GINI => Trainer::fit(
                ts,
                config.clone(),
                &mut tr,
                metrics::Gini::with_classes(num_classes),
            ),
            _ => panic!("Metric is not supported for classifier tree"),
        };

        tr
    }
}
