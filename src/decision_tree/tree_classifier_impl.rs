use serde::{Deserialize, Serialize};

use super::{tree_builder, DecisionTree};
use crate::{
    config::{Metric, TrainConfig},
    metrics::{self, WithClasses},
    ClassLabel, DatasetView, Weightable,
};
use super::Trainset;

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TreeClassifierImpl {
    proba: Vec<f32>,
    num_classes: usize,
    tree: DecisionTree<ClassLabel>,
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

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.tree.num_features()
    }

    #[inline(always)]
    pub fn predict_one(&self, sample: &[f32]) -> &[f32] {
        let i = self.tree.predict(sample).1 as usize * self.num_classes;
        &self.proba[i..i + self.num_classes]
    }

    pub fn fit(
        ts: Trainset<ClassLabel>,
        num_classes: usize,
        config: &TrainConfig,
    ) -> TreeClassifierImpl {
        let mut tr = TreeClassifierImpl {
            proba: Vec::new(),
            num_classes,
            tree: DecisionTree::new(ts.num_features() as u16),
        };

        tr.fit_internal(ts, config);
        tr
    }

    fn fit_internal(&mut self, ts: Trainset<ClassLabel>, config: &TrainConfig) {
        let (ranges, targets) = match config.metric {
            Metric::GINI => tree_builder::build(
                ts,
                &mut self.tree,
                config.clone(),
                metrics::Gini::with_classes(self.num_classes),
            ),
            _ => panic!("Metric is not supported for classifier tree"),
        };

        self.proba.resize(self.num_classes * ranges.len(), 0.);
        let mut offset = 0;
        for ((node, range), bins) in ranges.iter().zip(self.proba.chunks_mut(self.num_classes)) {
            let targets = &targets[range.clone()];

            let mut count = 0;
            for (x, w) in targets.iter().map(|t| ClassLabel::unweight(t)) {
                //let (x, w) = u32::unweight(t);
                bins[x as usize] += w as f32;
                count += w;
            }

            for x in bins.iter_mut() {
                *x /= count as f32;
            }

            self.tree.set_node_value(*node, offset);
            offset += 1;
        }
    }
}
