use super::splitter::GiniSplitter;
use super::trainer;
use super::DecisionTree;
use super::TrainView;
use crate::{
    config::{Metric, TrainConfig},
    ClassTarget, DatasetView, SampleWeight,
};
use trainer::TrainSpace;

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierModel {
    proba: Vec<f32>,
    num_classes: usize,
    tree: DecisionTree,
}

impl ClassifierModel {
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
        let i = self.tree.predict(sample) as usize * self.num_classes;
        &self.proba[i..i + self.num_classes]
    }

    pub fn fit(tv: TrainView<ClassTarget>, num_cls: usize, cfg: &TrainConfig) -> ClassifierModel {
        let mut tr = ClassifierModel {
            proba: Vec::new(),
            num_classes: num_cls,
            tree: DecisionTree::new(tv.dataview.num_features() as u16),
        };

        let mut space: TrainSpace<ClassTarget> = TrainSpace::new(tv);

        let (tree, ranges) = match cfg.metric {
            Metric::GINI => trainer::fit(
                &mut space,
                cfg.clone(),
                GiniSplitter::new(num_cls, cfg.min_samples_leaf),
            ),
            _ => panic!("Metric is not supported for classifier tree"),
        };

        tr.tree = tree;
        tr.proba.resize(tr.num_classes * ranges.len(), 0.);
        let mut offset = 0;
        for ((node, range), bins) in ranges.iter().zip(tr.proba.chunks_mut(tr.num_classes)) {
            let targets = &space.targets(&range);

            let mut count = 0;
            for &(x, w) in targets.iter() {
                bins[x as usize] += w as f32;
                count += w;
            }

            for x in bins.iter_mut() {
                *x /= count as f32;
            }

            tr.tree.set_leaf_value(&node, offset);
            offset += 1;
        }
        tr
    }
}
