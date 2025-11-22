use super::splitter::GiniSplitter;
use super::trainer;
use super::DecisionTree;
use super::TrainView;
use crate::{
    config::{Metric, TrainConfig},
    labels::*,
    DatasetView,
};
use trainer::TrainSpace;

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierModel {
    proba: Vec<f32>,
    num_classes: usize,
    tree: DecisionTree<ClassTarget>,
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
        let i = self.tree.predict(sample).1 as usize * self.num_classes;
        &self.proba[i..i + self.num_classes]
    }

    pub fn fit(tv: TrainView<ClassTarget>, num_cls: usize, cfg: &TrainConfig) -> ClassifierModel {
        let max_weight = *tv.weights.iter().max().unwrap();

        if num_cls < (1 << 8) && max_weight < (1 << 24) {
            Self::fit_internal::<DenseClass<8>>(tv, num_cls, cfg)
        } else if num_cls < (1 << 16) && max_weight < (1 << 16) {
            Self::fit_internal::<DenseClass<16>>(tv, num_cls, cfg)
        } else if num_cls < (1 << 24) && max_weight < (1 << 8) {
            Self::fit_internal::<DenseClass<24>>(tv, num_cls, cfg)
        } else {
            Self::fit_internal::<(ClassTarget, SampleWeight)>(tv, num_cls, cfg)
        }
    }

    pub fn fit_internal<P: Weighted<ClassTarget>>(
        tv: TrainView<ClassTarget>,
        num_cls: usize,
        cfg: &TrainConfig,
    ) -> ClassifierModel {
        let mut tr = ClassifierModel {
            proba: Vec::new(),
            num_classes: num_cls,
            tree: DecisionTree::new(tv.dataview.num_features() as u16),
        };

        let mut space: TrainSpace<P> = TrainSpace::new(tv);

        let ranges = match cfg.metric {
            Metric::GINI => trainer::fit(
                &mut space,
                &mut tr.tree,
                cfg.clone(),
                GiniSplitter::new(num_cls, cfg.min_samples_leaf),
            ),
            _ => panic!("Metric is not supported for classifier tree"),
        };

        tr.proba.resize(tr.num_classes * ranges.len(), 0.);
        let mut offset = 0;
        for ((node, range), bins) in ranges.iter().zip(tr.proba.chunks_mut(tr.num_classes)) {
            let targets = &space.targets(&range);

            let mut count = 0;
            for t in targets.iter() {
                let (x, w) = t.unweight();
                bins[x as usize] += w as f32;
                count += w;
            }

            for x in bins.iter_mut() {
                *x /= count as f32;
            }

            tr.tree.set_node_value(*node, offset);
            offset += 1;
        }
        tr
    }
}
