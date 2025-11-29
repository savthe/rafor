use super::splitter::GiniSplitter;
use super::trainer;
use super::DecisionTree;
use super::TrainView;
use crate::{
    config::{Metric, TrainConfig},
    ClassTarget, DatasetView, SampleWeight,
};

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierModel {
    proba: Vec<f32>,
    num_classes: usize,
    tree: DecisionTree,
}

#[derive(Default)]
struct ProbabilityAggregator {
    proba: Vec<f32>,
    num_classes: usize,
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

    pub fn train(tv: TrainView<ClassTarget>, num_cls: usize, cfg: &TrainConfig) -> ClassifierModel {
        let mut probability_aggr = ProbabilityAggregator::new(num_cls);
        let tree = match cfg.metric {
            Metric::GINI => trainer::train(
                tv,
                cfg.clone(),
                GiniSplitter::new(num_cls, cfg.min_samples_leaf),
                &mut probability_aggr,
            ),
            _ => panic!("Metric is not supported for classifier tree"),
        };

        ClassifierModel {
            proba: probability_aggr.proba,
            num_classes: num_cls,
            tree,
        }
    }
}

impl ProbabilityAggregator {
    fn new(num_classes: usize) -> Self {
        Self {
            proba: Vec::new(),
            num_classes,
        }
    }
}

impl trainer::Aggregator<ClassTarget> for ProbabilityAggregator {
    fn aggregate(&mut self, leaf_items: &[(ClassTarget, SampleWeight)]) -> u32 {
        let mut bins = vec![0.; self.num_classes];
        let mut count = 0;
        for &(x, w) in leaf_items.iter() {
            bins[x as usize] += w as f32;
            count += w;
        }

        for x in bins.iter_mut() {
            *x /= count as f32;
        }

        let offset = self.proba.len() / self.num_classes;
        self.proba.extend_from_slice(&bins);
        offset as u32
    }
}
