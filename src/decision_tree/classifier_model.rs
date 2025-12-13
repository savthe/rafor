use super::splitter::GiniSplitter;
use super::trainer;
use super::DecisionTree;
use super::TrainView;
use crate::{ClassTarget, DatasetView, SampleWeight};

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

    pub fn train(
        tv: TrainView<ClassTarget>,
        num_cls: usize,
        cfg: &trainer::Config,
    ) -> ClassifierModel {
        let mut probability_aggr = ProbabilityAggregator::new(num_cls);
        let tree = trainer::train(
            tv,
            cfg.clone(),
            GiniSplitter::new(num_cls, cfg.min_samples_leaf),
            &mut probability_aggr,
        );

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
        let mut bins = vec![0. as f64; self.num_classes];
        let mut total_weight: f64 = 0.;
        for &(x, w) in leaf_items.iter() {
            bins[x as usize] += w as f64;
            total_weight += w as f64;
        }

        let offset = self.proba.len() / self.num_classes;
        for x in bins.iter_mut() {
            self.proba.push((*x / total_weight) as f32);
        }

        offset as u32
    }
}
