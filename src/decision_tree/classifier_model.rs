use super::{splitter::GiniSplitter, trainer, Predictor, TrainConfig};

use crate::{ClassTarget, SampleWeight, Trainset};

use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifierModel<P: Predictor> {
    proba: Vec<f32>,
    num_classes: usize,
    predictor: P,
    num_features: usize,
}

#[derive(Default)]
struct ProbabilityAggregator {
    proba: Vec<f32>,
    num_classes: usize,
}

impl<P: Predictor> ClassifierModel<P> {
    pub fn predict(&self, dataset: &[f32]) -> Vec<f32> {
        assert!(dataset.len() % self.num_features == 0);
        let num_samples = dataset.len() / self.num_features;
        let mut result = vec![0.; num_samples * self.num_classes];

        for (r, sample) in result
            .chunks_exact_mut(self.num_classes)
            .zip(dataset.chunks_exact(self.num_features))
        {
            r.copy_from_slice(self.predict_one(sample));
        }
        result
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    #[inline(always)]
    pub fn predict_one(&self, sample: &[f32]) -> &[f32] {
        let i = self.predictor.resolve(sample) as usize * self.num_classes;
        &self.proba[i..i + self.num_classes]
    }

    pub fn train(
        ts: &Trainset<ClassTarget>,
        num_cls: usize,
        cfg: &TrainConfig,
    ) -> ClassifierModel<P> {
        let mut probability_aggr = ProbabilityAggregator::new(num_cls);
        let tree = trainer::train(
            ts,
            cfg.clone(),
            GiniSplitter::new(num_cls, cfg.min_samples_leaf),
            &mut probability_aggr,
        );

        ClassifierModel {
            proba: probability_aggr.proba,
            num_classes: num_cls,
            predictor: tree,
            num_features: ts.num_features,
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
