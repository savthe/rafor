use super::decision_tree::{DecisionTree, NodeHandle};
use super::splitter::Splitter;
use super::TrainView;
use crate::SampleWeight;
use crate::{
    config::{NumFeatures, TrainConfig},
    DatasetView, IndexRange,
};
use radsort;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

#[derive(Default)]
struct Split {
    feature: usize,
    pivot: usize,
    threshold: f32,
}

struct Trainer<'a, 'b, Target, S>
where
    Target: Copy,
    S: Splitter<Target>,
{
    max_features: usize,
    conf: TrainConfig,
    space: &'a mut TrainSpace<'b, Target>,
    tree: DecisionTree,
    splitter: S,
    features_perm: FeaturePermutation,
}

struct FeaturePermutation {
    rng: Option<SmallRng>,
    perm: Vec<usize>,
}

impl FeaturePermutation {
    fn new(num_features: usize, rng: Option<SmallRng>) -> Self {
        Self {
            rng,
            perm: (0..num_features).collect(),
        }
    }

    fn shake(&mut self) {
        if let Some(rng) = self.rng.as_mut() {
            self.perm.shuffle(rng);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &usize> + '_ {
        self.perm.iter()
    }
}

pub fn fit<Target: Copy>(
    space: &mut TrainSpace<Target>,
    conf: TrainConfig,
    splitter: impl Splitter<Target>,
) -> (DecisionTree, Vec<(NodeHandle, IndexRange)>) {
    let num_features = space.num_features();

    let max_features = match conf.max_features {
        NumFeatures::SQRT => (num_features as f32).sqrt() as usize,
        NumFeatures::LOG => (num_features as f32).log2() as usize,
        NumFeatures::NUMBER(n) => n.min(num_features),
    };

    let rng = (max_features < num_features).then_some(SmallRng::seed_from_u64(conf.seed));

    let num_features = space.num_features();
    let mut trainer = Trainer {
        max_features,
        features_perm: FeaturePermutation::new(num_features, rng),
        conf,
        space,
        tree: DecisionTree::new(num_features as u16),
        splitter,
    };

    let ranges = trainer.fit();
    (trainer.tree, ranges)
}

impl<'a, 'b, Target, S> Trainer<'a, 'b, Target, S>
where
    Target: Copy,
    S: Splitter<Target>,
{
    pub fn fit(&mut self) -> Vec<(NodeHandle, IndexRange)> {
        let mut stack: Vec<(NodeHandle, IndexRange, usize)> =
            vec![(self.tree.root(), 0..self.space.size(), 0); 1];

        let mut ranges: Vec<(NodeHandle, IndexRange)> = Vec::new();
        while let Some((node, range, depth)) = stack.pop() {
            let split = if depth < self.conf.max_depth
                && range.len() >= self.conf.min_samples_split
                && range.len() >= 2 * self.conf.min_samples_leaf
            {
                self.find_best_split(&range)
            } else {
                None
            };

            if let Some(s) = split {
                self.space.split(&range, s.feature, s.threshold);
                let (left_range, right_range) = (range.start..s.pivot, s.pivot..range.end);
                let (left_node, right_node) = self.tree.split(&node, s.feature as u16, s.threshold);

                stack.push((left_node, left_range, depth + 1));
                stack.push((right_node, right_range, depth + 1));
            } else {
                ranges.push((node, range));
            }
        }
        ranges
    }

    fn find_best_split(&mut self, range: &IndexRange) -> Option<Split> {
        let targets = self.space.targets(&range);
        let samples = self.space.samples(&range);

        // Splitter returns false if the range is pure.
        if !self.splitter.prepare(targets) {
            return None;
        }

        let mut split = Split::default();
        let mut best_impurity = f64::INFINITY;

        let proto: Vec<(f32, Target, SampleWeight)> =
            targets.iter().map(|&(t, w)| (0., t, w)).collect();

        self.features_perm.shake();
        for (i, &feature) in self.features_perm.iter().enumerate() {
            let mut ordered_samples = proto.clone();
            for ((x, _, _), y) in ordered_samples.iter_mut().zip(samples.iter()) {
                *x = self.space.feature_val(*y, feature);
            }

            radsort::sort_by_key(&mut ordered_samples, |k| k.0);
            let p = self.splitter.find_split(&ordered_samples, best_impurity);
            if p.pivot > 0 {
                split.pivot = p.pivot + range.start;
                split.feature = feature;
                split.threshold =
                    (ordered_samples[p.pivot - 1].0 + ordered_samples[p.pivot].0) / 2.;
                best_impurity = p.impurity;
            }

            if best_impurity == 0. || (i + 1 >= self.max_features && split.pivot > range.start) {
                break;
            }
        }
        (split.pivot > 0).then_some(split)
    }
}

pub struct TrainSpace<'a, T> {
    dataview: DatasetView<'a>,
    samples: Vec<u32>,
    targets: Vec<(T, SampleWeight)>,
}

impl<'a, T: Copy> TrainSpace<'a, T> {
    pub fn new(ts: TrainView<'a, T>) -> TrainSpace<'a, T> {
        let amount = ts.weights.iter().filter(|&x| *x > 0).count();
        let mut samples: Vec<u32> = Vec::with_capacity(amount);
        let mut weighted_targets: Vec<(T, SampleWeight)> = Vec::with_capacity(amount);

        for (i, (&t, &w)) in ts.targets.iter().zip(ts.weights.iter()).enumerate() {
            if w > 0 {
                samples.push(i as u32);
                weighted_targets.push((t, w));
            }
        }

        TrainSpace {
            dataview: ts.dataview,
            samples,
            targets: weighted_targets,
        }
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.dataview.num_features()
    }

    #[inline(always)]
    pub fn targets(&self, range: &IndexRange) -> &[(T, SampleWeight)] {
        &&self.targets[range.clone()]
    }

    #[inline(always)]
    fn samples(&self, range: &IndexRange) -> &[u32] {
        &self.samples[range.clone()]
    }

    fn split(&mut self, range: &IndexRange, feature: usize, threshold: f32) {
        let mut i = range.start;
        let mut j = range.end;
        while i < j {
            if self.feature_val(self.samples[i], feature) <= threshold {
                i += 1;
            } else {
                j -= 1;
                self.samples.swap(i, j);
                self.targets.swap(i, j);
            }
        }
    }

    #[inline(always)]
    fn feature_val(&self, id: u32, feature: usize) -> f32 {
        self.dataview.feature_val(id as usize, feature)
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.samples.len()
    }
}
