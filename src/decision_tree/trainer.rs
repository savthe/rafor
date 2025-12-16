use super::decision_tree::{DecisionTree, NodeHandle};
use super::splitter::Splitter;
use crate::SampleWeight;
use crate::{IndexRange, Trainset};
use radsort;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

#[derive(Default)]
struct Split {
    feature: usize,
    pivot: usize,
    threshold: f32,
}

// TODO check that we don't need to add one.
/// Defines the limiting strategy for a number of features that are selected at each split.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MaxFeaturesPolicy {
    /// Takes `sqrt(total_features)`.
    SQRT,
    /// Takes `log2(total_features)`.
    LOG,
    /// Sets the exact number.
    NUMBER(usize),
}

/// Configuration for training a decision tree.
#[derive(Clone, PartialEq, Debug)]
pub struct TrainConfig {
    /// Max depth of a tree.
    pub max_depth: usize,

    /// Seed for randomizing feature sets in splits if `max_features < num_features`. When used in
    /// ensemble config, it sets the seed for random numbers in bootstrapping and feature sets in
    /// splits.
    pub seed: u64,

    /// Maximum number of features to use in each split. If `max_features` is less than total
    /// amount of features, at each split will be used a random subset of features with size at
    /// least `num_features`.
    ///
    /// **Note**. If trainer is unable find a splitting value in `num_features` features, it will
    /// consider additional features.
    pub max_features: MaxFeaturesPolicy,

    /// Minimal number of samples in the node that can be splitted.
    pub min_samples_split: usize,

    /// Forces leaves to have at least min_samples_leaf samples.
    pub min_samples_leaf: usize,

    pub weights: Vec<SampleWeight>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_depth: usize::MAX,
            max_features: MaxFeaturesPolicy::NUMBER(usize::MAX),
            seed: 42,
            min_samples_leaf: 1,
            min_samples_split: 2,
            weights: Vec::new(),
        }
    }
}

impl TrainConfig {
    pub fn scale_weights(&mut self, scalars: &[SampleWeight]) {
        if self.weights.is_empty() {
            self.weights = scalars.to_vec();
        } else {
            assert!(self.weights.len() == scalars.len());
            for (w, &s) in self.weights.iter_mut().zip(scalars.iter()) {
                *w *= s;
            }
        }
    }
}

struct FeaturePermutation {
    rng: Option<SmallRng>,
    perm: Vec<usize>,
}

pub struct TrainSpace<'a, T> {
    data: &'a [f32],
    samples: Vec<u32>,
    targets: Vec<(T, SampleWeight)>,
    num_features: usize,
    dataset_size: usize,
}

struct Trainer<'a, Target, S, Aggr>
where
    Target: Copy,
    S: Splitter<Target>,
    Aggr: Aggregator<Target>,
{
    max_features: usize,
    config: TrainConfig,
    space: TrainSpace<'a, Target>,
    tree: DecisionTree,
    splitter: S,
    features_perm: FeaturePermutation,
    aggregator: &'a mut Aggr,
}

pub trait Aggregator<T> {
    fn aggregate(&mut self, leaf_items: &[(T, SampleWeight)]) -> u32;
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

pub fn train<Target: Copy>(
    ts: &Trainset<Target>,
    config: TrainConfig,
    splitter: impl Splitter<Target>,
    aggregator: &mut impl Aggregator<Target>,
) -> DecisionTree {
    let space = TrainSpace::new(ts, &config.weights);
    let num_features = space.num_features();

    let max_features = match config.max_features {
        MaxFeaturesPolicy::SQRT => (num_features as f32).sqrt() as usize,
        MaxFeaturesPolicy::LOG => (num_features as f32).log2() as usize,
        MaxFeaturesPolicy::NUMBER(n) => n.min(num_features),
    };

    let rng = (max_features < num_features).then_some(SmallRng::seed_from_u64(config.seed));

    let num_features = space.num_features();
    let mut trainer = Trainer {
        max_features,
        features_perm: FeaturePermutation::new(num_features, rng),
        config,
        space,
        tree: DecisionTree::new(num_features as u16),
        splitter,
        aggregator,
    };

    trainer.fit();
    trainer.tree
}

impl<'a, Target, S, Aggr> Trainer<'a, Target, S, Aggr>
where
    Target: Copy,
    S: Splitter<Target>,
    Aggr: Aggregator<Target>,
{
    pub fn fit(&mut self) {
        let mut stack: Vec<(NodeHandle, IndexRange, usize)> =
            vec![(self.tree.root(), 0..self.space.size(), 0); 1];

        while let Some((node, range, depth)) = stack.pop() {
            let split = if depth < self.config.max_depth
                && range.len() >= self.config.min_samples_split
                && range.len() >= 2 * self.config.min_samples_leaf
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
                let value = self.aggregator.aggregate(self.space.targets(&range));
                self.tree.set_leaf_value(&node, value);
            }
        }
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

impl<'a, T: Copy> TrainSpace<'a, T> {
    pub fn new(ts: &'a Trainset<'a, T>, weights: &[SampleWeight]) -> TrainSpace<'a, T> {
        let mut samples: Vec<u32> = Vec::new();
        let mut weighted_targets: Vec<(T, SampleWeight)> = Vec::new();

        if weights.is_empty() {
            samples = (0..ts.targets.len()).map(|x| x as u32).collect();
            weighted_targets = ts.targets.iter().map(|&t| (t, 1.0)).collect();
        } else {
            for (i, (&t, &w)) in ts.targets.iter().zip(weights.iter()).enumerate() {
                if w > 0. {
                    samples.push(i as u32);
                    weighted_targets.push((t, w));
                }
            }
        }

        TrainSpace {
            data: &ts.data,
            samples,
            targets: weighted_targets,
            num_features: ts.data.len() / ts.targets.len(),
            dataset_size: ts.targets.len(),
        }
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features
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
        //self.dataview.feature_val(id as usize, feature)
        self.data[self.dataset_size * feature + id as usize]
    }

    #[inline(always)]
    fn size(&self) -> usize {
        self.samples.len()
    }
}
