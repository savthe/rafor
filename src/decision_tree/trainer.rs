use super::decision_tree::{DecisionTree, NodeHandle};
use super::splitter::Splitter;
use super::TrainView;
use crate::{
    config::{NumFeatures, TrainConfig},
    labels::*,
    DatasetView, IndexRange,
};
use radsort;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

impl TrainConfig {
    fn setup_max_features(&self, num_features: usize) -> (usize, Option<SmallRng>) {
        let max_features = match self.max_features {
            NumFeatures::SQRT => (num_features as f32).sqrt() as usize,
            NumFeatures::LOG => (num_features as f32).log2() as usize,
            NumFeatures::NUMBER(n) => n,
        };

        let mut rng = None;
        if max_features < num_features {
            rng = Some(SmallRng::seed_from_u64(self.seed))
        }

        (max_features, rng)
    }
}

#[derive(Default)]
struct Split {
    feature: usize,
    pivot: usize,
    threshold: f32,
}

// Trainer gets mutable refs to the tree it should train, and the training space. Target is at this
// point is generic, but actually for random forest, it is a weighted target. A splitter knows how
// to handle it, trainer just rearranges targets and passes them to splitter.
struct Trainer<'a, 'b, Target, S>
where
    Target: Copy,
    S: Splitter<Target>,
{
    features_perm: Vec<usize>,
    max_features: usize,
    rng: Option<SmallRng>,
    conf: TrainConfig,
    space: &'a mut TrainSpace<'b, Target>,
    tree: DecisionTree,
    splitter: S,
}

pub fn fit<Target>(
    space: &mut TrainSpace<Target>,
    conf: TrainConfig,
    splitter: impl Splitter<Target>,
) -> (DecisionTree, Vec<(NodeHandle, IndexRange)>)
where
    Target: Copy,
{
    let (max_features, rng) = conf.setup_max_features(space.num_features());
    let num_features = space.num_features();
    let mut trainer = Trainer {
        max_features,
        rng,
        features_perm: (0..space.num_features()).collect(),
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
            let mut split = Split::default();
            if depth < self.conf.max_depth
                && range.len() >= self.conf.min_samples_split
                && range.len() >= 2 * self.conf.min_samples_leaf
            {
                split = self.find_best_split(&range);
            }

            if split.pivot > 0 {
                self.space.split(&range, split.feature, split.threshold);
                let (left_range, right_range) = (range.start..split.pivot, split.pivot..range.end);
                let (left_node, right_node) =
                    self.tree.split(&node, split.feature as u16, split.threshold);

                stack.push((left_node, left_range, depth + 1));
                stack.push((right_node, right_range, depth + 1));
            } else {
                ranges.push((node, range));
            }
        }
        ranges
    }

    fn find_best_split(&mut self, range: &IndexRange) -> Split {
        let mut split = Split::default();
        let targets = self.space.targets(&range);

        if !self.splitter.prepare(targets) {
            return split;
        }

        let mut best_impurity = f64::INFINITY;
        let proto: Vec<(f32, Target)> = targets.iter().map(|t| (0., *t)).collect();
        self.prepare_features();

        for (idx, &feature) in self.features_perm.iter().enumerate() {
            let ordered_samples = self.prepare_samples(&proto, &range, feature);
            assert!(ordered_samples.len() == range.len());
            let p = self.splitter.find_split(&ordered_samples, best_impurity);
            if p.pivot > 0 {
                split.pivot = p.pivot + range.start;
                split.feature = feature;
                split.threshold =
                    (ordered_samples[p.pivot - 1].0 + ordered_samples[p.pivot].0) / 2.;
                best_impurity = p.impurity;
            }

            if best_impurity == 0. {
                break;
            }
            if idx >= self.max_features && split.pivot > range.start {
                break;
            }
        }
        split
    }

    fn prepare_features(&mut self) {
        if let Some(rng) = self.rng.as_mut() {
            self.features_perm.shuffle(rng);
        }
    }

    fn prepare_samples(
        &self,
        proto: &Vec<(f32, Target)>,
        range: &IndexRange,
        feature: usize,
    ) -> Vec<(f32, Target)> {
        let mut v = proto.clone();
        for ((x, _), y) in v.iter_mut().zip(self.space.samples(&range).iter()) {
            *x = self.space.feature_val(*y, feature);
        }
        radsort::sort_by_key(&mut v, |k| k.0);
        v
    }
}

pub struct TrainSpace<'a, W> {
    dataview: DatasetView<'a>,
    samples: Vec<u32>,
    targets: Vec<W>,
}

impl<'a, W> TrainSpace<'a, W> {
    pub fn new<T>(ts: TrainView<'a, T>) -> TrainSpace<'a, W>
    where
        W: Weighted<T>,
    {
        let amount = ts.weights.iter().filter(|&x| *x > 0).count();
        let mut samples: Vec<u32> = Vec::with_capacity(amount);
        let mut weighted_targets: Vec<W> = Vec::with_capacity(amount);

        for (i, (t, &w)) in ts.targets.iter().zip(ts.weights.iter()).enumerate() {
            if w > 0 {
                samples.push(i as u32);
                weighted_targets.push(W::new(t, w));
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
    pub fn targets(&self, range: &IndexRange) -> &[W] {
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
