use super::Trainset;
use crate::{
    config::{NumFeatures, TrainConfig},
    metrics::ImpurityMetric,
    IndexRange, Weightable,
};
use radsort;
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

//TODO(Sav): we don't need this.
pub trait Trainable<T: Weightable> {
    fn split_node(&mut self, node: usize, feature: u16, threshold: f32) -> (usize, usize);
    fn handle_leaf(&mut self, node: usize, targets: &[T::Weighted]);
}

#[derive(Default)]
struct SplitDescriptor {
    feature: usize,
    pivot: usize,
    threshold: f32,
    left_impurity: f32,
    right_impurity: f32,
}

pub struct Trainer<'a, T, I>
where
    T: Weightable + Copy,
    I: ImpurityMetric<T> + Clone,
{
    impurity_proto: I,
    features_perm: Vec<usize>,
    max_features: usize,
    rng: Option<SmallRng>,
    conf: TrainConfig,
    trainset: Trainset<'a, T>,
}

impl<'a, T, I> Trainer<'a, T, I>
where
    T: Weightable + Copy,
    I: ImpurityMetric<T> + Clone,
{
    pub fn fit<Trainee>(
        trainset: Trainset<T>,
        conf: TrainConfig,
        trainee: &mut Trainee,
        impurity_proto: I,
    ) where
        T: Copy,
        I: ImpurityMetric<T> + Clone,
        Trainee: Trainable<T>,
    {
        let max_features = match conf.max_features {
            NumFeatures::SQRT => (trainset.num_features() as f32).sqrt() as usize,
            NumFeatures::LOG => (trainset.num_features() as f32).log2() as usize,
            NumFeatures::NUMBER(n) => n,
        };

        let mut rng = None;
        if max_features < trainset.num_features() {
            rng = Some(SmallRng::seed_from_u64(conf.seed))
        }

        let features_perm: Vec<usize> = (0..trainset.num_features()).collect();

        let mut trainer = Trainer {
            impurity_proto,
            max_features,
            rng,
            features_perm,
            conf,
            trainset,
        };

        trainer.fit_internal(trainee);
    }

    pub fn fit_internal<Trainee>(&mut self, trainee: &mut Trainee)
    where
        Trainee: Trainable<T>,
    {
        let mut stack: Vec<(usize, IndexRange, usize)> = vec![(0, 0..self.trainset.size(), 0); 1];
        while let Some((node, range, depth)) = stack.pop() {
            if let Some(split) = self.find_best_split(&range) {
                self.trainset.split(&range, split.feature, split.threshold);
                let (left_range, right_range) = (range.start..split.pivot, split.pivot..range.end);
                let (left_node, right_node) =
                    trainee.split_node(node, split.feature as u16, split.threshold);

                if depth + 1 == self.conf.max_depth || split.left_impurity == 0. {
                    trainee.handle_leaf(left_node, self.targets(&left_range));
                } else {
                    stack.push((left_node, left_range, depth + 1));
                }

                if depth + 1 == self.conf.max_depth || split.right_impurity == 0. {
                    trainee.handle_leaf(right_node, self.targets(&right_range));
                } else {
                    stack.push((right_node, right_range, depth + 1));
                }
            } else {
                trainee.handle_leaf(node, self.targets(&range));
            }
        }
    }

    fn find_best_split(&mut self, range: &IndexRange) -> Option<SplitDescriptor> {
        let mut split = SplitDescriptor::default();
        let imp_range = self.compute_impurity(range);
        let mut best_impurity = imp_range.impurity();

        let proto: Vec<(f32, T::Weighted)> = self.targets(range).iter().map(|t| (0., *t)).collect();

        self.prepare_features();

        for (idx, &feature) in self.features_perm.iter().enumerate() {
            let ordered_samples = self.prepare_samples(&proto, range, feature);
            let mut imp_left = self.impurity_proto.clone();
            let mut imp_right = imp_range.clone();

            for i in 0..ordered_samples.len() - 1 {
                let (value, t) = ordered_samples[i];
                let next = ordered_samples[i + 1].0;
                let (label, weight) = T::unweight(&t);
                imp_left.push(label, weight);
                imp_right.pop(label, weight);
                if value < next && imp_left.split_impurity(&imp_right) < best_impurity {
                    best_impurity = imp_left.split_impurity(&imp_right);
                    split.pivot = range.start + i + 1;
                    split.feature = feature;
                    split.threshold = (value + next) / 2.;
                    split.left_impurity = imp_left.impurity();
                    split.right_impurity = imp_right.impurity();
                }
            }

            if idx >= self.max_features && split.pivot > range.start {
                break;
            }
        }

        if best_impurity < imp_range.impurity() {
            Some(split)
        } else {
            None
        }
    }

    fn prepare_features(&mut self) {
        if let Some(rng) = self.rng.as_mut() {
            self.features_perm.shuffle(rng);
        }
    }

    fn compute_impurity(&self, range: &IndexRange) -> I {
        let mut imp = self.impurity_proto.clone();
        for t in self.targets(range).iter() {
            let (label, weight) = T::unweight(t);
            imp.push(label, weight);
        }
        imp
    }

    fn targets(&self, range: &IndexRange) -> &[T::Weighted] {
        self.trainset.targets(&range)
    }

    fn prepare_samples(
        &self,
        proto: &Vec<(f32, T::Weighted)>,
        r: &IndexRange,
        feature: usize,
    ) -> Vec<(f32, T::Weighted)> {
        let mut v = proto.clone();
        for ((x, _), y) in v.iter_mut().zip(self.trainset.samples(&r).iter()) {
            *x = self.trainset.feature_val(*y, feature);
        }
        radsort::sort_by_key(&mut v, |k| k.0);
        v
    }
}
