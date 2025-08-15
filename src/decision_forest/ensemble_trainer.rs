use crate::{
    decision_tree::Trainset,
    options::EnsembleOptions,
    weight::{Weightable, TARGET_WEIGHT_BITS},
    DatasetView, LabelWeight,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
    thread,
};

pub trait Trainable<Target: Weightable + Copy> {
    fn fit(&mut self, ts: Trainset<Target>, seed: u64);
}

pub fn fit<Target, Trainee>(
    proto: Trainee,
    view: DatasetView,
    targets: &[Target],
    opts: &EnsembleOptions,
) -> Vec<Trainee>
where
    Target: Weightable + Copy + Sync + Send,
    Trainee: Trainable<Target> + Clone + Send + Sync,
{
    let mut rng = SmallRng::seed_from_u64(opts.seed);
    let seeds: Vec<u64> = (0..opts.num_trees).map(|_| rng.random()).collect();

    let num_trees = opts.num_trees;
    let tree_idx = Arc::new(AtomicUsize::new(0));
    let mut ensemble: Vec<Trainee> = Vec::new();
    thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..opts.num_threads {
            let handle = s.spawn(|| {
                let mut trainees: Vec<Trainee> = Vec::new();
                let mut id = 0;
                while id < num_trees {
                    id = tree_idx.fetch_add(1, Ordering::Relaxed);
                    if id < num_trees {
                        let mut t = proto.clone();
                        let mut rng = SmallRng::seed_from_u64(seeds[id]);
                        let (samples, targets) = bootstrap(targets, &mut rng);
                        let ts = Trainset::from_bootstrap(view.clone(), samples, targets);
                        t.fit(ts, rng.random());
                        trainees.push(t);
                    }
                }
                trainees
            });

            handles.push(handle);
        }

        for handle in handles {
            ensemble.extend(handle.join().unwrap());
        }
    });

    ensemble
}

fn bootstrap<T: Copy>(targets: &[T], rng: &mut SmallRng) -> (Vec<u32>, Vec<(T, LabelWeight)>) {
    let weight_mask = (1 << TARGET_WEIGHT_BITS) - 1;
    let num_samples = targets.len();
    let mut weights: Vec<usize> = vec![0; num_samples];
    for _ in 0..num_samples {
        let i = rng.random_range(0..num_samples);
        weights[i] = (weights[i] + 1) & weight_mask;
    }

    let amount = weights.iter().filter(|x| **x > 0).count();
    let mut samples: Vec<u32> = Vec::with_capacity(amount);
    let mut weighted_targets: Vec<(T, LabelWeight)> = Vec::with_capacity(amount);

    for (i, &w) in weights.iter().enumerate() {
        if w > 0 {
            samples.push(i as u32);
            weighted_targets.push((targets[i], w as LabelWeight));
        }
    }

    (samples, weighted_targets)
}
