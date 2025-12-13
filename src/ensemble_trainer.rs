use crate::{SampleWeight, Trainset};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
    thread,
};

// Configuration for training the ensembles of trees.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EnsembleConfig {
    /// Number of decision trees in ensemble.
    pub num_trees: usize,

    /// Number of threads to use. Please note that there is no specific value for "use all cores".
    /// Maximun number of theads can be obtained useing, for instance, crate `num_cpus`.
    pub num_threads: usize,
}

pub trait Trainable<T: Copy> {
    fn fit(&mut self, ts: Trainset<T>, seed: u64);
}

pub fn fit<Target, Trainee>(
    proto: Trainee,
    trainset: &Trainset<Target>,
    config: &EnsembleConfig,
    seed: u64,
) -> Vec<Trainee>
where
    Target: Copy + Sync + Send,
    Trainee: Trainable<Target> + Clone + Send + Sync,
{
    assert!(config.num_threads > 0);
    let mut rng = SmallRng::seed_from_u64(seed);
    let seeds: Vec<u64> = (0..config.num_trees).map(|_| rng.random()).collect();

    let num_trees = config.num_trees;
    let tree_idx = Arc::new(AtomicUsize::new(0));
    let mut ensemble: Vec<Trainee> = Vec::new();
    thread::scope(|s| {
        let mut handles = Vec::new();
        for _ in 0..config.num_threads {
            let handle = s.spawn(|| {
                let mut trainees: Vec<Trainee> = Vec::new();
                let mut id = 0;
                while id < num_trees {
                    id = tree_idx.fetch_add(1, Ordering::Relaxed);
                    if id < num_trees {
                        let mut rng = SmallRng::seed_from_u64(seeds[id]);
                        let scalars = bootstrap(trainset.size(), &mut rng);
                        let mut ts = trainset.clone();
                        ts.scale_weights(&scalars);
                        let mut trainee = proto.clone();
                        trainee.fit(ts, rng.random());
                        trainees.push(trainee);
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

fn bootstrap(num_samples: usize, rng: &mut SmallRng) -> Vec<SampleWeight> {
    let mut weights: Vec<SampleWeight> = vec![0.; num_samples];
    for _ in 0..num_samples {
        let i = rng.random_range(0..num_samples);
        weights[i] += 1.
    }
    weights
}
