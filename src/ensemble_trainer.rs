use crate::{
    decision_tree,
    trainer_builders::{CommonTrainerBuilder, TrainConfigProvider},
    SampleWeight, Trainset,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
    thread,
};

// Configuration for training the ensembles of trees.
#[derive(Clone, PartialEq, Debug)]
pub struct EnsembleConfig {
    pub tree_config_proto: decision_tree::TrainConfig,

    /// Number of decision trees in ensemble.
    pub num_trees: usize,

    /// Number of threads to use. Please note that there is no specific value for "use all cores".
    /// Maximun number of theads can be obtained useing, for instance, crate `num_cpus`.
    pub num_threads: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            tree_config_proto: decision_tree::TrainConfig::default(),
            num_trees: 100,
            num_threads: 1,
        }
    }
}

impl TrainConfigProvider for EnsembleConfig {
    fn train_config(&mut self) -> &mut decision_tree::TrainConfig {
        &mut self.tree_config_proto
    }
}

impl CommonTrainerBuilder for EnsembleConfig {}

pub trait Trainable<T: Copy> {
    fn fit(&mut self, ts: &Trainset<T>, config: decision_tree::TrainConfig);
}

pub fn fit<Target, Trainee>(
    proto: Trainee,
    trainset: &Trainset<Target>,
    config: &EnsembleConfig,
) -> Vec<Trainee>
where
    Target: Copy + Sync + Send,
    Trainee: Trainable<Target> + Clone + Send + Sync,
{
    assert!(config.num_threads > 0);
    let seed = config.tree_config_proto.seed;
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
                        let mut trainee = proto.clone();
                        let mut train_config = config.tree_config_proto.clone();
                        train_config.scale_weights(&scalars);
                        train_config.seed = rng.random();
                        trainee.fit(trainset, train_config);
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
