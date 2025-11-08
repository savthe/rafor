use crate::{config::EnsembleConfig, labels::SampleWeight, DatasetView, TrainView};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
    thread,
};

pub trait Trainable<T: Copy> {
    fn fit(&mut self, ts: TrainView<T>, seed: u64);
}

pub fn fit<Target, Trainee>(
    proto: Trainee,
    view: DatasetView,
    targets: &[Target],
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
                        let mut t = proto.clone();
                        let mut rng = SmallRng::seed_from_u64(seeds[id]);
                        let weights = bootstrap(targets.len(), &mut rng);
                        let ts = TrainView::new(view.clone(), &targets, &weights);
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

fn bootstrap(num_samples: usize, rng: &mut SmallRng) -> Vec<SampleWeight> {
    let mut weights: Vec<SampleWeight> = vec![0; num_samples];
    for _ in 0..num_samples {
        let i = rng.random_range(0..num_samples);
        weights[i] += 1
    }
    weights
}
