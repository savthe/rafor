use crate::{decision_tree::TrainConfig, ensemble_trainer::EnsembleConfig, MaxFeaturesPolicy};

pub trait TrainConfigProvider: Sized {
    fn train_config(&mut self) -> &mut TrainConfig;
}

pub trait CommonTrainerBuilder: TrainConfigProvider {
    /// Sets max tree depth (`max_depth`).
    fn with_max_depth(&mut self, n: usize) -> &mut Self {
        self.train_config().max_depth = n;
        self
    }

    /// Sets tree `seed`.
    fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.train_config().seed = seed;
        self
    }

    /// Sets maximum features to consider in split (`max_features`).
    fn with_max_features(&mut self, max_features: MaxFeaturesPolicy) -> &mut Self {
        self.train_config().max_features = max_features;
        self
    }

    /// Sets minimal samples for splitting the node (`min_samples_split`).
    fn with_min_samples_split(&mut self, num_samples: usize) -> &mut Self {
        self.train_config().min_samples_split = num_samples;
        self
    }

    /// Sets the least number of samples in leaf (`min_samples_leaf`).
    fn with_min_samples_leaf(&mut self, num_samples: usize) -> &mut Self {
        self.train_config().min_samples_leaf = num_samples;
        self
    }

    /// Sets sample weights.
    fn with_weights(&mut self, weights: &[f32]) -> &mut Self {
        self.train_config().weights = weights.to_vec();
        self
    }
}

pub trait EnsembleConfigProvider: Sized {
    fn ensemble_config(&mut self) -> &mut EnsembleConfig;
}

pub trait EnsembleTrainerBuilder: EnsembleConfigProvider + CommonTrainerBuilder {
    /// Sets the number of threads to use. Panics if zero is specified (`num_threads`).
    fn with_threads(&mut self, n: usize) -> &mut Self {
        self.ensemble_config().num_threads = n;
        self
    }

    /// Sets number of trees in ensemble (`num_trees`).
    fn with_trees(&mut self, num_trees: usize) -> &mut Self {
        self.ensemble_config().num_trees = num_trees;
        self
    }
}
