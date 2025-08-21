use crate::config;
pub trait TrainConfigProvider: Sized {
    fn train_config(&mut self) -> &mut crate::config::TrainConfig;
}

pub trait CommonConfigBuilder: TrainConfigProvider {
    /// Sets max tree depth.
    fn with_max_depth(&mut self, n: usize) -> &mut Self {
        self.train_config().max_depth = n;
        self
    }

    /// Sets tree seed. It is used in Decision Trees if `max_features < total_features`. In De
    fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.train_config().seed = seed;
        self
    }

    /// Sets maximum features to consider in split.
    fn with_max_features(&mut self, max_features: config::NumFeatures) -> &mut Self {
        self.train_config().max_features = max_features;
        self
    }

    /// Sets minimal samples for splitting the node.
    fn with_min_samples_split(&mut self, num_samples: usize) -> &mut Self {
        self.train_config().min_samples_split = num_samples;
        self
    }

    /// Sets the least number of samples in leaf.
    fn with_min_samples_leaf(&mut self, num_samples: usize) -> &mut Self {
        self.train_config().min_samples_leaf = num_samples;
        self
    }
}

pub trait ClassifierConfigBuilder: TrainConfigProvider {
    /// Sets metric to Gini index.
    fn with_gini(&mut self) -> &mut Self {
        self.train_config().metric = config::Metric::GINI;
        self
    }
}

pub trait RegressorConfigBuilder: TrainConfigProvider {
    /// Sets metric to MSE.
    fn with_mse(&mut self) -> &mut Self {
        self.train_config().metric = config::Metric::MSE;
        self
    }
}

pub trait EnsembleConfigProvider: Sized {
    fn ensemble_config(&mut self) -> &mut crate::config::EnsembleConfig;
}

pub trait EnsembleConfigBuilder: EnsembleConfigProvider + CommonConfigBuilder {
    /// Sets the number of threads to use. Panics if zero is specified.
    fn with_threads(&mut self, n: usize) -> &mut Self {
        self.ensemble_config().num_threads = n;
        self
    }

    /// Sets number of trees in ensemble.
    fn with_trees(&mut self, n: usize) -> &mut Self {
        self.ensemble_config().num_trees = n;
        self
    }
}
