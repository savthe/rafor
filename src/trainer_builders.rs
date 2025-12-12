use crate::decision_tree::trainer;
use crate::ensemble_trainer;
use ensemble_trainer::EnsembleConfig;
use crate::SampleWeight;
pub trait TrainConfigProvider: Sized {
    fn train_config(&mut self) -> &mut trainer::Config;
}

pub trait SampleWeightsSetter: Sized {
    fn set_sample_weights(&mut self, weights: &[SampleWeight]);
}

pub trait CommonTrainerBuilder: TrainConfigProvider + SampleWeightsSetter {
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
    fn with_max_features(&mut self, max_features: trainer::MaxFeaturesPolicy) -> &mut Self {
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
}

// pub trait ClassifierConfigBuilder: TrainConfigProvider {
//     /// Sets metric to Gini index.
//     fn with_gini(&mut self) -> &mut Self {
//         self.train_config().metric = config::Metric::GINI;
//         self
//     }
// }

// pub trait RegressorConfigBuilder: TrainConfigProvider {
//     /// Sets metric to MSE.
//     fn with_mse(&mut self) -> &mut Self {
//         self.train_config().metric = config::Metric::MSE;
//         self
//     }
// }

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
