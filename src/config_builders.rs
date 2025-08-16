use crate::config;
pub trait TreeConfigProvider: Sized {
    fn tree_config(&mut self) -> &mut crate::config::TreeConfig;
}

pub trait CommonConfigBuilder: TreeConfigProvider {
    fn with_max_depth(&mut self, n: usize) -> &mut Self {
        self.tree_config().max_depth = n;
        self
    }

    fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.tree_config().seed = seed;
        self
    }
}

pub trait ClassifierConfigBuilder: TreeConfigProvider {
    fn with_gini(&mut self) -> &mut Self {
        self.tree_config().metric = config::Metric::GINI;
        self
    }
}

pub trait RegressorConfigBuilder: TreeConfigProvider {
    fn with_mse(&mut self) -> &mut Self {
        self.tree_config().metric = config::Metric::MSE;
        self
    }
}

pub trait EnsembleConfigProvider: Sized {
    fn ensemble_config(&mut self) -> &mut crate::config::EnsembleConfig;
}

pub trait EnsembleConfigBuilder: EnsembleConfigProvider {
    fn with_threads(&mut self, n: usize) -> &mut Self {
        self.ensemble_config().num_threads = n;
        self
    }

    fn with_trees(&mut self, n: usize) -> &mut Self {
        self.ensemble_config().num_trees = n;
        self
    }
}
