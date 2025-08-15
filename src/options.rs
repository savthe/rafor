#[derive(Clone)]
pub enum Metric {
    GINI,
    MSE
}

#[derive(Clone)]
pub struct TreeOptions {
    pub max_depth: usize,
    pub seed: u64,
    pub metric: Metric,
    pub max_features: NumFeatures

}

#[derive(Clone)]
pub struct EnsembleOptions {
    pub num_trees: usize,
    pub num_threads: usize,
    pub seed: u64,
}

#[derive(Clone)]
pub enum NumFeatures {
    SQRT,
    LOG,
    NUMBER(usize)
}

pub trait TreeOptionsProvider: Sized {
    fn tree_options(&mut self) -> &mut TreeOptions;
}

pub trait TreeOptionsBuilder: TreeOptionsProvider {
    fn with_max_depth(&mut self, n: usize) -> &mut Self {
        self.tree_options().max_depth = n;
        self
    }
}

pub trait ClassifierOptionsBuilder: TreeOptionsProvider {
    fn with_gini(&mut self) -> &mut Self {
        self.tree_options().metric = Metric::GINI;
        self
    }
}

pub trait RegressorOptionsBuilder: TreeOptionsProvider {
    fn with_mse(&mut self) -> &mut Self {
        self.tree_options().metric = Metric::MSE;
        self
    }
}

pub trait EnsembleOptionsProvider: Sized {
    fn ensemble_options(&mut self) -> &mut EnsembleOptions;
}

pub trait EnsembleOptionsBuilder: EnsembleOptionsProvider {
    fn with_threads(&mut self, n: usize) -> &mut Self {
        self.ensemble_options().num_threads = n;
        self
    }

    fn with_trees(&mut self, n: usize) -> &mut Self {
        self.ensemble_options().num_trees = n;
        self
    }

    fn with_seed(&mut self, n: u64) -> &mut Self {
        self.ensemble_options().seed = n;
        self
    }
}
