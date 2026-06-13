mod block_tree;
mod classifier_model;
mod compact_tree;
mod metrics;
mod regressor_model;
mod splitter;
mod trainer;

pub use block_tree::BlockTree;
pub use classifier_model::ClassifierModel;
pub use compact_tree::CompactTree;
pub use regressor_model::RegressorModel;
pub use trainer::MaxFeaturesPolicy;
pub use trainer::TrainConfig;

pub trait Trainable {
    type Handle: Clone;
    fn new() -> Self;
    fn root(&self) -> Self::Handle;
    fn split(
        &mut self,
        handle: &Self::Handle,
        feature: u16,
        threshold: f32,
    ) -> (Self::Handle, Self::Handle);
    fn set_leaf_value(&mut self, handle: &Self::Handle, value: u32);
}

pub trait Resolve {
    fn resolve(&self, sample: &[f32]) -> u32;
}

pub trait Predictor: Resolve + Trainable {}

impl<P: Resolve + Trainable> Predictor for P {}
