use crate::{ClassTarget, LabelWeight};

#[derive(Default, Clone, Debug)]
pub struct Gini {
    bins: Vec<u64>,
    num_items: u64,
    sum_squares: u64,
}

#[derive(Default, Clone)]
pub struct Mse {
    mean: f64,
    sum_sq_diffs: f64,
    num_items: u64,
}

pub trait ImpurityMetric<Target> {
    fn push(&mut self, item: Target, weight: LabelWeight);
    fn pop(&mut self, item: Target, weight: LabelWeight);
    fn impurity(&self) -> f64;
    fn split_impurity(&self, other: &Self) -> f64;
}

pub trait WithClasses {
    fn with_classes(num_classes: usize) -> Self;
}

impl ImpurityMetric<ClassTarget> for Gini {
    #[inline(always)]
    fn push(&mut self, bin_index: ClassTarget, weight: LabelWeight) {
        let weight = weight as u64;
        self.sum_squares += weight * (2 * self.bins[bin_index as usize] + weight);
        self.bins[bin_index as usize] += weight;
        self.num_items += weight;
    }

    #[inline(always)]
    fn pop(&mut self, bin_index: ClassTarget, weight: LabelWeight) {
        let weight = weight as u64;
        self.sum_squares =
            self.sum_squares + weight * weight - 2 * self.bins[bin_index as usize] * weight;
        self.bins[bin_index as usize] -= weight;
        self.num_items -= weight;
    }

    #[inline(always)]
    fn impurity(&self) -> f64 {
        1.0 - self.sum_squares as f64 / (self.num_items * self.num_items) as f64
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f64 {
        1.0 - (self.sum_squares * other.num_items + other.sum_squares * self.num_items) as f64
            / (self.num_items * other.num_items * (self.num_items + other.num_items)) as f64
    }
}

impl WithClasses for Gini {
    fn with_classes(num_classes: usize) -> Gini {
        Gini {
            bins: vec![0; num_classes],
            num_items: 0,
            sum_squares: 0,
        }
    }
}

impl ImpurityMetric<f32> for Mse {
    #[inline(always)]
    fn push(&mut self, y: f32, weight: LabelWeight) {
        let weight = weight as u64;
        let y = y as f64;
        let next_mean =
            self.mean + weight as f64 * (y - self.mean) / (self.num_items + weight) as f64;
        self.sum_sq_diffs += weight as f64 * (y - self.mean) * (y - next_mean);
        self.mean = next_mean;
        self.num_items += weight;
    }

    #[inline(always)]
    fn pop(&mut self, y: f32, weight: LabelWeight) {
        let weight = weight as u64;
        let y = y as f64;

        let next_mean =
            y + self.num_items as f64 * (self.mean - y) / (self.num_items - weight) as f64;
        self.sum_sq_diffs -= weight as f64 * (y - next_mean) * (y - self.mean);
        self.mean = next_mean;
        self.num_items -= weight;
    }

    #[inline(always)]
    fn impurity(&self) -> f64 {
        self.sum_sq_diffs / self.num_items as f64
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f64 {
        self.sum_sq_diffs + other.sum_sq_diffs
    }
}
