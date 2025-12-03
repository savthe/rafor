use crate::{ClassTarget, SampleWeight};

#[derive(Default, Clone, Debug)]
pub struct Gini {
    bins: Vec<f64>,
    total_weight: f64,
    sum_squares: f64,
}

#[derive(Default, Clone)]
pub struct Mse {
    mean: f64,
    sum_squares: f64,
    total_weight: f64,
}

pub trait ImpurityMetric<Target> {
    fn push(&mut self, item: Target, weight: SampleWeight);
    fn pop(&mut self, item: Target, weight: SampleWeight);
    fn pure(&self) -> bool;
    fn split_impurity(&self, other: &Self) -> f64;
}

pub trait WithClasses {
    fn with_classes(num_classes: usize) -> Self;
}

impl ImpurityMetric<ClassTarget> for Gini {
    #[inline(always)]
    fn push(&mut self, bin_index: ClassTarget, weight: SampleWeight) {
        let weight = weight as f64;
        self.sum_squares += weight * (2. * self.bins[bin_index as usize] + weight);
        self.bins[bin_index as usize] += weight;
        self.total_weight += weight;
    }

    #[inline(always)]
    fn pop(&mut self, bin_index: ClassTarget, weight: SampleWeight) {
        let weight = weight as f64;
        self.sum_squares =
            self.sum_squares + weight * (weight - 2. * self.bins[bin_index as usize]);
        self.bins[bin_index as usize] -= weight;
        self.total_weight -= weight;
    }

    #[inline(always)]
    fn pure(&self) -> bool {
        let empty_bins = self.bins.iter().filter(|&x| *x == 0.).count();
        self.bins.len() <= empty_bins + 1
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f64 {
        1.0 - (self.sum_squares * other.total_weight + other.sum_squares * self.total_weight) as f64
            / (self.total_weight * other.total_weight * (self.total_weight + other.total_weight))
                as f64
    }
}

impl WithClasses for Gini {
    fn with_classes(num_classes: usize) -> Gini {
        Gini {
            bins: vec![0.; num_classes],
            total_weight: 0.,
            sum_squares: 0.,
        }
    }
}

impl ImpurityMetric<f32> for Mse {
    #[inline(always)]
    fn push(&mut self, y: f32, weight: SampleWeight) {
        let weight = weight as f64;
        let y = y as f64;

        let next_mean =
            self.mean + weight as f64 * (y - self.mean) / (self.total_weight + weight) as f64;
        self.sum_squares += weight as f64 * (y - self.mean) * (y - next_mean);
        self.mean = next_mean;
        self.total_weight += weight;
    }

    #[inline(always)]
    fn pop(&mut self, y: f32, weight: SampleWeight) {
        let weight = weight as f64;
        let y = y as f64;

        let next_mean =
            y + self.total_weight * (self.mean - y) / (self.total_weight - weight) as f64;
        self.sum_squares -= weight as f64 * (y - next_mean) * (y - self.mean);
        self.mean = next_mean;
        self.total_weight -= weight;
    }

    #[inline(always)]
    fn pure(&self) -> bool {
        self.sum_squares == 0.
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f64 {
        self.sum_squares + other.sum_squares
    }
}
