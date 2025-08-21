use crate::{ClassTarget, LabelWeight};

#[derive(Default, Clone, Debug)]
pub struct Gini {
    bins: Vec<u64>,
    num_items: u64,
    sum_squares: u64,
}

#[derive(Default, Clone)]
pub struct Mse {
    sum: f32,
    sum_squares: f32,
    num_items: u64,
}

pub trait ImpurityMetric<Target> {
    fn push(&mut self, item: Target, weight: LabelWeight);
    fn pop(&mut self, item: Target, weight: LabelWeight);
    fn impurity(&self) -> f32;
    fn split_impurity(&self, other: &Self) -> f32;
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
    fn impurity(&self) -> f32 {
        1.0 - self.sum_squares as f32 / (self.num_items * self.num_items) as f32
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f32 {
        1.0 - (self.sum_squares * other.num_items + other.sum_squares * self.num_items) as f32
            / (self.num_items * other.num_items * (self.num_items + other.num_items)) as f32
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
        self.sum += y * weight as f32;
        self.sum_squares += y * y * weight as f32;
        self.num_items += weight;
    }

    #[inline(always)]
    fn pop(&mut self, y: f32, weight: LabelWeight) {
        let weight = weight as u64;
        self.sum -= y * weight as f32;
        self.sum_squares -= y * y * weight as f32;
        self.num_items -= weight;
    }

    #[inline(always)]
    fn impurity(&self) -> f32 {
        let s = self.num_items as f32 * self.sum_squares - self.sum * self.sum;
        s / (self.num_items * self.num_items) as f32
    }

    #[inline(always)]
    fn split_impurity(&self, other: &Self) -> f32 {
        let d1 = self.num_items as f32 * self.sum_squares - self.sum * self.sum;
        let d2 = other.num_items as f32 * other.sum_squares - other.sum * other.sum;
        (d1 * other.num_items as f32 + d2 * self.num_items as f32)
            / ((self.num_items * other.num_items) * (self.num_items + other.num_items)) as f32
    }
}
