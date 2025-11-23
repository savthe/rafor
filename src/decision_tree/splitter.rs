use crate::{ClassTarget, FloatTarget, SampleWeight};

use super::metrics::*;

#[derive(Default)]
pub struct Position {
    pub pivot: usize,
    pub impurity: f64,
}

pub trait Splitter<T> {
    // Calls before series of find_split calls for given samples range, but with different feature
    // orderings. Returns true if given range is not pure.
    fn prepare(&mut self, targets: &[(T, SampleWeight)]) -> bool;

    // Finds split point with impurity lower than upper_impurity. Data is a slice of pairs of some
    // feature value and weighted target.
    fn find_split(&self, data: &[(f32, T, SampleWeight)], upper_impurity: f64) -> Position;
}

pub struct GiniSplitter {
    num_classes: usize,
    min_samples_leaf: usize,
    range_imp: Gini,
}

pub struct MseSplitter {
    min_samples_leaf: usize,
    range_imp: Mse,
}

impl GiniSplitter {
    pub fn new(num_classes: usize, min_samples_leaf: usize) -> Self {
        Self {
            num_classes,
            min_samples_leaf,
            range_imp: Gini::with_classes(num_classes),
        }
    }
}

impl MseSplitter {
    pub fn new(min_samples_leaf: usize) -> Self {
        Self {
            min_samples_leaf,
            range_imp: Mse::default(),
        }
    }
}

impl Splitter<ClassTarget> for GiniSplitter {
    fn prepare(&mut self, targets: &[(ClassTarget, SampleWeight)]) -> bool {
        let mut gini = Gini::with_classes(self.num_classes);
        // TODO compute this without push.
        for &(label, weight) in targets.iter() {
            gini.push(label, weight);
        }
        self.range_imp = gini;
        !self.range_imp.pure()
    }

    fn find_split(&self, data: &[(f32, ClassTarget, SampleWeight)], upper_imp: f64) -> Position {
        let left = Gini::with_classes(self.num_classes);
        let right = self.range_imp.clone();
        find_split(left, right, data, upper_imp, self.min_samples_leaf)
    }
}

impl Splitter<FloatTarget> for MseSplitter {
    fn prepare(&mut self, targets: &[(FloatTarget, SampleWeight)]) -> bool {
        let mut mse = Mse::default();
        for &(label, weight) in targets.iter() {
            mse.push(label, weight);
        }
        self.range_imp = mse;
        !self.range_imp.pure()
    }

    fn find_split(&self, data: &[(f32, FloatTarget, SampleWeight)], upper_imp: f64) -> Position {
        let left = Mse::default();
        let right = self.range_imp.clone();
        find_split(left, right, data, upper_imp, self.min_samples_leaf)
    }
}

fn find_split<T: Copy, I: ImpurityMetric<T>>(
    mut left: I,
    mut right: I,
    data: &[(f32, T, SampleWeight)],
    upper_imp: f64,
    min_samples_leaf: usize,
) -> Position {
    let mut split = Position::default();
    split.impurity = upper_imp;
    for i in 0..data.len() - min_samples_leaf {
        let &(value, target, weight) = &data[i];
        left.push(target, weight);
        right.pop(target, weight);
        if value < data[i + 1].0
            && i + 1 >= min_samples_leaf
            && left.split_impurity(&right) < split.impurity
        {
            split.impurity = left.split_impurity(&right);
            split.pivot = i + 1;
            if split.impurity == 0. {
                break;
            }
        }
    }
    split
}
