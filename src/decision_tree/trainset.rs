use crate::{DatasetView, IndexRange, LabelWeight, Weightable};

pub struct Trainset<'a, T: Weightable + Copy> {
    dataset: DatasetView<'a>,
    samples: Vec<u32>,
    targets: Vec<T::Weighted>,
}

impl<'a, T: Weightable + Copy> Trainset<'a, T> {
    pub fn from_dataset(dataset: DatasetView<'a>, targets: &[T]) -> Trainset<'a, T> {
        Self {
            samples: (0..targets.len()).map(|i| i as u32).collect(),
            dataset,
            targets: targets.iter().map(|t| (t.weight(1))).collect(),
        }
    }

    pub fn from_bootstrap(
        dataset: DatasetView<'a>,
        samples: Vec<u32>,
        targets: Vec<(T, LabelWeight)>,
    ) -> Trainset<'a, T> {
        Trainset {
            dataset,
            samples,
            targets: targets.iter().map(|(t, w)| t.weight(*w)).collect(),
        }
    }

    #[inline(always)]
    pub fn targets(&self, range: &IndexRange) -> &[T::Weighted] {
        &self.targets[range.clone()]
    }

    #[inline(always)]
    pub fn samples(&self, range: &IndexRange) -> &[u32] {
        &self.samples[range.clone()]
    }

    pub fn split(&mut self, range: &IndexRange, feature: usize, threshold: f32) {
        let mut i = range.start;
        let mut j = range.end;
        while i < j {
            if self.feature_val(self.samples[i], feature) <= threshold {
                i += 1;
            } else {
                j -= 1;
                self.samples.swap(i, j);
                self.targets.swap(i, j);
            }
        }
    }

    #[inline(always)]
    pub fn feature_val(&self, id: u32, feature: usize) -> f32 {
        self.dataset.feature_val(id as usize, feature)
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.samples.len()
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.dataset.num_features()
    }
}
