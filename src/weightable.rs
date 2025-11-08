// The weighted target mechanic is for training classifiers. It allows to store class and its
// weight in a single u32 variable. This accelerates learning due to faster sorting of
// (feature, weighted_label) pairs as more data stays in cache. It is very unprobable that during
// bootstrapping we'll get weight larger than 15, thus we use 4 bits to encode label weight.

pub type SampleWeight = u32;
const WEIGHT_BITS: usize = 4;
pub const WEIGHT_MASK: SampleWeight = (1 << WEIGHT_BITS) - 1;

pub trait Weightable: Clone
where
    Self: Sized,
{
    type Weighted: Clone + Copy;
    fn weight(&self, weight: SampleWeight) -> Self::Weighted;
    fn unweight(weighted: &Self::Weighted) -> (Self, SampleWeight);
}

impl Weightable for u32 {
    type Weighted = u32;
    #[inline(always)]
    fn weight(&self, weight: SampleWeight) -> Self::Weighted {
        (*self << WEIGHT_BITS) + weight as Self::Weighted
    }

    #[inline(always)]
    fn unweight(weighted: &Self::Weighted) -> (Self, SampleWeight) {
        (
            weighted >> WEIGHT_BITS,
            (weighted & WEIGHT_MASK) as SampleWeight,
        )
    }
}

impl Weightable for f32 {
    type Weighted = (f32, SampleWeight);
    #[inline(always)]
    fn weight(&self, weight: SampleWeight) -> Self::Weighted {
        (*self, weight)
    }

    #[inline(always)]
    fn unweight(weighted: &Self::Weighted) -> (Self, SampleWeight) {
        *weighted
    }
}
