use crate::ClassLabel;
use crate::LabelWeight;

pub const TARGET_WEIGHT_BITS: usize = 4;

pub trait Weightable: Clone
where
    Self: Sized,
{
    type Weighted: Clone + Copy;
    fn weight(&self, weight: LabelWeight) -> Self::Weighted;
    fn unweight(weighted: &Self::Weighted) -> (Self, LabelWeight);
}

impl Weightable for ClassLabel {
    type Weighted = ClassLabel;
    #[inline(always)]
    fn weight(&self, weight: LabelWeight) -> Self::Weighted {
        (*self << TARGET_WEIGHT_BITS) + weight as Self::Weighted
    }

    #[inline(always)]
    fn unweight(weighted: &Self::Weighted) -> (Self, LabelWeight) {
        (
            weighted >> TARGET_WEIGHT_BITS,
            (weighted & 0xf) as LabelWeight,
        )
    }
}

impl Weightable for f32 {
    type Weighted = (f32, LabelWeight);
    #[inline(always)]
    fn weight(&self, weight: LabelWeight) -> Self::Weighted {
        (*self, weight)
    }

    #[inline(always)]
    fn unweight(weighted: &Self::Weighted) -> (Self, LabelWeight) {
        *weighted
    }
}
