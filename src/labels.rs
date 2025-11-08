// For classification trees training, a label can be packed with weight during labels sorting
// by a feature. So instead of sorting triples (value, class, weight), the tuples 
// (value, class_and_weight) are sorted which is faster. Here we define type DenseClass, which
// supports packing of class and weight depending on the size of class placeholder in bits.

pub type ClassTarget = u32;
pub type FloatTarget = f32;
pub type SampleWeight = u32;

#[derive(Copy, Clone)]
pub struct DenseClass<const CLASS_BITS: usize> {
    data: u32,
}

pub trait Weighted<T>: Copy + Clone {
    fn new(t: &T, w: SampleWeight) -> Self;
    fn unweight(&self) -> (T, SampleWeight);
}

impl Weighted<FloatTarget> for (FloatTarget, SampleWeight) {
    #[inline(always)]
    fn new(t: &f32, w: SampleWeight) -> Self {
        (*t, w)
    }

    #[inline(always)]
    fn unweight(&self) -> (FloatTarget, SampleWeight) {
        (self.0, self.1)
    }
}

impl Weighted<ClassTarget> for (ClassTarget, SampleWeight) {
    #[inline(always)]
    fn new(t: &ClassTarget, w: SampleWeight) -> Self {
        (*t, w)
    }

    #[inline(always)]
    fn unweight(&self) -> (ClassTarget, SampleWeight) {
        (self.0, self.1)
    }
}

impl Weighted<ClassTarget> for DenseClass<8> {
    #[inline(always)]
    fn new(t: &ClassTarget, w: SampleWeight) -> Self {
        Self {
            data: (w << 8) | t
        }
    }

    #[inline(always)]
    fn unweight(&self) -> (ClassTarget, SampleWeight) {
        (self.data & 0xff, self.data >> 8)
    }
}

impl Weighted<ClassTarget> for DenseClass<16> {
    #[inline(always)]
    fn new(t: &ClassTarget, w: SampleWeight) -> Self {
        Self {
            data: (w << 16) | t
        }
    }

    #[inline(always)]
    fn unweight(&self) -> (ClassTarget, SampleWeight) {
        (self.data & 0xffff, self.data >> 16)
    }
}

impl Weighted<ClassTarget> for DenseClass<24> {
    #[inline(always)]
    fn new(t: &ClassTarget, w: SampleWeight) -> Self {
        Self {
            data: (w << 24) | t
        }
    }

    #[inline(always)]
    fn unweight(&self) -> (ClassTarget, SampleWeight) {
        (self.data & 0xffffff, self.data >> 24)
    }
}

