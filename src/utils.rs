pub trait Aggregate {
    fn aggregate(&mut self, other: &[f32]);
}

impl Aggregate for [f32] {
    fn aggregate(&mut self, other: &[f32]) {
        assert!(self.len() == other.len());
        for (s, x) in self.iter_mut().zip(other.iter()) {
            *s += *x;
        }
    }
}
