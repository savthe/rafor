#[derive(Default, Clone)]
pub struct Dataset {
    data: Vec<f32>,
    num_features: usize,
}

#[derive(Clone)]
pub struct DatasetView<'a> {
    data: &'a [f32],
    num_features: usize,
    size: usize,
}

impl Dataset {
    pub fn as_view(&self) -> DatasetView {
        DatasetView {
            data: &self.data,
            num_features: self.num_features,
            size: self.data.len() / self.num_features,
        }
    }

    pub fn with_transposed(data: &[f32], num_samples: usize) -> Self {
        assert!(data.len() % num_samples == 0);
        let num_features = data.len() / num_samples;

        let mut transposed: Vec<f32> = Vec::with_capacity(data.len());
        for feature in 0..num_features {
            transposed.extend(data.iter().skip(feature).step_by(num_features));
        }

        Self {
            data: transposed,
            num_features,
        }
    }
}

impl<'a> DatasetView<'a> {
    pub fn new(data: &'a [f32], num_features: usize) -> Self {
        assert!(data.len() % num_features == 0);
        Self {
            num_features,
            data,
            size: data.len() / num_features,
        }
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    #[inline(always)]
    pub fn samples(&self) -> std::slice::ChunksExact<'_, f32> {
        self.data.chunks_exact(self.num_features)
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    #[inline(always)]
    pub fn feature_val(&self, sample: usize, feature: usize) -> f32 {
        //self.data[sample * self.num_features + feature]
        self.data[self.size * feature + sample]
    }
}
