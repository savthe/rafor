use crate::labels::ClassTarget;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Default, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ClassesMapping {
    decode_table: Vec<i64>,
}

impl ClassesMapping {
    pub fn with_encode(labels: &[i64]) -> (Self, Vec<ClassTarget>) {
        let mut m = Self {
            decode_table: Vec::new(),
        };

        let classes: BTreeSet<i64> = labels.iter().copied().collect();
        m.decode_table = classes.iter().copied().collect();
        let encode_table: HashMap<i64, usize> =
            classes.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        let encoded_labels = labels
            .iter()
            .map(|v| *encode_table.get(v).unwrap() as ClassTarget)
            .collect();
        (m, encoded_labels)
    }
}

pub trait ClassDecode {
    /// Returns a decode table of length num_classes().
    fn get_decode_table(&self) -> &[i64];

    /// Returns a number of classes stored int table.
    #[inline(always)]
    fn num_classes(&self) -> usize {
        self.get_decode_table().len()
    }

    /// Decodes a usize value into i64 according to decode table.
    #[inline(always)]
    fn decode(&self, class_enc: usize) -> i64 {
        self.get_decode_table()[class_enc]
    }
}

impl ClassDecode for ClassesMapping {
    #[inline(always)]
    fn get_decode_table(&self) -> &[i64] {
        &self.decode_table
    }
}
