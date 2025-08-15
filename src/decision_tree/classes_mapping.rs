use crate::ClassLabel;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct ClassesMapping {
    decode_table: Vec<i64>,
}

impl ClassesMapping {
    pub fn encode(&mut self, labels: &[i64]) -> Vec<ClassLabel> {
        let classes: BTreeSet<i64> = labels.iter().copied().collect();
        self.decode_table = classes.iter().copied().collect();
        let encode_table: HashMap<i64, usize> =
            classes.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        labels
            .iter()
            .map(|v| *encode_table.get(v).unwrap() as ClassLabel)
            .collect()
    }

    pub fn decode(&self, value: usize) -> i64 {
        self.decode_table[value]
    }

    pub fn num_classes(&self) -> usize {
        self.decode_table.len()
    }

    pub fn get_decode_table(&self) -> &[i64] {
        &self.decode_table
    }
}
