use serde::{Deserialize, Serialize};

pub trait Splittable {
    fn split(&mut self, node: usize, feature: u16, threshold: f32) -> (usize, usize);
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node<T> {
    value: T,
    threshold: f32,
    feature: u16,
    left: usize,
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionTree<T: Default + Copy> {
    tree: Vec<Node<T>>,
    num_features: usize,
}

impl<T: Default + Copy> DecisionTree<T> {
    pub fn new(num_features: u16) -> Self {
        Self {
            tree: vec![Node::default(); 1],
            num_features: num_features as usize,
        }
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn set_node_value(&mut self, node: usize, value: T) {
        self.tree[node].value = value;
    }

    pub fn set_node_threshold(&mut self, node: usize, threshold: f32) {
        self.tree[node].threshold = threshold;
    }

    pub fn is_leaf(&self, index: usize) -> bool {
        self.tree[index].left == 0
    }

    pub fn predict(&self, sample: &[f32]) -> (f32, T) {
        let mut id = 0;
        while self.tree[id].left > 0 {
            let node = &self.tree[id];
            if sample[node.feature as usize] <= node.threshold {
                id = node.left
            } else {
                id = node.left + 1
            }
        }
        (self.tree[id].threshold, self.tree[id].value)
    }
}

impl<T: Default + Copy> Splittable for DecisionTree<T> {
    fn split(&mut self, index: usize, feature: u16, threshold: f32) -> (usize, usize) {
        if !self.is_leaf(index) {
            panic!("Can't split non-leaf node")
        }
        self.tree[index].feature = feature;
        self.tree[index].threshold = threshold;

        let left = self.tree.len();
        self.tree.push(Node::default());
        self.tree.push(Node::default());

        let node = &mut self.tree[index];
        node.left = left;
        (left, left + 1)
    }
}
