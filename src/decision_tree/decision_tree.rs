use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Node<T> {
    value: T,
    threshold: f32,
    feature: u16,
    left: usize,
    right: usize,
}

#[derive(Default, Clone, Serialize, Deserialize)]
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

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn set_node(&mut self, node: usize, feature: u16, threshold: f32, value: T) {
        self.tree[node].value = value;
        self.tree[node].feature = feature;
        self.tree[node].threshold = threshold;
    }

    pub fn split_node(&mut self, index: usize) -> (usize, usize) {
        if !self.is_leaf(index) {
            panic!("Can't split non-leaf node")
        }

        let left = self.tree.len();
        let right = self.tree.len() + 1;
        self.tree.push(Node::default());
        self.tree.push(Node::default());

        let node = &mut self.tree[index];
        node.left = left;
        node.right = right;
        (left, right)
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
                id = node.right
            }
        }
        (self.tree[id].threshold, self.tree[id].value)
    }
}
