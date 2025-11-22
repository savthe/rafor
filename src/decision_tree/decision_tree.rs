use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
enum Child {
    LEFT,
    RIGHT,
    ROOT,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeHandle {
    parent: u32,
    child: Child,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct InternalNode {
    left: u32,
    right: u32,
    threshold: f32,
    feature: u16,
    left_is_leaf: bool,
    right_is_leaf: bool,
}

impl Default for InternalNode {
    fn default() -> Self {
        Self {
            left: 0,
            right: 0,
            threshold: 0.0,
            feature: 0,
            left_is_leaf: true,
            right_is_leaf: true,
        }
    }
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionTree {
    nodes: Vec<InternalNode>,
    num_features: u16,
}

impl DecisionTree {
    pub fn new(num_features: u16) -> Self {
        Self {
            nodes: Vec::new(),
            num_features,
        }
    }

    pub fn root(&self) -> NodeHandle {
        assert!(self.nodes.is_empty());
        NodeHandle {
            parent: 0,
            child: Child::ROOT,
        }
    }

    pub fn split(
        &mut self,
        handle: &NodeHandle,
        feature: u16,
        threshold: f32,
    ) -> (NodeHandle, NodeHandle) {
        let new_node = InternalNode {
            feature,
            left_is_leaf: true,
            right_is_leaf: true,
            threshold,
            left: 0,
            right: 0,
        };
        let new_index = self.nodes.len() as u32;
        self.nodes.push(new_node);

        let parent = &mut self.nodes[handle.parent as usize];
        match handle.child {
            Child::LEFT => {
                parent.left = new_index;
                parent.left_is_leaf = false;
            }
            Child::RIGHT => {
                parent.right = new_index;
                parent.right_is_leaf = false;
            }
            _ => {}
        };

        let left_handle = NodeHandle {
            parent: new_index,
            child: Child::LEFT,
        };
        let right_handle = NodeHandle {
            parent: new_index,
            child: Child::RIGHT,
        };
        (left_handle, right_handle)
    }

    #[inline(always)]
    pub fn num_features(&self) -> usize {
        self.num_features as usize
    }

    pub fn set_leaf_value(&mut self, handle: &NodeHandle, value: u32) {
        match handle.child {
            Child::LEFT => {
                self.nodes[handle.parent as usize].left = value;
            }
            Child::RIGHT => {
                self.nodes[handle.parent as usize].right = value;
            }
            Child::ROOT => {
                // If tree has only one node (a leaf), make single internal node with identical
                // leaves.
                let root = InternalNode {
                    left: value,
                    right: value,
                    threshold: 0.0,
                    feature: 0,
                    left_is_leaf: true,
                    right_is_leaf: true,
                };
                self.nodes = vec![root];
            }
        };
    }

    pub fn predict(&self, sample: &[f32]) -> u32 {
        // FIXME if tree is empty?
        let mut id = 0;
        let mut is_leaf = false;

        while !is_leaf {
            let node = &self.nodes[id as usize];
            if sample[node.feature as usize] <= node.threshold {
                id = node.left;
                is_leaf = node.left_is_leaf;
            } else {
                id = node.right;
                is_leaf = node.right_is_leaf;
            }
        }

        id as u32
    }
}
