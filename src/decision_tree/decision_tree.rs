use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
enum Child {
    LEFT, RIGHT, ROOT
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeHandle {
    parent: u32,
    child: Child,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct InternalNode {
    feature: u16, 
    left_leaf: bool,
    right_leaf: bool,
    threshold: f32,
    left: u32,
    right: u32
}

impl Default for InternalNode {
    fn default() -> Self {
        Self {
            feature: 0,
            left_leaf: true,
            right_leaf: true,
            threshold: 0.0,
            left: 0,
            right: 0,
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
            num_features
        }
    }

    pub fn root(&self) -> NodeHandle {
        assert!(self.nodes.is_empty());
        NodeHandle {
            parent: 0,
            child: Child::ROOT,
        }
    }

    pub fn split(&mut self, handle: &NodeHandle, feature: u16, threshold: f32) -> (NodeHandle, NodeHandle) {
        let new_node = InternalNode {
            feature,
            left_leaf: true,
            right_leaf: true,
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
                parent.left_leaf = false;
            },
            Child::RIGHT => {
                parent.right = new_index;
                parent.right_leaf = false;
            }
            _ => {}
        };

        let left_handle = NodeHandle { parent: new_index, child: Child::LEFT };
        let right_handle = NodeHandle { parent: new_index, child: Child::RIGHT};
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
            },
            Child::RIGHT => {
                self.nodes[handle.parent as usize].right = value;
            }
            Child::ROOT => {
                let root = InternalNode {
                    feature: 0,
                    left: value,
                    right: value,
                    left_leaf: true,
                    right_leaf: true,
                    threshold: 0.0
                };
                self.nodes = vec![root];
            }
        };
    }

    pub fn predict(&self, sample: &[f32]) -> u32 {
        // FIXME if tree is empty?
        let mut id = 0;
        loop {
            let node = &self.nodes[id as usize];
            if sample[node.feature as usize] <= node.threshold {
                id = node.left;
                if node.left_leaf {
                    break;
                }
                
            } else {
                id = node.right;
                if node.right_leaf {
                    break;
                }
            }
        }
        
        id as u32
    }
}
