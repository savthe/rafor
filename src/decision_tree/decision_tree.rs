use bitvec::prelude::*;
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

// Serialized tree doesn't need information about child indexes if tree traverse order is defined.
// We will use BFS and recreate indexes during deserialization.
// For serialization we use PackedTree structure which effectively holds node values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PackedTree {
    thresholds: Vec<f32>,
    features: Vec<u16>,
    left_leaves_mask: BitVec,
    right_leaves_mask: BitVec,
    leaves: Vec<u32>,
    num_features: u16,
}

impl From<DecisionTree> for PackedTree {
    fn from(tree: DecisionTree) -> Self {
        // Queue of tree node indexes on next layer.
        let mut pending: Vec<usize> = vec![0];

        let mut packed = PackedTree {
            thresholds: Vec::new(),
            features: Vec::new(),
            left_leaves_mask: BitVec::new(),
            right_leaves_mask: BitVec::new(),
            leaves: Vec::new(),
            num_features: tree.num_features,
        };

        // While we have nodes to pack.
        while !pending.is_empty() {
            let mut next_pending: Vec<usize> = Vec::new();
            for &i in &pending {
                let node = &tree.nodes[i];
                packed.thresholds.push(node.threshold);
                packed.features.push(node.feature);
                packed.left_leaves_mask.push(node.left_is_leaf);
                packed.right_leaves_mask.push(node.right_is_leaf);

                // If node is a leaf, store its payload as leaf data, else is is a child index. 
                if node.left_is_leaf {
                    packed.leaves.push(node.left);
                } else {
                    next_pending.push(node.left as usize);
                }

                if node.right_is_leaf {
                    packed.leaves.push(node.right);
                } else {
                    next_pending.push(node.right as usize);
                }
            }
            pending = next_pending;
        }

        packed
    }
}

impl From<PackedTree> for DecisionTree {
    fn from(packed: PackedTree) -> Self {
        let mut tree = DecisionTree {
            nodes: Vec::new(),
            num_features: packed.num_features,
        };

        // Number of nodes to be read from current layer. Initially -- only root.
        let mut pending_size = 1;

        // Leaves are accessed sequentially, leaf_data_index is an index of leaf data in leaves
        // vector of a next leaf to be added to tree.
        let mut leaf_data_index = 0;

        // Index of first node on previous layer.
        let mut parents_offset = 0;

        while pending_size > 0 {
            let cur_layer_offset = tree.nodes.len();
            let mut next_pending_size = 0;
            // Create layer nodes and partially initialize them.
            for i in 0..pending_size {
                let offset = cur_layer_offset + i;
                let mut node = InternalNode {
                    left_is_leaf: packed.left_leaves_mask[offset],
                    right_is_leaf: packed.right_leaves_mask[offset],
                    feature: packed.features[offset],
                    threshold: packed.thresholds[offset],
                    left: 0,
                    right: 0,
                };

                if node.left_is_leaf {
                    node.left = packed.leaves[leaf_data_index];
                    leaf_data_index += 1;
                }
                else {
                    next_pending_size += 1;
                }

                if node.right_is_leaf {
                    node.right = packed.leaves[leaf_data_index];
                    leaf_data_index += 1;
                }
                else {
                    next_pending_size += 1;
                }

                tree.nodes.push(node);
            }

            // Connect parent nodes with current layer nodes.
            if tree.nodes.len() > 1 {
                let mut child_index = cur_layer_offset;
                while parents_offset < cur_layer_offset {
                    let parent = &mut tree.nodes[parents_offset];
                    if !parent.left_is_leaf {
                        parent.left = child_index as u32;
                        child_index += 1;
                    }

                    if !parent.right_is_leaf {
                        parent.right = child_index as u32;
                        child_index += 1;
                    }

                    parents_offset += 1;
                }
                assert!(child_index == tree.nodes.len());
            }

            pending_size = next_pending_size;
        }

        assert!(leaf_data_index == packed.leaves.len());

        tree
    }
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "PackedTree", into = "PackedTree")]
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
        assert!(!self.nodes.is_empty());
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
