use serde::{Deserialize, Serialize};

use super::Resolve;
use super::Trainable;
// Block decision tree consists of 7-vertex balanced trees, where some vertices may become unused
// during training. Each 7-vertex tree is packed into block of size not larger than 64 bytes.
// Blocks are aligned by 64 bytes, which makes blocks cache-friendly during inference.
//       0
//     /  \
//    1    2
//   / \  / \
//  3  4  5  6
//
// In current implementation we don't store the information about whether a node is child or
// parent. Each terminal node of a block (nodes 3, 4, 5, 6) could be a parent of another block, so
// we keep 4 offsets. If offset is 0, then it is invalid, so the corresponding node is a leaf.
// During the training we propagate leaf value to all child nodes within current block. This means
// that during the inference we will always finish in one of terminal vertices of a block which
// makes up to 2 redundant comparisons, but still it is more efficient than finding the exact node
// to stop.

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[repr(C, align(64))]
struct Block {
    // Values are thresholds for parent nodes, and values for leaf nodes.
    values: [f32; 7],
    // Indexes of features for each node.
    features: [u16; 7],
    // Offsets for children blocks. Two offsets correspond to terminal node i: offsets[i - 3] and
    // offsets[i - 3] + 1.
    offsets: [u32; 4],
}

#[derive(Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockTree {
    tree: Vec<Block>,
}

#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Handle {
    // Index of a block.
    block: usize,
    // Index of a node withing block.
    node: usize,
}

impl Trainable for BlockTree {
    type Handle = Handle;
    fn root(&self) -> Self::Handle {
        Self::Handle::default()
    }

    fn new() -> Self {
        Self {
            tree: vec![Block::default(); 1],
        }
    }

    fn split(
        &mut self,
        handle: &Self::Handle,
        feature: u16,
        threshold: f32,
    ) -> (Self::Handle, Self::Handle) {
        let offset = self.tree.len();
        let block = &mut self.tree[handle.block];
        let i = handle.node;
        block.features[i] = feature;
        block.values[i] = threshold;
        if i >= 3 {
            // Splitting terminal node of a block. It will point to two child blocks.
            block.offsets[i - 3] = offset as u32;
            self.tree.push(Block::default());
            self.tree.push(Block::default());

            // Return handles to new blocks. Local node index of both blocks set to block root.
            (
                Self::Handle {
                    block: offset,
                    node: 0,
                },
                Self::Handle {
                    block: offset + 1,
                    node: 0,
                },
            )
        } else {
            // Splitting non-terminal node. It has child nodes within block, just provide their
            // indexes.
            (
                Self::Handle {
                    block: handle.block,
                    node: (i << 1) + 1,
                },
                Self::Handle {
                    block: handle.block,
                    node: (i << 1) + 2,
                },
            )
        }
    }

    fn set_leaf_value(&mut self, handle: &Self::Handle, value: u32) {
        let block = &mut self.tree[handle.block];
        let i = handle.node;
        let value = f32::from_bits(value);
        block.values[i] = value;

        // Propagate current leaf value to lower nodes.
        if i == 0 {
            block.values[3] = value;
            block.values[4] = value;
            block.values[5] = value;
            block.values[6] = value;
        } else if i == 1 {
            block.values[3] = value;
            block.values[4] = value;
        } else if i == 2 {
            block.values[5] = value;
            block.values[6] = value;
        }
    }
}

impl Block {
    #[inline(always)]
    fn compare(&self, index: usize, sample: &[f32]) -> bool {
        sample[self.features[index] as usize] <= self.values[index]
    }
}

impl BlockTree {
    #[inline(always)]
    pub fn num_blocks(&self) -> usize {
        self.tree.len()
    }
}

impl Resolve for BlockTree {
    fn resolve(&self, sample: &[f32]) -> u32 {
        assert!(!self.tree.is_empty());
        let mut cur_block = 0;
        loop {
            let b = &self.tree[cur_block];
            // Cute, but slower.
            // let c0 = !b.compare(0, sample) as usize;
            // let c1 = !b.compare(1 + c0, sample) as usize;
            // let i = 2*c0 + c1;
            // if b.offsets[i] == 0 {
            //     return b.values[i + 3].to_bits();
            // }
            // cur_block = b.offsets[i] as usize + !b.compare(i + 3, sample) as usize;

            macro_rules! process_term {
                ($idx:expr) => {
                    if b.offsets[$idx - 3] == 0 {
                        return b.values[$idx].to_bits();
                    }
                    cur_block = b.offsets[$idx - 3] as usize + !b.compare($idx, sample) as usize;
                };
            }

            if b.compare(0, sample) {
                if b.compare(1, sample) {
                    process_term!(3);
                } else {
                    process_term!(4);
                }
            } else {
                if b.compare(2, sample) {
                    process_term!(5);
                } else {
                    process_term!(6);
                }
            }
        }
    }
}
