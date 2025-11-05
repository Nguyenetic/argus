//! Prioritized Experience Replay Buffer
//!
//! Implements a circular replay buffer with prioritized sampling based on TD-error.
//! Uses a SumTree data structure for efficient O(log n) sampling.

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::action::Action;
use crate::state::State;

/// A single transition in the replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Current state
    pub state: State,

    /// Action taken
    pub action: Action,

    /// Reward received
    pub reward: f32,

    /// Next state
    pub next_state: State,

    /// Whether episode ended
    pub done: bool,

    /// Priority for sampling (based on TD-error)
    pub priority: f32,
}

/// SumTree node for efficient prioritized sampling
#[derive(Debug, Clone)]
struct SumTreeNode {
    /// Sum of priorities in this subtree
    sum: f32,

    /// Left child index (if internal node)
    left: Option<usize>,

    /// Right child index (if internal node)
    right: Option<usize>,

    /// Data index (if leaf node)
    data_index: Option<usize>,
}

/// SumTree for O(log n) prioritized sampling
///
/// A binary tree where each node stores the sum of priorities in its subtree.
/// This enables efficient sampling proportional to priority.
#[derive(Debug)]
pub struct SumTree {
    nodes: Vec<SumTreeNode>,
    capacity: usize,
    size: usize,
}

impl SumTree {
    /// Create a new SumTree with given capacity
    pub fn new(capacity: usize) -> Self {
        // Tree needs 2*capacity - 1 nodes (complete binary tree)
        let num_nodes = 2 * capacity - 1;
        let mut nodes = Vec::with_capacity(num_nodes);

        // Initialize tree structure
        for i in 0..num_nodes {
            let is_leaf = i >= capacity - 1;

            if is_leaf {
                // Leaf node
                let data_index = i - (capacity - 1);
                nodes.push(SumTreeNode {
                    sum: 0.0,
                    left: None,
                    right: None,
                    data_index: Some(data_index),
                });
            } else {
                // Internal node
                let left = 2 * i + 1;
                let right = 2 * i + 2;
                nodes.push(SumTreeNode {
                    sum: 0.0,
                    left: Some(left),
                    right: Some(right),
                    data_index: None,
                });
            }
        }

        Self {
            nodes,
            capacity,
            size: 0,
        }
    }

    /// Update priority at given index
    pub fn update(&mut self, data_index: usize, priority: f32) {
        if data_index >= self.capacity {
            return;
        }

        let leaf_index = data_index + self.capacity - 1;
        let delta = priority - self.nodes[leaf_index].sum;

        // Update leaf
        self.nodes[leaf_index].sum = priority;

        // Propagate change up the tree
        let mut current = leaf_index;
        while current > 0 {
            let parent = (current - 1) / 2;
            self.nodes[parent].sum += delta;
            current = parent;
        }

        if data_index >= self.size {
            self.size = data_index + 1;
        }
    }

    /// Sample an index proportional to priority
    pub fn sample(&self, value: f32) -> Option<usize> {
        if self.size == 0 {
            return None;
        }

        let mut current = 0;
        let mut remaining = value;

        loop {
            let node = &self.nodes[current];

            // If leaf, return data index
            if let Some(data_index) = node.data_index {
                return Some(data_index);
            }

            // Otherwise, go left or right based on cumulative sum
            let left_idx = node.left.unwrap();
            let left_sum = self.nodes[left_idx].sum;

            if remaining <= left_sum {
                current = left_idx;
            } else {
                remaining -= left_sum;
                current = node.right.unwrap();
            }
        }
    }

    /// Get total sum of all priorities
    pub fn total(&self) -> f32 {
        self.nodes[0].sum
    }
}

/// Prioritized Experience Replay Buffer
///
/// Stores transitions and samples them with probability proportional to TD-error.
/// Higher TD-error = more surprising transition = sample more frequently.
pub struct ReplayBuffer {
    /// Circular buffer of transitions
    buffer: VecDeque<Transition>,

    /// SumTree for prioritized sampling
    sum_tree: SumTree,

    /// Maximum buffer size
    capacity: usize,

    /// Current write position
    position: usize,

    /// Alpha parameter for prioritization (0 = uniform, 1 = full prioritization)
    alpha: f32,

    /// Beta parameter for importance sampling (0 = no correction, 1 = full correction)
    beta: f32,

    /// Small constant to ensure non-zero priorities
    epsilon: f32,
}

impl ReplayBuffer {
    /// Create a new replay buffer
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions to store
    /// * `alpha` - Prioritization exponent (0 = uniform, 1 = full prioritization)
    /// * `beta` - Importance sampling exponent (increases to 1 during training)
    pub fn new(capacity: usize, alpha: f32, beta: f32) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            sum_tree: SumTree::new(capacity),
            capacity,
            position: 0,
            alpha,
            beta,
            epsilon: 1e-6,
        }
    }

    /// Add a transition to the buffer
    pub fn push(&mut self, transition: Transition) {
        // Use max priority for new transitions (ensures they're sampled at least once)
        let max_priority = if self.buffer.is_empty() {
            1.0
        } else {
            self.buffer
                .iter()
                .map(|t| t.priority)
                .fold(0.0f32, f32::max)
        };

        let mut transition = transition;
        transition.priority = max_priority;

        if self.buffer.len() < self.capacity {
            self.buffer.push_back(transition);
        } else {
            self.buffer[self.position] = transition;
        }

        self.sum_tree
            .update(self.position, max_priority.powf(self.alpha));
        self.position = (self.position + 1) % self.capacity;
    }

    /// Sample a batch of transitions
    ///
    /// Returns (indices, transitions, weights) where weights are importance sampling weights
    pub fn sample(&self, batch_size: usize) -> Result<(Vec<usize>, Vec<Transition>, Vec<f32>)> {
        if self.buffer.is_empty() {
            anyhow::bail!("Cannot sample from empty buffer");
        }

        let mut rng = rand::thread_rng();
        let total = self.sum_tree.total();

        let mut indices = Vec::with_capacity(batch_size);
        let mut transitions = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let segment = total / batch_size as f32;

        // Stratified sampling: divide [0, total] into batch_size segments
        for i in 0..batch_size {
            let a = segment * i as f32;
            let b = segment * (i + 1) as f32;
            let value = rng.gen_range(a..b);

            if let Some(idx) = self.sum_tree.sample(value) {
                if idx < self.buffer.len() {
                    indices.push(idx);
                    transitions.push(self.buffer[idx].clone());

                    // Calculate importance sampling weight
                    let priority = self.buffer[idx].priority;
                    let prob = (priority.powf(self.alpha) + self.epsilon) / (total + self.epsilon);
                    let weight = (self.buffer.len() as f32 * prob).powf(-self.beta);
                    weights.push(weight);
                }
            }
        }

        // Normalize weights
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        if max_weight > 0.0 {
            for w in &mut weights {
                *w /= max_weight;
            }
        }

        Ok((indices, transitions, weights))
    }

    /// Update priorities for a batch of transitions
    pub fn update_priorities(&mut self, indices: &[usize], priorities: &[f32]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            if idx < self.buffer.len() {
                self.buffer[idx].priority = priority.abs() + self.epsilon;
                self.sum_tree
                    .update(idx, (priority.abs() + self.epsilon).powf(self.alpha));
            }
        }
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Update beta parameter (should increase to 1.0 during training)
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.sum_tree = SumTree::new(self.capacity);
        self.position = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sumtree_basic() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert_eq!(tree.total(), 10.0);
    }

    #[test]
    fn test_sumtree_sampling() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        // Sample with value 0.5 should get index 0 (priority 1.0)
        assert_eq!(tree.sample(0.5), Some(0));

        // Sample with value 2.0 should get index 1 (cumulative 1.0 + 2.0)
        assert_eq!(tree.sample(2.0), Some(1));
    }

    #[test]
    fn test_replay_buffer_push() {
        let mut buffer = ReplayBuffer::new(10, 0.6, 0.4);

        let transition = Transition {
            state: State::new(),
            action: Action::WaitShort,
            reward: 1.0,
            next_state: State::new(),
            done: false,
            priority: 1.0,
        };

        buffer.push(transition);

        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer = ReplayBuffer::new(100, 0.6, 0.4);

        // Add some transitions
        for i in 0..50 {
            let transition = Transition {
                state: State::new(),
                action: Action::WaitShort,
                reward: i as f32,
                next_state: State::new(),
                done: false,
                priority: 1.0,
            };
            buffer.push(transition);
        }

        // Sample a batch
        let (indices, transitions, weights) = buffer.sample(32).unwrap();

        assert_eq!(indices.len(), 32);
        assert_eq!(transitions.len(), 32);
        assert_eq!(weights.len(), 32);

        // All weights should be normalized (max = 1.0)
        let max_weight = weights.iter().cloned().fold(0.0f32, f32::max);
        assert!((max_weight - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_replay_buffer_update_priorities() {
        let mut buffer = ReplayBuffer::new(100, 0.6, 0.4);

        for _ in 0..10 {
            let transition = Transition {
                state: State::new(),
                action: Action::WaitShort,
                reward: 1.0,
                next_state: State::new(),
                done: false,
                priority: 1.0,
            };
            buffer.push(transition);
        }

        // Update priorities
        let indices = vec![0, 1, 2];
        let new_priorities = vec![5.0, 10.0, 3.0];
        buffer.update_priorities(&indices, &new_priorities);

        // Verify priorities were updated
        assert!((buffer.buffer[0].priority - 5.0).abs() < 1e-4);
        assert!((buffer.buffer[1].priority - 10.0).abs() < 1e-4);
        assert!((buffer.buffer[2].priority - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buffer = ReplayBuffer::new(5, 0.6, 0.4);

        // Add more than capacity
        for i in 0..10 {
            let transition = Transition {
                state: State::new(),
                action: Action::WaitShort,
                reward: i as f32,
                next_state: State::new(),
                done: false,
                priority: 1.0,
            };
            buffer.push(transition);
        }

        // Buffer should not exceed capacity
        assert_eq!(buffer.len(), 5);

        // Should contain the last 5 transitions (rewards 5-9)
        assert!((buffer.buffer[0].reward - 5.0).abs() < 0.01);
    }
}
