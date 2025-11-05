//! Argus RL - Reinforcement Learning agent for anti-bot evasion
//!
//! This crate implements a Discrete SAC (Stable Discrete Soft Actor-Critic)
//! reinforcement learning agent for intelligent anti-bot evasion.
//!
//! # Architecture
//!
//! - **State Space**: 15-dimensional continuous (timing, behavior, detection signals)
//! - **Action Space**: 8 discrete actions (wait, scroll, mouse, interact)
//! - **Algorithm**: Discrete SAC with entropy-penalty and double Q-learning
//! - **Framework**: tch-rs (PyTorch Rust bindings)
//!
//! See RL_RESEARCH_2025.md for comprehensive research and design decisions.

pub mod action;
pub mod agent;
pub mod reward;
pub mod state;

pub use action::Action;
pub use agent::RLAgent;
pub use reward::{RewardCalculator, StepOutcome};
pub use state::State;
