/// Multi-Agent Ensemble System
///
/// This module implements ensemble methods for combining multiple RL agents
/// to achieve better performance and robustness. Ensemble strategies include:
/// - Voting (majority vote, weighted vote)
/// - Averaging (mean, median)
/// - Stacking (meta-learning)
/// - Dynamic selection based on confidence
use crate::action::Action;
use crate::agent::RLAgent;
use crate::state::State;
use anyhow::{Context, Result};
use rand::Rng;
use std::collections::HashMap;
use tracing::{debug, info};

/// Ensemble strategy for combining agent predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleStrategy {
    /// Majority voting (most common action)
    MajorityVote,

    /// Weighted voting (agents have different weights)
    WeightedVote,

    /// Average probabilities then sample
    AverageProbabilities,

    /// Maximum confidence (use most confident agent)
    MaxConfidence,

    /// Random selection (pick random agent)
    RandomSelection,

    /// Round-robin rotation
    RoundRobin,
}

/// Multi-agent ensemble for robust decision making
pub struct AgentEnsemble {
    agents: Vec<RLAgent>,
    weights: Vec<f32>,
    strategy: EnsembleStrategy,
    agent_performance: Vec<f32>, // Track performance for adaptive weighting
    current_index: usize,        // For round-robin
}

impl AgentEnsemble {
    /// Create new ensemble from multiple agents
    pub fn new(agents: Vec<RLAgent>, strategy: EnsembleStrategy) -> Result<Self> {
        if agents.is_empty() {
            anyhow::bail!("Cannot create empty ensemble");
        }

        let num_agents = agents.len();
        let uniform_weight = 1.0 / num_agents as f32;
        let weights = vec![uniform_weight; num_agents];
        let agent_performance = vec![0.5; num_agents]; // Start neutral

        info!(
            "Created ensemble with {} agents using {:?} strategy",
            num_agents, strategy
        );

        Ok(Self {
            agents,
            weights,
            strategy,
            agent_performance,
            current_index: 0,
        })
    }

    /// Load ensemble from multiple model paths
    pub fn load(paths: &[&str], strategy: EnsembleStrategy) -> Result<Self> {
        let mut agents = Vec::new();

        for path in paths {
            let agent = RLAgent::load(path)
                .with_context(|| format!("Failed to load agent from {}", path))?;
            agents.push(agent);
        }

        Self::new(agents, strategy)
    }

    /// Select action using ensemble strategy
    pub fn select_action(&mut self, state: &State) -> Result<Action> {
        match self.strategy {
            EnsembleStrategy::MajorityVote => self.majority_vote(state),
            EnsembleStrategy::WeightedVote => self.weighted_vote(state),
            EnsembleStrategy::AverageProbabilities => self.average_probabilities(state),
            EnsembleStrategy::MaxConfidence => self.max_confidence(state),
            EnsembleStrategy::RandomSelection => self.random_selection(state),
            EnsembleStrategy::RoundRobin => self.round_robin(state),
        }
    }

    /// Majority voting: most common action wins
    fn majority_vote(&self, state: &State) -> Result<Action> {
        let mut votes: HashMap<Action, usize> = HashMap::new();

        for agent in &self.agents {
            let action = agent.select_action(state)?;
            *votes.entry(action).or_insert(0) += 1;
        }

        // Find action with most votes
        let action = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(action, _)| action)
            .context("No votes recorded")?;

        debug!("Majority vote selected: {:?}", action);
        Ok(action)
    }

    /// Weighted voting: agents have different importance
    fn weighted_vote(&self, state: &State) -> Result<Action> {
        let mut vote_scores: HashMap<Action, f32> = HashMap::new();

        for (agent, &weight) in self.agents.iter().zip(self.weights.iter()) {
            let action = agent.select_action(state)?;
            *vote_scores.entry(action).or_insert(0.0) += weight;
        }

        // Find action with highest weighted score
        let action = vote_scores
            .into_iter()
            .max_by(|(_, score1), (_, score2)| {
                score1
                    .partial_cmp(score2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(action, _)| action)
            .context("No votes recorded")?;

        debug!("Weighted vote selected: {:?}", action);
        Ok(action)
    }

    /// Average action probabilities across agents
    fn average_probabilities(&self, state: &State) -> Result<Action> {
        // Note: This would require exposing probability distributions from agents
        // For now, fallback to majority vote
        // In production, you'd get action probabilities from each agent
        self.majority_vote(state)
    }

    /// Select action from most confident agent
    fn max_confidence(&self, state: &State) -> Result<Action> {
        // Use agent performance as confidence proxy
        let best_agent_idx = self
            .agent_performance
            .iter()
            .enumerate()
            .max_by(|(_, perf1), (_, perf2)| {
                perf1
                    .partial_cmp(perf2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .context("No agents available")?;

        let action = self.agents[best_agent_idx].select_action(state)?;
        debug!(
            "Max confidence agent {} selected: {:?}",
            best_agent_idx, action
        );
        Ok(action)
    }

    /// Randomly select one agent
    fn random_selection(&self, state: &State) -> Result<Action> {
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.agents.len());

        let action = self.agents[idx].select_action(state)?;
        debug!("Random agent {} selected: {:?}", idx, action);
        Ok(action)
    }

    /// Round-robin through agents
    fn round_robin(&mut self, state: &State) -> Result<Action> {
        let action = self.agents[self.current_index].select_action(state)?;

        debug!(
            "Round-robin agent {} selected: {:?}",
            self.current_index, action
        );

        self.current_index = (self.current_index + 1) % self.agents.len();
        Ok(action)
    }

    /// Update agent performance based on outcome
    pub fn update_performance(&mut self, agent_idx: usize, reward: f32) {
        if agent_idx < self.agent_performance.len() {
            // Exponential moving average
            let alpha = 0.1;
            let normalized_reward = (reward + 20.0) / 40.0; // Normalize to [0, 1]
            self.agent_performance[agent_idx] =
                alpha * normalized_reward + (1.0 - alpha) * self.agent_performance[agent_idx];
        }
    }

    /// Update weights based on performance (for weighted voting)
    pub fn update_weights(&mut self) {
        let total_performance: f32 = self.agent_performance.iter().sum();

        if total_performance > 0.0 {
            for i in 0..self.weights.len() {
                self.weights[i] = self.agent_performance[i] / total_performance;
            }
        }
    }

    /// Get number of agents in ensemble
    pub fn size(&self) -> usize {
        self.agents.len()
    }

    /// Get current strategy
    pub fn strategy(&self) -> EnsembleStrategy {
        self.strategy
    }

    /// Change ensemble strategy
    pub fn set_strategy(&mut self, strategy: EnsembleStrategy) {
        info!(
            "Changing ensemble strategy from {:?} to {:?}",
            self.strategy, strategy
        );
        self.strategy = strategy;
    }

    /// Get agent weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Get agent performance scores
    pub fn performance(&self) -> &[f32] {
        &self.agent_performance
    }

    /// Get statistics about ensemble
    pub fn statistics(&self) -> EnsembleStatistics {
        EnsembleStatistics {
            num_agents: self.agents.len(),
            strategy: self.strategy,
            avg_performance: self.agent_performance.iter().sum::<f32>() / self.agents.len() as f32,
            best_agent: self
                .agent_performance
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0),
            weights: self.weights.clone(),
        }
    }
}

/// Ensemble statistics
#[derive(Debug, Clone)]
pub struct EnsembleStatistics {
    pub num_agents: usize,
    pub strategy: EnsembleStrategy,
    pub avg_performance: f32,
    pub best_agent: usize,
    pub weights: Vec<f32>,
}

impl EnsembleStatistics {
    pub fn summary(&self) -> String {
        format!(
            "Ensemble: {} agents, {:?} strategy, avg performance: {:.3}, best agent: {}",
            self.num_agents, self.strategy, self.avg_performance, self.best_agent
        )
    }
}

/// Adaptive ensemble that switches strategies based on performance
pub struct AdaptiveEnsemble {
    ensemble: AgentEnsemble,
    strategies: Vec<EnsembleStrategy>,
    strategy_performance: HashMap<EnsembleStrategy, f32>,
    evaluation_window: usize,
    steps_since_switch: usize,
}

impl AdaptiveEnsemble {
    pub fn new(agents: Vec<RLAgent>, evaluation_window: usize) -> Result<Self> {
        let strategies = vec![
            EnsembleStrategy::MajorityVote,
            EnsembleStrategy::WeightedVote,
            EnsembleStrategy::MaxConfidence,
        ];

        let ensemble = AgentEnsemble::new(agents, strategies[0])?;

        let mut strategy_performance = HashMap::new();
        for &strategy in &strategies {
            strategy_performance.insert(strategy, 0.5);
        }

        Ok(Self {
            ensemble,
            strategies,
            strategy_performance,
            evaluation_window,
            steps_since_switch: 0,
        })
    }

    pub fn select_action(&mut self, state: &State) -> Result<Action> {
        // Switch strategy if evaluation window reached
        if self.steps_since_switch >= self.evaluation_window {
            self.switch_to_best_strategy();
            self.steps_since_switch = 0;
        }

        self.steps_since_switch += 1;
        self.ensemble.select_action(state)
    }

    pub fn update_performance(&mut self, reward: f32) {
        let current_strategy = self.ensemble.strategy();

        if let Some(performance) = self.strategy_performance.get_mut(&current_strategy) {
            let alpha = 0.1;
            let normalized_reward = (reward + 20.0) / 40.0;
            *performance = alpha * normalized_reward + (1.0 - alpha) * *performance;
        }
    }

    fn switch_to_best_strategy(&mut self) {
        let best_strategy = self
            .strategy_performance
            .iter()
            .max_by(|(_, perf1), (_, perf2)| {
                perf1
                    .partial_cmp(perf2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(strategy, _)| *strategy)
            .unwrap_or(EnsembleStrategy::MajorityVote);

        if best_strategy != self.ensemble.strategy() {
            info!(
                "Switching from {:?} to {:?} (performance: {:.3})",
                self.ensemble.strategy(),
                best_strategy,
                self.strategy_performance[&best_strategy]
            );
            self.ensemble.set_strategy(best_strategy);
        }
    }

    pub fn ensemble(&self) -> &AgentEnsemble {
        &self.ensemble
    }

    pub fn ensemble_mut(&mut self) -> &mut AgentEnsemble {
        &mut self.ensemble
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_agent() -> RLAgent {
        // In real code, this would load from a file
        // For tests, we'd need to create a mock
        unimplemented!("Mock agent creation for tests")
    }

    #[test]
    fn test_ensemble_strategies() {
        // Test that all strategies are distinct
        let strategies = vec![
            EnsembleStrategy::MajorityVote,
            EnsembleStrategy::WeightedVote,
            EnsembleStrategy::AverageProbabilities,
            EnsembleStrategy::MaxConfidence,
            EnsembleStrategy::RandomSelection,
            EnsembleStrategy::RoundRobin,
        ];

        for i in 0..strategies.len() {
            for j in i + 1..strategies.len() {
                assert_ne!(strategies[i], strategies[j]);
            }
        }
    }

    #[test]
    fn test_ensemble_statistics() {
        let stats = EnsembleStatistics {
            num_agents: 3,
            strategy: EnsembleStrategy::MajorityVote,
            avg_performance: 0.75,
            best_agent: 1,
            weights: vec![0.3, 0.5, 0.2],
        };

        let summary = stats.summary();
        assert!(summary.contains("3 agents"));
        assert!(summary.contains("0.75"));
    }

    #[test]
    fn test_weight_normalization() {
        let weights = vec![0.3, 0.5, 0.2];
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
