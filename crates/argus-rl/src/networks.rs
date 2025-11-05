//! Neural networks for Discrete SAC agent
//!
//! Implements:
//! - Actor network: State → Action probabilities
//! - Critic network: State + Action → Q-value
//! - Temperature parameter (α) for entropy tuning

use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

use crate::action::Action;
use crate::state::State;

/// Actor network that outputs action probabilities
#[derive(Debug)]
pub struct ActorNetwork {
    vs: nn::VarStore,
    fc1: nn::Linear,
    fc2: nn::Linear,
    output: nn::Linear,
}

impl ActorNetwork {
    /// Create a new Actor network
    ///
    /// Architecture: State(15) → FC(256) → ReLU → FC(256) → ReLU → FC(8) → Softmax
    pub fn new(device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let fc1 = nn::linear(&root / "fc1", State::dim() as i64, 256, Default::default());
        let fc2 = nn::linear(&root / "fc2", 256, 256, Default::default());
        let output = nn::linear(
            &root / "output",
            256,
            Action::dim() as i64,
            Default::default(),
        );

        Ok(Self {
            vs,
            fc1,
            fc2,
            output,
        })
    }

    /// Forward pass: State → Action probabilities
    pub fn forward(&self, state: &Tensor) -> Tensor {
        let x = state
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.output);

        // Softmax to get probabilities
        x.softmax(-1, tch::Kind::Float)
    }

    /// Sample action from policy
    pub fn sample_action(&self, state: &State) -> Result<Action> {
        let state_tensor = Tensor::from_slice(&state.to_tensor())
            .to_device(self.vs.device())
            .unsqueeze(0); // Add batch dimension

        let probs = self.forward(&state_tensor);

        // Sample from categorical distribution
        let action_idx = probs.multinomial(1, true).int64_value(&[0, 0]);

        Action::from_index(action_idx as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid action index: {}", action_idx))
    }

    /// Get action probabilities for a state
    pub fn get_probabilities(&self, state: &State) -> Result<Vec<f32>> {
        let state_tensor = Tensor::from_slice(&state.to_tensor())
            .to_device(self.vs.device())
            .unsqueeze(0);

        let probs = self.forward(&state_tensor);

        let probs_vec: Vec<f32> = probs.squeeze_dim(0).try_into()?;

        Ok(probs_vec)
    }

    /// Get VarStore for optimizer
    pub fn var_store(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }
}

/// Critic network that outputs Q-value for state-action pair
#[derive(Debug)]
pub struct CriticNetwork {
    vs: nn::VarStore,
    fc1: nn::Linear,
    fc2: nn::Linear,
    output: nn::Linear,
}

impl CriticNetwork {
    /// Create a new Critic network
    ///
    /// Architecture: State(15) + Action(8) → FC(256) → ReLU → FC(256) → ReLU → FC(1)
    pub fn new(device: Device) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let input_dim = (State::dim() + Action::dim()) as i64;
        let fc1 = nn::linear(&root / "fc1", input_dim, 256, Default::default());
        let fc2 = nn::linear(&root / "fc2", 256, 256, Default::default());
        let output = nn::linear(&root / "output", 256, 1, Default::default());

        Ok(Self {
            vs,
            fc1,
            fc2,
            output,
        })
    }

    /// Forward pass: (State, Action) → Q-value
    pub fn forward(&self, state: &Tensor, action_onehot: &Tensor) -> Tensor {
        // Concatenate state and action
        let input = Tensor::cat(&[state, action_onehot], -1);

        input
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.output)
    }

    /// Get Q-value for state-action pair
    pub fn q_value(&self, state: &State, action: Action) -> Result<f32> {
        let state_tensor = Tensor::from_slice(&state.to_tensor())
            .to_device(self.vs.device())
            .unsqueeze(0);

        // One-hot encode action
        let mut action_onehot = vec![0.0f32; Action::dim()];
        action_onehot[action.to_index()] = 1.0;
        let action_tensor = Tensor::from_slice(&action_onehot)
            .to_device(self.vs.device())
            .unsqueeze(0);

        let q = self.forward(&state_tensor, &action_tensor);

        Ok(q.double_value(&[0, 0]) as f32)
    }

    /// Get VarStore for optimizer
    pub fn var_store(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Save model to file
    pub fn save(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    /// Load model from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }
}

/// Temperature parameter (α) for entropy tuning
///
/// In SAC, the temperature controls the exploration-exploitation tradeoff.
/// Higher temperature = more exploration (more random actions)
/// Lower temperature = more exploitation (more greedy actions)
#[derive(Debug)]
pub struct TemperatureParameter {
    vs: nn::VarStore,
    log_alpha: Tensor,
    target_entropy: f64,
}

impl TemperatureParameter {
    /// Create a new temperature parameter
    ///
    /// # Arguments
    /// * `device` - Device to run on (CPU or CUDA)
    /// * `initial_value` - Initial temperature value (default: 0.2)
    /// * `target_entropy` - Target entropy (default: -dim(A) = -8)
    pub fn new(device: Device, initial_value: f64, target_entropy: Option<f64>) -> Result<Self> {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Store log(alpha) for numerical stability
        let log_alpha = root.var("log_alpha", &[1], nn::Init::Const(initial_value.ln()));

        // Default target entropy: -dim(A) (from SAC paper)
        let target_entropy = target_entropy.unwrap_or(-(Action::dim() as f64));

        Ok(Self {
            vs,
            log_alpha,
            target_entropy,
        })
    }

    /// Get current temperature value (α)
    pub fn alpha(&self) -> Tensor {
        self.log_alpha.exp()
    }

    /// Get log(α) for loss calculation
    pub fn log_alpha(&self) -> &Tensor {
        &self.log_alpha
    }

    /// Get target entropy
    pub fn target_entropy(&self) -> f64 {
        self.target_entropy
    }

    /// Get VarStore for optimizer
    pub fn var_store(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Get current alpha value as f32
    pub fn alpha_value(&self) -> f32 {
        self.alpha().double_value(&[0]) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_network_creation() {
        let device = Device::Cpu;
        let actor = ActorNetwork::new(device).unwrap();

        // Verify network was created
        assert!(actor.vs.variables().len() > 0);
    }

    #[test]
    fn test_actor_forward_pass() {
        let device = Device::Cpu;
        let actor = ActorNetwork::new(device).unwrap();

        let state = State::new();
        let state_tensor = Tensor::from_slice(&state.to_tensor()).unsqueeze(0);

        let probs = actor.forward(&state_tensor);

        // Check output shape: [1, 8]
        assert_eq!(probs.size(), vec![1, 8]);

        // Check probabilities sum to 1
        let sum: f32 = probs.sum(tch::Kind::Float).try_into().unwrap();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_actor_sample_action() {
        let device = Device::Cpu;
        let actor = ActorNetwork::new(device).unwrap();

        let state = State::new();
        let action = actor.sample_action(&state).unwrap();

        // Verify action is valid
        assert!(action.to_index() < Action::dim());
    }

    #[test]
    fn test_critic_network_creation() {
        let device = Device::Cpu;
        let critic = CriticNetwork::new(device).unwrap();

        assert!(critic.vs.variables().len() > 0);
    }

    #[test]
    fn test_critic_forward_pass() {
        let device = Device::Cpu;
        let critic = CriticNetwork::new(device).unwrap();

        let state = State::new();
        let state_tensor = Tensor::from_slice(&state.to_tensor()).unsqueeze(0);

        let action_onehot = Tensor::zeros(&[1, Action::dim() as i64], (tch::Kind::Float, device));

        let q_value = critic.forward(&state_tensor, &action_onehot);

        // Check output shape: [1, 1]
        assert_eq!(q_value.size(), vec![1, 1]);
    }

    #[test]
    fn test_critic_q_value() {
        let device = Device::Cpu;
        let critic = CriticNetwork::new(device).unwrap();

        let state = State::new();
        let action = Action::WaitShort;

        let q = critic.q_value(&state, action).unwrap();

        // Q-value should be a finite number
        assert!(q.is_finite());
    }

    #[test]
    fn test_temperature_parameter() {
        let device = Device::Cpu;
        let temp = TemperatureParameter::new(device, 0.2, None).unwrap();

        // Check initial value
        let alpha = temp.alpha_value();
        assert!((alpha - 0.2).abs() < 0.01);

        // Check target entropy
        assert_eq!(temp.target_entropy(), -8.0);
    }

    #[test]
    fn test_actor_get_probabilities() {
        let device = Device::Cpu;
        let actor = ActorNetwork::new(device).unwrap();

        let state = State::new();
        let probs = actor.get_probabilities(&state).unwrap();

        // Check we got 8 probabilities
        assert_eq!(probs.len(), 8);

        // Check they sum to ~1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Check all are non-negative
        assert!(probs.iter().all(|&p| p >= 0.0));
    }
}
