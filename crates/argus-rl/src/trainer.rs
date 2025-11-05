//! Discrete SAC (Stable Discrete SAC) Training Loop
//!
//! Implements the SDSAC algorithm from arXiv:2209.10081 with:
//! - Entropy-penalty (not bonus)
//! - Double average Q-learning
//! - Q-clip for stability
//! - Soft target network updates

use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};
use tracing::{debug, info};

use crate::action::Action;
use crate::buffer::{ReplayBuffer, Transition};
use crate::networks::{ActorNetwork, CriticNetwork, TemperatureParameter};
use crate::state::State;

/// Configuration for SDSAC trainer
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Learning rate for actor network
    pub actor_lr: f64,

    /// Learning rate for critic networks
    pub critic_lr: f64,

    /// Learning rate for temperature parameter
    pub alpha_lr: f64,

    /// Discount factor (gamma)
    pub gamma: f64,

    /// Soft update coefficient (tau)
    pub tau: f64,

    /// Batch size for training
    pub batch_size: usize,

    /// Minimum buffer size before training starts
    pub min_buffer_size: usize,

    /// Q-value clip range (for stability)
    pub q_clip: Option<(f64, f64)>,

    /// Gradient clipping value
    pub grad_clip: Option<f64>,

    /// Device to run on (CPU or CUDA)
    pub device: Device,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            alpha_lr: 3e-4,
            gamma: 0.99,
            tau: 0.005,
            batch_size: 256,
            min_buffer_size: 1000,
            q_clip: Some((-10.0, 10.0)), // Prevent extreme Q-values
            grad_clip: Some(1.0),
            device: Device::Cpu,
        }
    }
}

/// SDSAC Trainer
///
/// Implements Stable Discrete Soft Actor-Critic training algorithm
pub struct SdsacTrainer {
    /// Actor network (policy)
    actor: ActorNetwork,

    /// Critic network 1
    critic1: CriticNetwork,

    /// Critic network 2 (for double Q-learning)
    critic2: CriticNetwork,

    /// Target critic network 1
    target_critic1: CriticNetwork,

    /// Target critic network 2
    target_critic2: CriticNetwork,

    /// Temperature parameter (alpha)
    temperature: TemperatureParameter,

    /// Actor optimizer
    actor_opt: nn::Optimizer,

    /// Critic 1 optimizer
    critic1_opt: nn::Optimizer,

    /// Critic 2 optimizer
    critic2_opt: nn::Optimizer,

    /// Temperature optimizer
    alpha_opt: nn::Optimizer,

    /// Training configuration
    config: TrainerConfig,

    /// Training step counter
    step: usize,
}

impl SdsacTrainer {
    /// Create a new SDSAC trainer
    pub fn new(config: TrainerConfig) -> Result<Self> {
        let device = config.device;

        // Initialize networks
        let mut actor = ActorNetwork::new(device)?;
        let mut critic1 = CriticNetwork::new(device)?;
        let mut critic2 = CriticNetwork::new(device)?;
        let target_critic1 = CriticNetwork::new(device)?;
        let target_critic2 = CriticNetwork::new(device)?;
        let mut temperature = TemperatureParameter::new(device, 0.2, None)?;

        // Create optimizers
        let actor_opt = nn::Adam::default().build(actor.var_store(), config.actor_lr)?;
        let critic1_opt = nn::Adam::default().build(critic1.var_store(), config.critic_lr)?;
        let critic2_opt = nn::Adam::default().build(critic2.var_store(), config.critic_lr)?;
        let alpha_opt = nn::Adam::default().build(temperature.var_store(), config.alpha_lr)?;

        Ok(Self {
            actor,
            critic1,
            critic2,
            target_critic1,
            target_critic2,
            temperature,
            actor_opt,
            critic1_opt,
            critic2_opt,
            alpha_opt,
            config,
            step: 0,
        })
    }

    /// Train on a batch from replay buffer
    pub fn train_step(&mut self, buffer: &mut ReplayBuffer) -> Result<TrainingMetrics> {
        if buffer.len() < self.config.min_buffer_size {
            return Ok(TrainingMetrics::default());
        }

        // Sample batch
        let (indices, transitions, weights) = buffer.sample(self.config.batch_size)?;

        // Convert to tensors
        let (states, actions, rewards, next_states, dones) =
            self.transitions_to_tensors(&transitions)?;
        let weights = Tensor::from_slice(&weights).to_device(self.config.device);

        // Update critics
        let (critic_loss, td_errors) =
            self.update_critics(&states, &actions, &rewards, &next_states, &dones, &weights)?;

        // Update actor
        let actor_loss = self.update_actor(&states)?;

        // Update temperature
        let alpha_loss = self.update_temperature(&states)?;

        // Update target networks (soft update)
        self.soft_update_targets();

        // Update priorities in replay buffer
        let td_errors_vec: Vec<f32> = td_errors.try_into()?;
        buffer.update_priorities(&indices, &td_errors_vec);

        // Update beta (importance sampling) - linearly increase to 1.0
        let beta = 0.4 + (self.step as f32 / 100_000.0) * 0.6;
        buffer.set_beta(beta.min(1.0));

        self.step += 1;

        Ok(TrainingMetrics {
            actor_loss: actor_loss.double_value(&[]),
            critic_loss: critic_loss.double_value(&[]),
            alpha_loss: alpha_loss.double_value(&[]),
            alpha_value: self.temperature.alpha_value() as f64,
            step: self.step,
        })
    }

    /// Update critic networks
    fn update_critics(
        &mut self,
        states: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        next_states: &Tensor,
        dones: &Tensor,
        weights: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Get current Q-values
        let q1 = self.critic1.forward(states, actions);
        let q2 = self.critic2.forward(states, actions);

        // Compute target Q-values
        let target_q = Tensor::no_grad(|| {
            // Get next action probabilities from actor
            let next_probs = self.actor.forward(next_states);

            // Get Q-values for all actions from target critics
            let batch_size = next_states.size()[0];
            let num_actions = Action::dim() as i64;

            let mut next_q1_all = Vec::new();
            let mut next_q2_all = Vec::new();

            for a in 0..num_actions {
                let mut action_onehot = vec![0.0f32; Action::dim()];
                action_onehot[a as usize] = 1.0;
                let action_tensor = Tensor::from_slice(&action_onehot)
                    .to_device(self.config.device)
                    .unsqueeze(0)
                    .repeat(&[batch_size, 1]);

                next_q1_all.push(self.target_critic1.forward(next_states, &action_tensor));
                next_q2_all.push(self.target_critic2.forward(next_states, &action_tensor));
            }

            let next_q1_tensor = Tensor::cat(&next_q1_all, 1);
            let next_q2_tensor = Tensor::cat(&next_q2_all, 1);

            // Double average Q-learning: use minimum of both critics
            let next_q = next_q1_tensor.min_other(&next_q2_tensor);

            // Apply Q-clip if configured
            let next_q = if let Some((min, max)) = self.config.q_clip {
                next_q.clamp(min, max)
            } else {
                next_q
            };

            // Expected Q-value: sum over action probabilities
            let alpha = self.temperature.alpha();
            let log_probs = (next_probs + 1e-8).log();
            let entropy = -(next_probs * log_probs).sum_dim_intlist(&[1], true, next_probs.kind());

            // SDSAC: Use entropy-penalty (subtract alpha * H)
            let expected_q = (next_probs * next_q).sum_dim_intlist(&[1], true, next_q.kind());
            let v_next = expected_q - alpha * entropy;

            // Bellman backup: r + gamma * (1 - done) * V(s')
            let not_dones = 1.0 - dones;
            rewards + self.config.gamma * not_dones * &v_next
        });

        // Compute TD errors
        let td_error1 = &q1 - &target_q;
        let td_error2 = &q2 - &target_q;

        // Importance-weighted MSE loss
        let critic1_loss = (weights * td_error1.pow_tensor_scalar(2)).mean(tch::Kind::Float);
        let critic2_loss = (weights * td_error2.pow_tensor_scalar(2)).mean(tch::Kind::Float);

        // Combined critic loss
        let critic_loss = &critic1_loss + &critic2_loss;

        // Backprop and update
        self.critic1_opt.zero_grad();
        self.critic2_opt.zero_grad();
        critic_loss.backward();

        // Gradient clipping
        if let Some(max_norm) = self.config.grad_clip {
            let _ = tch::nn::utils::clip_grad_norm_(
                self.critic1.var_store().trainable_variables(),
                max_norm,
            );
            let _ = tch::nn::utils::clip_grad_norm_(
                self.critic2.var_store().trainable_variables(),
                max_norm,
            );
        }

        self.critic1_opt.step();
        self.critic2_opt.step();

        // Return average TD error for priority updates
        let td_errors = (td_error1.abs() + td_error2.abs()) / 2.0;
        let td_errors = td_errors.squeeze_dim(1);

        Ok((critic_loss, td_errors))
    }

    /// Update actor network
    fn update_actor(&mut self, states: &Tensor) -> Result<Tensor> {
        // Get action probabilities
        let probs = self.actor.forward(states);

        // Get Q-values for all actions
        let batch_size = states.size()[0];
        let num_actions = Action::dim() as i64;

        let mut q1_all = Vec::new();
        let mut q2_all = Vec::new();

        for a in 0..num_actions {
            let mut action_onehot = vec![0.0f32; Action::dim()];
            action_onehot[a as usize] = 1.0;
            let action_tensor = Tensor::from_slice(&action_onehot)
                .to_device(self.config.device)
                .unsqueeze(0)
                .repeat(&[batch_size, 1]);

            q1_all.push(self.critic1.forward(states, &action_tensor));
            q2_all.push(self.critic2.forward(states, &action_tensor));
        }

        let q1_tensor = Tensor::cat(&q1_all, 1);
        let q2_tensor = Tensor::cat(&q2_all, 1);

        // Use minimum Q-value (conservative)
        let q_values = q1_tensor.min_other(&q2_tensor);

        // Entropy
        let log_probs = (probs + 1e-8).log();
        let entropy = -(probs * &log_probs).sum_dim_intlist(&[1], true, probs.kind());

        // SDSAC: Actor loss with entropy-penalty
        // J_π = E[π(a|s) * (α * log π(a|s) - Q(s,a))]
        let alpha = self.temperature.alpha().detach();
        let actor_loss = (probs * (alpha * log_probs - q_values))
            .sum_dim_intlist(&[1], true, probs.kind())
            .mean(tch::Kind::Float);

        // Backprop and update
        self.actor_opt.zero_grad();
        actor_loss.backward();

        if let Some(max_norm) = self.config.grad_clip {
            let _ = tch::nn::utils::clip_grad_norm_(
                self.actor.var_store().trainable_variables(),
                max_norm,
            );
        }

        self.actor_opt.step();

        Ok(actor_loss)
    }

    /// Update temperature parameter
    fn update_temperature(&mut self, states: &Tensor) -> Result<Tensor> {
        // Get action probabilities
        let probs = self.actor.forward(states).detach();

        // Compute entropy
        let log_probs = (probs + 1e-8).log();
        let entropy = -(probs * log_probs).sum_dim_intlist(&[1], false, probs.kind());

        // Temperature loss: J_α = -α * (entropy - target_entropy)
        let target_entropy =
            Tensor::from(self.temperature.target_entropy()).to_device(self.config.device);
        let alpha_loss = -(self.temperature.log_alpha().exp()
            * (entropy - target_entropy).detach())
        .mean(tch::Kind::Float);

        // Backprop and update
        self.alpha_opt.zero_grad();
        alpha_loss.backward();
        self.alpha_opt.step();

        Ok(alpha_loss)
    }

    /// Soft update of target networks
    fn soft_update_targets(&mut self) {
        let tau = self.config.tau;

        // Update target critic 1
        tch::no_grad(|| {
            for (target, source) in self
                .target_critic1
                .var_store()
                .variables()
                .iter()
                .zip(self.critic1.var_store().variables().iter())
            {
                let _ = target.1.copy_(&(tau * source.1 + (1.0 - tau) * &*target.1));
            }

            // Update target critic 2
            for (target, source) in self
                .target_critic2
                .var_store()
                .variables()
                .iter()
                .zip(self.critic2.var_store().variables().iter())
            {
                let _ = target.1.copy_(&(tau * source.1 + (1.0 - tau) * &*target.1));
            }
        });
    }

    /// Convert transitions to tensors
    fn transitions_to_tensors(
        &self,
        transitions: &[Transition],
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let batch_size = transitions.len();
        let state_dim = State::dim();
        let action_dim = Action::dim();

        let mut states_vec = Vec::with_capacity(batch_size * state_dim);
        let mut actions_vec = Vec::with_capacity(batch_size * action_dim);
        let mut rewards_vec = Vec::with_capacity(batch_size);
        let mut next_states_vec = Vec::with_capacity(batch_size * state_dim);
        let mut dones_vec = Vec::with_capacity(batch_size);

        for t in transitions {
            states_vec.extend_from_slice(&t.state.to_tensor());

            // One-hot encode action
            let mut action_onehot = vec![0.0f32; action_dim];
            action_onehot[t.action.to_index()] = 1.0;
            actions_vec.extend_from_slice(&action_onehot);

            rewards_vec.push(t.reward);
            next_states_vec.extend_from_slice(&t.next_state.to_tensor());
            dones_vec.push(if t.done { 1.0 } else { 0.0 });
        }

        let device = self.config.device;

        let states = Tensor::from_slice(&states_vec)
            .to_device(device)
            .view([batch_size as i64, state_dim as i64]);
        let actions = Tensor::from_slice(&actions_vec)
            .to_device(device)
            .view([batch_size as i64, action_dim as i64]);
        let rewards = Tensor::from_slice(&rewards_vec)
            .to_device(device)
            .view([batch_size as i64, 1]);
        let next_states = Tensor::from_slice(&next_states_vec)
            .to_device(device)
            .view([batch_size as i64, state_dim as i64]);
        let dones = Tensor::from_slice(&dones_vec)
            .to_device(device)
            .view([batch_size as i64, 1]);

        Ok((states, actions, rewards, next_states, dones))
    }

    /// Select action using current policy (for inference)
    pub fn select_action(&self, state: &State) -> Result<Action> {
        self.actor.sample_action(state)
    }

    /// Get current training step
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Save models to files
    pub fn save(&self, path_prefix: &str) -> Result<()> {
        self.actor.save(&format!("{}_actor.pt", path_prefix))?;
        self.critic1.save(&format!("{}_critic1.pt", path_prefix))?;
        self.critic2.save(&format!("{}_critic2.pt", path_prefix))?;
        info!("Saved models to {}_*.pt", path_prefix);
        Ok(())
    }

    /// Load models from files
    pub fn load(&mut self, path_prefix: &str) -> Result<()> {
        self.actor.load(&format!("{}_actor.pt", path_prefix))?;
        self.critic1.load(&format!("{}_critic1.pt", path_prefix))?;
        self.critic2.load(&format!("{}_critic2.pt", path_prefix))?;
        info!("Loaded models from {}_*.pt", path_prefix);
        Ok(())
    }
}

/// Training metrics for logging
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub actor_loss: f64,
    pub critic_loss: f64,
    pub alpha_loss: f64,
    pub alpha_value: f64,
    pub step: usize,
}

impl TrainingMetrics {
    pub fn log(&self) {
        info!(
            "Step {}: Actor={:.4}, Critic={:.4}, Alpha={:.4}, α={:.4}",
            self.step, self.actor_loss, self.critic_loss, self.alpha_loss, self.alpha_value
        );
    }
}
