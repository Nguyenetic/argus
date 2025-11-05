//! Training environment with synthetic bot detector
//!
//! Simulates a website with bot detection that the RL agent must learn to evade.

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::reward::{RewardCalculator, StepOutcome};
use crate::state::State;

/// Detection result from bot detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Whether bot was detected
    pub detected: bool,

    /// Confidence score (0-1)
    pub confidence: f32,

    /// Detection reason
    pub reason: DetectionReason,
}

/// Reasons for bot detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectionReason {
    /// No detection
    None,

    /// Request frequency too high
    HighFrequency,

    /// Mouse movement too linear
    LinearMouseMovement,

    /// Scroll pattern unnatural
    UnaturalScrolling,

    /// No interaction observed
    NoInteraction,

    /// Timing too regular
    RegularTiming,

    /// Multiple signals combined
    MultipleSignals(Vec<String>),
}

/// Synthetic bot detector
///
/// Rule-based detector that checks for bot-like behavior patterns
#[derive(Debug)]
pub struct BotDetector {
    /// Threshold for detection (0-1)
    detection_threshold: f32,

    /// Whether detector is strict
    strict_mode: bool,
}

impl BotDetector {
    /// Create a new bot detector
    pub fn new(detection_threshold: f32, strict_mode: bool) -> Self {
        Self {
            detection_threshold,
            strict_mode,
        }
    }

    /// Detect bot behavior from state
    pub fn detect(&self, state: &State, action_history: &[Action]) -> DetectionResult {
        let mut suspicion_score = 0.0;
        let mut reasons = Vec::new();

        // Rule 1: Check request frequency (high frequency = suspicious)
        if state.time_since_last_request < 0.2 {
            suspicion_score += 0.3;
            reasons.push("high_frequency".to_string());
        }

        // Rule 2: Check timing regularity (too regular = bot)
        if state.request_variance < 0.1 {
            suspicion_score += 0.25;
            reasons.push("regular_timing".to_string());
        }

        // Rule 3: Check mouse movement entropy (low entropy = linear = bot)
        if state.mouse_movement_entropy < 0.3 {
            suspicion_score += 0.2;
            reasons.push("linear_mouse".to_string());
        }

        // Rule 4: Check scroll pattern (low score = unnatural)
        if state.scroll_pattern_score < 0.3 {
            suspicion_score += 0.15;
            reasons.push("unnatural_scroll".to_string());
        }

        // Rule 5: Check interaction count (too few = suspicious)
        if state.interaction_count < 0.2 && state.requests_in_session > 0.3 {
            suspicion_score += 0.1;
            reasons.push("no_interaction".to_string());
        }

        // Rule 6: Strict mode - check action patterns
        if self.strict_mode && action_history.len() > 3 {
            // Check for repetitive actions
            let last_3 = &action_history[action_history.len() - 3..];
            if last_3[0] == last_3[1] && last_3[1] == last_3[2] {
                suspicion_score += 0.2;
                reasons.push("repetitive_actions".to_string());
            }
        }

        // Normalize suspicion score
        suspicion_score = suspicion_score.min(1.0);

        // Determine detection
        let detected = suspicion_score > self.detection_threshold;

        let reason = if !detected {
            DetectionReason::None
        } else if reasons.is_empty() {
            DetectionReason::None
        } else if reasons.len() == 1 {
            match reasons[0].as_str() {
                "high_frequency" => DetectionReason::HighFrequency,
                "regular_timing" => DetectionReason::RegularTiming,
                "linear_mouse" => DetectionReason::LinearMouseMovement,
                "unnatural_scroll" => DetectionReason::UnaturalScrolling,
                "no_interaction" => DetectionReason::NoInteraction,
                _ => DetectionReason::None,
            }
        } else {
            DetectionReason::MultipleSignals(reasons)
        };

        DetectionResult {
            detected,
            confidence: suspicion_score,
            reason,
        }
    }
}

/// Training environment
///
/// Simulates a website with bot detection for RL agent training
pub struct TrainingEnvironment {
    /// Current state
    state: State,

    /// Bot detector
    detector: BotDetector,

    /// Reward calculator
    reward_calc: RewardCalculator,

    /// Action history for current episode
    action_history: Vec<Action>,

    /// Episode step counter
    step: usize,

    /// Maximum steps per episode
    max_steps: usize,

    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl TrainingEnvironment {
    /// Create a new training environment
    pub fn new(detection_threshold: f32, strict_mode: bool, max_steps: usize) -> Self {
        Self {
            state: State::new(),
            detector: BotDetector::new(detection_threshold, strict_mode),
            reward_calc: RewardCalculator::default(),
            action_history: Vec::new(),
            step: 0,
            max_steps,
            rng: rand::thread_rng(),
        }
    }

    /// Reset environment for new episode
    pub fn reset(&mut self) -> State {
        self.state = State::new();
        self.action_history.clear();
        self.step = 0;
        self.state.clone()
    }

    /// Take a step in the environment
    pub fn step(&mut self, action: Action) -> Result<(State, f32, bool, StepOutcome)> {
        self.step += 1;
        self.action_history.push(action);

        // Update state based on action
        self.update_state_from_action(&action);

        // Add some randomness to simulate real-world variability
        self.add_state_noise();

        // Detect bot behavior
        let detection = self.detector.detect(&self.state, &self.action_history);

        // Calculate reward
        let outcome = self.create_outcome(&detection);
        let reward = self.reward_calc.calculate(&outcome, &self.state);

        // Check if episode is done
        let done = detection.detected || self.step >= self.max_steps;

        Ok((self.state.clone(), reward, done, outcome))
    }

    /// Update state based on action taken
    fn update_state_from_action(&mut self, action: &Action) {
        match action {
            Action::WaitShort => {
                self.state.time_since_last_request = self.rng.gen_range(0.5..2.0);
                self.state.avg_request_interval = (self.state.avg_request_interval * 0.9
                    + self.state.time_since_last_request * 0.1)
                    .min(1.0);
            }
            Action::WaitLong => {
                self.state.time_since_last_request = self.rng.gen_range(2.0..10.0);
                self.state.avg_request_interval = (self.state.avg_request_interval * 0.9
                    + self.state.time_since_last_request * 0.1)
                    .min(1.0);
                self.state.request_variance += 0.05;
            }
            Action::ScrollSmall => {
                self.state.scroll_pattern_score = (self.state.scroll_pattern_score + 0.1).min(1.0);
                self.state.interaction_count += 0.05;
            }
            Action::ScrollLarge => {
                self.state.scroll_pattern_score = (self.state.scroll_pattern_score + 0.15).min(1.0);
                self.state.interaction_count += 0.08;
            }
            Action::MouseMovement => {
                self.state.mouse_movement_entropy =
                    (self.state.mouse_movement_entropy + 0.1).min(1.0);
                self.state.interaction_count += 0.03;
            }
            Action::MouseClick => {
                self.state.mouse_movement_entropy =
                    (self.state.mouse_movement_entropy + 0.15).min(1.0);
                self.state.interaction_count += 0.1;
            }
            Action::Interact => {
                self.state.mouse_movement_entropy =
                    (self.state.mouse_movement_entropy + 0.2).min(1.0);
                self.state.scroll_pattern_score = (self.state.scroll_pattern_score + 0.1).min(1.0);
                self.state.interaction_count += 0.15;
            }
            Action::Navigate => {
                self.state.requests_in_session += 0.1;
                self.state.time_since_last_request = self.rng.gen_range(1.0..5.0);
            }
        }

        // Update request variance
        self.state.request_variance =
            (self.state.request_variance * 0.95 + self.rng.gen_range(0.0..0.1)).min(1.0);

        // Normalize state
        self.state.normalize();
    }

    /// Add noise to state to simulate real-world variability
    fn add_state_noise(&mut self) {
        let noise_level = 0.02; // Small random variations

        self.state.mouse_movement_entropy += self.rng.gen_range(-noise_level..noise_level);
        self.state.scroll_pattern_score += self.rng.gen_range(-noise_level..noise_level);

        // Clamp to [0, 1]
        self.state.mouse_movement_entropy = self.state.mouse_movement_entropy.clamp(0.0, 1.0);
        self.state.scroll_pattern_score = self.state.scroll_pattern_score.clamp(0.0, 1.0);
    }

    /// Create step outcome from detection result
    fn create_outcome(&self, detection: &DetectionResult) -> StepOutcome {
        // Simulate CAPTCHA, rate limiting, access denied based on detection
        let captcha_detected = detection.detected && detection.confidence > 0.7;
        let rate_limited = detection.detected && detection.confidence > 0.8;
        let access_denied = detection.detected && detection.confidence > 0.9;

        // Calculate human-like score (inverse of suspicion)
        let human_like_score = 1.0 - detection.confidence;

        // Success if not detected
        let success = !detection.detected;

        StepOutcome {
            success,
            captcha_detected,
            rate_limited,
            access_denied,
            human_like_score,
            time_elapsed: self.state.time_since_last_request,
            done: detection.detected || self.step >= self.max_steps,
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &State {
        &self.state
    }

    /// Get current step count
    pub fn get_step(&self) -> usize {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bot_detector_creation() {
        let detector = BotDetector::new(0.5, false);
        assert_eq!(detector.detection_threshold, 0.5);
        assert!(!detector.strict_mode);
    }

    #[test]
    fn test_bot_detector_suspicious_behavior() {
        let detector = BotDetector::new(0.5, false);

        // Create a suspicious state (bot-like)
        let mut state = State::new();
        state.time_since_last_request = 0.1; // Very fast
        state.request_variance = 0.05; // Very regular
        state.mouse_movement_entropy = 0.1; // Linear
        state.scroll_pattern_score = 0.1; // Unnatural

        let result = detector.detect(&state, &[]);

        // Should be detected as bot
        assert!(result.detected);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_bot_detector_human_behavior() {
        let detector = BotDetector::new(0.5, false);

        // Create a human-like state
        let mut state = State::new();
        state.time_since_last_request = 3.0; // Normal pace
        state.request_variance = 0.5; // Variable timing
        state.mouse_movement_entropy = 0.8; // Random movement
        state.scroll_pattern_score = 0.9; // Natural scrolling
        state.interaction_count = 0.7; // Good interaction

        let result = detector.detect(&state, &[]);

        // Should NOT be detected as bot
        assert!(!result.detected);
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_environment_creation() {
        let env = TrainingEnvironment::new(0.5, false, 100);
        assert_eq!(env.step, 0);
        assert_eq!(env.max_steps, 100);
    }

    #[test]
    fn test_environment_reset() {
        let mut env = TrainingEnvironment::new(0.5, false, 100);

        // Take some steps
        let _ = env.step(Action::WaitShort);
        let _ = env.step(Action::ScrollSmall);
        assert!(env.step > 0);

        // Reset
        let state = env.reset();
        assert_eq!(env.step, 0);
        assert_eq!(env.action_history.len(), 0);
    }

    #[test]
    fn test_environment_step() {
        let mut env = TrainingEnvironment::new(0.5, false, 100);
        env.reset();

        let (next_state, reward, done, outcome) = env.step(Action::WaitShort).unwrap();

        // Verify step incremented
        assert_eq!(env.get_step(), 1);

        // Verify state changed
        assert!(next_state.time_since_last_request > 0.0);

        // Verify outcome created
        assert!(outcome.time_elapsed > 0.0);
    }

    #[test]
    fn test_environment_episode_completion() {
        let mut env = TrainingEnvironment::new(0.5, false, 10);
        env.reset();

        let mut steps = 0;
        let mut done = false;

        while !done && steps < 20 {
            let (_, _, is_done, _) = env.step(Action::WaitLong).unwrap();
            done = is_done;
            steps += 1;
        }

        // Should complete within max_steps
        assert!(done);
        assert!(steps <= 10);
    }

    #[test]
    fn test_state_updates_from_actions() {
        let mut env = TrainingEnvironment::new(0.5, false, 100);
        env.reset();

        let initial_mouse_entropy = env.state.mouse_movement_entropy;

        // Mouse movement should increase entropy
        env.step(Action::MouseMovement).unwrap();

        assert!(env.state.mouse_movement_entropy >= initial_mouse_entropy);
    }
}
