//! Reward function for RL agent

use crate::state::State;
use serde::{Deserialize, Serialize};

/// Reward calculator for RL agent
#[derive(Debug, Clone)]
pub struct RewardCalculator {
    /// Base reward for successful scrape
    pub success_reward: f32,

    /// Bonus for no detection signals
    pub no_detection_bonus: f32,

    /// Bonus for human-like behavior
    pub human_like_bonus: f32,

    /// Penalty for CAPTCHA
    pub captcha_penalty: f32,

    /// Penalty for rate limiting
    pub rate_limit_penalty: f32,

    /// Penalty for access denied
    pub access_denied_penalty: f32,

    /// Time penalty per second (encourage efficiency)
    pub time_penalty: f32,
}

impl Default for RewardCalculator {
    fn default() -> Self {
        Self {
            success_reward: 10.0,
            no_detection_bonus: 5.0,
            human_like_bonus: 2.0,
            captcha_penalty: -5.0,
            rate_limit_penalty: -10.0,
            access_denied_penalty: -20.0,
            time_penalty: -1.0,
        }
    }
}

/// Outcome of a scraping episode step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOutcome {
    /// Whether the scrape was successful
    pub success: bool,

    /// Whether CAPTCHA was encountered
    pub captcha_detected: bool,

    /// Whether rate limited
    pub rate_limited: bool,

    /// Whether access was denied
    pub access_denied: bool,

    /// Human-like behavior score (0-1)
    pub human_like_score: f32,

    /// Time elapsed in seconds
    pub time_elapsed: f32,

    /// Whether episode is done
    pub done: bool,
}

impl RewardCalculator {
    /// Calculate reward based on outcome and state
    pub fn calculate(&self, outcome: &StepOutcome, _state: &State) -> f32 {
        let mut reward = 0.0;

        // Positive rewards
        if outcome.success {
            reward += self.success_reward;
        }

        // No detection signals bonus
        if !outcome.captcha_detected && !outcome.rate_limited && !outcome.access_denied {
            reward += self.no_detection_bonus;
        }

        // Human-like behavior bonus
        if outcome.human_like_score > 0.8 {
            reward += self.human_like_bonus * outcome.human_like_score;
        }

        // Penalties
        if outcome.captcha_detected {
            reward += self.captcha_penalty;
        }

        if outcome.rate_limited {
            reward += self.rate_limit_penalty;
        }

        if outcome.access_denied {
            reward += self.access_denied_penalty;
        }

        // Time penalty (encourage efficiency)
        reward += self.time_penalty * outcome.time_elapsed;

        reward
    }

    /// Calculate cumulative reward for an episode
    pub fn calculate_episode_reward(&self, outcomes: &[StepOutcome]) -> f32 {
        // For now, use simple sum
        // Could add discount factor (gamma) later
        outcomes
            .iter()
            .enumerate()
            .map(|(i, outcome)| {
                let state = State::new(); // Would use actual state here
                self.calculate(outcome, &state)
            })
            .sum()
    }

    /// Calculate discounted return with gamma
    pub fn calculate_discounted_return(&self, outcomes: &[StepOutcome], gamma: f32) -> f32 {
        outcomes
            .iter()
            .enumerate()
            .map(|(i, outcome)| {
                let state = State::new();
                let reward = self.calculate(outcome, &state);
                reward * gamma.powi(i as i32)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successful_scrape() {
        let calc = RewardCalculator::default();
        let outcome = StepOutcome {
            success: true,
            captcha_detected: false,
            rate_limited: false,
            access_denied: false,
            human_like_score: 0.9,
            time_elapsed: 2.0,
            done: false,
        };

        let reward = calc.calculate(&outcome, &State::new());

        // success (10) + no_detection (5) + human_like (2 * 0.9) + time (-1 * 2)
        // = 10 + 5 + 1.8 - 2 = 14.8
        assert!((reward - 14.8).abs() < 0.01);
    }

    #[test]
    fn test_captcha_penalty() {
        let calc = RewardCalculator::default();
        let outcome = StepOutcome {
            success: false,
            captcha_detected: true,
            rate_limited: false,
            access_denied: false,
            human_like_score: 0.5,
            time_elapsed: 1.0,
            done: true,
        };

        let reward = calc.calculate(&outcome, &State::new());

        // captcha (-5) + time (-1 * 1) = -6
        assert!((reward + 6.0).abs() < 0.01);
    }

    #[test]
    fn test_access_denied() {
        let calc = RewardCalculator::default();
        let outcome = StepOutcome {
            success: false,
            captcha_detected: false,
            rate_limited: false,
            access_denied: true,
            human_like_score: 0.3,
            time_elapsed: 0.5,
            done: true,
        };

        let reward = calc.calculate(&outcome, &State::new());

        // access_denied (-20) + time (-1 * 0.5) = -20.5
        assert!((reward + 20.5).abs() < 0.01);
    }

    #[test]
    fn test_discount_factor() {
        let calc = RewardCalculator::default();
        let outcomes = vec![
            StepOutcome {
                success: true,
                captcha_detected: false,
                rate_limited: false,
                access_denied: false,
                human_like_score: 0.9,
                time_elapsed: 1.0,
                done: false,
            },
            StepOutcome {
                success: true,
                captcha_detected: false,
                rate_limited: false,
                access_denied: false,
                human_like_score: 0.9,
                time_elapsed: 1.0,
                done: true,
            },
        ];

        let gamma = 0.99;
        let discounted = calc.calculate_discounted_return(&outcomes, gamma);
        let undiscounted = calc.calculate_episode_reward(&outcomes);

        // Discounted should be slightly less than undiscounted
        assert!(discounted < undiscounted);
        assert!(discounted > 0.0);
    }
}
