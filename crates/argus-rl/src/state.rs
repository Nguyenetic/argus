//! State space representation for RL agent

use serde::{Deserialize, Serialize};

/// State representation for the RL agent
///
/// Captures features that might indicate bot detection:
/// - Timing patterns (request frequency, delays)
/// - Behavioral signals (mouse movement, scrolling patterns)
/// - Page characteristics (complexity, dynamic content)
/// - Detection signals (CAPTCHAs, rate limits, blocks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    // Timing features (normalized 0-1)
    pub time_since_last_request: f32, // Seconds since last request
    pub avg_request_interval: f32,    // Average time between requests
    pub request_variance: f32,        // Variance in request timing

    // Behavioral features
    pub mouse_movement_entropy: f32, // Randomness of mouse movement (0=linear, 1=random)
    pub scroll_pattern_score: f32,   // Natural scrolling score (0=bot-like, 1=human-like)
    pub interaction_count: f32,      // Number of interactions per page

    // Page characteristics
    pub page_load_time: f32,        // Time to load page (seconds)
    pub dynamic_content_ratio: f32, // Ratio of dynamic vs static content
    pub page_complexity: f32,       // DOM complexity score

    // Detection signals
    pub captcha_detected: f32, // 1.0 if CAPTCHA present, 0.0 otherwise
    pub rate_limit_hit: f32,   // 1.0 if rate limited, 0.0 otherwise
    pub access_denied: f32,    // 1.0 if blocked, 0.0 otherwise
    pub challenge_score: f32,  // Overall challenge score (0-1)

    // Environment context
    pub requests_in_session: f32, // Number of requests in current session
    pub success_rate: f32,        // Recent success rate (0-1)
}

impl State {
    /// Create a new initial state
    pub fn new() -> Self {
        Self {
            time_since_last_request: 0.0,
            avg_request_interval: 5.0,
            request_variance: 0.5,
            mouse_movement_entropy: 0.7,
            scroll_pattern_score: 0.7,
            interaction_count: 0.0,
            page_load_time: 0.0,
            dynamic_content_ratio: 0.5,
            page_complexity: 0.5,
            captcha_detected: 0.0,
            rate_limit_hit: 0.0,
            access_denied: 0.0,
            challenge_score: 0.0,
            requests_in_session: 0.0,
            success_rate: 1.0,
        }
    }

    /// Convert state to feature vector for neural network
    pub fn to_tensor(&self) -> Vec<f32> {
        vec![
            self.time_since_last_request,
            self.avg_request_interval,
            self.request_variance,
            self.mouse_movement_entropy,
            self.scroll_pattern_score,
            self.interaction_count,
            self.page_load_time,
            self.dynamic_content_ratio,
            self.page_complexity,
            self.captcha_detected,
            self.rate_limit_hit,
            self.access_denied,
            self.challenge_score,
            self.requests_in_session,
            self.success_rate,
        ]
    }

    /// Get the dimensionality of the state space
    pub const fn dim() -> usize {
        15 // Number of features
    }

    /// Normalize state values to [0, 1] range
    pub fn normalize(&mut self) {
        // Time features already in reasonable range
        self.avg_request_interval = (self.avg_request_interval / 60.0).min(1.0); // Cap at 60 seconds
        self.page_load_time = (self.page_load_time / 10.0).min(1.0); // Cap at 10 seconds
        self.requests_in_session = (self.requests_in_session / 100.0).min(1.0); // Cap at 100
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dimensions() {
        let state = State::new();
        let tensor = state.to_tensor();
        assert_eq!(tensor.len(), State::dim());
    }

    #[test]
    fn test_state_normalization() {
        let mut state = State::new();
        state.avg_request_interval = 120.0;
        state.page_load_time = 20.0;
        state.requests_in_session = 500.0;

        state.normalize();

        assert!(state.avg_request_interval <= 1.0);
        assert!(state.page_load_time <= 1.0);
        assert!(state.requests_in_session <= 1.0);
    }
}
