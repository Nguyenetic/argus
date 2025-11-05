/// RL Agent Integration Module
///
/// This module provides the complete integration between the RL agent,
/// browser executor, and state tracking for real-world bot evasion.
/// It orchestrates the agent's decision-making with browser actions.
use crate::action::Action;
use crate::agent::RLAgent;
use crate::executor::{ActionExecutor, ExecutionResult};
use crate::state::State;
use anyhow::{Context, Result};
use chromiumoxide::page::Page;
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Integrated RL agent for browser automation with bot evasion
pub struct IntegratedAgent {
    rl_agent: RLAgent,
    executor: ActionExecutor,
    state_tracker: StateTracker,
    config: AgentConfig,
}

impl IntegratedAgent {
    /// Create new integrated agent
    pub fn new(
        model_path: &str,
        viewport_width: f32,
        viewport_height: f32,
        config: AgentConfig,
    ) -> Result<Self> {
        let rl_agent = RLAgent::load(model_path)?;
        let executor = ActionExecutor::new(viewport_width, viewport_height);
        let state_tracker = StateTracker::new(config.state_window_size);

        Ok(Self {
            rl_agent,
            executor,
            state_tracker,
            config,
        })
    }

    /// Execute one step of the agent
    /// Returns (action_taken, execution_result, current_state)
    pub async fn step(&mut self, page: &Page) -> Result<AgentStep> {
        // Update state from page
        let state = self.state_tracker.update(page).await?;

        // Select action from RL agent
        let action = self.rl_agent.select_action(&state)?;

        info!("Agent selected action: {:?}", action);

        // Execute action on browser
        let result = self.executor.execute(page, action).await?;

        // Update state tracker with execution result
        self.state_tracker.record_action(action, &result);

        // Check for bot detection
        let detection = self.check_detection(page).await?;

        Ok(AgentStep {
            state,
            action,
            result,
            detection,
        })
    }

    /// Run the agent for multiple steps
    pub async fn run(&mut self, page: &Page, num_steps: usize) -> Result<SessionResult> {
        let session_start = Instant::now();
        let mut steps = Vec::new();
        let mut detections = 0;

        info!("Starting agent session with {} steps", num_steps);

        for step_num in 0..num_steps {
            debug!("Step {}/{}", step_num + 1, num_steps);

            let step = self.step(page).await?;

            if step.detection.detected {
                detections += 1;
                warn!(
                    "Bot detection at step {}: {:?}",
                    step_num + 1,
                    step.detection.reason
                );

                if self.config.stop_on_detection {
                    info!("Stopping due to detection");
                    break;
                }
            }

            steps.push(step);

            // Check if we should stop
            if self.should_stop(&steps) {
                info!("Stopping early due to termination condition");
                break;
            }
        }

        let session_duration = session_start.elapsed();

        Ok(SessionResult {
            steps,
            detections,
            session_duration,
            success_rate: self.calculate_success_rate(&steps),
            behavior_score: self.calculate_behavior_score(),
        })
    }

    /// Check for bot detection signals
    async fn check_detection(&self, page: &Page) -> Result<DetectionInfo> {
        // Check for CAPTCHA
        let captcha_detected = self.check_captcha(page).await?;

        // Check for rate limiting (HTTP 429)
        let rate_limited = self.check_rate_limit(page).await?;

        // Check for access denied (HTTP 403)
        let access_denied = self.check_access_denied(page).await?;

        // Check for challenge pages
        let challenge_detected = self.check_challenge(page).await?;

        let detected = captcha_detected || rate_limited || access_denied || challenge_detected;

        let reason = if captcha_detected {
            Some("CAPTCHA detected".to_string())
        } else if rate_limited {
            Some("Rate limit (429)".to_string())
        } else if access_denied {
            Some("Access denied (403)".to_string())
        } else if challenge_detected {
            Some("Challenge page detected".to_string())
        } else {
            None
        };

        Ok(DetectionInfo { detected, reason })
    }

    /// Check for CAPTCHA on page
    async fn check_captcha(&self, page: &Page) -> Result<bool> {
        let js = r#"
            document.body.innerHTML.toLowerCase().includes('captcha') ||
            document.querySelector('[data-sitekey]') !== null ||
            document.querySelector('.g-recaptcha') !== null ||
            document.querySelector('.h-captcha') !== null ||
            document.querySelector('#cf-challenge-running') !== null
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Check for rate limiting
    async fn check_rate_limit(&self, page: &Page) -> Result<bool> {
        let js = r#"
            document.body.innerHTML.toLowerCase().includes('rate limit') ||
            document.body.innerHTML.toLowerCase().includes('too many requests')
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Check for access denied
    async fn check_access_denied(&self, page: &Page) -> Result<bool> {
        let js = r#"
            document.body.innerHTML.toLowerCase().includes('access denied') ||
            document.body.innerHTML.toLowerCase().includes('forbidden')
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Check for challenge pages (Cloudflare, etc.)
    async fn check_challenge(&self, page: &Page) -> Result<bool> {
        let js = r#"
            document.title.toLowerCase().includes('just a moment') ||
            document.title.toLowerCase().includes('checking your browser') ||
            document.querySelector('.cf-browser-verification') !== null
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Check if we should stop the session
    fn should_stop(&self, steps: &[AgentStep]) -> bool {
        if steps.is_empty() {
            return false;
        }

        // Stop if too many recent failures
        let recent_failures = steps
            .iter()
            .rev()
            .take(5)
            .filter(|s| !s.result.success)
            .count();

        if recent_failures >= 3 {
            return true;
        }

        // Stop if detected multiple times recently
        let recent_detections = steps
            .iter()
            .rev()
            .take(10)
            .filter(|s| s.detection.detected)
            .count();

        if recent_detections >= 2 {
            return true;
        }

        false
    }

    /// Calculate success rate
    fn calculate_success_rate(&self, steps: &[AgentStep]) -> f32 {
        if steps.is_empty() {
            return 0.0;
        }

        let successes = steps.iter().filter(|s| s.result.success).count();
        successes as f32 / steps.len() as f32
    }

    /// Calculate overall behavior score
    fn calculate_behavior_score(&self) -> f32 {
        let actions: Vec<Action> = self.state_tracker.recent_actions.iter().copied().collect();
        self.executor.calculate_behavior_score(&actions)
    }

    /// Get current state
    pub fn current_state(&self) -> &State {
        &self.state_tracker.current_state
    }

    /// Get session statistics
    pub fn statistics(&self) -> SessionStatistics {
        SessionStatistics {
            total_actions: self.state_tracker.action_count,
            requests_in_session: self.state_tracker.request_count,
            success_rate: self.state_tracker.calculate_success_rate(),
            avg_request_interval: self.state_tracker.calculate_avg_interval(),
        }
    }
}

/// State tracking for agent
struct StateTracker {
    current_state: State,
    recent_actions: VecDeque<Action>,
    action_times: VecDeque<Instant>,
    action_count: usize,
    request_count: usize,
    success_count: usize,
    window_size: usize,
}

impl StateTracker {
    fn new(window_size: usize) -> Self {
        Self {
            current_state: State::default(),
            recent_actions: VecDeque::new(),
            action_times: VecDeque::new(),
            action_count: 0,
            request_count: 0,
            success_count: 0,
            window_size,
        }
    }

    /// Update state from page
    async fn update(&mut self, page: &Page) -> Result<State> {
        // Get page metrics
        let page_load_time = self.measure_page_load_time(page).await?;
        let dynamic_content_ratio = self.measure_dynamic_content(page).await?;
        let page_complexity = self.measure_page_complexity(page).await?;
        let word_count = self.count_words(page).await?;

        // Calculate timing features
        let time_since_last = if let Some(last_time) = self.action_times.back() {
            last_time.elapsed().as_secs_f32()
        } else {
            0.0
        };

        let avg_interval = self.calculate_avg_interval();
        let variance = self.calculate_interval_variance();

        // Calculate behavioral features
        let mouse_entropy = 0.7; // Will be updated by executor
        let scroll_score = 0.7; // Will be updated by executor
        let interaction_count = self.action_count as f32;

        // Detection signals (will be updated by detection checks)
        let captcha = 0.0;
        let rate_limit = 0.0;
        let access_denied = 0.0;
        let challenge = 0.0;

        // Context
        let requests = self.request_count as f32;
        let success_rate = self.calculate_success_rate();

        self.current_state = State {
            time_since_last_request: time_since_last,
            avg_request_interval: avg_interval,
            request_variance: variance,
            mouse_movement_entropy: mouse_entropy,
            scroll_pattern_score: scroll_score,
            interaction_count,
            page_load_time,
            dynamic_content_ratio,
            page_complexity,
            captcha_detected: captcha,
            rate_limit_hit: rate_limit,
            access_denied,
            challenge_score: challenge,
            requests_in_session: requests,
            success_rate,
        };

        Ok(self.current_state.clone())
    }

    /// Record an action
    fn record_action(&mut self, action: Action, result: &ExecutionResult) {
        self.recent_actions.push_back(action);
        self.action_times.push_back(Instant::now());
        self.action_count += 1;

        if result.success {
            self.success_count += 1;
        }

        // Only count certain actions as "requests"
        if matches!(action, Action::Navigate | Action::Interact) {
            self.request_count += 1;
        }

        // Maintain window size
        while self.recent_actions.len() > self.window_size {
            self.recent_actions.pop_front();
            self.action_times.pop_front();
        }
    }

    /// Measure page load time
    async fn measure_page_load_time(&self, page: &Page) -> Result<f32> {
        let js = "performance.timing.loadEventEnd - performance.timing.navigationStart";
        let result = page.evaluate(js).await?;
        let ms: f64 = result.into_value().unwrap_or(0.0);
        Ok((ms / 1000.0) as f32) // Convert to seconds
    }

    /// Measure dynamic content ratio
    async fn measure_dynamic_content(&self, page: &Page) -> Result<f32> {
        let js = r#"
            (function() {
                const scripts = document.querySelectorAll('script').length;
                const total = document.querySelectorAll('*').length;
                return total > 0 ? scripts / total : 0;
            })()
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value().unwrap_or(0.0))
    }

    /// Measure page complexity
    async fn measure_page_complexity(&self, page: &Page) -> Result<f32> {
        let js = r#"
            (function() {
                const elements = document.querySelectorAll('*').length;
                const depth = Math.max(...Array.from(document.querySelectorAll('*')).map(el => {
                    let d = 0;
                    let e = el;
                    while (e.parentElement) { d++; e = e.parentElement; }
                    return d;
                }));
                return (elements * depth) / 10000;
            })()
        "#;

        let result = page.evaluate(js).await?;
        Ok(result.into_value().unwrap_or(0.5))
    }

    /// Count words on page
    async fn count_words(&self, page: &Page) -> Result<usize> {
        let js = "document.body.innerText.split(/\\s+/).length";
        let result = page.evaluate(js).await?;
        Ok(result.into_value().unwrap_or(0))
    }

    /// Calculate average interval
    fn calculate_avg_interval(&self) -> f32 {
        if self.action_times.len() < 2 {
            return 1.0;
        }

        let mut intervals = Vec::new();
        for i in 1..self.action_times.len() {
            let interval = self.action_times[i]
                .duration_since(self.action_times[i - 1])
                .as_secs_f32();
            intervals.push(interval);
        }

        intervals.iter().sum::<f32>() / intervals.len() as f32
    }

    /// Calculate interval variance
    fn calculate_interval_variance(&self) -> f32 {
        if self.action_times.len() < 2 {
            return 0.0;
        }

        let avg = self.calculate_avg_interval();
        let mut sum_sq_diff = 0.0;

        for i in 1..self.action_times.len() {
            let interval = self.action_times[i]
                .duration_since(self.action_times[i - 1])
                .as_secs_f32();
            sum_sq_diff += (interval - avg).powi(2);
        }

        (sum_sq_diff / (self.action_times.len() - 1) as f32).sqrt()
    }

    /// Calculate success rate
    fn calculate_success_rate(&self) -> f32 {
        if self.action_count == 0 {
            return 1.0;
        }
        self.success_count as f32 / self.action_count as f32
    }
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Number of recent actions to track for state
    pub state_window_size: usize,

    /// Stop session on first detection
    pub stop_on_detection: bool,

    /// Maximum session duration
    pub max_duration: Duration,

    /// Enable debug logging
    pub debug: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            state_window_size: 50,
            stop_on_detection: false,
            max_duration: Duration::from_secs(300), // 5 minutes
            debug: false,
        }
    }
}

/// Information about a single agent step
#[derive(Debug, Clone)]
pub struct AgentStep {
    pub state: State,
    pub action: Action,
    pub result: ExecutionResult,
    pub detection: DetectionInfo,
}

/// Bot detection information
#[derive(Debug, Clone)]
pub struct DetectionInfo {
    pub detected: bool,
    pub reason: Option<String>,
}

/// Result of an agent session
#[derive(Debug)]
pub struct SessionResult {
    pub steps: Vec<AgentStep>,
    pub detections: usize,
    pub session_duration: Duration,
    pub success_rate: f32,
    pub behavior_score: f32,
}

impl SessionResult {
    /// Check if session was successful (no detections, good success rate)
    pub fn is_successful(&self) -> bool {
        self.detections == 0 && self.success_rate > 0.7
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        format!(
            "Session: {} steps, {} detections, {:.1}% success, {:.2} behavior score, {:.1}s duration",
            self.steps.len(),
            self.detections,
            self.success_rate * 100.0,
            self.behavior_score,
            self.session_duration.as_secs_f32()
        )
    }
}

/// Session statistics
#[derive(Debug, Clone)]
pub struct SessionStatistics {
    pub total_actions: usize,
    pub requests_in_session: usize,
    pub success_rate: f32,
    pub avg_request_interval: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.state_window_size, 50);
        assert!(!config.stop_on_detection);
        assert_eq!(config.max_duration, Duration::from_secs(300));
    }

    #[test]
    fn test_state_tracker_creation() {
        let tracker = StateTracker::new(50);
        assert_eq!(tracker.action_count, 0);
        assert_eq!(tracker.success_count, 0);
        assert_eq!(tracker.request_count, 0);
    }

    #[test]
    fn test_state_tracker_record_action() {
        let mut tracker = StateTracker::new(50);
        let result = ExecutionResult {
            success: true,
            duration_ms: 100,
            action_type: "test".to_string(),
            details: "test".to_string(),
        };

        tracker.record_action(Action::WaitShort, &result);

        assert_eq!(tracker.action_count, 1);
        assert_eq!(tracker.success_count, 1);
        assert_eq!(tracker.recent_actions.len(), 1);
    }

    #[test]
    fn test_state_tracker_window_size() {
        let mut tracker = StateTracker::new(5);
        let result = ExecutionResult {
            success: true,
            duration_ms: 100,
            action_type: "test".to_string(),
            details: "test".to_string(),
        };

        // Add 10 actions
        for _ in 0..10 {
            tracker.record_action(Action::WaitShort, &result);
        }

        // Should only keep last 5
        assert_eq!(tracker.recent_actions.len(), 5);
        assert_eq!(tracker.action_count, 10); // But total count should be 10
    }

    #[test]
    fn test_session_result_is_successful() {
        let result = SessionResult {
            steps: vec![],
            detections: 0,
            session_duration: Duration::from_secs(60),
            success_rate: 0.8,
            behavior_score: 0.9,
        };

        assert!(result.is_successful());

        let failed_result = SessionResult {
            steps: vec![],
            detections: 1,
            session_duration: Duration::from_secs(60),
            success_rate: 0.8,
            behavior_score: 0.9,
        };

        assert!(!failed_result.is_successful());
    }
}
