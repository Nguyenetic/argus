/// Browser Action Executor
///
/// This module connects the RL agent to the actual browser by executing
/// actions using chromiumoxide. It translates abstract RL actions into
/// concrete browser interactions with human-like behavior.
use crate::action::Action;
use crate::behavior::{
    AttentionModel, MousePathGenerator, Point, ScrollGenerator, TimingGenerator,
};
use crate::state::State;
use anyhow::{Context, Result};
use chromiumoxide::cdp::browser_protocol::page::Viewport;
use chromiumoxide::page::Page;
use rand::Rng;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, info, warn};

/// Browser action executor with human behavior emulation
pub struct ActionExecutor {
    mouse_path_gen: MousePathGenerator,
    timing_gen: TimingGenerator,
    scroll_gen: ScrollGenerator,
    attention: AttentionModel,
    current_mouse_pos: Point,
    viewport_size: (f32, f32),
}

impl ActionExecutor {
    /// Create new action executor
    pub fn new(viewport_width: f32, viewport_height: f32) -> Self {
        Self {
            mouse_path_gen: MousePathGenerator::default(),
            timing_gen: TimingGenerator::default(),
            scroll_gen: ScrollGenerator::default(),
            attention: AttentionModel::default(),
            current_mouse_pos: Point::new(viewport_width / 2.0, viewport_height / 2.0),
            viewport_size: (viewport_width, viewport_height),
        }
    }

    /// Execute an action on the browser page
    pub async fn execute(&mut self, page: &Page, action: Action) -> Result<ExecutionResult> {
        info!("Executing action: {:?}", action);

        let result = match action {
            Action::WaitShort => self.execute_wait_short().await?,
            Action::WaitLong => self.execute_wait_long().await?,
            Action::ScrollSmall => self.execute_scroll_small(page).await?,
            Action::ScrollLarge => self.execute_scroll_large(page).await?,
            Action::MouseMovement => self.execute_mouse_movement(page).await?,
            Action::MouseClick => self.execute_mouse_click(page).await?,
            Action::Interact => self.execute_interact(page).await?,
            Action::Navigate => self.execute_navigate(page).await?,
        };

        Ok(result)
    }

    /// Execute short wait (0.5-2s)
    async fn execute_wait_short(&self) -> Result<ExecutionResult> {
        let duration = TimingGenerator::new(1.0, 0.5).generate_delay();
        debug!("Waiting {:.2}s (short)", duration);
        sleep(Duration::from_secs_f32(duration)).await;

        Ok(ExecutionResult {
            success: true,
            duration_ms: (duration * 1000.0) as u64,
            action_type: "wait_short".to_string(),
            details: format!("Waited {:.2}s", duration),
        })
    }

    /// Execute long wait (2-10s)
    async fn execute_wait_long(&self) -> Result<ExecutionResult> {
        let duration = TimingGenerator::new(5.0, 2.0).generate_delay();
        debug!("Waiting {:.2}s (long)", duration);
        sleep(Duration::from_secs_f32(duration)).await;

        Ok(ExecutionResult {
            success: true,
            duration_ms: (duration * 1000.0) as u64,
            action_type: "wait_long".to_string(),
            details: format!("Waited {:.2}s", duration),
        })
    }

    /// Execute small scroll (10-30% of viewport)
    async fn execute_scroll_small(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        // Calculate scroll distance (10-30% of viewport height)
        let scroll_percent = rng.gen_range(0.1..0.3);
        let scroll_distance = self.viewport_size.1 * scroll_percent;

        debug!(
            "Scrolling {:.0}px ({:.0}% of viewport)",
            scroll_distance,
            scroll_percent * 100.0
        );

        // Generate natural scroll pattern
        let num_steps = rng.gen_range(5..10);
        let deltas = self.scroll_gen.generate_scroll(scroll_distance, num_steps);

        // Execute scroll with natural timing
        for delta in deltas {
            // Scroll using JavaScript evaluation
            let js = format!("window.scrollBy(0, {});", delta);
            page.evaluate(js).await.context("Failed to scroll")?;

            // Small delay between scroll steps
            let step_delay = rng.gen_range(0.02..0.05);
            sleep(Duration::from_secs_f32(step_delay)).await;
        }

        // Pause after scrolling (reading time)
        let pause = self.scroll_gen.generate_scroll_pause();
        sleep(Duration::from_secs_f32(pause)).await;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "scroll_small".to_string(),
            details: format!("Scrolled {:.0}px in {} steps", scroll_distance, num_steps),
        })
    }

    /// Execute large scroll (30-70% of viewport)
    async fn execute_scroll_large(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        // Calculate scroll distance (30-70% of viewport height)
        let scroll_percent = rng.gen_range(0.3..0.7);
        let scroll_distance = self.viewport_size.1 * scroll_percent;

        debug!(
            "Scrolling {:.0}px ({:.0}% of viewport)",
            scroll_distance,
            scroll_percent * 100.0
        );

        // Generate natural scroll pattern with more steps
        let num_steps = rng.gen_range(10..20);
        let deltas = self.scroll_gen.generate_scroll(scroll_distance, num_steps);

        // Execute scroll with natural timing
        for delta in deltas {
            let js = format!("window.scrollBy(0, {});", delta);
            page.evaluate(js).await.context("Failed to scroll")?;

            let step_delay = rng.gen_range(0.02..0.05);
            sleep(Duration::from_secs_f32(step_delay)).await;
        }

        // Longer pause after large scroll (more reading time)
        let pause = self.scroll_gen.generate_scroll_pause() * 1.5;
        sleep(Duration::from_secs_f32(pause)).await;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "scroll_large".to_string(),
            details: format!("Scrolled {:.0}px in {} steps", scroll_distance, num_steps),
        })
    }

    /// Execute mouse movement with Perlin noise
    async fn execute_mouse_movement(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        let mut rng = rand::thread_rng();

        // Generate random target within viewport
        let target = Point::new(
            rng.gen_range(0.0..self.viewport_size.0),
            rng.gen_range(0.0..self.viewport_size.1),
        );

        debug!(
            "Moving mouse from ({:.0}, {:.0}) to ({:.0}, {:.0})",
            self.current_mouse_pos.x, self.current_mouse_pos.y, target.x, target.y
        );

        // Generate natural path
        let distance = self.current_mouse_pos.distance(&target);
        let num_points = ((distance / 50.0).ceil() as usize).max(10).min(50);
        let path = self
            .mouse_path_gen
            .generate_path(self.current_mouse_pos, target, num_points);

        // Execute path with realistic timing
        for point in &path {
            let js = format!(
                "document.dispatchEvent(new MouseEvent('mousemove', {{ clientX: {}, clientY: {} }}));",
                point.x, point.y
            );
            page.evaluate(js).await.context("Failed to move mouse")?;

            // Small delay between movements
            let delay = rng.gen_range(0.01..0.03);
            sleep(Duration::from_secs_f32(delay)).await;
        }

        self.current_mouse_pos = target;
        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "mouse_movement".to_string(),
            details: format!("Moved {:.0}px in {} steps", distance, num_points),
        })
    }

    /// Execute mouse click with realistic timing
    async fn execute_mouse_click(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();

        // Think time before clicking
        let think_time = self.attention.generate_think_time();
        sleep(Duration::from_secs_f32(think_time)).await;

        debug!(
            "Clicking at ({:.0}, {:.0})",
            self.current_mouse_pos.x, self.current_mouse_pos.y
        );

        // Mouse down
        let js_down = format!(
            "document.dispatchEvent(new MouseEvent('mousedown', {{ clientX: {}, clientY: {}, button: 0 }}));",
            self.current_mouse_pos.x, self.current_mouse_pos.y
        );
        page.evaluate(js_down)
            .await
            .context("Failed to mouse down")?;

        // Hold duration
        let hold_duration = self.timing_gen.generate_click_duration();
        sleep(Duration::from_secs_f32(hold_duration)).await;

        // Mouse up
        let js_up = format!(
            "document.dispatchEvent(new MouseEvent('mouseup', {{ clientX: {}, clientY: {}, button: 0 }}));",
            self.current_mouse_pos.x, self.current_mouse_pos.y
        );
        page.evaluate(js_up).await.context("Failed to mouse up")?;

        // Click event
        let js_click = format!(
            "document.dispatchEvent(new MouseEvent('click', {{ clientX: {}, clientY: {}, button: 0 }}));",
            self.current_mouse_pos.x, self.current_mouse_pos.y
        );
        page.evaluate(js_click).await.context("Failed to click")?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "mouse_click".to_string(),
            details: format!(
                "Clicked at ({:.0}, {:.0})",
                self.current_mouse_pos.x, self.current_mouse_pos.y
            ),
        })
    }

    /// Execute interact (hover + click on clickable element)
    async fn execute_interact(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();

        // Find clickable elements
        let js_find = r#"
            Array.from(document.querySelectorAll('a, button, input[type="button"], input[type="submit"]'))
                .filter(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.top >= 0 && rect.left >= 0 &&
                           rect.bottom <= window.innerHeight &&
                           rect.right <= window.innerWidth;
                })
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        tag: el.tagName
                    };
                });
        "#;

        let result = page
            .evaluate(js_find)
            .await
            .context("Failed to find clickable elements")?;
        let elements: Vec<serde_json::Value> = result.into_value()?;

        if elements.is_empty() {
            warn!("No clickable elements found, executing random click instead");
            return self.execute_mouse_click(page).await;
        }

        // Pick random element
        let mut rng = rand::thread_rng();
        let element = &elements[rng.gen_range(0..elements.len())];
        let target_x = element["x"].as_f64().unwrap_or(0.0) as f32;
        let target_y = element["y"].as_f64().unwrap_or(0.0) as f32;
        let tag = element["tag"].as_str().unwrap_or("unknown");

        debug!(
            "Interacting with {} at ({:.0}, {:.0})",
            tag, target_x, target_y
        );

        // Move mouse to element
        let target = Point::new(target_x, target_y);
        let distance = self.current_mouse_pos.distance(&target);
        let num_points = ((distance / 50.0).ceil() as usize).max(10).min(50);
        let path = self
            .mouse_path_gen
            .generate_path(self.current_mouse_pos, target, num_points);

        for point in &path {
            let js = format!(
                "document.dispatchEvent(new MouseEvent('mousemove', {{ clientX: {}, clientY: {} }}));",
                point.x, point.y
            );
            page.evaluate(js).await.context("Failed to move mouse")?;

            let delay = rng.gen_range(0.01..0.03);
            sleep(Duration::from_secs_f32(delay)).await;
        }

        self.current_mouse_pos = target;

        // Hover
        let js_hover = format!(
            "document.elementFromPoint({}, {}).dispatchEvent(new MouseEvent('mouseenter'));",
            target_x, target_y
        );
        page.evaluate(js_hover).await.ok(); // Ignore errors

        // Pause before clicking
        let think_time = self.attention.generate_think_time();
        sleep(Duration::from_secs_f32(think_time)).await;

        // Click
        let click_result = self.execute_mouse_click(page).await?;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "interact".to_string(),
            details: format!(
                "Interacted with {} at ({:.0}, {:.0})",
                tag, target_x, target_y
            ),
        })
    }

    /// Execute navigate (find and click link)
    async fn execute_navigate(&mut self, page: &Page) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();

        // Find links
        let js_find = r#"
            Array.from(document.querySelectorAll('a[href]'))
                .filter(el => {
                    const rect = el.getBoundingClientRect();
                    const href = el.getAttribute('href');
                    return rect.top >= 0 && rect.left >= 0 &&
                           rect.bottom <= window.innerHeight &&
                           rect.right <= window.innerWidth &&
                           href && !href.startsWith('#') && !href.startsWith('javascript:');
                })
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        href: el.getAttribute('href'),
                        text: el.textContent.trim().substring(0, 50)
                    };
                });
        "#;

        let result = page
            .evaluate(js_find)
            .await
            .context("Failed to find links")?;
        let links: Vec<serde_json::Value> = result.into_value()?;

        if links.is_empty() {
            warn!("No navigable links found");
            return Ok(ExecutionResult {
                success: false,
                duration_ms: start_time.elapsed().as_millis() as u64,
                action_type: "navigate".to_string(),
                details: "No links found".to_string(),
            });
        }

        // Pick random link
        let mut rng = rand::thread_rng();
        let link = &links[rng.gen_range(0..links.len())];
        let href = link["href"].as_str().unwrap_or("");
        let text = link["text"].as_str().unwrap_or("");

        debug!("Navigating to: {} ({})", text, href);

        // Execute interact on the link
        self.execute_interact(page).await?;

        // Wait for navigation
        sleep(Duration::from_secs(2)).await;

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ExecutionResult {
            success: true,
            duration_ms,
            action_type: "navigate".to_string(),
            details: format!("Navigated to: {}", text),
        })
    }

    /// Update viewport size
    pub fn update_viewport(&mut self, width: f32, height: f32) {
        self.viewport_size = (width, height);
    }

    /// Get current mouse position
    pub fn mouse_position(&self) -> Point {
        self.current_mouse_pos
    }

    /// Calculate behavior score (0-1, higher = more human-like)
    pub fn calculate_behavior_score(&self, actions: &[Action]) -> f32 {
        if actions.is_empty() {
            return 0.5;
        }

        let mut score = 1.0;

        // Check for repetitive patterns (bot-like)
        let mut consecutive_same = 0;
        for i in 1..actions.len() {
            if actions[i] == actions[i - 1] {
                consecutive_same += 1;
            } else {
                consecutive_same = 0;
            }

            if consecutive_same > 3 {
                score -= 0.1; // Penalty for repetition
            }
        }

        // Check for variety (human-like)
        let unique_actions: std::collections::HashSet<_> = actions.iter().collect();
        let variety = unique_actions.len() as f32 / Action::all().len() as f32;
        score += variety * 0.2;

        // Check for wait actions (humans pause)
        let wait_ratio = actions
            .iter()
            .filter(|a| matches!(a, Action::WaitShort | Action::WaitLong))
            .count() as f32
            / actions.len() as f32;
        if wait_ratio > 0.1 && wait_ratio < 0.5 {
            score += 0.1; // Good wait ratio
        }

        score.clamp(0.0, 1.0)
    }
}

/// Result of executing an action
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub duration_ms: u64,
    pub action_type: String,
    pub details: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = ActionExecutor::new(1920.0, 1080.0);
        assert_eq!(executor.viewport_size, (1920.0, 1080.0));
        assert_eq!(executor.mouse_position(), Point::new(960.0, 540.0));
    }

    #[test]
    fn test_behavior_score_empty() {
        let executor = ActionExecutor::new(1920.0, 1080.0);
        let score = executor.calculate_behavior_score(&[]);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn test_behavior_score_repetitive() {
        let executor = ActionExecutor::new(1920.0, 1080.0);
        let actions = vec![
            Action::WaitShort,
            Action::WaitShort,
            Action::WaitShort,
            Action::WaitShort,
            Action::WaitShort,
        ];
        let score = executor.calculate_behavior_score(&actions);
        assert!(score < 0.7, "Repetitive actions should have low score");
    }

    #[test]
    fn test_behavior_score_varied() {
        let executor = ActionExecutor::new(1920.0, 1080.0);
        let actions = vec![
            Action::WaitShort,
            Action::MouseMovement,
            Action::ScrollSmall,
            Action::WaitLong,
            Action::MouseClick,
            Action::Interact,
        ];
        let score = executor.calculate_behavior_score(&actions);
        assert!(
            score > 0.7,
            "Varied actions should have high score: {}",
            score
        );
    }

    #[test]
    fn test_viewport_update() {
        let mut executor = ActionExecutor::new(1920.0, 1080.0);
        executor.update_viewport(1280.0, 720.0);
        assert_eq!(executor.viewport_size, (1280.0, 720.0));
    }
}
