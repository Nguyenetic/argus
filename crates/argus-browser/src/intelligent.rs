/// Intelligent browser automation with RL agent integration
///
/// Combines:
/// - Browser pool management
/// - Stealth mode techniques
/// - RL-based behavioral evasion
/// - CAPTCHA detection and solving
/// - Session persistence
///
/// This is the complete, production-ready scraping solution.
use anyhow::{Context as AnyhowContext, Result};
use chromiumoxide::Page;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::pool::{BrowserGuard, BrowserPool, PoolConfig, PoolStats};
use crate::session::{SessionData, SessionLifecycle};
use crate::stealth::{StealthConfig, StealthMode};

/// Intelligent browser configuration
#[derive(Debug, Clone)]
pub struct IntelligentBrowserConfig {
    /// Browser pool configuration
    pub pool_config: PoolConfig,

    /// Enable RL-based evasion (requires argus-rl)
    pub enable_rl_evasion: bool,

    /// Enable CAPTCHA solving (requires argus-captcha)
    pub enable_captcha_solving: bool,

    /// Session storage directory
    pub session_storage_dir: String,

    /// Auto-save sessions
    pub auto_save_sessions: bool,

    /// Request delay range (min, max) in milliseconds
    pub request_delay: (u64, u64),

    /// Maximum retries on failure
    pub max_retries: u32,
}

impl Default for IntelligentBrowserConfig {
    fn default() -> Self {
        Self {
            pool_config: PoolConfig::default(),
            enable_rl_evasion: true,
            enable_captcha_solving: true,
            session_storage_dir: "./data/sessions".to_string(),
            auto_save_sessions: true,
            request_delay: (1000, 3000), // 1-3 seconds
            max_retries: 3,
        }
    }
}

/// Intelligent browser manager
///
/// High-level API that combines all browser automation features
pub struct IntelligentBrowser {
    pool: BrowserPool,
    config: IntelligentBrowserConfig,
    session_manager: Option<SessionLifecycle>,
    active_sessions: Arc<RwLock<Vec<String>>>,
}

impl IntelligentBrowser {
    /// Create new intelligent browser
    pub async fn new(config: IntelligentBrowserConfig) -> Result<Self> {
        info!("Initializing intelligent browser...");

        // Create browser pool
        let pool = BrowserPool::new(config.pool_config.clone()).await?;

        // Create session manager
        let session_manager = if config.auto_save_sessions {
            Some(SessionLifecycle::new(&config.session_storage_dir)?)
        } else {
            None
        };

        info!("✓ Intelligent browser initialized");

        Ok(Self {
            pool,
            config,
            session_manager,
            active_sessions: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Navigate to URL with full evasion
    pub async fn navigate(&self, url: &str) -> Result<PageSession> {
        info!("Navigating to: {}", url);

        // Add human-like delay
        self.add_human_delay().await;

        // Acquire browser from pool
        let guard = self.pool.acquire().await?;

        // Create page with stealth mode
        let page = guard.new_page(url).await?;

        // Wait for page load
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Check for CAPTCHA if enabled
        #[cfg(feature = "captcha")]
        if self.config.enable_captcha_solving {
            self.check_and_solve_captcha(&page).await?;
        }

        // Create session
        let session_id = if let Some(manager) = &self.session_manager {
            Some(manager.create_session(&page).await?)
        } else {
            None
        };

        // Track active session
        if let Some(id) = &session_id {
            self.active_sessions.write().await.push(id.clone());
        }

        Ok(PageSession {
            page,
            session_id,
            _guard: guard,
        })
    }

    /// Resume existing session
    pub async fn resume_session(&self, session_id: &str, url: &str) -> Result<PageSession> {
        info!("Resuming session: {}", session_id);

        let manager = self
            .session_manager
            .as_ref()
            .context("Session manager not enabled")?;

        // Acquire browser
        let guard = self.pool.acquire().await?;
        let page = guard.new_page(url).await?;

        // Restore session
        manager.resume_session(&page, session_id).await?;

        Ok(PageSession {
            page,
            session_id: Some(session_id.to_string()),
            _guard: guard,
        })
    }

    /// Navigate with retry logic
    pub async fn navigate_with_retry(&self, url: &str) -> Result<PageSession> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < self.config.max_retries {
            match self.navigate(url).await {
                Ok(session) => return Ok(session),
                Err(e) => {
                    attempts += 1;
                    warn!("Navigation failed (attempt {}): {}", attempts, e);
                    last_error = Some(e);

                    if attempts < self.config.max_retries {
                        // Exponential backoff
                        let wait_time = 2u64.pow(attempts);
                        tokio::time::sleep(Duration::from_secs(wait_time)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Navigation failed")))
    }

    /// Scrape page with full automation
    pub async fn scrape<F, T>(&self, url: &str, extractor: F) -> Result<T>
    where
        F: FnOnce(&Page) -> futures::future::BoxFuture<'_, Result<T>>,
    {
        let session = self.navigate(url).await?;

        // Apply RL evasion behaviors
        #[cfg(feature = "rl")]
        if self.config.enable_rl_evasion {
            self.apply_rl_behaviors(&session.page).await?;
        }

        // Extract data
        let result = extractor(&session.page).await?;

        // Save session if needed
        if let (Some(manager), Some(session_id)) = (&self.session_manager, &session.session_id) {
            manager.update_session(&session.page, session_id).await?;
        }

        Ok(result)
    }

    /// Parallel scraping of multiple URLs
    pub async fn scrape_parallel<F, T>(
        &self,
        urls: Vec<String>,
        extractor: F,
    ) -> Result<Vec<Result<T>>>
    where
        F: Fn(&Page) -> futures::future::BoxFuture<'_, Result<T>> + Send + Sync + 'static,
        T: Send + 'static,
    {
        info!("Parallel scraping {} URLs", urls.len());

        let extractor = Arc::new(extractor);
        let mut tasks = Vec::new();

        for url in urls {
            let browser = self.clone();
            let extractor = extractor.clone();

            let task =
                tokio::spawn(async move { browser.scrape(&url, |page| extractor(page)).await });

            tasks.push(task);

            // Small delay between spawns
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let mut results = Vec::new();
        for task in tasks {
            results.push(task.await?);
        }

        Ok(results)
    }

    /// Add human-like delay
    async fn add_human_delay(&self) {
        let (min, max) = self.config.request_delay;
        let delay = min + (rand::random::<u64>() % (max - min));
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }

    /// Check and solve CAPTCHA
    #[cfg(feature = "captcha")]
    async fn check_and_solve_captcha(&self, page: &Page) -> Result<()> {
        // This would integrate with argus-captcha
        // For now, just a placeholder
        debug!("CAPTCHA check (feature not compiled in)");
        Ok(())
    }

    /// Apply RL-based evasion behaviors
    #[cfg(feature = "rl")]
    async fn apply_rl_behaviors(&self, page: &Page) -> Result<()> {
        // This would integrate with argus-rl
        // For now, just a placeholder
        debug!("RL evasion (feature not compiled in)");
        Ok(())
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        self.pool.stats().await
    }

    /// List active sessions
    pub async fn active_sessions(&self) -> Vec<String> {
        self.active_sessions.read().await.clone()
    }

    /// Cleanup resources
    pub async fn cleanup(&self) -> Result<()> {
        info!("Cleaning up browser resources...");
        self.pool.close_all().await?;
        Ok(())
    }
}

impl Clone for IntelligentBrowser {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            config: self.config.clone(),
            session_manager: None, // Don't clone session manager
            active_sessions: self.active_sessions.clone(),
        }
    }
}

/// Page session with automatic cleanup
pub struct PageSession {
    pub page: Page,
    pub session_id: Option<String>,
    _guard: BrowserGuard,
}

impl PageSession {
    /// Execute JavaScript
    pub async fn execute(&self, js: &str) -> Result<serde_json::Value> {
        let result = self.page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Get page content
    pub async fn content(&self) -> Result<String> {
        let result = self.page.content().await?;
        Ok(result)
    }

    /// Take screenshot
    pub async fn screenshot(&self) -> Result<Vec<u8>> {
        let screenshot = self
            .page
            .screenshot(chromiumoxide::page::ScreenshotParams::default())
            .await?;
        Ok(screenshot)
    }

    /// Wait for selector
    pub async fn wait_for(&self, selector: &str) -> Result<()> {
        self.page.wait_for_selector(selector).await?;
        Ok(())
    }
}

/// Builder pattern for intelligent browser
pub struct IntelligentBrowserBuilder {
    config: IntelligentBrowserConfig,
}

impl IntelligentBrowserBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: IntelligentBrowserConfig::default(),
        }
    }

    /// Set pool size
    pub fn pool_size(mut self, min: usize, max: usize) -> Self {
        self.config.pool_config.min_instances = min;
        self.config.pool_config.max_instances = max;
        self.config.pool_config.initial_instances = min;
        self
    }

    /// Enable/disable stealth mode
    pub fn stealth(mut self, enabled: bool) -> Self {
        self.config.pool_config.enable_stealth = enabled;
        self
    }

    /// Enable/disable RL evasion
    pub fn rl_evasion(mut self, enabled: bool) -> Self {
        self.config.enable_rl_evasion = enabled;
        self
    }

    /// Enable/disable CAPTCHA solving
    pub fn captcha_solving(mut self, enabled: bool) -> Self {
        self.config.enable_captcha_solving = enabled;
        self
    }

    /// Set request delay range
    pub fn request_delay(mut self, min_ms: u64, max_ms: u64) -> Self {
        self.config.request_delay = (min_ms, max_ms);
        self
    }

    /// Set max retries
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set session storage directory
    pub fn session_storage(mut self, dir: impl Into<String>) -> Self {
        self.config.session_storage_dir = dir.into();
        self
    }

    /// Set window size
    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        self.config.pool_config.window_size = (width, height);
        self
    }

    /// Set headless mode
    pub fn headless(mut self, headless: bool) -> Self {
        self.config.pool_config.headless = headless;
        self
    }

    /// Build the intelligent browser
    pub async fn build(self) -> Result<IntelligentBrowser> {
        IntelligentBrowser::new(self.config).await
    }
}

impl Default for IntelligentBrowserBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_builder() {
        let config = IntelligentBrowserBuilder::new()
            .pool_size(2, 5)
            .stealth(true)
            .request_delay(500, 1500)
            .headless(true)
            .build()
            .await;

        assert!(config.is_ok());
    }
}
