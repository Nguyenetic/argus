//! Chrome/Chromium browser automation using chromiumoxide

use argus_core::{Error, Result};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::cdp::browser_protocol::page::CaptureScreenshotFormat;
use futures::StreamExt;
use rand::Rng;
use std::time::Duration;
use tracing::{debug, info};

/// Chrome browser instance with stealth capabilities
pub struct ChromeBrowser {
    browser: Browser,
}

impl ChromeBrowser {
    /// Create a new Chrome browser instance
    pub async fn new(headless: bool) -> Result<Self> {
        info!("Initializing Chrome browser (headless: {})", headless);

        let (browser, mut handler) =
            Browser::launch(BrowserConfig::builder().with_head().build().map_err(|e| {
                Error::BrowserError(format!("Failed to build browser config: {}", e))
            })?)
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to launch browser: {}", e)))?;

        // Spawn the browser handler
        tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                debug!("Browser event: {:?}", event);
            }
        });

        info!("Chrome browser launched successfully");

        Ok(Self { browser })
    }

    /// Create a new page with stealth mode enabled
    pub async fn new_page(&self) -> Result<chromiumoxide::Page> {
        let page = self
            .browser
            .new_page("about:blank")
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to create page: {}", e)))?;

        // Apply stealth techniques
        self.apply_stealth_mode(&page).await?;

        Ok(page)
    }

    /// Navigate to a URL and wait for content to load
    pub async fn navigate(&self, url: &str, _wait_for: Option<&str>) -> Result<String> {
        info!("Navigating to: {}", url);

        let page = self.new_page().await?;

        // Navigate with timeout
        page.goto(url)
            .await
            .map_err(|e| Error::BrowserError(format!("Navigation failed: {}", e)))?;

        // Wait for network to be idle
        page.wait_for_navigation()
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to wait for navigation: {}", e)))?;

        // Add random delay to mimic human behavior
        self.random_delay(500, 2000).await;

        // Get page content
        let html = page
            .content()
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to get content: {}", e)))?;

        Ok(html)
    }

    /// Take a screenshot of the page
    pub async fn screenshot(&self, url: &str, full_page: bool) -> Result<Vec<u8>> {
        info!("Taking screenshot of: {}", url);

        let page = self.new_page().await?;
        page.goto(url)
            .await
            .map_err(|e| Error::BrowserError(format!("Navigation failed: {}", e)))?;

        page.wait_for_navigation()
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to wait for navigation: {}", e)))?;

        let screenshot = if full_page {
            page.screenshot(
                chromiumoxide::page::ScreenshotParams::builder()
                    .format(CaptureScreenshotFormat::Png)
                    .full_page(true)
                    .build(),
            )
            .await
        } else {
            page.screenshot(
                chromiumoxide::page::ScreenshotParams::builder()
                    .format(CaptureScreenshotFormat::Png)
                    .build(),
            )
            .await
        }
        .map_err(|e| Error::BrowserError(format!("Screenshot failed: {}", e)))?;

        Ok(screenshot)
    }

    /// Apply stealth mode techniques to avoid detection
    async fn apply_stealth_mode(&self, page: &chromiumoxide::Page) -> Result<()> {
        debug!("Applying stealth mode");

        // Remove webdriver flag
        page.evaluate(
            r#"
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        "#,
        )
        .await
        .map_err(|e| Error::BrowserError(format!("Failed to remove webdriver flag: {}", e)))?;

        // Randomize user agent
        let user_agents = vec![
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ];

        let user_agent = user_agents[rand::thread_rng().gen_range(0..user_agents.len())];
        page.set_user_agent(user_agent)
            .await
            .map_err(|e| Error::BrowserError(format!("Failed to set user agent: {}", e)))?;

        // Override permissions
        page.evaluate(
            r#"
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        "#,
        )
        .await
        .map_err(|e| Error::BrowserError(format!("Failed to override permissions: {}", e)))?;

        // Spoof plugins
        page.evaluate(
            r#"
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        "#,
        )
        .await
        .map_err(|e| Error::BrowserError(format!("Failed to spoof plugins: {}", e)))?;

        // Spoof languages
        page.evaluate(
            r#"
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        "#,
        )
        .await
        .map_err(|e| Error::BrowserError(format!("Failed to spoof languages: {}", e)))?;

        debug!("Stealth mode applied successfully");
        Ok(())
    }

    /// Add random delay to mimic human behavior
    async fn random_delay(&self, min_ms: u64, max_ms: u64) {
        let delay = rand::thread_rng().gen_range(min_ms..=max_ms);
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
}

impl Drop for ChromeBrowser {
    fn drop(&mut self) {
        info!("Closing Chrome browser");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Ignore by default as it requires Chrome to be installed
    async fn test_chrome_browser_creation() {
        let result = ChromeBrowser::new(true).await;
        assert!(result.is_ok());
    }
}
