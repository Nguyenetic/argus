/// Stealth mode techniques for browser automation
///
/// Implements anti-detection techniques to evade bot detection systems:
/// - Disable automation flags (navigator.webdriver)
/// - Randomize user agents
/// - Spoof browser properties
/// - Mimic human-like behavior
use anyhow::{Context, Result};
use chromiumoxide::cdp::browser_protocol::emulation::{
    SetNavigatorOverridesParams, SetUserAgentOverrideParams,
};
use chromiumoxide::cdp::browser_protocol::page::AddScriptToEvaluateOnNewDocumentParams;
use chromiumoxide::Page;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stealth configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StealthConfig {
    /// Randomize user agent
    pub randomize_user_agent: bool,

    /// Disable navigator.webdriver flag
    pub hide_webdriver: bool,

    /// Randomize canvas fingerprint
    pub randomize_canvas: bool,

    /// Randomize WebGL fingerprint
    pub randomize_webgl: bool,

    /// Spoof navigator properties
    pub spoof_navigator: bool,

    /// Randomize screen dimensions
    pub randomize_screen: bool,

    /// Emulate real browser plugins
    pub emulate_plugins: bool,
}

impl Default for StealthConfig {
    fn default() -> Self {
        Self {
            randomize_user_agent: true,
            hide_webdriver: true,
            randomize_canvas: true,
            randomize_webgl: true,
            spoof_navigator: true,
            randomize_screen: true,
            emulate_plugins: true,
        }
    }
}

/// Stealth mode manager
pub struct StealthMode {
    config: StealthConfig,
    user_agents: Vec<String>,
    screen_resolutions: Vec<(u32, u32)>,
}

impl StealthMode {
    /// Create new stealth mode manager
    pub fn new(config: StealthConfig) -> Self {
        Self {
            config,
            user_agents: Self::default_user_agents(),
            screen_resolutions: Self::default_resolutions(),
        }
    }

    /// Apply stealth techniques to a page
    pub async fn apply(&self, page: &Page) -> Result<()> {
        // 1. Hide webdriver flag
        if self.config.hide_webdriver {
            self.hide_webdriver_flag(page).await?;
        }

        // 2. Randomize user agent
        if self.config.randomize_user_agent {
            self.set_random_user_agent(page).await?;
        }

        // 3. Spoof navigator properties
        if self.config.spoof_navigator {
            self.spoof_navigator_properties(page).await?;
        }

        // 4. Randomize canvas fingerprint
        if self.config.randomize_canvas {
            self.randomize_canvas_fingerprint(page).await?;
        }

        // 5. Randomize WebGL fingerprint
        if self.config.randomize_webgl {
            self.randomize_webgl_fingerprint(page).await?;
        }

        // 6. Randomize screen dimensions
        if self.config.randomize_screen {
            self.randomize_screen_dimensions(page).await?;
        }

        // 7. Emulate plugins
        if self.config.emulate_plugins {
            self.emulate_browser_plugins(page).await?;
        }

        tracing::info!("Stealth mode applied successfully");
        Ok(())
    }

    /// Hide navigator.webdriver flag
    async fn hide_webdriver_flag(&self, page: &Page) -> Result<()> {
        let script = r#"
            // Method 1: Delete property
            delete Object.getPrototypeOf(navigator).webdriver;

            // Method 2: Override getter
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
                configurable: true
            });

            // Method 3: Override toString
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
        "#;

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script.to_string(),
            ..Default::default()
        })
        .await?;

        tracing::debug!("Webdriver flag hidden");
        Ok(())
    }

    /// Set random user agent
    async fn set_random_user_agent(&self, page: &Page) -> Result<()> {
        let user_agent = self.random_user_agent();

        page.execute(SetUserAgentOverrideParams {
            user_agent: user_agent.clone(),
            accept_language: Some("en-US,en;q=0.9".to_string()),
            platform: Some(self.random_platform()),
            ..Default::default()
        })
        .await?;

        tracing::debug!("User agent set: {}", user_agent);
        Ok(())
    }

    /// Spoof navigator properties
    async fn spoof_navigator_properties(&self, page: &Page) -> Result<()> {
        let script = r#"
            // Spoof navigator properties to match real browser
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
                configurable: true
            });

            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {
                        0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "Portable Document Format",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chrome PDF Plugin"
                    },
                    {
                        0: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format"},
                        description: "Portable Document Format",
                        filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                        length: 1,
                        name: "Chrome PDF Viewer"
                    },
                    {
                        description: "Portable Document Format",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chromium PDF Plugin"
                    }
                ],
                configurable: true
            });

            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8,
                configurable: true
            });

            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8,
                configurable: true
            });
        "#;

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script.to_string(),
            ..Default::default()
        })
        .await?;

        tracing::debug!("Navigator properties spoofed");
        Ok(())
    }

    /// Randomize canvas fingerprint
    async fn randomize_canvas_fingerprint(&self, page: &Page) -> Result<()> {
        let noise = rand::thread_rng().gen::<f32>() * 0.001;

        let script = format!(
            r#"
            // Add slight noise to canvas fingerprint
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            const originalToBlob = HTMLCanvasElement.prototype.toBlob;
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;

            const noise = {};

            // Randomize toDataURL
            HTMLCanvasElement.prototype.toDataURL = function(type) {{
                const context = this.getContext('2d');
                if (context) {{
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {{
                        imageData.data[i] = Math.min(255, imageData.data[i] + noise);
                    }}
                    context.putImageData(imageData, 0, 0);
                }}
                return originalToDataURL.apply(this, arguments);
            }};

            // Randomize getImageData
            CanvasRenderingContext2D.prototype.getImageData = function() {{
                const imageData = originalGetImageData.apply(this, arguments);
                for (let i = 0; i < imageData.data.length; i += 4) {{
                    imageData.data[i] = Math.min(255, imageData.data[i] + noise);
                }}
                return imageData;
            }};
        "#,
            noise
        );

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script,
            ..Default::default()
        })
        .await?;

        tracing::debug!("Canvas fingerprint randomized");
        Ok(())
    }

    /// Randomize WebGL fingerprint
    async fn randomize_webgl_fingerprint(&self, page: &Page) -> Result<()> {
        let noise = rand::thread_rng().gen::<f32>() * 0.001;

        let script = format!(
            r#"
            const noise = {};

            // Randomize WebGL parameters
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                const result = getParameter.call(this, parameter);

                // Add noise to renderer and vendor
                if (parameter === 37445) {{ // UNMASKED_VENDOR_WEBGL
                    return 'Intel Inc.';
                }}
                if (parameter === 37446) {{ // UNMASKED_RENDERER_WEBGL
                    return 'Intel Iris OpenGL Engine';
                }}

                return result;
            }};

            // Randomize WebGL2 as well
            if (typeof WebGL2RenderingContext !== 'undefined') {{
                const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
                WebGL2RenderingContext.prototype.getParameter = function(parameter) {{
                    const result = getParameter2.call(this, parameter);

                    if (parameter === 37445) {{
                        return 'Intel Inc.';
                    }}
                    if (parameter === 37446) {{
                        return 'Intel Iris OpenGL Engine';
                    }}

                    return result;
                }};
            }}
        "#,
            noise
        );

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script,
            ..Default::default()
        })
        .await?;

        tracing::debug!("WebGL fingerprint randomized");
        Ok(())
    }

    /// Randomize screen dimensions
    async fn randomize_screen_dimensions(&self, page: &Page) -> Result<()> {
        let (width, height) = self.random_screen_resolution();

        let script = format!(
            r#"
            Object.defineProperty(screen, 'width', {{
                get: () => {},
                configurable: true
            }});

            Object.defineProperty(screen, 'height', {{
                get: () => {},
                configurable: true
            }});

            Object.defineProperty(screen, 'availWidth', {{
                get: () => {},
                configurable: true
            }});

            Object.defineProperty(screen, 'availHeight', {{
                get: () => {} - 40, // Account for taskbar
                configurable: true
            }});
        "#,
            width, height, width, height
        );

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script,
            ..Default::default()
        })
        .await?;

        tracing::debug!("Screen dimensions randomized: {}x{}", width, height);
        Ok(())
    }

    /// Emulate browser plugins
    async fn emulate_browser_plugins(&self, page: &Page) -> Result<()> {
        let script = r#"
            // Emulate PDF plugin
            Object.defineProperty(navigator, 'pdfViewerEnabled', {
                get: () => true,
                configurable: true
            });

            // Emulate permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = function(parameters) {
                const allowedPermissions = ['notifications', 'geolocation'];
                if (allowedPermissions.includes(parameters.name)) {
                    return Promise.resolve({ state: 'prompt' });
                }
                return originalQuery(parameters);
            };

            // Emulate battery API
            if (!navigator.getBattery) {
                navigator.getBattery = function() {
                    return Promise.resolve({
                        charging: true,
                        chargingTime: 0,
                        dischargingTime: Infinity,
                        level: 1.0
                    });
                };
            }
        "#;

        page.execute(AddScriptToEvaluateOnNewDocumentParams {
            source: script.to_string(),
            ..Default::default()
        })
        .await?;

        tracing::debug!("Browser plugins emulated");
        Ok(())
    }

    /// Get random user agent
    fn random_user_agent(&self) -> String {
        let mut rng = rand::thread_rng();
        self.user_agents[rng.gen_range(0..self.user_agents.len())].clone()
    }

    /// Get random platform
    fn random_platform(&self) -> String {
        let platforms = vec!["Win32", "MacIntel", "Linux x86_64"];
        let mut rng = rand::thread_rng();
        platforms[rng.gen_range(0..platforms.len())].to_string()
    }

    /// Get random screen resolution
    fn random_screen_resolution(&self) -> (u32, u32) {
        let mut rng = rand::thread_rng();
        self.screen_resolutions[rng.gen_range(0..self.screen_resolutions.len())]
    }

    /// Default user agents (recent Chrome/Firefox/Safari)
    fn default_user_agents() -> Vec<String> {
        vec![
            // Chrome on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string(),
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36".to_string(),

            // Chrome on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string(),
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36".to_string(),

            // Firefox on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0".to_string(),
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0".to_string(),

            // Firefox on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0".to_string(),

            // Safari on macOS
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15".to_string(),

            // Edge on Windows
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0".to_string(),
        ]
    }

    /// Default screen resolutions (common desktop sizes)
    fn default_resolutions() -> Vec<(u32, u32)> {
        vec![
            (1920, 1080), // Full HD
            (2560, 1440), // 2K
            (3840, 2160), // 4K
            (1680, 1050), // WSXGA+
            (1440, 900),  // WXGA+
            (1366, 768),  // HD
            (2560, 1600), // WQXGA
        ]
    }
}

/// User agent manager for rotation
pub struct UserAgentManager {
    agents: Vec<String>,
    current_index: usize,
}

impl UserAgentManager {
    /// Create new user agent manager
    pub fn new() -> Self {
        Self {
            agents: StealthMode::default_user_agents(),
            current_index: 0,
        }
    }

    /// Get next user agent (round-robin)
    pub fn next(&mut self) -> String {
        let agent = self.agents[self.current_index].clone();
        self.current_index = (self.current_index + 1) % self.agents.len();
        agent
    }

    /// Get random user agent
    pub fn random(&self) -> String {
        let mut rng = rand::thread_rng();
        self.agents[rng.gen_range(0..self.agents.len())].clone()
    }

    /// Add custom user agent
    pub fn add(&mut self, user_agent: String) {
        self.agents.push(user_agent);
    }
}

impl Default for UserAgentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_agent_manager() {
        let mut manager = UserAgentManager::new();

        let agent1 = manager.next();
        let agent2 = manager.next();

        assert!(!agent1.is_empty());
        assert!(!agent2.is_empty());
        assert_ne!(agent1, agent2);
    }

    #[test]
    fn test_random_user_agent() {
        let manager = UserAgentManager::new();
        let agent = manager.random();

        assert!(!agent.is_empty());
        assert!(agent.contains("Mozilla"));
    }

    #[test]
    fn test_stealth_config_default() {
        let config = StealthConfig::default();

        assert!(config.randomize_user_agent);
        assert!(config.hide_webdriver);
        assert!(config.randomize_canvas);
    }
}
