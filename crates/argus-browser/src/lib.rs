//! Argus Browser - Intelligent browser automation with anti-detection
//!
//! Features:
//! - Browser pool management with auto-scaling
//! - Stealth mode (hide automation flags, randomize fingerprints)
//! - Session persistence (cookies, local storage)
//! - RL-based behavioral evasion (optional)
//! - CAPTCHA detection and solving (optional)
//! - Parallel scraping support

use argus_core::Result;

pub mod chrome;
pub mod intelligent;
pub mod pool;
pub mod session;
pub mod stealth;

pub use chrome::ChromeBrowser;
pub use intelligent::{
    IntelligentBrowser, IntelligentBrowserBuilder, IntelligentBrowserConfig, PageSession,
};
pub use pool::{BrowserGuard, BrowserPool, PoolConfig, PoolStats};
pub use session::{
    SerializableCookie, SessionData, SessionExtractor, SessionLifecycle, SessionManager,
    SessionRestorer,
};
pub use stealth::{StealthConfig, StealthMode, UserAgentManager};

/// Trait for browser automation backends
#[async_trait::async_trait]
pub trait Browser {
    async fn navigate(&self, url: &str) -> Result<String>;
    async fn screenshot(&self, full_page: bool) -> Result<Vec<u8>>;
    async fn get_content(&self) -> Result<String>;
}
