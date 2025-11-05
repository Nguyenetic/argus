//! Argus Browser - Browser automation for intelligent scraping

use argus_core::Result;

pub mod chrome;
pub mod pool;

pub use chrome::ChromeBrowser;
pub use pool::{BrowserGuard, BrowserPool};

/// Trait for browser automation backends
#[async_trait::async_trait]
pub trait Browser {
    async fn navigate(&self, url: &str) -> Result<String>;
    async fn screenshot(&self, full_page: bool) -> Result<Vec<u8>>;
    async fn get_content(&self) -> Result<String>;
}
