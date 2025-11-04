//! Argus Browser - Browser automation for intelligent scraping

use argus_core::{Error, Result};

pub mod chrome;

pub use chrome::ChromeBrowser;

/// Trait for browser automation backends
#[async_trait::async_trait]
pub trait Browser {
    async fn navigate(&self, url: &str) -> Result<String>;
    async fn screenshot(&self, full_page: bool) -> Result<Vec<u8>>;
    async fn get_content(&self) -> Result<String>;
}
