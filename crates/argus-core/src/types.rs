//! Core types for Argus

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a scraped web page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub id: Uuid,
    pub url: String,
    pub title: Option<String>,
    pub content: String,
    pub html: Option<String>,
    pub status: PageStatus,
    pub scraped_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Status of a page scraping operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PageStatus {
    Pending,
    InProgress,
    Success,
    Failed,
}

/// Configuration for scraping a page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeConfig {
    pub url: String,
    pub use_browser: bool,
    pub wait_for_js: bool,
    pub extract_links: bool,
    pub max_depth: u32,
    pub timeout_ms: u64,
}

impl Default for ScrapeConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            use_browser: false,
            wait_for_js: false,
            extract_links: true,
            max_depth: 1,
            timeout_ms: 30000,
        }
    }
}

/// Result of a scraping operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrapeResult {
    pub success: bool,
    pub page: Option<Page>,
    pub error: Option<String>,
    pub duration_ms: u64,
}
