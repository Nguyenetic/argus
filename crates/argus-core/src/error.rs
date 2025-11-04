//! Error types for Argus

use thiserror::Error;

/// Main error type for Argus operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Parsing error: {0}")]
    ParseError(String),

    #[error("Scraping failed: {0}")]
    ScrapingError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Browser automation error: {0}")]
    BrowserError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
