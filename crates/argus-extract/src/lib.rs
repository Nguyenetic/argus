//! # argus-extract
//!
//! Generic configuration-driven structured data extraction for Argus.
//!
//! This crate provides a flexible extraction system that can extract structured
//! data (tables, lists, paired content, sections) from any website using
//! YAML configuration files.
//!
//! ## Example
//!
//! ```rust,ignore
//! use argus_extract::ExtractorEngine;
//!
//! let config_yaml = r#"
//! version: "1.0"
//! name: "My Extractor"
//! extractors:
//!   - name: "tables"
//!     type: table
//!     selector: "table"
//!     columns:
//!       term: { index: 0 }
//!       definition: { index: 1 }
//! "#;
//!
//! let engine = ExtractorEngine::from_yaml(config_yaml)?;
//! let result = engine.extract("https://example.com", html_content)?;
//! ```

pub mod config;
pub mod engine;
pub mod extractors;
pub mod result;

pub use config::ExtractionConfig;
pub use engine::ExtractorEngine;
pub use result::{ExtractedData, ExtractionResult, Section};
