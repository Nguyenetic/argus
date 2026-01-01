//! Main extraction engine that orchestrates the extraction process.

use crate::config::{ExtractionConfig, Extractor, ExtractorType};
use crate::extractors::{
    extract_keyvalue, extract_list, extract_paired, extract_repeating, extract_sections,
    extract_table,
};
use crate::result::{ExtractedData, ExtractionMetadata, ExtractionResult};
use anyhow::{Context, Result};
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Main extraction engine that runs configured extractors against HTML content.
pub struct ExtractorEngine {
    config: ExtractionConfig,
}

impl ExtractorEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: ExtractionConfig) -> Self {
        Self { config }
    }

    /// Load configuration from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let config: ExtractionConfig =
            serde_yaml::from_str(yaml).context("Failed to parse extraction config YAML")?;
        Ok(Self::new(config))
    }

    /// Load configuration from a YAML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context("Failed to read extraction config file")?;
        Self::from_yaml(&content)
    }

    /// Get the configuration name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get access to the underlying configuration.
    pub fn config(&self) -> &ExtractionConfig {
        &self.config
    }

    /// Check if this configuration applies to the given URL.
    pub fn matches_url(&self, url: &str) -> bool {
        if self.config.url_patterns.is_empty() {
            return true; // No patterns = match all
        }

        for pattern in &self.config.url_patterns {
            if pattern.starts_with("regex:") {
                if let Ok(re) = Regex::new(&pattern[6..]) {
                    if re.is_match(url) {
                        return true;
                    }
                }
            } else {
                // Glob-style pattern
                if glob_match(pattern, url) {
                    return true;
                }
            }
        }
        false
    }

    /// Extract structured data from HTML content.
    pub fn extract(&self, url: &str, html: &str) -> Result<ExtractionResult> {
        let start = Instant::now();
        let doc = Html::parse_document(html);

        let mut data = HashMap::new();
        let mut extractors_matched = Vec::new();
        let mut extractors_failed = Vec::new();

        let trim = self.config.settings.trim_whitespace;

        // Run each extractor
        for extractor in &self.config.extractors {
            match self.run_extractor(&doc, extractor, trim) {
                Ok(Some(extracted)) => {
                    // Apply max_items limit if set
                    let limited = if self.config.settings.max_items > 0 {
                        limit_items(extracted, self.config.settings.max_items)
                    } else {
                        extracted
                    };

                    // Filter empty if enabled
                    let filtered = if self.config.settings.filter_empty {
                        filter_empty(limited)
                    } else {
                        Some(limited)
                    };

                    if let Some(d) = filtered {
                        data.insert(extractor.name.clone(), d);
                        extractors_matched.push(extractor.name.clone());
                    } else {
                        extractors_failed.push(extractor.name.clone());
                    }
                }
                Ok(None) => {
                    tracing::debug!("Extractor '{}' found no matches", extractor.name);
                    extractors_failed.push(extractor.name.clone());
                }
                Err(e) => {
                    tracing::warn!("Extractor '{}' failed: {}", extractor.name, e);
                    extractors_failed.push(extractor.name.clone());
                }
            }
        }

        // Validate results
        let validation_errors = self.validate(&data);

        // Fallback to raw content if needed
        let raw_content = if self.config.settings.fallback_to_raw
            && data.is_empty()
            && validation_errors.is_empty()
        {
            Some(extract_raw_text(&doc))
        } else {
            None
        };

        Ok(ExtractionResult {
            config_name: self.config.name.clone(),
            url: url.to_string(),
            data,
            raw_content,
            metadata: ExtractionMetadata {
                extracted_at: chrono::Utc::now(),
                duration_ms: start.elapsed().as_millis() as u64,
                extractors_matched,
                extractors_failed,
            },
            validation_errors,
        })
    }

    /// Run a single extractor against the document.
    fn run_extractor(
        &self,
        doc: &Html,
        extractor: &Extractor,
        trim: bool,
    ) -> Result<Option<ExtractedData>> {
        match &extractor.config {
            ExtractorType::Table(cfg) => extract_table(doc, cfg, trim),
            ExtractorType::List(cfg) => extract_list(doc, cfg, trim),
            ExtractorType::Paired(cfg) => extract_paired(doc, cfg, trim),
            ExtractorType::KeyValue(cfg) => extract_keyvalue(doc, cfg, trim),
            ExtractorType::Repeating(cfg) => extract_repeating(doc, cfg, trim),
            ExtractorType::Sections(cfg) => extract_sections(doc, cfg, trim),
        }
    }

    /// Validate extracted data against configured rules.
    fn validate(&self, data: &HashMap<String, ExtractedData>) -> Vec<String> {
        let mut errors = Vec::new();

        // Check required fields
        for required in &self.config.validation.required_fields {
            if !data.contains_key(required) {
                errors.push(format!("Required field '{}' not found", required));
            }
        }

        // Check minimum item counts
        for (field, min) in &self.config.validation.min_items {
            if let Some(extracted) = data.get(field) {
                let count = match extracted {
                    ExtractedData::Array(arr) => arr.len(),
                    ExtractedData::Sections(secs) => secs.len(),
                    ExtractedData::Single(m) => m.len(),
                };
                if count < *min {
                    errors.push(format!(
                        "Field '{}' has {} items, minimum is {}",
                        field, count, min
                    ));
                }
            }
        }

        errors
    }
}

/// Simple glob pattern matching (supports * and ?).
fn glob_match(pattern: &str, text: &str) -> bool {
    let regex_pattern = pattern
        .replace('.', r"\.")
        .replace('*', ".*")
        .replace('?', ".");

    Regex::new(&format!("^{}$", regex_pattern))
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}

/// Extract raw text content from body.
fn extract_raw_text(doc: &Html) -> String {
    let body_sel = Selector::parse("body").unwrap();
    doc.select(&body_sel)
        .next()
        .map(|el| {
            el.text()
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

/// Limit the number of items in extracted data.
fn limit_items(data: ExtractedData, max: usize) -> ExtractedData {
    match data {
        ExtractedData::Array(mut arr) => {
            arr.truncate(max);
            ExtractedData::Array(arr)
        }
        ExtractedData::Sections(mut secs) => {
            secs.truncate(max);
            ExtractedData::Sections(secs)
        }
        other => other,
    }
}

/// Filter out empty extracted data.
fn filter_empty(data: ExtractedData) -> Option<ExtractedData> {
    match &data {
        ExtractedData::Array(arr) if arr.is_empty() => None,
        ExtractedData::Sections(secs) if secs.is_empty() => None,
        ExtractedData::Single(m) if m.is_empty() => None,
        _ => Some(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match(
            "*.example.com/*",
            "https://www.example.com/page"
        ));
        assert!(glob_match(
            "*.tofugu.com/*",
            "https://www.tofugu.com/grammar/"
        ));
        assert!(!glob_match("*.example.com/*", "https://other.com/page"));
    }

    #[test]
    fn test_full_extraction() {
        let config_yaml = r#"
version: "1.0"
name: "Test Extractor"
extractors:
  - name: "vocab"
    type: table
    selector: "table"
    columns:
      word:
        index: 0
      meaning:
        index: 1
    skip_rows: 1
  - name: "examples"
    type: list
    container: "ul.examples"
    item_selector: "li"
"#;

        let html = r#"
        <html>
        <body>
            <table>
                <tr><th>Word</th><th>Meaning</th></tr>
                <tr><td>猫</td><td>cat</td></tr>
            </table>
            <ul class="examples">
                <li>猫が好きです。</li>
            </ul>
        </body>
        </html>
        "#;

        let engine = ExtractorEngine::from_yaml(config_yaml).unwrap();
        let result = engine.extract("https://example.com", html).unwrap();

        assert_eq!(result.config_name, "Test Extractor");
        assert!(result.data.contains_key("vocab"));
        assert!(result.data.contains_key("examples"));
        assert_eq!(result.metadata.extractors_matched.len(), 2);
    }

    #[test]
    fn test_validation() {
        let config_yaml = r#"
version: "1.0"
name: "Validation Test"
extractors:
  - name: "missing"
    type: table
    selector: "table.nonexistent"
    columns:
      col:
        index: 0
validation:
  required_fields:
    - "missing"
"#;

        let html = "<html><body><p>No table here</p></body></html>";

        let engine = ExtractorEngine::from_yaml(config_yaml).unwrap();
        let result = engine.extract("https://example.com", html).unwrap();

        assert!(!result.validation_errors.is_empty());
        assert!(result.validation_errors[0].contains("missing"));
    }
}
