//! Result types for structured extraction output.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Complete extraction result from a single page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Name of the config used
    pub config_name: String,

    /// Source URL
    pub url: String,

    /// Extracted structured data (keyed by extractor name)
    pub data: HashMap<String, ExtractedData>,

    /// Raw content fallback (if enabled and extractors failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_content: Option<String>,

    /// Extraction metadata
    pub metadata: ExtractionMetadata,

    /// Validation errors (if any)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_errors: Vec<String>,
}

/// Extracted data from a single extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ExtractedData {
    /// Single value (from keyvalue extractor)
    Single(HashMap<String, Value>),

    /// Array of items (from table, list, repeating extractors)
    Array(Vec<HashMap<String, Value>>),

    /// Hierarchical sections
    Sections(Vec<Section>),
}

impl ExtractedData {
    /// Get the number of items in this extracted data.
    pub fn len(&self) -> usize {
        match self {
            ExtractedData::Single(m) => m.len(),
            ExtractedData::Array(arr) => arr.len(),
            ExtractedData::Sections(secs) => secs.len(),
        }
    }

    /// Check if the extracted data is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Hierarchical section with heading and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Heading level (2 for h2, 3 for h3, etc.)
    pub level: u8,

    /// Heading text
    pub heading: String,

    /// Content under this heading
    pub content: String,

    /// Nested child sections
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Section>,
}

/// Metadata about the extraction process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMetadata {
    /// When extraction occurred
    pub extracted_at: DateTime<Utc>,

    /// Duration in milliseconds
    pub duration_ms: u64,

    /// Names of extractors that matched
    pub extractors_matched: Vec<String>,

    /// Names of extractors that failed/didn't match
    pub extractors_failed: Vec<String>,
}

impl ExtractionResult {
    /// Create a new extraction result.
    pub fn new(config_name: String, url: String) -> Self {
        Self {
            config_name,
            url,
            data: HashMap::new(),
            raw_content: None,
            metadata: ExtractionMetadata {
                extracted_at: Utc::now(),
                duration_ms: 0,
                extractors_matched: Vec::new(),
                extractors_failed: Vec::new(),
            },
            validation_errors: Vec::new(),
        }
    }

    /// Check if any data was extracted.
    pub fn has_data(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get total items extracted across all extractors.
    pub fn total_items(&self) -> usize {
        self.data.values().map(|d| d.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_result_serialization() {
        let mut result =
            ExtractionResult::new("Test Config".to_string(), "https://example.com".to_string());

        let mut row = HashMap::new();
        row.insert("word".to_string(), Value::String("猫".to_string()));
        row.insert("meaning".to_string(), Value::String("cat".to_string()));

        result
            .data
            .insert("vocab".to_string(), ExtractedData::Array(vec![row]));

        let json = serde_json::to_string_pretty(&result).unwrap();
        assert!(json.contains("猫"));
        assert!(json.contains("cat"));
    }

    #[test]
    fn test_section_hierarchy() {
        let section = Section {
            level: 2,
            heading: "Grammar".to_string(),
            content: "Main content".to_string(),
            children: vec![Section {
                level: 3,
                heading: "Verbs".to_string(),
                content: "Verb content".to_string(),
                children: vec![],
            }],
        };

        let json = serde_json::to_string(&section).unwrap();
        assert!(json.contains("Grammar"));
        assert!(json.contains("Verbs"));
    }
}
