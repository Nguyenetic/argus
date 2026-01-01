//! Configuration types for structured extraction.
//!
//! Extraction patterns are defined in YAML configuration files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root extraction configuration loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Config version (e.g., "1.0")
    pub version: String,

    /// Human-readable name for this config
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// Global extraction settings
    #[serde(default)]
    pub settings: ExtractionSettings,

    /// URL patterns this config applies to (glob or regex)
    #[serde(default)]
    pub url_patterns: Vec<String>,

    /// Ordered list of extractors to run
    pub extractors: Vec<Extractor>,

    /// Post-processing transformations
    #[serde(default)]
    pub transformations: Vec<Transformation>,

    /// Validation rules for extracted data
    #[serde(default)]
    pub validation: ValidationRules,
}

/// Global settings for extraction behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSettings {
    /// Fall back to raw text if no extractors match
    #[serde(default = "default_true")]
    pub fallback_to_raw: bool,

    /// Trim whitespace from extracted text
    #[serde(default = "default_true")]
    pub trim_whitespace: bool,

    /// Filter out empty results
    #[serde(default = "default_true")]
    pub filter_empty: bool,

    /// Maximum items per extractor (0 = unlimited)
    #[serde(default)]
    pub max_items: usize,
}

impl Default for ExtractionSettings {
    fn default() -> Self {
        Self {
            fallback_to_raw: true,
            trim_whitespace: true,
            filter_empty: true,
            max_items: 0,
        }
    }
}

fn default_true() -> bool {
    true
}

/// Individual extractor definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extractor {
    /// Unique name for this extractor (used as key in output)
    pub name: String,

    /// The extractor type and its configuration
    #[serde(flatten)]
    pub config: ExtractorType,
}

/// Supported extractor types with their configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ExtractorType {
    /// Extract data from HTML tables
    Table(TableExtractor),

    /// Extract term/definition pairs
    Paired(PairedExtractor),

    /// Extract hierarchical sections (headings + content)
    Sections(SectionsExtractor),

    /// Extract list items
    List(ListExtractor),

    /// Extract single key-value metadata
    #[serde(rename = "keyvalue")]
    KeyValue(KeyValueExtractor),

    /// Extract repeated blocks/cards
    Repeating(RepeatingExtractor),
}

/// Configuration for table extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableExtractor {
    /// CSS selector to find tables
    pub selector: String,

    /// Column definitions (name -> column config)
    pub columns: HashMap<String, ColumnDef>,

    /// Number of header rows to skip
    #[serde(default)]
    pub skip_rows: usize,
}

/// Column definition within a table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    /// Column index (0-based)
    pub index: usize,

    /// Alternative selectors to try within the cell
    #[serde(default)]
    pub selectors: Vec<String>,

    /// Whether this column is optional
    #[serde(default)]
    pub optional: bool,

    /// Extract attribute instead of text
    #[serde(default)]
    pub attribute: Option<String>,
}

/// Configuration for paired content extraction (term + definition).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedExtractor {
    /// CSS selector for the container holding pairs
    pub container: String,

    /// Selector for individual pair elements (optional)
    #[serde(default)]
    pub pair_selector: Option<String>,

    /// First element of pair (e.g., term)
    pub first: PairElement,

    /// Second element of pair (e.g., definition)
    pub second: PairElement,
}

/// Element definition within a pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairElement {
    /// CSS selector for this element
    pub selector: String,

    /// Field name in output
    pub field: String,

    /// Whether second element is a sibling (vs nested)
    #[serde(default)]
    pub sibling: bool,
}

/// Configuration for hierarchical section extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionsExtractor {
    /// Heading selectors in priority order (e.g., ["h2", "h3"])
    pub headings: Vec<String>,

    /// How to extract content
    pub content: ContentSelector,

    /// Nest sections by heading level
    #[serde(default)]
    pub nest_by_level: bool,
}

/// Content selection strategy for sections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSelector {
    /// Collect all content until next heading
    #[serde(default)]
    pub until_next_heading: bool,

    /// Specific selector for content elements
    #[serde(default)]
    pub selector: Option<String>,
}

/// Configuration for list extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListExtractor {
    /// CSS selector for the list container
    pub container: String,

    /// CSS selector for list items
    pub item_selector: String,

    /// Fields to extract from each item
    #[serde(default)]
    pub fields: HashMap<String, FieldDef>,
}

/// Field definition for complex extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// CSS selector within the item
    #[serde(default)]
    pub selector: Option<String>,

    /// Default behavior if no selector (":self" = use item itself)
    #[serde(default = "default_self")]
    pub default: String,

    /// Whether this field is optional
    #[serde(default)]
    pub optional: bool,

    /// Extract attribute instead of text
    #[serde(default)]
    pub attribute: Option<String>,

    /// Collect all matches as array
    #[serde(default)]
    pub multiple: bool,

    /// Transform to apply (e.g., "number", "date")
    #[serde(default)]
    pub transform: Option<String>,
}

fn default_self() -> String {
    ":self".to_string()
}

/// Configuration for key-value metadata extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValueExtractor {
    /// Map of field names to their definitions
    pub pairs: HashMap<String, FieldDef>,
}

/// Configuration for repeated block extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatingExtractor {
    /// Optional container selector
    #[serde(default)]
    pub container: Option<String>,

    /// CSS selector for each block
    pub block_selector: String,

    /// Fields to extract from each block
    pub fields: HashMap<String, FieldDef>,
}

/// Post-processing transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    /// Field path (e.g., "vocab_tables.term")
    pub field: String,

    /// Operations to apply
    pub operations: Vec<TransformOp>,
}

/// Transformation operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformOp {
    /// Trim whitespace
    Trim,
    /// Convert to lowercase
    Lowercase,
    /// Convert to uppercase
    Uppercase,
    /// Remove HTML tags
    RemoveHtml,
    /// Parse as date
    #[serde(rename = "parse_date")]
    ParseDate(String),
    /// Regex replacement
    Regex { pattern: String, replace: String },
}

/// Validation rules for extracted data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationRules {
    /// Fields that must be present
    #[serde(default)]
    pub required_fields: Vec<String>,

    /// Minimum item counts per field
    #[serde(default)]
    pub min_items: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_config() {
        let yaml = r#"
version: "1.0"
name: "Test Extractor"
extractors:
  - name: "tables"
    type: table
    selector: "table.vocab"
    columns:
      word:
        index: 0
      meaning:
        index: 1
        optional: true
    skip_rows: 1
"#;

        let config: ExtractionConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.name, "Test Extractor");
        assert_eq!(config.extractors.len(), 1);

        match &config.extractors[0].config {
            ExtractorType::Table(t) => {
                assert_eq!(t.selector, "table.vocab");
                assert_eq!(t.skip_rows, 1);
                assert!(t.columns.contains_key("word"));
            }
            _ => panic!("Expected table extractor"),
        }
    }

    #[test]
    fn test_parse_list_extractor() {
        let yaml = r#"
version: "1.0"
name: "List Test"
extractors:
  - name: "examples"
    type: list
    container: "ul.examples"
    item_selector: "li"
    fields:
      sentence:
        selector: ".jp"
      translation:
        selector: ".en"
        optional: true
"#;

        let config: ExtractionConfig = serde_yaml::from_str(yaml).unwrap();
        match &config.extractors[0].config {
            ExtractorType::List(l) => {
                assert_eq!(l.container, "ul.examples");
                assert_eq!(l.item_selector, "li");
            }
            _ => panic!("Expected list extractor"),
        }
    }
}
