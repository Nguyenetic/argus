//! Key-value extractor - extracts single metadata values.

use crate::config::KeyValueExtractor;
use crate::result::ExtractedData;
use anyhow::Result;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;

use super::{extract_attribute, extract_text, try_parse_selector};

/// Extract key-value metadata from HTML document.
pub fn extract_keyvalue(
    doc: &Html,
    config: &KeyValueExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    let mut result = HashMap::new();

    for (field_name, field_def) in &config.pairs {
        let value = if let Some(selector_str) = &field_def.selector {
            // Try each selector (comma-separated)
            let selectors: Vec<_> = selector_str
                .split(',')
                .filter_map(|s| Selector::parse(s.trim()).ok())
                .collect();

            let mut found_value = None;

            for sel in &selectors {
                if let Some(el) = doc.select(sel).next() {
                    let text = if let Some(attr) = &field_def.attribute {
                        extract_attribute(&el, attr)
                    } else {
                        Some(extract_text(&el, trim_whitespace))
                    };

                    if let Some(t) = text {
                        if !t.is_empty() {
                            found_value = Some(t);
                            break;
                        }
                    }
                }
            }

            // Handle multiple matches if requested
            if field_def.multiple && found_value.is_none() {
                let mut values = Vec::new();
                for sel in &selectors {
                    for el in doc.select(sel) {
                        let text = if let Some(attr) = &field_def.attribute {
                            extract_attribute(&el, attr).unwrap_or_default()
                        } else {
                            extract_text(&el, trim_whitespace)
                        };

                        if !text.is_empty() {
                            values.push(Value::String(text));
                        }
                    }
                }

                if !values.is_empty() {
                    result.insert(field_name.clone(), Value::Array(values));
                    continue;
                }
            }

            found_value
        } else {
            None
        };

        if let Some(v) = value {
            // Apply transform if specified
            let transformed = apply_transform(&v, field_def.transform.as_deref());
            result.insert(field_name.clone(), transformed);
        } else if !field_def.optional {
            // Required field missing - return None
            return Ok(None);
        }
    }

    if result.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ExtractedData::Single(result)))
    }
}

/// Apply a transform to a string value.
fn apply_transform(value: &str, transform: Option<&str>) -> Value {
    match transform {
        Some("number") => {
            if let Ok(n) = value.trim().parse::<f64>() {
                Value::Number(
                    serde_json::Number::from_f64(n)
                        .unwrap_or_else(|| serde_json::Number::from(n as i64)),
                )
            } else {
                Value::String(value.to_string())
            }
        }
        Some("boolean") => {
            let lower = value.to_lowercase();
            Value::Bool(lower == "true" || lower == "yes" || lower == "1" || lower == "on")
        }
        Some("date") => {
            // Try to parse common date formats
            // For now, just return as string - can be enhanced later
            Value::String(value.to_string())
        }
        _ => Value::String(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FieldDef;

    #[test]
    fn test_extract_metadata() {
        let html = r#"
        <html>
        <head>
            <meta name="author" content="Tofugu">
            <meta name="date" content="2024-01-15">
        </head>
        <body>
            <h1 class="title">Japanese Grammar Guide</h1>
            <span class="category">Grammar</span>
            <span class="category">Beginner</span>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = KeyValueExtractor {
            pairs: [
                (
                    "author".to_string(),
                    FieldDef {
                        selector: Some("meta[name='author']".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: Some("content".to_string()),
                        multiple: false,
                        transform: None,
                    },
                ),
                (
                    "title".to_string(),
                    FieldDef {
                        selector: Some("h1.title".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: None,
                        multiple: false,
                        transform: None,
                    },
                ),
                (
                    "categories".to_string(),
                    FieldDef {
                        selector: Some(".category".to_string()),
                        default: ":self".to_string(),
                        optional: true,
                        attribute: None,
                        multiple: true,
                        transform: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };

        let result = extract_keyvalue(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Single(data)) = result {
            assert_eq!(data.get("author").unwrap(), "Tofugu");
            assert_eq!(data.get("title").unwrap(), "Japanese Grammar Guide");

            if let Some(Value::Array(cats)) = data.get("categories") {
                assert_eq!(cats.len(), 2);
                assert_eq!(cats[0], "Grammar");
                assert_eq!(cats[1], "Beginner");
            }
        }
    }
}
