//! List extractor - extracts data from HTML lists (ul, ol).

use crate::config::ListExtractor;
use crate::result::ExtractedData;
use anyhow::Result;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;

use super::{extract_attribute, extract_text, try_parse_selector};

/// Extract list data from HTML document.
pub fn extract_list(
    doc: &Html,
    config: &ListExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    // Parse container selector (may have multiple selectors separated by comma)
    let container_selectors: Vec<_> = config
        .container
        .split(',')
        .filter_map(|s| Selector::parse(s.trim()).ok())
        .collect();

    if container_selectors.is_empty() {
        anyhow::bail!("Invalid container selector: {}", config.container);
    }

    let item_selector = Selector::parse(&config.item_selector)
        .map_err(|_| anyhow::anyhow!("Invalid item selector: {}", config.item_selector))?;

    let mut results = Vec::new();

    // Try each container selector
    for container_sel in &container_selectors {
        for container in doc.select(container_sel) {
            for item in container.select(&item_selector) {
                let mut item_data = HashMap::new();

                if config.fields.is_empty() {
                    // No fields defined - just extract the item text
                    let text = extract_text(&item, trim_whitespace);
                    if !text.is_empty() {
                        item_data.insert("text".to_string(), Value::String(text));
                    }
                } else {
                    // Extract each defined field
                    for (field_name, field_def) in &config.fields {
                        let value = if let Some(selector) = &field_def.selector {
                            // Use specific selector
                            if let Some(sel) = try_parse_selector(selector) {
                                if field_def.multiple {
                                    // Collect all matches
                                    let values: Vec<Value> = item
                                        .select(&sel)
                                        .map(|el| {
                                            let text = if let Some(attr) = &field_def.attribute {
                                                extract_attribute(&el, attr).unwrap_or_default()
                                            } else {
                                                extract_text(&el, trim_whitespace)
                                            };
                                            Value::String(text)
                                        })
                                        .collect();

                                    if !values.is_empty() {
                                        Some(Value::Array(values))
                                    } else {
                                        None
                                    }
                                } else {
                                    // Single match
                                    item.select(&sel).next().map(|el| {
                                        let text = if let Some(attr) = &field_def.attribute {
                                            extract_attribute(&el, attr).unwrap_or_default()
                                        } else {
                                            extract_text(&el, trim_whitespace)
                                        };
                                        Value::String(text)
                                    })
                                }
                            } else {
                                None
                            }
                        } else if field_def.default == ":self" {
                            // Use the item itself
                            Some(Value::String(extract_text(&item, trim_whitespace)))
                        } else {
                            None
                        };

                        if let Some(v) = value {
                            // Apply transform if specified
                            let transformed =
                                apply_field_transform(v, field_def.transform.as_deref());
                            item_data.insert(field_name.clone(), transformed);
                        } else if !field_def.optional {
                            // Required field missing - skip this item
                            item_data.clear();
                            break;
                        }
                    }
                }

                if !item_data.is_empty() {
                    results.push(item_data);
                }
            }
        }
    }

    if results.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ExtractedData::Array(results)))
    }
}

/// Apply a transform to a field value.
fn apply_field_transform(value: Value, transform: Option<&str>) -> Value {
    match transform {
        Some("number") => {
            if let Value::String(s) = &value {
                // Try to parse as number
                if let Ok(n) = s.trim().parse::<f64>() {
                    return Value::Number(
                        serde_json::Number::from_f64(n)
                            .unwrap_or_else(|| serde_json::Number::from(n as i64)),
                    );
                }
            }
            value
        }
        Some("boolean") => {
            if let Value::String(s) = &value {
                let lower = s.to_lowercase();
                return Value::Bool(
                    lower == "true" || lower == "yes" || lower == "1" || lower == "on",
                );
            }
            value
        }
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FieldDef;

    #[test]
    fn test_extract_simple_list() {
        let html = r#"
        <html>
        <body>
            <ul class="examples">
                <li>今日は暑いです。</li>
                <li>明日は寒いです。</li>
            </ul>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = ListExtractor {
            container: "ul.examples".to_string(),
            item_selector: "li".to_string(),
            fields: HashMap::new(),
        };

        let result = extract_list(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(items)) = result {
            assert_eq!(items.len(), 2);
            assert!(items[0]
                .get("text")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("暑い"));
        }
    }

    #[test]
    fn test_extract_list_with_fields() {
        let html = r#"
        <html>
        <body>
            <ul class="vocab">
                <li><span class="jp">猫</span><span class="en">cat</span></li>
                <li><span class="jp">犬</span><span class="en">dog</span></li>
            </ul>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = ListExtractor {
            container: "ul.vocab".to_string(),
            item_selector: "li".to_string(),
            fields: [
                (
                    "japanese".to_string(),
                    FieldDef {
                        selector: Some(".jp".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: None,
                        multiple: false,
                        transform: None,
                    },
                ),
                (
                    "english".to_string(),
                    FieldDef {
                        selector: Some(".en".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: None,
                        multiple: false,
                        transform: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
        };

        let result = extract_list(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(items)) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0].get("japanese").unwrap(), "猫");
            assert_eq!(items[0].get("english").unwrap(), "cat");
        }
    }
}
