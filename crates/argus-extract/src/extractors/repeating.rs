//! Repeating block extractor - extracts repeated cards/blocks.

use crate::config::RepeatingExtractor;
use crate::result::ExtractedData;
use anyhow::Result;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;

use super::{extract_attribute, extract_text, try_parse_selector};

/// Extract repeating blocks from HTML document.
pub fn extract_repeating(
    doc: &Html,
    config: &RepeatingExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    let block_selector = Selector::parse(&config.block_selector)
        .map_err(|_| anyhow::anyhow!("Invalid block selector: {}", config.block_selector))?;

    let mut results = Vec::new();

    // If container specified, search within it
    let blocks: Box<dyn Iterator<Item = scraper::ElementRef>> =
        if let Some(container) = &config.container {
            let container_selectors: Vec<_> = container
                .split(',')
                .filter_map(|s| Selector::parse(s.trim()).ok())
                .collect();

            if container_selectors.is_empty() {
                anyhow::bail!("Invalid container selector: {}", container);
            }

            // Collect all blocks from all matching containers
            let blocks: Vec<_> = container_selectors
                .iter()
                .flat_map(|sel| doc.select(sel))
                .flat_map(|container| container.select(&block_selector))
                .collect();

            Box::new(blocks.into_iter())
        } else {
            // Search entire document
            Box::new(doc.select(&block_selector))
        };

    for block in blocks {
        let mut block_data = HashMap::new();
        let mut has_required = true;

        for (field_name, field_def) in &config.fields {
            let value = if let Some(selector_str) = &field_def.selector {
                // Try each selector
                let selectors: Vec<_> = selector_str
                    .split(',')
                    .filter_map(|s| try_parse_selector(s.trim()))
                    .collect();

                let mut found = None;
                for sel in &selectors {
                    if field_def.multiple {
                        // Collect all matches
                        let values: Vec<Value> = block
                            .select(&sel)
                            .map(|el| {
                                let text = if let Some(attr) = &field_def.attribute {
                                    extract_attribute(&el, attr).unwrap_or_default()
                                } else {
                                    extract_text(&el, trim_whitespace)
                                };
                                Value::String(text)
                            })
                            .filter(|v| {
                                if let Value::String(s) = v {
                                    !s.is_empty()
                                } else {
                                    true
                                }
                            })
                            .collect();

                        if !values.is_empty() {
                            found = Some(Value::Array(values));
                            break;
                        }
                    } else if let Some(el) = block.select(&sel).next() {
                        let text = if let Some(attr) = &field_def.attribute {
                            extract_attribute(&el, attr)
                        } else {
                            Some(extract_text(&el, trim_whitespace))
                        };

                        if let Some(t) = text {
                            if !t.is_empty() {
                                found = Some(Value::String(t));
                                break;
                            }
                        }
                    }
                }
                found
            } else if field_def.default == ":self" {
                Some(Value::String(extract_text(&block, trim_whitespace)))
            } else {
                None
            };

            if let Some(v) = value {
                // Apply transform if specified
                let transformed = apply_transform(v, field_def.transform.as_deref());
                block_data.insert(field_name.clone(), transformed);
            } else if !field_def.optional {
                has_required = false;
                break;
            }
        }

        if has_required && !block_data.is_empty() {
            results.push(block_data);
        }
    }

    if results.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ExtractedData::Array(results)))
    }
}

/// Apply a transform to a value.
fn apply_transform(value: Value, transform: Option<&str>) -> Value {
    match (transform, &value) {
        (Some("number"), Value::String(s)) => {
            if let Ok(n) = s.trim().parse::<f64>() {
                Value::Number(
                    serde_json::Number::from_f64(n)
                        .unwrap_or_else(|| serde_json::Number::from(n as i64)),
                )
            } else {
                value
            }
        }
        (Some("boolean"), Value::String(s)) => {
            let lower = s.to_lowercase();
            Value::Bool(lower == "true" || lower == "yes" || lower == "1" || lower == "on")
        }
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FieldDef;

    #[test]
    fn test_extract_review_cards() {
        let html = r#"
        <html>
        <body>
            <div class="reviews">
                <div class="review">
                    <h3 class="author">山田太郎</h3>
                    <span class="rating" data-score="5">★★★★★</span>
                    <p class="text">とても良い!</p>
                </div>
                <div class="review">
                    <h3 class="author">田中花子</h3>
                    <span class="rating" data-score="4">★★★★</span>
                    <p class="text">分かりやすい</p>
                </div>
            </div>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = RepeatingExtractor {
            container: Some(".reviews".to_string()),
            block_selector: ".review".to_string(),
            fields: [
                (
                    "author".to_string(),
                    FieldDef {
                        selector: Some(".author".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: None,
                        multiple: false,
                        transform: None,
                    },
                ),
                (
                    "rating".to_string(),
                    FieldDef {
                        selector: Some(".rating".to_string()),
                        default: ":self".to_string(),
                        optional: false,
                        attribute: Some("data-score".to_string()),
                        multiple: false,
                        transform: Some("number".to_string()),
                    },
                ),
                (
                    "text".to_string(),
                    FieldDef {
                        selector: Some(".text".to_string()),
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

        let result = extract_repeating(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(reviews)) = result {
            assert_eq!(reviews.len(), 2);
            assert_eq!(reviews[0].get("author").unwrap(), "山田太郎");
            assert_eq!(reviews[0].get("rating").unwrap(), 5);
            assert_eq!(reviews[0].get("text").unwrap(), "とても良い!");
        }
    }
}
