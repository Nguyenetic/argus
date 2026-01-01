//! Paired content extractor - extracts term/definition pairs.

use crate::config::PairedExtractor;
use crate::result::ExtractedData;
use anyhow::Result;
use scraper::{ElementRef, Html, Selector};
use serde_json::Value;
use std::collections::HashMap;

use super::{extract_text, try_parse_selector};

/// Extract paired content (term/definition) from HTML document.
pub fn extract_paired(
    doc: &Html,
    config: &PairedExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    // Parse container selector(s)
    let container_selectors: Vec<_> = config
        .container
        .split(',')
        .filter_map(|s| Selector::parse(s.trim()).ok())
        .collect();

    if container_selectors.is_empty() {
        anyhow::bail!("Invalid container selector: {}", config.container);
    }

    let first_selector = Selector::parse(&config.first.selector)
        .map_err(|_| anyhow::anyhow!("Invalid first selector: {}", config.first.selector))?;

    let second_selector = Selector::parse(&config.second.selector)
        .map_err(|_| anyhow::anyhow!("Invalid second selector: {}", config.second.selector))?;

    let mut results = Vec::new();

    for container_sel in &container_selectors {
        for container in doc.select(container_sel) {
            if config.second.sibling {
                // Sibling mode: <dt>Term</dt><dd>Definition</dd>
                results.extend(extract_sibling_pairs(
                    &container,
                    &first_selector,
                    &second_selector,
                    &config.first.field,
                    &config.second.field,
                    trim_whitespace,
                ));
            } else if let Some(pair_sel) = &config.pair_selector {
                // Nested mode with pair selector
                if let Some(pair_selector) = try_parse_selector(pair_sel) {
                    results.extend(extract_nested_pairs(
                        &container,
                        &pair_selector,
                        &first_selector,
                        &second_selector,
                        &config.first.field,
                        &config.second.field,
                        trim_whitespace,
                    ));
                }
            } else {
                // Direct nested mode: find first elements and look for second within
                results.extend(extract_direct_pairs(
                    &container,
                    &first_selector,
                    &second_selector,
                    &config.first.field,
                    &config.second.field,
                    trim_whitespace,
                ));
            }
        }
    }

    if results.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ExtractedData::Array(results)))
    }
}

/// Extract pairs where second element is a sibling of first.
fn extract_sibling_pairs(
    container: &ElementRef,
    first_sel: &Selector,
    second_sel: &Selector,
    first_field: &str,
    second_field: &str,
    trim: bool,
) -> Vec<HashMap<String, Value>> {
    let mut pairs = Vec::new();

    for first_el in container.select(first_sel) {
        // Find next sibling that matches second selector
        let mut current = first_el.next_sibling();

        while let Some(node) = current {
            if let Some(el) = ElementRef::wrap(node) {
                if second_sel.matches(&el) {
                    let first_text = extract_text(&first_el, trim);
                    let second_text = extract_text(&el, trim);

                    if !first_text.is_empty() && !second_text.is_empty() {
                        let mut pair = HashMap::new();
                        pair.insert(first_field.to_string(), Value::String(first_text));
                        pair.insert(second_field.to_string(), Value::String(second_text));
                        pairs.push(pair);
                    }
                    break;
                }
            }
            current = node.next_sibling();
        }
    }

    pairs
}

/// Extract pairs from nested structure with explicit pair selector.
fn extract_nested_pairs(
    container: &ElementRef,
    pair_sel: &Selector,
    first_sel: &Selector,
    second_sel: &Selector,
    first_field: &str,
    second_field: &str,
    trim: bool,
) -> Vec<HashMap<String, Value>> {
    let mut pairs = Vec::new();

    for pair_el in container.select(pair_sel) {
        let first = pair_el.select(first_sel).next();
        let second = pair_el.select(second_sel).next();

        if let (Some(f), Some(s)) = (first, second) {
            let first_text = extract_text(&f, trim);
            let second_text = extract_text(&s, trim);

            if !first_text.is_empty() && !second_text.is_empty() {
                let mut pair = HashMap::new();
                pair.insert(first_field.to_string(), Value::String(first_text));
                pair.insert(second_field.to_string(), Value::String(second_text));
                pairs.push(pair);
            }
        }
    }

    pairs
}

/// Extract pairs where both elements are direct children/descendants.
fn extract_direct_pairs(
    container: &ElementRef,
    first_sel: &Selector,
    second_sel: &Selector,
    first_field: &str,
    second_field: &str,
    trim: bool,
) -> Vec<HashMap<String, Value>> {
    let mut pairs = Vec::new();

    // Get all first and second elements
    let firsts: Vec<_> = container.select(first_sel).collect();
    let seconds: Vec<_> = container.select(second_sel).collect();

    // Pair them up by position
    for (f, s) in firsts.iter().zip(seconds.iter()) {
        let first_text = extract_text(f, trim);
        let second_text = extract_text(s, trim);

        if !first_text.is_empty() && !second_text.is_empty() {
            let mut pair = HashMap::new();
            pair.insert(first_field.to_string(), Value::String(first_text));
            pair.insert(second_field.to_string(), Value::String(second_text));
            pairs.push(pair);
        }
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PairElement;

    #[test]
    fn test_extract_dl_pairs() {
        let html = r#"
        <html>
        <body>
            <dl class="vocab">
                <dt>猫</dt>
                <dd>cat</dd>
                <dt>犬</dt>
                <dd>dog</dd>
            </dl>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = PairedExtractor {
            container: "dl.vocab".to_string(),
            pair_selector: None,
            first: PairElement {
                selector: "dt".to_string(),
                field: "term".to_string(),
                sibling: false,
            },
            second: PairElement {
                selector: "dd".to_string(),
                field: "definition".to_string(),
                sibling: true,
            },
        };

        let result = extract_paired(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(pairs)) = result {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].get("term").unwrap(), "猫");
            assert_eq!(pairs[0].get("definition").unwrap(), "cat");
            assert_eq!(pairs[1].get("term").unwrap(), "犬");
            assert_eq!(pairs[1].get("definition").unwrap(), "dog");
        }
    }

    #[test]
    fn test_extract_nested_pairs() {
        let html = r#"
        <html>
        <body>
            <div class="glossary">
                <div class="entry">
                    <span class="word">水</span>
                    <span class="meaning">water</span>
                </div>
                <div class="entry">
                    <span class="word">火</span>
                    <span class="meaning">fire</span>
                </div>
            </div>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = PairedExtractor {
            container: "div.glossary".to_string(),
            pair_selector: Some(".entry".to_string()),
            first: PairElement {
                selector: ".word".to_string(),
                field: "japanese".to_string(),
                sibling: false,
            },
            second: PairElement {
                selector: ".meaning".to_string(),
                field: "english".to_string(),
                sibling: false,
            },
        };

        let result = extract_paired(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(pairs)) = result {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].get("japanese").unwrap(), "水");
            assert_eq!(pairs[0].get("english").unwrap(), "water");
        }
    }
}
