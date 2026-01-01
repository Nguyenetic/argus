//! Table extractor - extracts data from HTML tables.

use crate::config::TableExtractor;
use crate::result::ExtractedData;
use anyhow::{Context, Result};
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;

use super::{extract_attribute, extract_text, try_parse_selector};

/// Extract table data from HTML document.
pub fn extract_table(
    doc: &Html,
    config: &TableExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    let table_selector = Selector::parse(&config.selector)
        .map_err(|_| anyhow::anyhow!("Invalid table selector: {}", config.selector))?;

    let row_selector = Selector::parse("tr").unwrap();
    let cell_selector = Selector::parse("td, th").unwrap();

    let mut results = Vec::new();

    for table in doc.select(&table_selector) {
        for (row_idx, row) in table.select(&row_selector).enumerate() {
            // Skip header rows
            if row_idx < config.skip_rows {
                continue;
            }

            let cells: Vec<_> = row.select(&cell_selector).collect();
            let mut row_data = HashMap::new();
            let mut has_required = true;

            for (col_name, col_def) in &config.columns {
                let value = if col_def.index < cells.len() {
                    let cell = &cells[col_def.index];

                    // Try alternative selectors within the cell
                    let text = if !col_def.selectors.is_empty() {
                        col_def
                            .selectors
                            .iter()
                            .find_map(|sel| {
                                try_parse_selector(sel).and_then(|s| {
                                    cell.select(&s)
                                        .next()
                                        .map(|el| extract_text(&el, trim_whitespace))
                                })
                            })
                            .unwrap_or_else(|| extract_text(cell, trim_whitespace))
                    } else if let Some(attr) = &col_def.attribute {
                        extract_attribute(cell, attr).unwrap_or_default()
                    } else {
                        extract_text(cell, trim_whitespace)
                    };

                    if text.is_empty() && !col_def.optional {
                        has_required = false;
                    }
                    text
                } else if col_def.optional {
                    String::new()
                } else {
                    has_required = false;
                    String::new()
                };

                if !value.is_empty() || !col_def.optional {
                    row_data.insert(col_name.clone(), Value::String(value));
                }
            }

            // Only add row if all required columns have data
            if has_required && !row_data.is_empty() {
                results.push(row_data);
            }
        }
    }

    if results.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ExtractedData::Array(results)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_table() {
        let html = r#"
        <html>
        <body>
            <table class="vocab">
                <tr><th>Word</th><th>Meaning</th></tr>
                <tr><td>猫</td><td>cat</td></tr>
                <tr><td>犬</td><td>dog</td></tr>
            </table>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = TableExtractor {
            selector: "table.vocab".to_string(),
            columns: [
                (
                    "word".to_string(),
                    crate::config::ColumnDef {
                        index: 0,
                        selectors: vec![],
                        optional: false,
                        attribute: None,
                    },
                ),
                (
                    "meaning".to_string(),
                    crate::config::ColumnDef {
                        index: 1,
                        selectors: vec![],
                        optional: false,
                        attribute: None,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            skip_rows: 1,
        };

        let result = extract_table(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Array(rows)) = result {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].get("word").unwrap(), "猫");
            assert_eq!(rows[0].get("meaning").unwrap(), "cat");
        }
    }
}
