//! Sections extractor - extracts hierarchical heading + content structure.

use crate::config::SectionsExtractor;
use crate::result::{ExtractedData, Section};
use anyhow::Result;
use scraper::{ElementRef, Html, Node, Selector};

use super::extract_text;

/// Extract hierarchical sections from HTML document.
pub fn extract_sections(
    doc: &Html,
    config: &SectionsExtractor,
    trim_whitespace: bool,
) -> Result<Option<ExtractedData>> {
    // Build combined selector for all heading levels
    let heading_selector_str = config.headings.join(", ");
    let heading_selector = Selector::parse(&heading_selector_str)
        .map_err(|_| anyhow::anyhow!("Invalid heading selectors: {:?}", config.headings))?;

    // Find all headings in document order
    let body_selector = Selector::parse("body").unwrap();
    let body = doc.select(&body_selector).next();

    if body.is_none() {
        return Ok(None);
    }

    let body = body.unwrap();
    let mut sections = Vec::new();

    // Collect all headings with their levels
    let headings: Vec<(ElementRef, u8)> = body
        .select(&heading_selector)
        .filter_map(|el| {
            let tag = el.value().name();
            let level = match tag {
                "h1" => Some(1),
                "h2" => Some(2),
                "h3" => Some(3),
                "h4" => Some(4),
                "h5" => Some(5),
                "h6" => Some(6),
                _ => None,
            };
            level.map(|l| (el, l))
        })
        .collect();

    if headings.is_empty() {
        return Ok(None);
    }

    // Process each heading
    for (i, (heading_el, level)) in headings.iter().enumerate() {
        let heading_text = extract_text(heading_el, trim_whitespace);

        // Collect content after this heading until next heading
        let content = if config.content.until_next_heading {
            collect_content_until_next_heading(heading_el, &headings, i, trim_whitespace)
        } else if let Some(sel) = &config.content.selector {
            // Use specific content selector
            if let Ok(content_sel) = Selector::parse(sel) {
                heading_el
                    .next_siblings()
                    .filter_map(|node| ElementRef::wrap(node))
                    .take_while(|el| !heading_selector.matches(el))
                    .filter(|el| content_sel.matches(el))
                    .map(|el| extract_text(&el, trim_whitespace))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        if !heading_text.is_empty() {
            sections.push(Section {
                level: *level,
                heading: heading_text,
                content,
                children: Vec::new(),
            });
        }
    }

    if sections.is_empty() {
        return Ok(None);
    }

    // Optionally nest sections by level
    if config.nest_by_level {
        sections = nest_sections(sections);
    }

    Ok(Some(ExtractedData::Sections(sections)))
}

/// Collect all content between this heading and the next.
fn collect_content_until_next_heading(
    heading: &ElementRef,
    all_headings: &[(ElementRef, u8)],
    current_index: usize,
    trim: bool,
) -> String {
    let mut content_parts = Vec::new();

    // Get the next heading element (if any)
    let next_heading = all_headings.get(current_index + 1).map(|(el, _)| el);

    // Traverse siblings until we hit the next heading or end
    let mut current = heading.next_sibling();

    while let Some(node) = current {
        // Check if this is the next heading
        if let Some(el) = ElementRef::wrap(node) {
            if let Some(next) = next_heading {
                if el == *next {
                    break;
                }
            }

            // Check if this is any heading element
            let tag = el.value().name();
            if matches!(tag, "h1" | "h2" | "h3" | "h4" | "h5" | "h6") {
                break;
            }

            // Extract text from this element
            let text = extract_text(&el, trim);
            if !text.is_empty() {
                content_parts.push(text);
            }
        } else if let Some(text_node) = node.value().as_text() {
            let text = if trim {
                text_node.text.trim().to_string()
            } else {
                text_node.text.to_string()
            };
            if !text.is_empty() {
                content_parts.push(text);
            }
        }

        current = node.next_sibling();
    }

    content_parts.join("\n\n")
}

/// Nest sections based on heading level.
fn nest_sections(flat_sections: Vec<Section>) -> Vec<Section> {
    if flat_sections.is_empty() {
        return flat_sections;
    }

    let mut result = Vec::new();
    let mut stack: Vec<Section> = Vec::new();

    for section in flat_sections {
        // Pop sections from stack that are at same or higher level
        while let Some(top) = stack.last() {
            if top.level >= section.level {
                let completed = stack.pop().unwrap();
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(completed);
                } else {
                    result.push(completed);
                }
            } else {
                break;
            }
        }

        stack.push(section);
    }

    // Empty remaining stack
    while let Some(completed) = stack.pop() {
        if let Some(parent) = stack.last_mut() {
            parent.children.push(completed);
        } else {
            result.push(completed);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ContentSelector;

    #[test]
    fn test_extract_flat_sections() {
        let html = r#"
        <html>
        <body>
            <h2>Introduction</h2>
            <p>This is the intro.</p>
            <h2>Grammar Basics</h2>
            <p>Grammar content here.</p>
            <p>More grammar.</p>
            <h3>Verbs</h3>
            <p>Verb info.</p>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = SectionsExtractor {
            headings: vec!["h2".to_string(), "h3".to_string()],
            content: ContentSelector {
                until_next_heading: true,
                selector: None,
            },
            nest_by_level: false,
        };

        let result = extract_sections(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Sections(sections)) = result {
            assert_eq!(sections.len(), 3);
            assert_eq!(sections[0].heading, "Introduction");
            assert!(sections[0].content.contains("intro"));
            assert_eq!(sections[1].heading, "Grammar Basics");
            assert!(sections[1].content.contains("grammar"));
            assert_eq!(sections[2].heading, "Verbs");
        }
    }

    #[test]
    fn test_extract_nested_sections() {
        let html = r#"
        <html>
        <body>
            <h2>Chapter 1</h2>
            <p>Chapter content.</p>
            <h3>Section 1.1</h3>
            <p>Section content.</p>
            <h3>Section 1.2</h3>
            <p>More content.</p>
            <h2>Chapter 2</h2>
            <p>Chapter 2 content.</p>
        </body>
        </html>
        "#;

        let doc = Html::parse_document(html);
        let config = SectionsExtractor {
            headings: vec!["h2".to_string(), "h3".to_string()],
            content: ContentSelector {
                until_next_heading: true,
                selector: None,
            },
            nest_by_level: true,
        };

        let result = extract_sections(&doc, &config, true).unwrap();
        assert!(result.is_some());

        if let Some(ExtractedData::Sections(sections)) = result {
            assert_eq!(sections.len(), 2); // Two chapters
            assert_eq!(sections[0].heading, "Chapter 1");
            assert_eq!(sections[0].children.len(), 2); // Two subsections
            assert_eq!(sections[0].children[0].heading, "Section 1.1");
            assert_eq!(sections[1].heading, "Chapter 2");
        }
    }
}
