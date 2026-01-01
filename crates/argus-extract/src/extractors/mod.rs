//! Extractor implementations for different content types.

mod keyvalue;
mod list;
mod paired;
mod repeating;
mod sections;
mod table;

pub use keyvalue::extract_keyvalue;
pub use list::extract_list;
pub use paired::extract_paired;
pub use repeating::extract_repeating;
pub use sections::extract_sections;
pub use table::extract_table;

use scraper::ElementRef;

/// Extract text content from an element, optionally trimming whitespace.
pub fn extract_text(element: &ElementRef, trim: bool) -> String {
    let text: String = element.text().collect::<Vec<_>>().join(" ");
    if trim {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        text
    }
}

/// Extract an attribute value from an element.
pub fn extract_attribute(element: &ElementRef, attr: &str) -> Option<String> {
    element.value().attr(attr).map(|s| s.to_string())
}

/// Try to parse a selector, returning None if invalid.
pub fn try_parse_selector(selector: &str) -> Option<scraper::Selector> {
    scraper::Selector::parse(selector).ok()
}
