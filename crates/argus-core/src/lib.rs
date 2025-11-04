//! Argus Core - The foundation of the Argus web intelligence system
//!
//! This crate contains core types, traits, and utilities used across all Argus components.

pub mod error;
pub mod types;

pub use error::{Error, Result};
pub use types::*;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
