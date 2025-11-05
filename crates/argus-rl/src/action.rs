//! Action space for RL agent

use serde::{Deserialize, Serialize};

/// Discrete actions the RL agent can take
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// Wait for a short duration (0.5-2 seconds)
    WaitShort,

    /// Wait for a longer duration (2-10 seconds)
    WaitLong,

    /// Smooth scroll, small distance (10-30% of page)
    ScrollSmall,

    /// Smooth scroll, large distance (30-70% of page)
    ScrollLarge,

    /// Random mouse movement using Perlin noise
    MouseMovement,

    /// Mouse movement + click using Gaussian curve
    MouseClick,

    /// Hover + click interaction
    Interact,

    /// Navigate to new page with delay
    Navigate,
}

impl Action {
    /// Get all possible actions
    pub fn all() -> Vec<Self> {
        vec![
            Action::WaitShort,
            Action::WaitLong,
            Action::ScrollSmall,
            Action::ScrollLarge,
            Action::MouseMovement,
            Action::MouseClick,
            Action::Interact,
            Action::Navigate,
        ]
    }

    /// Get the number of possible actions
    pub const fn dim() -> usize {
        8
    }

    /// Convert action to index (0-7)
    pub fn to_index(&self) -> usize {
        match self {
            Action::WaitShort => 0,
            Action::WaitLong => 1,
            Action::ScrollSmall => 2,
            Action::ScrollLarge => 3,
            Action::MouseMovement => 4,
            Action::MouseClick => 5,
            Action::Interact => 6,
            Action::Navigate => 7,
        }
    }

    /// Convert index to action
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Action::WaitShort),
            1 => Some(Action::WaitLong),
            2 => Some(Action::ScrollSmall),
            3 => Some(Action::ScrollLarge),
            4 => Some(Action::MouseMovement),
            5 => Some(Action::MouseClick),
            6 => Some(Action::Interact),
            7 => Some(Action::Navigate),
            _ => None,
        }
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Action::WaitShort => "Wait 0.5-2s",
            Action::WaitLong => "Wait 2-10s",
            Action::ScrollSmall => "Scroll 10-30%",
            Action::ScrollLarge => "Scroll 30-70%",
            Action::MouseMovement => "Move mouse (Perlin)",
            Action::MouseClick => "Mouse + click (Gaussian)",
            Action::Interact => "Hover + click",
            Action::Navigate => "Navigate to new page",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_count() {
        assert_eq!(Action::all().len(), Action::dim());
    }

    #[test]
    fn test_index_conversion() {
        for action in Action::all() {
            let index = action.to_index();
            let recovered = Action::from_index(index).unwrap();
            assert_eq!(action, recovered);
        }
    }

    #[test]
    fn test_invalid_index() {
        assert!(Action::from_index(8).is_none());
        assert!(Action::from_index(100).is_none());
    }
}
