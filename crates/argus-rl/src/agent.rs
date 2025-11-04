//! RL Agent implementation (placeholder)

use argus_core::Result;

pub struct RLAgent {
    // Will implement later
}

impl RLAgent {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn select_action(&self) -> Result<Action> {
        // Placeholder
        Ok(Action::Wait)
    }
}

#[derive(Debug, Clone)]
pub enum Action {
    Wait,
    Scroll,
    Click,
}
