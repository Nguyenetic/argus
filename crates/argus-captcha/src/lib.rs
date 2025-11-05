//! Argus CAPTCHA - Computer Vision CAPTCHA Solver
//!
//! This crate implements comprehensive CAPTCHA solving using computer vision,
//! machine learning, and audio processing techniques.
//!
//! # Supported CAPTCHA Types
//!
//! - **Text CAPTCHAs**: OCR-based recognition
//! - **Image CAPTCHAs**: CNN-based object detection (reCAPTCHA v2)
//! - **Puzzle CAPTCHAs**: Template matching and geometric analysis
//! - **Audio CAPTCHAs**: Speech-to-text transcription
//! - **Cloudflare Turnstile**: Behavioral analysis
//!
//! # Architecture
//!
//! ```text
//! CaptchaSolver
//!     ├── TextSolver (OCR)
//!     ├── ImageSolver (CNN + Object Detection)
//!     ├── PuzzleSolver (Template Matching)
//!     └── AudioSolver (Speech Recognition)
//! ```

pub mod audio;
pub mod common;
pub mod image_captcha;
pub mod models;
pub mod ocr;
pub mod puzzle;
pub mod rl_integration;
pub mod solver;

pub use common::{CaptchaImage, DetectedObject, Point};
pub use rl_integration::{
    CaptchaDetection, CaptchaOutcome, IntegratedBotAgent, IntegrationConfig, IntegrationMetrics,
};
pub use solver::{CaptchaSolver, CaptchaType, Solution, SolverConfig, SolverResult};
