use anyhow::{bail, Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::audio::AudioSolver;
use crate::common::CaptchaImage;
use crate::image_captcha::ImageCaptchaSolver;
use crate::ocr::OcrSolver;
use crate::puzzle::PuzzleSolver;

/// Main CAPTCHA solver orchestrator
///
/// Automatically detects CAPTCHA type and routes to appropriate solver.
/// Implements fallback strategies and metrics tracking.
pub struct CaptchaSolver {
    ocr_solver: OcrSolver,
    image_solver: ImageCaptchaSolver,
    audio_solver: Option<AudioSolver>,
    puzzle_solver: PuzzleSolver,
    config: SolverConfig,
    metrics: SolverMetrics,
}

/// Configuration for CAPTCHA solver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum time to spend solving a CAPTCHA
    pub timeout: Duration,

    /// Enable audio CAPTCHA solving (requires Whisper)
    pub enable_audio: bool,

    /// Enable fallback strategies (e.g., OCR → CNN)
    pub enable_fallback: bool,

    /// Minimum confidence threshold (0.0-1.0)
    pub min_confidence: f32,

    /// Maximum number of retry attempts
    pub max_retries: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            enable_audio: true,
            enable_fallback: true,
            min_confidence: 0.7,
            max_retries: 3,
        }
    }
}

/// CAPTCHA type detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptchaType {
    /// Text-based CAPTCHA (distorted characters)
    Text,

    /// Image selection (reCAPTCHA v2 style)
    ImageGrid,

    /// Audio CAPTCHA
    Audio,

    /// Slider puzzle
    Slider,

    /// Rotation puzzle
    Rotation,

    /// Jigsaw puzzle
    Jigsaw,

    /// Unknown/unsupported type
    Unknown,
}

/// Result from CAPTCHA solving attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    /// Type of CAPTCHA that was solved
    pub captcha_type: CaptchaType,

    /// Solution (format depends on CAPTCHA type)
    pub solution: Solution,

    /// Confidence score (0.0-1.0)
    pub confidence: f32,

    /// Time taken to solve
    pub solve_time: Duration,

    /// Whether fallback strategy was used
    pub used_fallback: bool,
}

/// CAPTCHA solution (type-specific)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Solution {
    /// Text solution (OCR result)
    Text(String),

    /// Image grid indices (reCAPTCHA)
    ImageIndices(Vec<usize>),

    /// Audio transcription
    AudioText(String),

    /// Slider X offset in pixels
    SliderOffset(i32),

    /// Rotation angle in degrees
    RotationAngle(f32),

    /// Jigsaw piece coordinates
    JigsawCoordinates { x: i32, y: i32 },
}

/// Metrics tracking
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SolverMetrics {
    pub total_attempts: u64,
    pub successful_solves: u64,
    pub failed_solves: u64,
    pub total_solve_time: Duration,
    pub by_type: std::collections::HashMap<String, TypeMetrics>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TypeMetrics {
    pub attempts: u64,
    pub successes: u64,
    pub avg_confidence: f32,
    pub avg_solve_time: Duration,
}

impl CaptchaSolver {
    /// Create a new CAPTCHA solver with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new CAPTCHA solver with custom configuration
    pub fn with_config(config: SolverConfig) -> Result<Self> {
        let ocr_solver = OcrSolver::new()?;
        let image_solver = ImageCaptchaSolver::new()?;
        let puzzle_solver = PuzzleSolver::new();

        let audio_solver = if config.enable_audio {
            #[cfg(feature = "audio")]
            {
                Some(AudioSolver::new()?)
            }
            #[cfg(not(feature = "audio"))]
            {
                log::warn!("Audio CAPTCHA solving disabled (compile with --features audio)");
                None
            }
        } else {
            None
        };

        Ok(Self {
            ocr_solver,
            image_solver,
            audio_solver,
            puzzle_solver,
            config,
            metrics: SolverMetrics::default(),
        })
    }

    /// Detect CAPTCHA type from image characteristics
    pub fn detect_type(&self, image: &CaptchaImage) -> CaptchaType {
        let (width, height) = (image.original.width(), image.original.height());
        let aspect_ratio = width as f32 / height as f32;

        // reCAPTCHA v2 is typically 300x300 or 450x450
        if (width == 300 || width == 450) && width == height {
            return CaptchaType::ImageGrid;
        }

        // Slider puzzles are typically wide and short
        if aspect_ratio > 2.0 && height < 150 {
            return CaptchaType::Slider;
        }

        // Rotation puzzles are square or nearly square
        if (0.9..=1.1).contains(&aspect_ratio) && width > 200 {
            return CaptchaType::Rotation;
        }

        // Text CAPTCHAs are typically wide rectangles
        if aspect_ratio > 1.5 && height < 100 {
            return CaptchaType::Text;
        }

        CaptchaType::Unknown
    }

    /// Solve a text CAPTCHA
    pub fn solve_text(&mut self, image: CaptchaImage) -> Result<SolverResult> {
        let start_time = Instant::now();

        let result = self.ocr_solver.solve(image)?;

        let solve_time = start_time.elapsed();

        Ok(SolverResult {
            captcha_type: CaptchaType::Text,
            solution: Solution::Text(result.text),
            confidence: result.confidence,
            solve_time,
            used_fallback: result.used_cnn,
        })
    }

    /// Solve an image grid CAPTCHA (reCAPTCHA v2)
    pub fn solve_image_grid(
        &mut self,
        images: Vec<CaptchaImage>,
        query: &str,
    ) -> Result<SolverResult> {
        let start_time = Instant::now();

        let indices = self.image_solver.solve_grid(images, query)?;

        let solve_time = start_time.elapsed();

        // Calculate average confidence (simplified)
        let confidence = 0.85; // TODO: Get actual confidence from detector

        Ok(SolverResult {
            captcha_type: CaptchaType::ImageGrid,
            solution: Solution::ImageIndices(indices),
            confidence,
            solve_time,
            used_fallback: false,
        })
    }

    /// Solve an audio CAPTCHA
    #[cfg(feature = "audio")]
    pub async fn solve_audio(&mut self, audio_bytes: &[u8]) -> Result<SolverResult> {
        let start_time = Instant::now();

        let audio_solver = self
            .audio_solver
            .as_mut()
            .context("Audio solver not initialized")?;

        let transcription = audio_solver.solve_from_bytes(audio_bytes).await?;

        let solve_time = start_time.elapsed();

        Ok(SolverResult {
            captcha_type: CaptchaType::Audio,
            solution: Solution::AudioText(transcription),
            confidence: 0.95, // Whisper is very confident
            solve_time,
            used_fallback: false,
        })
    }

    #[cfg(not(feature = "audio"))]
    pub async fn solve_audio(&mut self, _audio_bytes: &[u8]) -> Result<SolverResult> {
        bail!("Audio CAPTCHA solving not enabled (compile with --features audio)")
    }

    /// Solve a slider CAPTCHA
    pub fn solve_slider(
        &mut self,
        background: CaptchaImage,
        puzzle_piece: CaptchaImage,
    ) -> Result<SolverResult> {
        let start_time = Instant::now();

        let result = self
            .puzzle_solver
            .solve_slider(&background, &puzzle_piece)?;

        let solve_time = start_time.elapsed();

        Ok(SolverResult {
            captcha_type: CaptchaType::Slider,
            solution: Solution::SliderOffset(result.x_offset),
            confidence: result.confidence,
            solve_time,
            used_fallback: false,
        })
    }

    /// Solve a rotation CAPTCHA
    pub fn solve_rotation(&mut self, image: CaptchaImage) -> Result<SolverResult> {
        let start_time = Instant::now();

        let result = self.puzzle_solver.solve_rotation(&image)?;

        let solve_time = start_time.elapsed();

        Ok(SolverResult {
            captcha_type: CaptchaType::Rotation,
            solution: Solution::RotationAngle(result.angle),
            confidence: result.confidence,
            solve_time,
            used_fallback: false,
        })
    }

    /// Solve a jigsaw CAPTCHA
    pub fn solve_jigsaw(
        &mut self,
        background: CaptchaImage,
        puzzle_piece: CaptchaImage,
    ) -> Result<SolverResult> {
        let start_time = Instant::now();

        let result = self
            .puzzle_solver
            .solve_jigsaw(&background, &puzzle_piece)?;

        let solve_time = start_time.elapsed();

        Ok(SolverResult {
            captcha_type: CaptchaType::Jigsaw,
            solution: Solution::JigsawCoordinates {
                x: result.x,
                y: result.y,
            },
            confidence: result.confidence,
            solve_time,
            used_fallback: false,
        })
    }

    /// Automatically solve a CAPTCHA (detect type and route to solver)
    pub async fn solve_auto(&mut self, image: CaptchaImage) -> Result<SolverResult> {
        let captcha_type = self.detect_type(&image);

        match captcha_type {
            CaptchaType::Text => self.solve_text(image),
            CaptchaType::ImageGrid => {
                // Need query string for image grids
                bail!("Image grid CAPTCHAs require query string (use solve_image_grid)")
            }
            CaptchaType::Rotation => self.solve_rotation(image),
            _ => {
                bail!("Cannot auto-solve CAPTCHA type: {:?}", captcha_type)
            }
        }
    }

    /// Update metrics after solving attempt
    pub fn update_metrics(&mut self, result: &SolverResult, success: bool) {
        self.metrics.total_attempts += 1;

        if success {
            self.metrics.successful_solves += 1;
        } else {
            self.metrics.failed_solves += 1;
        }

        self.metrics.total_solve_time += result.solve_time;

        // Update type-specific metrics
        let type_key = format!("{:?}", result.captcha_type);
        let type_metrics = self
            .metrics
            .by_type
            .entry(type_key)
            .or_insert_with(TypeMetrics::default);

        type_metrics.attempts += 1;
        if success {
            type_metrics.successes += 1;
        }

        // Update rolling average
        let alpha = 0.1; // Exponential moving average factor
        type_metrics.avg_confidence =
            type_metrics.avg_confidence * (1.0 - alpha) + result.confidence * alpha;

        let new_avg_time = Duration::from_secs_f32(
            type_metrics.avg_solve_time.as_secs_f32() * (1.0 - alpha)
                + result.solve_time.as_secs_f32() * alpha,
        );
        type_metrics.avg_solve_time = new_avg_time;
    }

    /// Get current metrics
    pub fn metrics(&self) -> &SolverMetrics {
        &self.metrics
    }

    /// Get success rate (0.0-1.0)
    pub fn success_rate(&self) -> f32 {
        if self.metrics.total_attempts == 0 {
            return 0.0;
        }
        self.metrics.successful_solves as f32 / self.metrics.total_attempts as f32
    }

    /// Get average solve time
    pub fn avg_solve_time(&self) -> Duration {
        if self.metrics.total_attempts == 0 {
            return Duration::ZERO;
        }
        self.metrics.total_solve_time / self.metrics.total_attempts as u32
    }
}

impl Default for CaptchaSolver {
    fn default() -> Self {
        Self::new().expect("Failed to create default CaptchaSolver")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    fn create_test_image(width: u32, height: u32) -> CaptchaImage {
        let img = DynamicImage::new_rgb8(width, height);
        CaptchaImage {
            original: img,
            preprocessed: None,
        }
    }

    #[test]
    fn test_detect_recaptcha() {
        let solver = CaptchaSolver::new().unwrap();
        let image = create_test_image(300, 300);
        assert_eq!(solver.detect_type(&image), CaptchaType::ImageGrid);
    }

    #[test]
    fn test_detect_slider() {
        let solver = CaptchaSolver::new().unwrap();
        let image = create_test_image(400, 100);
        assert_eq!(solver.detect_type(&image), CaptchaType::Slider);
    }

    #[test]
    fn test_detect_rotation() {
        let solver = CaptchaSolver::new().unwrap();
        let image = create_test_image(250, 250);
        assert_eq!(solver.detect_type(&image), CaptchaType::Rotation);
    }

    #[test]
    fn test_detect_text() {
        let solver = CaptchaSolver::new().unwrap();
        let image = create_test_image(200, 60);
        assert_eq!(solver.detect_type(&image), CaptchaType::Text);
    }

    #[test]
    fn test_metrics_tracking() {
        let mut solver = CaptchaSolver::new().unwrap();

        let result = SolverResult {
            captcha_type: CaptchaType::Text,
            solution: Solution::Text("test".to_string()),
            confidence: 0.9,
            solve_time: Duration::from_millis(500),
            used_fallback: false,
        };

        solver.update_metrics(&result, true);

        assert_eq!(solver.metrics.total_attempts, 1);
        assert_eq!(solver.metrics.successful_solves, 1);
        assert_eq!(solver.success_rate(), 1.0);
    }
}
