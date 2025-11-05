/// Integration between CAPTCHA solver and RL agent
///
/// This module provides seamless integration between:
/// 1. RL Agent (argus-rl) - Behavioral evasion to prevent detection
/// 2. CAPTCHA Solver (argus-captcha) - Solving CAPTCHAs when detected
///
/// The combined system provides complete bot evasion:
/// - RL agent mimics human behavior to avoid triggering CAPTCHAs
/// - When CAPTCHA appears, solver automatically handles it
/// - Continues normal operation after solving
use anyhow::{bail, Context as AnyhowContext, Result};
use chromiumoxide::Page;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::{CaptchaImage, CaptchaSolver, CaptchaType, Solution, SolverResult};

/// Integrated agent with RL behavioral evasion and CAPTCHA solving
pub struct IntegratedBotAgent {
    /// CAPTCHA solver
    captcha_solver: CaptchaSolver,

    /// Configuration
    config: IntegrationConfig,

    /// Metrics
    metrics: IntegrationMetrics,
}

/// Configuration for integrated agent
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Automatically solve CAPTCHAs when detected
    pub auto_solve_captcha: bool,

    /// Maximum CAPTCHA solve attempts before giving up
    pub max_captcha_attempts: u32,

    /// Wait time after solving CAPTCHA before continuing
    pub post_captcha_wait: Duration,

    /// Enable detailed logging
    pub debug: bool,

    /// Retry failed CAPTCHAs with different strategy
    pub enable_fallback: bool,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            auto_solve_captcha: true,
            max_captcha_attempts: 3,
            post_captcha_wait: Duration::from_secs(2),
            debug: false,
            enable_fallback: true,
        }
    }
}

/// Metrics for integrated agent
#[derive(Debug, Default, Clone)]
pub struct IntegrationMetrics {
    pub total_captchas_encountered: u64,
    pub captchas_solved: u64,
    pub captchas_failed: u64,
    pub total_solve_time: Duration,
    pub avg_solve_time: Duration,
}

impl IntegratedBotAgent {
    /// Create new integrated agent
    pub fn new(config: IntegrationConfig) -> Result<Self> {
        let captcha_solver = CaptchaSolver::with_config(crate::SolverConfig {
            enable_audio: true,
            enable_fallback: config.enable_fallback,
            ..Default::default()
        })?;

        Ok(Self {
            captcha_solver,
            config,
            metrics: IntegrationMetrics::default(),
        })
    }

    /// Check if CAPTCHA is present on page
    pub async fn check_for_captcha(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        debug!("Checking for CAPTCHA...");

        // Check for reCAPTCHA v2
        let recaptcha = self.detect_recaptcha(page).await?;
        if recaptcha.is_some() {
            return Ok(recaptcha);
        }

        // Check for hCaptcha
        let hcaptcha = self.detect_hcaptcha(page).await?;
        if hcaptcha.is_some() {
            return Ok(hcaptcha);
        }

        // Check for text CAPTCHA
        let text_captcha = self.detect_text_captcha(page).await?;
        if text_captcha.is_some() {
            return Ok(text_captcha);
        }

        // Check for slider CAPTCHA
        let slider = self.detect_slider_captcha(page).await?;
        if slider.is_some() {
            return Ok(slider);
        }

        // Check for Cloudflare challenge
        let cloudflare = self.detect_cloudflare(page).await?;
        if cloudflare.is_some() {
            return Ok(cloudflare);
        }

        Ok(None)
    }

    /// Handle CAPTCHA on page (detect and solve)
    pub async fn handle_captcha(&mut self, page: &Page) -> Result<CaptchaOutcome> {
        let start_time = Instant::now();

        // Detect CAPTCHA type
        let detection = match self.check_for_captcha(page).await? {
            Some(d) => d,
            None => {
                debug!("No CAPTCHA detected");
                return Ok(CaptchaOutcome {
                    was_present: false,
                    solved: false,
                    captcha_type: None,
                    attempts: 0,
                    solve_time: Duration::ZERO,
                });
            }
        };

        info!("CAPTCHA detected: {:?}", detection.captcha_type);
        self.metrics.total_captchas_encountered += 1;

        if !self.config.auto_solve_captcha {
            warn!("Auto-solve disabled, skipping CAPTCHA");
            return Ok(CaptchaOutcome {
                was_present: true,
                solved: false,
                captcha_type: Some(detection.captcha_type),
                attempts: 0,
                solve_time: start_time.elapsed(),
            });
        }

        // Attempt to solve
        let mut attempts = 0;
        let mut solved = false;

        while attempts < self.config.max_captcha_attempts && !solved {
            attempts += 1;
            info!(
                "CAPTCHA solve attempt {}/{}",
                attempts, self.config.max_captcha_attempts
            );

            match self.solve_captcha_by_type(page, &detection).await {
                Ok(true) => {
                    info!("✓ CAPTCHA solved successfully");
                    solved = true;
                    self.metrics.captchas_solved += 1;
                }
                Ok(false) => {
                    warn!("CAPTCHA solution submitted but verification failed");
                }
                Err(e) => {
                    error!("CAPTCHA solve error: {}", e);
                }
            }

            if !solved && attempts < self.config.max_captcha_attempts {
                // Wait before retry
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }

        if !solved {
            self.metrics.captchas_failed += 1;
        }

        let solve_time = start_time.elapsed();
        self.metrics.total_solve_time += solve_time;

        // Update average
        let total_attempts = self.metrics.captchas_solved + self.metrics.captchas_failed;
        if total_attempts > 0 {
            self.metrics.avg_solve_time = self.metrics.total_solve_time / total_attempts as u32;
        }

        // Wait after solving
        if solved {
            tokio::time::sleep(self.config.post_captcha_wait).await;
        }

        Ok(CaptchaOutcome {
            was_present: true,
            solved,
            captcha_type: Some(detection.captcha_type),
            attempts,
            solve_time,
        })
    }

    /// Solve CAPTCHA based on detected type
    async fn solve_captcha_by_type(
        &mut self,
        page: &Page,
        detection: &CaptchaDetection,
    ) -> Result<bool> {
        match detection.captcha_type {
            CaptchaType::ImageGrid => self.solve_recaptcha_v2(page).await,
            CaptchaType::Text => self.solve_text_captcha(page).await,
            CaptchaType::Audio => self.solve_audio_captcha(page).await,
            CaptchaType::Slider => self.solve_slider_captcha(page).await,
            CaptchaType::Rotation => self.solve_rotation_captcha(page).await,
            _ => bail!("Unsupported CAPTCHA type: {:?}", detection.captcha_type),
        }
    }

    /// Detect reCAPTCHA v2
    async fn detect_recaptcha(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        let js = r#"
            document.querySelector('iframe[src*="recaptcha"]') !== null ||
            document.querySelector('.g-recaptcha') !== null ||
            document.querySelector('[data-sitekey]') !== null
        "#;

        let detected = page.evaluate(js).await?.into_value::<bool>()?;

        if detected {
            Ok(Some(CaptchaDetection {
                captcha_type: CaptchaType::ImageGrid,
                selector: Some("iframe[src*=\"recaptcha\"]".to_string()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect hCaptcha
    async fn detect_hcaptcha(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        let js = "document.querySelector('.h-captcha') !== null";
        let detected = page.evaluate(js).await?.into_value::<bool>()?;

        if detected {
            Ok(Some(CaptchaDetection {
                captcha_type: CaptchaType::ImageGrid,
                selector: Some(".h-captcha".to_string()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect text CAPTCHA
    async fn detect_text_captcha(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        let js = r#"
            document.querySelector('img[alt*="CAPTCHA"]') !== null ||
            document.querySelector('img.captcha-image') !== null ||
            document.querySelector('input[name="captcha"]') !== null
        "#;

        let detected = page.evaluate(js).await?.into_value::<bool>()?;

        if detected {
            Ok(Some(CaptchaDetection {
                captcha_type: CaptchaType::Text,
                selector: Some("img.captcha-image".to_string()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect slider CAPTCHA
    async fn detect_slider_captcha(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        let js = r#"
            document.querySelector('.slider-captcha') !== null ||
            document.querySelector('.geetest_slider') !== null
        "#;

        let detected = page.evaluate(js).await?.into_value::<bool>()?;

        if detected {
            Ok(Some(CaptchaDetection {
                captcha_type: CaptchaType::Slider,
                selector: Some(".slider-captcha".to_string()),
            }))
        } else {
            Ok(None)
        }
    }

    /// Detect Cloudflare challenge
    async fn detect_cloudflare(&self, page: &Page) -> Result<Option<CaptchaDetection>> {
        let js = r#"
            document.title.toLowerCase().includes('just a moment') ||
            document.querySelector('#cf-challenge-running') !== null ||
            document.querySelector('.cf-browser-verification') !== null
        "#;

        let detected = page.evaluate(js).await?.into_value::<bool>()?;

        if detected {
            info!("Cloudflare challenge detected - waiting for clearance...");
            // Cloudflare often clears automatically, just wait
            tokio::time::sleep(Duration::from_secs(5)).await;

            Ok(Some(CaptchaDetection {
                captcha_type: CaptchaType::Unknown,
                selector: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Solve reCAPTCHA v2
    async fn solve_recaptcha_v2(&mut self, page: &Page) -> Result<bool> {
        info!("Solving reCAPTCHA v2...");

        // Click checkbox first
        let checkbox_js = r#"
            const checkbox = document.querySelector('.recaptcha-checkbox-border');
            if (checkbox) checkbox.click();
        "#;
        page.evaluate(checkbox_js).await?;

        // Wait for challenge or success
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Check if challenge appeared
        let has_challenge = page
            .evaluate("document.querySelector('.rc-imageselect-challenge') !== null")
            .await?
            .into_value::<bool>()?;

        if !has_challenge {
            info!("✓ Passed reCAPTCHA without challenge");
            return Ok(true);
        }

        // Extract query
        let query: String = page
            .evaluate("document.querySelector('.rc-imageselect-desc-wrapper').innerText")
            .await?
            .into_value()?;

        info!("Challenge query: \"{}\"", query);

        // Extract grid images
        let images = self.extract_recaptcha_images(page).await?;

        // Solve with CAPTCHA solver
        let result = self.captcha_solver.solve_image_grid(images, &query)?;

        if let Solution::ImageIndices(indices) = result.solution {
            info!(
                "Solution: select tiles {:?} (confidence: {:.2}%)",
                indices,
                result.confidence * 100.0
            );

            // Click selected tiles
            for &idx in &indices {
                let script = format!(
                    "document.querySelectorAll('.rc-imageselect-tile')[{}].click()",
                    idx
                );
                page.evaluate(&script).await?;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }

            // Submit
            page.evaluate("document.querySelector('.rc-imageselect-verify-button').click()")
                .await?;

            // Wait for verification
            tokio::time::sleep(Duration::from_secs(2)).await;

            // Check if solved
            let still_has_challenge = page
                .evaluate("document.querySelector('.rc-imageselect-challenge') !== null")
                .await?
                .into_value::<bool>()?;

            self.captcha_solver
                .update_metrics(&result, !still_has_challenge);

            Ok(!still_has_challenge)
        } else {
            bail!("Unexpected solution type from image grid solver");
        }
    }

    /// Extract reCAPTCHA grid images
    async fn extract_recaptcha_images(&self, page: &Page) -> Result<Vec<CaptchaImage>> {
        // Get image count
        let count: usize = page
            .evaluate("document.querySelectorAll('.rc-image-tile-wrapper img').length")
            .await?
            .into_value()?;

        let mut images = Vec::new();

        for i in 0..count {
            // Get image data URL
            let script = format!(
                r#"
                (function() {{
                    const img = document.querySelectorAll('.rc-image-tile-wrapper img')[{}];
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    return canvas.toDataURL('image/png');
                }})()
                "#,
                i
            );

            let data_url: String = page.evaluate(&script).await?.into_value()?;

            // Decode data URL to image
            let img = self.decode_data_url(&data_url)?;
            images.push(CaptchaImage {
                original: img,
                preprocessed: None,
            });
        }

        Ok(images)
    }

    /// Solve text CAPTCHA
    async fn solve_text_captcha(&mut self, page: &Page) -> Result<bool> {
        info!("Solving text CAPTCHA...");

        // Extract CAPTCHA image
        let data_url: String = page
            .evaluate("document.querySelector('img.captcha-image').src")
            .await?
            .into_value()?;

        let img = self.decode_data_url(&data_url)?;
        let captcha = CaptchaImage {
            original: img,
            preprocessed: None,
        };

        // Solve
        let result = self.captcha_solver.solve_text(captcha)?;

        if let Solution::Text(text) = &result.solution {
            info!(
                "Solution: \"{}\" (confidence: {:.2}%)",
                text,
                result.confidence * 100.0
            );

            // Fill input
            let script = format!(
                "document.querySelector('input[name=\"captcha\"]').value = '{}'",
                text
            );
            page.evaluate(&script).await?;

            // Submit
            page.evaluate("document.querySelector('form').submit()")
                .await?;

            tokio::time::sleep(Duration::from_secs(1)).await;

            self.captcha_solver.update_metrics(&result, true);
            Ok(true)
        } else {
            bail!("Unexpected solution type from text solver");
        }
    }

    /// Solve audio CAPTCHA
    async fn solve_audio_captcha(&mut self, page: &Page) -> Result<bool> {
        info!("Solving audio CAPTCHA...");

        // Click audio button
        page.evaluate("document.querySelector('.rc-audiochallenge-tdownload-link').click()")
            .await?;

        tokio::time::sleep(Duration::from_secs(1)).await;

        // Get audio URL
        let audio_url: String = page
            .evaluate("document.querySelector('.rc-audiochallenge-tdownload-link').href")
            .await?
            .into_value()?;

        // Download audio
        let audio_bytes = reqwest::get(&audio_url).await?.bytes().await?.to_vec();

        // Solve
        let result = self.captcha_solver.solve_audio(&audio_bytes).await?;

        if let Solution::AudioText(text) = &result.solution {
            info!(
                "Transcription: \"{}\" (confidence: {:.2}%)",
                text,
                result.confidence * 100.0
            );

            // Fill input
            let script = format!(
                "document.querySelector('#audio-response').value = '{}'",
                text
            );
            page.evaluate(&script).await?;

            // Submit
            page.evaluate("document.querySelector('#recaptcha-verify-button').click()")
                .await?;

            tokio::time::sleep(Duration::from_secs(2)).await;

            self.captcha_solver.update_metrics(&result, true);
            Ok(true)
        } else {
            bail!("Unexpected solution type from audio solver");
        }
    }

    /// Solve slider CAPTCHA
    async fn solve_slider_captcha(&mut self, page: &Page) -> Result<bool> {
        info!("Solving slider CAPTCHA...");

        // Extract images (implementation depends on slider type)
        // This is a simplified example
        let background = self.extract_element_screenshot(page, ".slider-bg").await?;
        let piece = self
            .extract_element_screenshot(page, ".slider-piece")
            .await?;

        // Solve
        let result = self.captcha_solver.solve_slider(background, piece)?;

        if let Solution::SliderOffset(offset) = result.solution {
            info!(
                "Solution: slide {} pixels (confidence: {:.2}%)",
                offset,
                result.confidence * 100.0
            );

            // Simulate drag
            let script = format!(
                r#"
                const slider = document.querySelector('.slider-handle');
                const event = new MouseEvent('mousedown', {{ clientX: 0 }});
                slider.dispatchEvent(event);

                setTimeout(() => {{
                    const moveEvent = new MouseEvent('mousemove', {{ clientX: {} }});
                    document.dispatchEvent(moveEvent);

                    setTimeout(() => {{
                        const upEvent = new MouseEvent('mouseup');
                        document.dispatchEvent(upEvent);
                    }}, 100);
                }}, 100);
                "#,
                offset
            );
            page.evaluate(&script).await?;

            tokio::time::sleep(Duration::from_secs(1)).await;

            self.captcha_solver.update_metrics(&result, true);
            Ok(true)
        } else {
            bail!("Unexpected solution type from slider solver");
        }
    }

    /// Solve rotation CAPTCHA
    async fn solve_rotation_captcha(&mut self, page: &Page) -> Result<bool> {
        info!("Solving rotation CAPTCHA...");

        let image = self
            .extract_element_screenshot(page, ".rotation-image")
            .await?;
        let result = self.captcha_solver.solve_rotation(image)?;

        if let Solution::RotationAngle(angle) = result.solution {
            info!(
                "Solution: rotate {:.1}° (confidence: {:.2}%)",
                angle,
                result.confidence * 100.0
            );

            // Apply rotation (implementation varies)
            let script = format!(
                "document.querySelector('.rotation-slider').value = {}",
                angle
            );
            page.evaluate(&script).await?;

            self.captcha_solver.update_metrics(&result, true);
            Ok(true)
        } else {
            bail!("Unexpected solution type from rotation solver");
        }
    }

    /// Decode data URL to image
    fn decode_data_url(&self, data_url: &str) -> Result<image::DynamicImage> {
        if !data_url.starts_with("data:image") {
            bail!("Invalid data URL");
        }

        let parts: Vec<&str> = data_url.split(',').collect();
        if parts.len() != 2 {
            bail!("Invalid data URL format");
        }

        let bytes = base64::decode(parts[1])?;
        let img = image::load_from_memory(&bytes)?;

        Ok(img)
    }

    /// Extract element screenshot
    async fn extract_element_screenshot(
        &self,
        page: &Page,
        selector: &str,
    ) -> Result<CaptchaImage> {
        // Take screenshot of element
        let script = format!(
            r#"
            (function() {{
                const el = document.querySelector('{}');
                const canvas = document.createElement('canvas');
                const rect = el.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;
                const ctx = canvas.getContext('2d');
                // Note: This won't work for cross-origin images
                // In production, use proper screenshot API
                return canvas.toDataURL('image/png');
            }})()
            "#,
            selector
        );

        let data_url: String = page.evaluate(&script).await?.into_value()?;
        let img = self.decode_data_url(&data_url)?;

        Ok(CaptchaImage {
            original: img,
            preprocessed: None,
        })
    }

    /// Get metrics
    pub fn metrics(&self) -> &IntegrationMetrics {
        &self.metrics
    }

    /// Get CAPTCHA solver metrics
    pub fn captcha_solver_metrics(&self) -> &crate::SolverMetrics {
        self.captcha_solver.metrics()
    }
}

/// CAPTCHA detection information
#[derive(Debug, Clone)]
pub struct CaptchaDetection {
    pub captcha_type: CaptchaType,
    pub selector: Option<String>,
}

/// Outcome of CAPTCHA handling
#[derive(Debug, Clone)]
pub struct CaptchaOutcome {
    pub was_present: bool,
    pub solved: bool,
    pub captcha_type: Option<CaptchaType>,
    pub attempts: u32,
    pub solve_time: Duration,
}

impl CaptchaOutcome {
    pub fn summary(&self) -> String {
        if !self.was_present {
            return "No CAPTCHA detected".to_string();
        }

        let type_str = self
            .captcha_type
            .as_ref()
            .map(|t| format!("{:?}", t))
            .unwrap_or_else(|| "Unknown".to_string());

        if self.solved {
            format!(
                "✓ {} CAPTCHA solved in {} attempts ({:.1}s)",
                type_str,
                self.attempts,
                self.solve_time.as_secs_f32()
            )
        } else {
            format!(
                "✗ {} CAPTCHA failed after {} attempts ({:.1}s)",
                type_str,
                self.attempts,
                self.solve_time.as_secs_f32()
            )
        }
    }
}
