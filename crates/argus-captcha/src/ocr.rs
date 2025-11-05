/// OCR-based text CAPTCHA solver
///
/// Handles simple text CAPTCHAs using:
/// - Tesseract OCR for character recognition
/// - Image preprocessing for noise removal
/// - Custom character segmentation
/// - CNN-based character classification
use crate::common::{BoundingBox, CaptchaImage};
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use imageproc::contrast::threshold;
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate;
use std::collections::HashMap;
use tch::{nn, nn::Module, Device, Tensor};
use tracing::{debug, info};

/// OCR-based text CAPTCHA solver
pub struct OcrSolver {
    tesseract: Option<tesseract::Tesseract>,
    cnn_model: Option<CharacterCNN>,
    config: OcrConfig,
}

/// OCR configuration
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Use Tesseract OCR
    pub use_tesseract: bool,

    /// Use CNN for character recognition
    pub use_cnn: bool,

    /// Minimum confidence threshold
    pub confidence_threshold: f32,

    /// Apply aggressive preprocessing
    pub aggressive_preprocess: bool,

    /// Expected character set
    pub charset: String,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            use_tesseract: true,
            use_cnn: false,
            confidence_threshold: 0.6,
            aggressive_preprocess: true,
            charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".to_string(),
        }
    }
}

impl OcrSolver {
    pub fn new(config: OcrConfig) -> Result<Self> {
        let tesseract = if config.use_tesseract {
            Some(Self::init_tesseract()?)
        } else {
            None
        };

        let cnn_model = if config.use_cnn {
            Some(CharacterCNN::new()?)
        } else {
            None
        };

        Ok(Self {
            tesseract,
            cnn_model,
            config,
        })
    }

    fn init_tesseract() -> Result<tesseract::Tesseract> {
        let mut tess = tesseract::Tesseract::new(None, Some("eng"))?;

        // Configure for CAPTCHA-like text
        tess.set_variable(
            "tessedit_char_whitelist",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )?;
        tess.set_variable("tessedit_pageseg_mode", "7")?; // Single line
        tess.set_variable("load_system_dawg", "0")?;
        tess.set_variable("load_freq_dawg", "0")?;

        Ok(tess)
    }

    /// Solve text CAPTCHA
    pub fn solve(&mut self, image: &mut CaptchaImage) -> Result<String> {
        info!(
            "Solving text CAPTCHA ({}x{})",
            image.width(),
            image.height()
        );

        // Preprocess image
        let preprocessed = self.preprocess_for_ocr(&image.original)?;

        // Try multiple methods
        let mut results = Vec::new();

        // Method 1: Tesseract OCR
        if let Some(ref mut tess) = self.tesseract {
            if let Ok(text) = self.solve_with_tesseract(&preprocessed, tess) {
                debug!("Tesseract result: {}", text);
                results.push(text);
            }
        }

        // Method 2: Character segmentation + CNN
        if self.cnn_model.is_some() {
            if let Ok(text) = self.solve_with_segmentation(&preprocessed) {
                debug!("Segmentation result: {}", text);
                results.push(text);
            }
        }

        // Method 3: Brute force with variations
        if results.is_empty() {
            if let Ok(text) = self.solve_with_variations(&image.original) {
                debug!("Variation result: {}", text);
                results.push(text);
            }
        }

        // Return most confident result
        if !results.is_empty() {
            // For now, return first result
            // In production, you'd implement confidence scoring
            Ok(results[0].clone())
        } else {
            anyhow::bail!("Could not solve text CAPTCHA")
        }
    }

    /// Preprocess image specifically for OCR
    fn preprocess_for_ocr(&self, image: &DynamicImage) -> Result<DynamicImage> {
        let mut img = image.clone();

        // Convert to grayscale
        img = img.grayscale();

        // Resize for better recognition
        let (w, h) = img.dimensions();
        if w < 200 {
            let scale = 200.0 / w as f32;
            img = img.resize(
                (w as f32 * scale) as u32,
                (h as f32 * scale) as u32,
                image::imageops::FilterType::Lanczos3,
            );
        }

        if self.config.aggressive_preprocess {
            // Apply threshold
            let gray = img.to_luma8();
            let binary = threshold(&gray, 128);
            img = DynamicImage::ImageLuma8(binary);

            // Remove noise
            img = self.remove_noise(&img)?;
        }

        Ok(img)
    }

    /// Remove noise from image
    fn remove_noise(&self, image: &DynamicImage) -> Result<DynamicImage> {
        use imageproc::distance_transform::Norm;
        use imageproc::morphology::{close, open};

        let gray = image.to_luma8();

        // Morphological operations
        let opened = open(&gray, Norm::LInf, 1);
        let closed = close(&opened, Norm::LInf, 1);

        Ok(DynamicImage::ImageLuma8(closed))
    }

    /// Solve using Tesseract OCR
    fn solve_with_tesseract(
        &self,
        image: &DynamicImage,
        tess: &mut tesseract::Tesseract,
    ) -> Result<String> {
        // Convert to format Tesseract can use
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();

        // Set image
        tess.set_image_from_mem(gray.as_raw(), width, height, 1, width)?;

        // Get text
        let text = tess.get_text()?.trim().to_uppercase();

        // Filter based on charset
        let filtered: String = text
            .chars()
            .filter(|c| self.config.charset.contains(*c))
            .collect();

        if filtered.is_empty() {
            anyhow::bail!("No valid characters found");
        }

        Ok(filtered)
    }

    /// Solve using character segmentation
    fn solve_with_segmentation(&self, image: &DynamicImage) -> Result<String> {
        // Segment individual characters
        let segments = self.segment_characters(image)?;

        if segments.is_empty() {
            anyhow::bail!("Could not segment characters");
        }

        // Recognize each character
        let mut result = String::new();
        for segment in segments {
            if let Some(ref model) = self.cnn_model {
                let char = model.predict(&segment)?;
                result.push(char);
            }
        }

        Ok(result)
    }

    /// Segment individual characters
    fn segment_characters(&self, image: &DynamicImage) -> Result<Vec<DynamicImage>> {
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();

        // Find vertical projections
        let mut projection = vec![0u32; width as usize];
        for x in 0..width {
            for y in 0..height {
                let pixel = gray.get_pixel(x, y);
                if pixel[0] < 128 {
                    // Black pixel
                    projection[x as usize] += 1;
                }
            }
        }

        // Find character boundaries
        let mut segments = Vec::new();
        let mut in_char = false;
        let mut start_x = 0;
        let threshold = height / 4; // At least 25% vertical fill

        for (x, &count) in projection.iter().enumerate() {
            if count > threshold && !in_char {
                start_x = x;
                in_char = true;
            } else if count <= threshold && in_char {
                // End of character
                let char_width = x - start_x;
                if char_width > 5 {
                    // Minimum width
                    let segment = image.crop_imm(start_x as u32, 0, char_width as u32, height);
                    segments.push(segment);
                }
                in_char = false;
            }
        }

        Ok(segments)
    }

    /// Try multiple image variations
    fn solve_with_variations(&mut self, image: &DynamicImage) -> Result<String> {
        let variations = vec![
            image.clone(),
            image.brighten(20),
            image.brighten(-20),
            image.adjust_contrast(20.0),
            image.adjust_contrast(-20.0),
        ];

        for variation in variations {
            let preprocessed = self.preprocess_for_ocr(&variation)?;

            if let Some(ref mut tess) = self.tesseract {
                if let Ok(text) = self.solve_with_tesseract(&preprocessed, tess) {
                    if text.len() >= 4 {
                        // Minimum reasonable length
                        return Ok(text);
                    }
                }
            }
        }

        anyhow::bail!("All variations failed")
    }
}

/// CNN-based character classifier
pub struct CharacterCNN {
    vs: nn::VarStore,
    model: nn::Sequential,
}

impl CharacterCNN {
    pub fn new() -> Result<Self> {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        // Simple CNN architecture for character recognition
        let model = nn::seq()
            .add(nn::conv2d(&root / "conv1", 1, 32, 3, Default::default()))
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(2))
            .add(nn::conv2d(&root / "conv2", 32, 64, 3, Default::default()))
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(2))
            .add_fn(|x| x.flat_view())
            .add(nn::linear(
                &root / "fc1",
                64 * 5 * 5,
                128,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(&root / "fc2", 128, 36, Default::default())); // 26 letters + 10 digits

        Ok(Self { vs, model })
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    pub fn predict(&self, image: &DynamicImage) -> Result<char> {
        // Resize to 28x28
        let resized = image.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);
        let gray = resized.to_luma8();

        // Convert to tensor
        let mut data = Vec::with_capacity(28 * 28);
        for pixel in gray.pixels() {
            data.push(pixel[0] as f32 / 255.0);
        }

        let tensor = Tensor::of_slice(&data).view([1, 1, 28, 28]);

        // Forward pass
        let output = self.model.forward(&tensor);
        let prediction = output.argmax(-1, false).int64_value(&[]);

        // Convert to character
        let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let char = charset
            .chars()
            .nth(prediction as usize)
            .context("Invalid prediction index")?;

        Ok(char)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert!(config.use_tesseract);
        assert_eq!(config.confidence_threshold, 0.6);
    }

    #[test]
    fn test_charset_filter() {
        let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let text = "ABC123!@#";
        let filtered: String = text.chars().filter(|c| charset.contains(*c)).collect();
        assert_eq!(filtered, "ABC123");
    }
}
