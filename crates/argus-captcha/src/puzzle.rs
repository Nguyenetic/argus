/// Puzzle CAPTCHA solver
///
/// Solves various types of puzzle CAPTCHAs:
/// - Slider CAPTCHAs (slide piece to fit)
/// - Rotation CAPTCHAs (rotate image upright)
/// - Jigsaw CAPTCHAs (drag piece to correct position)
use crate::common::{BoundingBox, CaptchaImage, Point};
use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma};
use imageproc::edges::canny;
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use imageproc::hough::{detect_lines, LineDetectionOptions, PolarLine};
use std::f32::consts::PI;
use tracing::{debug, info};

/// Puzzle CAPTCHA solver
pub struct PuzzleSolver {
    config: PuzzleConfig,
}

/// Puzzle solver configuration
#[derive(Debug, Clone)]
pub struct PuzzleConfig {
    /// Edge detection low threshold
    pub canny_low: f32,

    /// Edge detection high threshold
    pub canny_high: f32,

    /// Similarity threshold for template matching
    pub similarity_threshold: f32,

    /// Angle step for rotation search (degrees)
    pub angle_step: f32,
}

impl Default for PuzzleConfig {
    fn default() -> Self {
        Self {
            canny_low: 50.0,
            canny_high: 150.0,
            similarity_threshold: 0.7,
            angle_step: 1.0,
        }
    }
}

/// Slider CAPTCHA solution
#[derive(Debug, Clone)]
pub struct SliderSolution {
    /// X-offset to slide to
    pub x_offset: i32,

    /// Confidence score (0-1)
    pub confidence: f32,
}

/// Rotation CAPTCHA solution
#[derive(Debug, Clone)]
pub struct RotationSolution {
    /// Angle to rotate (degrees)
    pub angle: f32,

    /// Confidence score (0-1)
    pub confidence: f32,
}

impl PuzzleSolver {
    pub fn new(config: PuzzleConfig) -> Self {
        Self { config }
    }

    /// Solve slider CAPTCHA
    ///
    /// # Arguments
    /// * `background` - Background image with missing piece
    /// * `puzzle_piece` - Piece to fit into background
    ///
    /// # Returns
    /// X-offset to slide the piece to
    pub fn solve_slider(
        &self,
        background: &CaptchaImage,
        puzzle_piece: &CaptchaImage,
    ) -> Result<SliderSolution> {
        info!(
            "Solving slider CAPTCHA (bg: {}x{}, piece: {}x{})",
            background.width(),
            background.height(),
            puzzle_piece.width(),
            puzzle_piece.height()
        );

        // Convert to edge maps
        let bg_edges = self.detect_edges(&background.original)?;
        let piece_edges = self.detect_edges(&puzzle_piece.original)?;

        // Template matching across x-axis
        let (x_offset, confidence) = self.find_best_match(&bg_edges, &piece_edges)?;

        info!(
            "Found match at x={} with confidence={:.3}",
            x_offset, confidence
        );

        Ok(SliderSolution {
            x_offset,
            confidence,
        })
    }

    /// Solve rotation CAPTCHA
    ///
    /// Finds the angle to rotate an image to make it upright
    pub fn solve_rotation(&self, image: &CaptchaImage) -> Result<RotationSolution> {
        info!(
            "Solving rotation CAPTCHA ({}x{})",
            image.width(),
            image.height()
        );

        // Detect edges
        let edges = self.detect_edges(&image.original)?;

        // Detect lines using Hough transform
        let lines = self.detect_lines(&edges)?;

        if lines.is_empty() {
            debug!("No lines detected, trying alternative method");
            return self.solve_rotation_alternative(image);
        }

        // Calculate dominant angle
        let angles: Vec<f32> = lines.iter().map(|line| line.angle_in_degrees).collect();
        let dominant_angle = self.find_dominant_angle(&angles)?;

        // Calculate rotation needed to make upright
        let rotation = self.calculate_rotation_to_upright(dominant_angle);

        info!(
            "Dominant angle: {:.1}°, rotation needed: {:.1}°",
            dominant_angle, rotation
        );

        Ok(RotationSolution {
            angle: rotation,
            confidence: 0.8, // TODO: Calculate actual confidence
        })
    }

    /// Alternative rotation solving using content-based analysis
    fn solve_rotation_alternative(&self, image: &CaptchaImage) -> Result<RotationSolution> {
        // Try multiple angles and find the one that looks most "upright"
        let mut best_angle = 0.0;
        let mut best_score = f32::NEG_INFINITY;

        for angle in (0..360).step_by(self.config.angle_step as usize) {
            let angle_f = angle as f32;
            let score = self.calculate_uprightness_score(image, angle_f)?;

            if score > best_score {
                best_score = score;
                best_angle = angle_f;
            }
        }

        Ok(RotationSolution {
            angle: best_angle,
            confidence: 0.6,
        })
    }

    /// Calculate how "upright" an image appears at a given angle
    fn calculate_uprightness_score(&self, image: &CaptchaImage, angle: f32) -> Result<f32> {
        // Rotate image
        let rotated = self.rotate_image(&image.original, angle)?;

        // Calculate score based on:
        // 1. Horizontal edge density (more horizontal edges = more upright)
        // 2. Vertical symmetry

        let edges = self.detect_edges(&rotated)?;
        let horizontal_score = self.calculate_horizontal_edge_ratio(&edges);

        Ok(horizontal_score)
    }

    /// Calculate ratio of horizontal edges
    fn calculate_horizontal_edge_ratio(&self, edges: &GrayImage) -> f32 {
        let (width, height) = edges.dimensions();
        let mut horizontal_count = 0;
        let mut total_count = 0;

        // Sample rows
        for y in (0..height).step_by(10) {
            let mut consecutive = 0;
            for x in 0..width {
                let pixel = edges.get_pixel(x, y)[0];
                if pixel > 128 {
                    consecutive += 1;
                    total_count += 1;
                } else {
                    if consecutive > 5 {
                        horizontal_count += consecutive;
                    }
                    consecutive = 0;
                }
            }
        }

        if total_count == 0 {
            return 0.0;
        }

        horizontal_count as f32 / total_count as f32
    }

    /// Detect edges using Canny algorithm
    fn detect_edges(&self, image: &DynamicImage) -> Result<GrayImage> {
        let gray = image.to_luma8();
        Ok(canny(&gray, self.config.canny_low, self.config.canny_high))
    }

    /// Detect lines using Hough transform
    fn detect_lines(&self, edges: &GrayImage) -> Result<Vec<PolarLine>> {
        let options = LineDetectionOptions {
            vote_threshold: 100,
            suppression_radius: 8,
        };

        let lines = detect_lines(edges, options);
        Ok(lines)
    }

    /// Find best matching position for template
    fn find_best_match(&self, background: &GrayImage, template: &GrayImage) -> Result<(i32, f32)> {
        let (bg_width, bg_height) = background.dimensions();
        let (tmpl_width, tmpl_height) = template.dimensions();

        if tmpl_width > bg_width || tmpl_height > bg_height {
            anyhow::bail!("Template larger than background");
        }

        let mut best_match = (0, f32::MAX);

        // Search across x-axis (assuming y is fixed)
        for x in 0..=(bg_width - tmpl_width) {
            let similarity = self.calculate_similarity(background, template, x, 0)?;

            if similarity < best_match.1 {
                best_match = (x as i32, similarity);
            }
        }

        // Convert distance to confidence (lower distance = higher confidence)
        let max_distance = (tmpl_width * tmpl_height * 255 * 255) as f32;
        let confidence = 1.0 - (best_match.1 / max_distance).sqrt();

        Ok((best_match.0, confidence))
    }

    /// Calculate similarity between template and region
    fn calculate_similarity(
        &self,
        background: &GrayImage,
        template: &GrayImage,
        offset_x: u32,
        offset_y: u32,
    ) -> Result<f32> {
        let (tmpl_width, tmpl_height) = template.dimensions();

        let mut sum_squared_diff = 0.0;

        for y in 0..tmpl_height {
            for x in 0..tmpl_width {
                let bg_pixel = background.get_pixel(offset_x + x, offset_y + y)[0] as f32;
                let tmpl_pixel = template.get_pixel(x, y)[0] as f32;

                let diff = bg_pixel - tmpl_pixel;
                sum_squared_diff += diff * diff;
            }
        }

        Ok(sum_squared_diff)
    }

    /// Find dominant angle from a list of angles
    fn find_dominant_angle(&self, angles: &[f32]) -> Result<f32> {
        if angles.is_empty() {
            anyhow::bail!("No angles provided");
        }

        // Bin angles into 10-degree buckets
        let num_bins = 36; // 360 / 10
        let mut bins = vec![0; num_bins];

        for &angle in angles {
            let normalized = ((angle % 360.0) + 360.0) % 360.0;
            let bin = (normalized / 10.0) as usize % num_bins;
            bins[bin] += 1;
        }

        // Find bin with most votes
        let max_bin = bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .context("No dominant angle found")?;

        // Return center of bin
        let dominant = (max_bin as f32 * 10.0) + 5.0;

        Ok(dominant)
    }

    /// Calculate rotation needed to make image upright
    fn calculate_rotation_to_upright(&self, current_angle: f32) -> f32 {
        // Normalize to [-180, 180]
        let mut angle = current_angle % 360.0;
        if angle > 180.0 {
            angle -= 360.0;
        }

        // Snap to nearest 90-degree orientation
        let snapped = if angle.abs() < 45.0 {
            0.0
        } else if angle > 0.0 && angle < 135.0 {
            90.0
        } else if angle < 0.0 && angle > -135.0 {
            -90.0
        } else {
            180.0
        };

        // Return rotation needed
        snapped - angle
    }

    /// Rotate image by angle
    fn rotate_image(&self, image: &DynamicImage, angle: f32) -> Result<DynamicImage> {
        let gray = image.to_luma8();
        let (width, height) = gray.dimensions();

        let center = ((width / 2) as f32, (height / 2) as f32);
        let angle_rad = angle * PI / 180.0;

        let rotated = rotate_about_center(
            &gray,
            center,
            angle_rad,
            Interpolation::Bilinear,
            Luma([255u8]),
        );

        Ok(DynamicImage::ImageLuma8(rotated))
    }

    /// Solve jigsaw CAPTCHA
    pub fn solve_jigsaw(&self, background: &CaptchaImage, piece: &CaptchaImage) -> Result<Point> {
        // Similar to slider, but search in 2D
        info!("Solving jigsaw CAPTCHA");

        let bg_edges = self.detect_edges(&background.original)?;
        let piece_edges = self.detect_edges(&piece.original)?;

        let (bg_width, bg_height) = bg_edges.dimensions();
        let (piece_width, piece_height) = piece_edges.dimensions();

        let mut best_match = (0, 0, f32::MAX);

        // Search entire background
        for y in 0..=(bg_height - piece_height) {
            for x in 0..=(bg_width - piece_width) {
                let similarity = self.calculate_similarity(&bg_edges, &piece_edges, x, y)?;

                if similarity < best_match.2 {
                    best_match = (x, y, similarity);
                }
            }
        }

        Ok(Point::new(best_match.0 as f32, best_match.1 as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puzzle_config_default() {
        let config = PuzzleConfig::default();
        assert_eq!(config.canny_low, 50.0);
        assert_eq!(config.canny_high, 150.0);
        assert_eq!(config.similarity_threshold, 0.7);
    }

    #[test]
    fn test_calculate_rotation_to_upright() {
        let solver = PuzzleSolver::new(PuzzleConfig::default());

        // Nearly upright
        assert!((solver.calculate_rotation_to_upright(5.0) - (-5.0)).abs() < 0.1);

        // 90 degrees off
        assert!((solver.calculate_rotation_to_upright(95.0) - (-5.0)).abs() < 0.1);

        // 180 degrees
        assert!((solver.calculate_rotation_to_upright(185.0) - (-5.0)).abs() < 0.1);
    }

    #[test]
    fn test_find_dominant_angle() {
        let solver = PuzzleSolver::new(PuzzleConfig::default());

        let angles = vec![45.0, 46.0, 44.0, 47.0, 90.0];
        let dominant = solver.find_dominant_angle(&angles).unwrap();

        // Should be around 45 degrees
        assert!((dominant - 45.0).abs() < 10.0);
    }

    #[test]
    fn test_calculate_horizontal_edge_ratio() {
        let solver = PuzzleSolver::new(PuzzleConfig::default());

        // Create test image with horizontal lines
        let mut img = GrayImage::new(100, 100);
        for y in (0..100).step_by(10) {
            for x in 0..100 {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }

        let ratio = solver.calculate_horizontal_edge_ratio(&img);
        assert!(ratio > 0.0);
    }
}
