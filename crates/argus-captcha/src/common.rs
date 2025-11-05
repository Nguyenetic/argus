/// Common types and utilities for CAPTCHA solving
use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

/// 2D point coordinate
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// Bounding box for detected objects
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BoundingBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn center(&self) -> Point {
        Point::new(
            self.x as f32 + self.width as f32 / 2.0,
            self.y as f32 + self.height as f32 / 2.0,
        )
    }

    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    pub fn intersects(&self, other: &BoundingBox) -> bool {
        !(self.x + self.width < other.x
            || other.x + other.width < self.x
            || self.y + self.height < other.y
            || other.y + other.height < self.y)
    }

    pub fn iou(&self, other: &BoundingBox) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);

        let intersection = (x2 - x1) * (y2 - y1);
        let union = self.area() + other.area() - intersection;

        intersection as f32 / union as f32
    }
}

/// Detected object with class and confidence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedObject {
    pub label: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

impl DetectedObject {
    pub fn new(label: String, confidence: f32, bbox: BoundingBox) -> Self {
        Self {
            label,
            confidence,
            bbox,
        }
    }
}

/// CAPTCHA image wrapper with preprocessing
#[derive(Clone)]
pub struct CaptchaImage {
    pub original: DynamicImage,
    pub preprocessed: Option<DynamicImage>,
}

impl CaptchaImage {
    pub fn new(image: DynamicImage) -> Self {
        Self {
            original: image,
            preprocessed: None,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let image = image::load_from_memory(bytes)?;
        Ok(Self::new(image))
    }

    pub fn from_base64(base64: &str) -> Result<Self> {
        let bytes = base64::decode(base64)?;
        Self::from_bytes(&bytes)
    }

    pub fn width(&self) -> u32 {
        self.original.width()
    }

    pub fn height(&self) -> u32 {
        self.original.height()
    }

    /// Convert to grayscale
    pub fn to_grayscale(&self) -> DynamicImage {
        self.original.grayscale()
    }

    /// Apply preprocessing pipeline
    pub fn preprocess(&mut self) -> Result<()> {
        let mut img = self.original.clone();

        // Convert to grayscale
        img = img.grayscale();

        // Resize to standard size (224x224 for most models)
        img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);

        // Normalize brightness
        img = normalize_brightness(&img);

        // Apply denoising
        img = denoise(&img);

        self.preprocessed = Some(img);
        Ok(())
    }

    /// Get preprocessed image or preprocess if not done
    pub fn get_preprocessed(&mut self) -> Result<&DynamicImage> {
        if self.preprocessed.is_none() {
            self.preprocess()?;
        }
        Ok(self.preprocessed.as_ref().unwrap())
    }

    /// Crop region of interest
    pub fn crop(&self, bbox: &BoundingBox) -> DynamicImage {
        self.original
            .crop_imm(bbox.x, bbox.y, bbox.width, bbox.height)
    }

    /// Convert to RGB tensor for PyTorch
    pub fn to_tensor(&self) -> Result<Vec<f32>> {
        let img = self.preprocessed.as_ref().unwrap_or(&self.original);
        let rgb = img.to_rgb8();

        let mut tensor = Vec::with_capacity(3 * 224 * 224);

        // Convert to CHW format (Channels, Height, Width)
        // Normalize to [0, 1]
        for channel in 0..3 {
            for y in 0..224 {
                for x in 0..224 {
                    let pixel = rgb.get_pixel(x, y);
                    tensor.push(pixel[channel] as f32 / 255.0);
                }
            }
        }

        Ok(tensor)
    }
}

/// Normalize image brightness
fn normalize_brightness(img: &DynamicImage) -> DynamicImage {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Calculate average brightness
    let mut sum = 0u64;
    for pixel in gray.pixels() {
        sum += pixel[0] as u64;
    }
    let avg = (sum / (width * height) as u64) as u8;

    // Adjust if too dark or too bright
    if avg < 80 || avg > 180 {
        let adjustment = 128i16 - avg as i16;
        let mut adjusted = gray.clone();

        for pixel in adjusted.pixels_mut() {
            let new_val = (pixel[0] as i16 + adjustment).clamp(0, 255) as u8;
            pixel[0] = new_val;
        }

        DynamicImage::ImageLuma8(adjusted)
    } else {
        DynamicImage::ImageLuma8(gray)
    }
}

/// Simple denoising filter
fn denoise(img: &DynamicImage) -> DynamicImage {
    use imageproc::filter::median_filter;

    let gray = img.to_luma8();
    let filtered = median_filter(&gray, 1, 1);

    DynamicImage::ImageLuma8(filtered)
}

/// Non-maximum suppression for object detection
pub fn non_max_suppression(
    detections: Vec<DetectedObject>,
    iou_threshold: f32,
) -> Vec<DetectedObject> {
    let mut result = Vec::new();
    let mut sorted = detections;

    // Sort by confidence (descending)
    sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    while !sorted.is_empty() {
        let current = sorted.remove(0);
        result.push(current.clone());

        // Remove overlapping detections
        sorted.retain(|det| current.bbox.iou(&det.bbox) < iou_threshold);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert_eq!(p1.distance(&p2), 5.0);
    }

    #[test]
    fn test_bbox_intersection() {
        let box1 = BoundingBox::new(0, 0, 10, 10);
        let box2 = BoundingBox::new(5, 5, 10, 10);
        assert!(box1.intersects(&box2));

        let box3 = BoundingBox::new(20, 20, 10, 10);
        assert!(!box1.intersects(&box3));
    }

    #[test]
    fn test_bbox_iou() {
        let box1 = BoundingBox::new(0, 0, 10, 10);
        let box2 = BoundingBox::new(5, 5, 10, 10);

        let iou = box1.iou(&box2);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BoundingBox::new(10, 10, 20, 20);
        let center = bbox.center();
        assert_eq!(center.x, 20.0);
        assert_eq!(center.y, 20.0);
    }

    #[test]
    fn test_non_max_suppression() {
        let detections = vec![
            DetectedObject::new("car".to_string(), 0.9, BoundingBox::new(0, 0, 10, 10)),
            DetectedObject::new("car".to_string(), 0.8, BoundingBox::new(2, 2, 10, 10)),
            DetectedObject::new("car".to_string(), 0.7, BoundingBox::new(50, 50, 10, 10)),
        ];

        let result = non_max_suppression(detections, 0.3);
        assert_eq!(result.len(), 2); // First two overlap, third is separate
    }
}
