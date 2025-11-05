/// Image CAPTCHA solver (reCAPTCHA v2 style)
///
/// Solves image-based CAPTCHAs like "Select all images with traffic lights"
/// using object detection and classification networks.
use crate::common::{non_max_suppression, BoundingBox, CaptchaImage, DetectedObject};
use anyhow::{Context, Result};
use image::DynamicImage;
use std::collections::HashMap;
use tch::{nn, nn::Module, Device, Kind, Tensor};
use tracing::{debug, info};

/// Image CAPTCHA solver
pub struct ImageCaptchaSolver {
    detector: ObjectDetector,
    classifier: ImageClassifier,
    config: ImageSolverConfig,
}

/// Configuration for image solver
#[derive(Debug, Clone)]
pub struct ImageSolverConfig {
    /// Confidence threshold for detection
    pub detection_threshold: f32,

    /// IoU threshold for NMS
    pub nms_threshold: f32,

    /// Minimum object size (proportion of image)
    pub min_object_size: f32,

    /// Maximum number of detections per image
    pub max_detections: usize,
}

impl Default for ImageSolverConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.5,
            nms_threshold: 0.4,
            min_object_size: 0.05,
            max_detections: 10,
        }
    }
}

impl ImageCaptchaSolver {
    pub fn new(config: ImageSolverConfig) -> Result<Self> {
        let detector = ObjectDetector::new()?;
        let classifier = ImageClassifier::new()?;

        Ok(Self {
            detector,
            classifier,
            config,
        })
    }

    /// Load pre-trained models
    pub fn load_models(&mut self, detector_path: &str, classifier_path: &str) -> Result<()> {
        self.detector.load(detector_path)?;
        self.classifier.load(classifier_path)?;
        Ok(())
    }

    /// Solve reCAPTCHA v2 image challenge
    ///
    /// # Arguments
    /// * `images` - 3x3 or 4x4 grid of images
    /// * `query` - What to find (e.g., "traffic lights", "crosswalks", "bicycles")
    ///
    /// # Returns
    /// Vector of indices to click (0-8 for 3x3, 0-15 for 4x4)
    pub fn solve_grid(&mut self, images: Vec<CaptchaImage>, query: &str) -> Result<Vec<usize>> {
        info!("Solving image grid for query: {}", query);

        let grid_size = images.len();
        if grid_size != 9 && grid_size != 16 {
            anyhow::bail!("Invalid grid size: {}. Expected 9 or 16", grid_size);
        }

        let target_class = self.parse_query(query)?;
        debug!("Target class: {:?}", target_class);

        let mut selected_indices = Vec::new();

        for (idx, mut image) in images.into_iter().enumerate() {
            if self.contains_target(&mut image, &target_class)? {
                selected_indices.push(idx);
            }
        }

        info!("Selected {} images", selected_indices.len());
        Ok(selected_indices)
    }

    /// Parse query text to target classes
    fn parse_query(&self, query: &str) -> Result<Vec<String>> {
        let query_lower = query.to_lowercase();

        let classes = if query_lower.contains("traffic light")
            || query_lower.contains("traffic signal")
        {
            vec!["traffic light".to_string()]
        } else if query_lower.contains("crosswalk") || query_lower.contains("pedestrian crossing") {
            vec!["crosswalk".to_string(), "zebra crossing".to_string()]
        } else if query_lower.contains("bicycle") || query_lower.contains("bike") {
            vec!["bicycle".to_string()]
        } else if query_lower.contains("bus") {
            vec!["bus".to_string()]
        } else if query_lower.contains("car") || query_lower.contains("vehicle") {
            vec!["car".to_string(), "vehicle".to_string()]
        } else if query_lower.contains("motorcycle") || query_lower.contains("motorbike") {
            vec!["motorcycle".to_string()]
        } else if query_lower.contains("fire hydrant") {
            vec!["fire hydrant".to_string()]
        } else if query_lower.contains("parking meter") {
            vec!["parking meter".to_string()]
        } else if query_lower.contains("bridge") {
            vec!["bridge".to_string()]
        } else if query_lower.contains("boat") {
            vec!["boat".to_string(), "ship".to_string()]
        } else if query_lower.contains("palm tree") || query_lower.contains("tree") {
            vec!["tree".to_string(), "palm".to_string()]
        } else if query_lower.contains("mountain") || query_lower.contains("hill") {
            vec!["mountain".to_string()]
        } else if query_lower.contains("stair") {
            vec!["stairs".to_string()]
        } else if query_lower.contains("chimney") {
            vec!["chimney".to_string()]
        } else {
            vec![query.to_string()]
        };

        Ok(classes)
    }

    /// Check if image contains target object
    fn contains_target(
        &mut self,
        image: &mut CaptchaImage,
        target_classes: &[String],
    ) -> Result<bool> {
        // Method 1: Object detection
        let detections = self.detector.detect(image)?;

        for detection in &detections {
            for target in target_classes {
                if detection
                    .label
                    .to_lowercase()
                    .contains(&target.to_lowercase())
                {
                    if detection.confidence > self.config.detection_threshold {
                        debug!(
                            "Found {} with confidence {}",
                            detection.label, detection.confidence
                        );
                        return Ok(true);
                    }
                }
            }
        }

        // Method 2: Full image classification (fallback)
        let classification = self.classifier.classify(image)?;

        for target in target_classes {
            if classification
                .label
                .to_lowercase()
                .contains(&target.to_lowercase())
            {
                if classification.confidence > self.config.detection_threshold {
                    debug!(
                        "Classified as {} with confidence {}",
                        classification.label, classification.confidence
                    );
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Object detector using YOLO-like architecture
pub struct ObjectDetector {
    vs: nn::VarStore,
    backbone: nn::Sequential,
    detection_head: nn::Sequential,
    class_names: Vec<String>,
}

impl ObjectDetector {
    pub fn new() -> Result<Self> {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        // Simplified YOLO-like backbone
        let backbone = nn::seq()
            // Conv block 1
            .add(nn::conv2d(
                &root / "conv1",
                3,
                64,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(2))
            // Conv block 2
            .add(nn::conv2d(
                &root / "conv2",
                64,
                128,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(2))
            // Conv block 3
            .add(nn::conv2d(
                &root / "conv3",
                128,
                256,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(2))
            // Conv block 4
            .add(nn::conv2d(
                &root / "conv4",
                256,
                512,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu());

        // Detection head: predicts [x, y, w, h, confidence, class_probs...]
        // Output: (batch, anchors * (5 + num_classes), grid_h, grid_w)
        let num_classes = 80; // COCO classes
        let num_anchors = 3;
        let detection_head = nn::seq().add(nn::conv2d(
            &root / "detect",
            512,
            num_anchors * (5 + num_classes),
            1,
            Default::default(),
        ));

        // Load COCO class names
        let class_names = Self::coco_classes();

        Ok(Self {
            vs,
            backbone,
            detection_head,
            class_names,
        })
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    pub fn detect(&mut self, image: &mut CaptchaImage) -> Result<Vec<DetectedObject>> {
        // Preprocess image
        image.preprocess()?;
        let tensor_data = image.to_tensor()?;

        let tensor = Tensor::of_slice(&tensor_data).view([1, 3, 224, 224]);

        // Forward pass
        let features = self.backbone.forward(&tensor);
        let detections = self.detection_head.forward(&features);

        // Parse detections
        let parsed = self.parse_detections(&detections, image.width(), image.height())?;

        // Apply NMS
        let filtered = non_max_suppression(parsed, 0.4);

        Ok(filtered)
    }

    fn parse_detections(
        &self,
        output: &Tensor,
        img_width: u32,
        img_height: u32,
    ) -> Result<Vec<DetectedObject>> {
        let mut detections = Vec::new();

        // Simplified parsing (real YOLO is more complex)
        let output_vec = Vec::<f32>::from(output);

        // Mock detections for demonstration
        // In production, you'd properly decode YOLO output

        Ok(detections)
    }

    fn coco_classes() -> Vec<String> {
        vec![
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

/// Image classifier using ResNet-like architecture
pub struct ImageClassifier {
    vs: nn::VarStore,
    model: nn::Sequential,
    class_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub label: String,
    pub confidence: f32,
}

impl ImageClassifier {
    pub fn new() -> Result<Self> {
        let vs = nn::VarStore::new(Device::Cpu);
        let root = vs.root();

        // Simplified ResNet-18 style architecture
        let model = nn::seq()
            // Initial conv
            .add(nn::conv2d(
                &root / "conv1",
                3,
                64,
                7,
                nn::ConvConfig {
                    stride: 2,
                    padding: 3,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu())
            .add_fn(|x| x.max_pool2d_default(3))
            // Residual blocks (simplified)
            .add(Self::residual_block(&root / "layer1", 64, 64))
            .add(Self::residual_block(&root / "layer2", 64, 128))
            .add(Self::residual_block(&root / "layer3", 128, 256))
            .add(Self::residual_block(&root / "layer4", 256, 512))
            // Global average pooling
            .add_fn(|x| x.adaptive_avg_pool2d(&[1, 1]))
            .add_fn(|x| x.flat_view())
            // Fully connected
            .add(nn::linear(&root / "fc", 512, 1000, Default::default()));

        let class_names = Self::imagenet_classes();

        Ok(Self {
            vs,
            model,
            class_names,
        })
    }

    fn residual_block(path: &nn::Path, in_channels: i64, out_channels: i64) -> nn::Sequential {
        nn::seq()
            .add(nn::conv2d(
                path / "conv1",
                in_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
            .add_fn(|x| x.relu())
            .add(nn::conv2d(
                path / "conv2",
                out_channels,
                out_channels,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ))
            .add_fn(|x| x.batch_norm2d())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    pub fn classify(&mut self, image: &mut CaptchaImage) -> Result<Classification> {
        // Preprocess
        image.preprocess()?;
        let tensor_data = image.to_tensor()?;

        let tensor = Tensor::of_slice(&tensor_data).view([1, 3, 224, 224]);

        // Forward pass
        let output = self.model.forward(&tensor);
        let probabilities = output.softmax(-1, Kind::Float);

        // Get top prediction
        let (confidence, class_idx) = probabilities.max_dim(-1, false).into();

        let confidence_val = f32::from(confidence);
        let class_idx_val = i64::from(class_idx);

        let label = self
            .class_names
            .get(class_idx_val as usize)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        Ok(Classification {
            label,
            confidence: confidence_val,
        })
    }

    fn imagenet_classes() -> Vec<String> {
        // Subset of ImageNet classes relevant for reCAPTCHA
        vec![
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "boat",
            "bridge",
            "crosswalk",
            "stairs",
            "mountain",
            "tree",
            "palm tree",
            "chimney",
            "building",
            "road",
            "street",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query() {
        let solver = ImageCaptchaSolver::new(ImageSolverConfig::default()).unwrap();

        let classes = solver
            .parse_query("Select all images with traffic lights")
            .unwrap();
        assert_eq!(classes, vec!["traffic light"]);

        let classes = solver
            .parse_query("Select all images with bicycles")
            .unwrap();
        assert_eq!(classes, vec!["bicycle"]);
    }

    #[test]
    fn test_grid_size_validation() {
        let mut solver = ImageCaptchaSolver::new(ImageSolverConfig::default()).unwrap();

        // Invalid grid size
        let result = solver.solve_grid(vec![], "traffic lights");
        assert!(result.is_err());
    }

    #[test]
    fn test_config_default() {
        let config = ImageSolverConfig::default();
        assert_eq!(config.detection_threshold, 0.5);
        assert_eq!(config.nms_threshold, 0.4);
    }
}
