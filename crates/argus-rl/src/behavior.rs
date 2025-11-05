/// Human Behavior Emulation Module
///
/// This module provides realistic human-like behavior patterns for browser automation,
/// designed to evade bot detection systems. It implements:
///
/// - **Perlin Noise**: Smooth, natural mouse movement trajectories
/// - **Gaussian Curves**: Realistic timing distributions for clicks and actions
/// - **Bezier Curves**: Human-like curved mouse paths
/// - **Natural Scrolling**: Variable speed scrolling with acceleration/deceleration
/// - **Attention Modeling**: Realistic pause patterns and focus behavior
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// 2D point for mouse coordinates
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

/// Perlin noise generator for smooth, natural-looking randomness
pub struct PerlinNoise {
    permutation: Vec<usize>,
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with random permutation
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut permutation: Vec<usize> = (0..256).collect();

        // Fisher-Yates shuffle
        for i in (1..256).rev() {
            let j = rng.gen_range(0..=i);
            permutation.swap(i, j);
        }

        // Duplicate for wrapping
        permutation.extend(permutation.clone());

        Self { permutation }
    }

    /// Generate 2D Perlin noise value at (x, y)
    /// Returns value in range [-1, 1]
    pub fn noise2d(&self, x: f32, y: f32) -> f32 {
        // Find unit grid cell containing point
        let x0 = x.floor() as i32 & 255;
        let y0 = y.floor() as i32 & 255;

        // Relative x, y within cell
        let x = x - x.floor();
        let y = y - y.floor();

        // Compute fade curves
        let u = Self::fade(x);
        let v = Self::fade(y);

        // Hash coordinates of 4 cube corners
        let aa = self.permutation[self.permutation[x0 as usize] + y0 as usize];
        let ab = self.permutation[self.permutation[x0 as usize] + (y0 + 1) as usize];
        let ba = self.permutation[self.permutation[(x0 + 1) as usize] + y0 as usize];
        let bb = self.permutation[self.permutation[(x0 + 1) as usize] + (y0 + 1) as usize];

        // Blend results from corners
        let x1 = Self::lerp(u, Self::grad(aa, x, y), Self::grad(ba, x - 1.0, y));
        let x2 = Self::lerp(
            u,
            Self::grad(ab, x, y - 1.0),
            Self::grad(bb, x - 1.0, y - 1.0),
        );

        Self::lerp(v, x1, x2)
    }

    /// Fade function for smooth interpolation (6t^5 - 15t^4 + 10t^3)
    fn fade(t: f32) -> f32 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Linear interpolation
    fn lerp(t: f32, a: f32, b: f32) -> f32 {
        a + t * (b - a)
    }

    /// Gradient function for Perlin noise
    fn grad(hash: usize, x: f32, y: f32) -> f32 {
        let h = hash & 3;
        let u = if h < 2 { x } else { y };
        let v = if h < 2 { y } else { x };
        (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
    }
}

impl Default for PerlinNoise {
    fn default() -> Self {
        Self::new()
    }
}

/// Mouse movement path generator using Bezier curves and Perlin noise
pub struct MousePathGenerator {
    perlin: PerlinNoise,
    noise_scale: f32,
    noise_strength: f32,
}

impl MousePathGenerator {
    /// Create new mouse path generator
    ///
    /// # Parameters
    /// - `noise_scale`: Frequency of noise (lower = smoother)
    /// - `noise_strength`: Amplitude of noise (lower = straighter path)
    pub fn new(noise_scale: f32, noise_strength: f32) -> Self {
        Self {
            perlin: PerlinNoise::new(),
            noise_scale,
            noise_strength,
        }
    }

    /// Generate natural mouse movement path from start to end
    /// Returns list of intermediate points
    pub fn generate_path(&self, start: Point, end: Point, num_points: usize) -> Vec<Point> {
        let mut rng = rand::thread_rng();

        // Generate control points for Bezier curve
        let distance = start.distance(&end);
        let control_offset = distance * 0.3; // 30% of distance for natural curve

        let control1 = Point::new(
            start.x + (end.x - start.x) * 0.33 + rng.gen_range(-control_offset..control_offset),
            start.y + (end.y - start.y) * 0.33 + rng.gen_range(-control_offset..control_offset),
        );

        let control2 = Point::new(
            start.x + (end.x - start.x) * 0.66 + rng.gen_range(-control_offset..control_offset),
            start.y + (end.y - start.y) * 0.66 + rng.gen_range(-control_offset..control_offset),
        );

        let mut points = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let t = i as f32 / (num_points - 1) as f32;

            // Cubic Bezier curve
            let point = self.bezier_cubic(start, control1, control2, end, t);

            // Add Perlin noise for natural jitter
            let noise_x = self.perlin.noise2d(t * self.noise_scale, 0.0) * self.noise_strength;
            let noise_y = self.perlin.noise2d(0.0, t * self.noise_scale) * self.noise_strength;

            points.push(Point::new(point.x + noise_x, point.y + noise_y));
        }

        points
    }

    /// Cubic Bezier curve interpolation
    fn bezier_cubic(&self, p0: Point, p1: Point, p2: Point, p3: Point, t: f32) -> Point {
        let u = 1.0 - t;
        let tt = t * t;
        let uu = u * u;
        let uuu = uu * u;
        let ttt = tt * t;

        Point::new(
            uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x,
            uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y,
        )
    }
}

impl Default for MousePathGenerator {
    fn default() -> Self {
        Self::new(5.0, 2.0)
    }
}

/// Timing distribution generator using Gaussian curves
pub struct TimingGenerator {
    mean: f32,
    std_dev: f32,
}

impl TimingGenerator {
    /// Create new timing generator
    ///
    /// # Parameters
    /// - `mean`: Average time in seconds
    /// - `std_dev`: Standard deviation in seconds
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self { mean, std_dev }
    }

    /// Generate realistic delay time using Gaussian distribution
    /// Clamps to positive values with minimum threshold
    pub fn generate_delay(&self) -> f32 {
        let mut rng = rand::thread_rng();

        // Box-Muller transform for Gaussian distribution
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        let delay = self.mean + self.std_dev * z0;

        // Clamp to reasonable range (0.05s to 10s)
        delay.max(0.05).min(10.0)
    }

    /// Generate click duration (time mouse is held down)
    /// Returns realistic click duration in seconds
    pub fn generate_click_duration(&self) -> f32 {
        let mut rng = rand::thread_rng();

        // Human clicks typically 0.05s to 0.2s
        rng.gen_range(0.05..0.20)
    }

    /// Generate typing speed variation
    /// Returns delay between keystrokes in seconds
    pub fn generate_keystroke_delay(&self) -> f32 {
        let mut rng = rand::thread_rng();

        // Average typist: 40-60 WPM = ~0.2s per character
        // With realistic variation
        let base_delay = 0.2;
        let variation = rng.gen_range(-0.1..0.15);

        (base_delay + variation).max(0.05)
    }
}

impl Default for TimingGenerator {
    fn default() -> Self {
        Self::new(1.0, 0.3) // 1s mean, 0.3s std dev
    }
}

/// Natural scrolling behavior generator
pub struct ScrollGenerator {
    perlin: PerlinNoise,
}

impl ScrollGenerator {
    pub fn new() -> Self {
        Self {
            perlin: PerlinNoise::new(),
        }
    }

    /// Generate natural scroll pattern with acceleration/deceleration
    /// Returns list of scroll deltas (pixel amounts)
    pub fn generate_scroll(&self, total_distance: f32, num_steps: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut deltas = Vec::with_capacity(num_steps);

        let mut remaining = total_distance;

        for i in 0..num_steps {
            let progress = i as f32 / num_steps as f32;

            // Ease-in-out curve for natural acceleration/deceleration
            let ease = if progress < 0.5 {
                2.0 * progress * progress
            } else {
                -1.0 + (4.0 - 2.0 * progress) * progress
            };

            // Base scroll amount
            let base_delta = (total_distance / num_steps as f32) * (1.0 + ease * 0.5);

            // Add Perlin noise for natural variation
            let noise = self.perlin.noise2d(progress * 10.0, 0.0) * base_delta * 0.2;

            let delta = (base_delta + noise).min(remaining);
            deltas.push(delta);
            remaining -= delta;
        }

        // Ensure we reach exactly the target distance
        if remaining > 0.0 {
            if let Some(last) = deltas.last_mut() {
                *last += remaining;
            }
        }

        deltas
    }

    /// Generate scroll pause duration (human reading time)
    pub fn generate_scroll_pause(&self) -> f32 {
        let mut rng = rand::thread_rng();

        // Humans pause 0.5s to 3s while reading
        rng.gen_range(0.5..3.0)
    }

    /// Generate wheel scroll events (discrete scrolling)
    /// Returns realistic number of scroll wheel "ticks"
    pub fn generate_wheel_ticks(&self, distance: f32) -> Vec<i32> {
        let mut rng = rand::thread_rng();

        // Typical scroll wheel: 100-120 pixels per tick
        let pixels_per_tick = rng.gen_range(100.0..120.0);
        let num_ticks = (distance / pixels_per_tick).ceil() as usize;

        let mut ticks = Vec::with_capacity(num_ticks);

        for _ in 0..num_ticks {
            // Occasionally vary direction for natural "overshooting"
            let tick = if rng.gen::<f32>() < 0.95 { 1 } else { -1 };
            ticks.push(tick);
        }

        ticks
    }
}

impl Default for ScrollGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Human attention and focus behavior
pub struct AttentionModel {
    perlin: PerlinNoise,
}

impl AttentionModel {
    pub fn new() -> Self {
        Self {
            perlin: PerlinNoise::new(),
        }
    }

    /// Generate realistic "thinking" pause before action
    pub fn generate_think_time(&self) -> f32 {
        let mut rng = rand::thread_rng();

        // Humans think 0.3s to 2s before acting
        rng.gen_range(0.3..2.0)
    }

    /// Generate page reading time based on content complexity
    pub fn generate_read_time(&self, word_count: usize, complexity: f32) -> f32 {
        let mut rng = rand::thread_rng();

        // Average reading: 200-250 words per minute
        let base_wpm = 225.0;
        let base_time = (word_count as f32 / base_wpm) * 60.0;

        // Adjust for complexity (0.5 = simple, 1.0 = normal, 1.5 = complex)
        let adjusted_time = base_time * complexity;

        // Add natural variation (±20%)
        let variation = rng.gen_range(0.8..1.2);

        adjusted_time * variation
    }

    /// Generate "distraction" probability and duration
    /// Simulates human attention lapses
    pub fn generate_distraction(&self) -> Option<f32> {
        let mut rng = rand::thread_rng();

        // 5% chance of distraction
        if rng.gen::<f32>() < 0.05 {
            // Distraction lasts 1-5 seconds
            Some(rng.gen_range(1.0..5.0))
        } else {
            None
        }
    }

    /// Calculate mouse movement entropy (randomness measure)
    /// Higher entropy = more human-like
    pub fn calculate_movement_entropy(&self, points: &[Point]) -> f32 {
        if points.len() < 2 {
            return 0.0;
        }

        let mut angles = Vec::new();
        for i in 1..points.len() {
            let dx = points[i].x - points[i - 1].x;
            let dy = points[i].y - points[i - 1].y;
            let angle = dy.atan2(dx);
            angles.push(angle);
        }

        // Calculate entropy using angle distribution
        let mut histogram = vec![0; 36]; // 10-degree bins
        for &angle in &angles {
            let normalized = ((angle + PI) / (2.0 * PI) * 36.0) as usize % 36;
            histogram[normalized] += 1;
        }

        let total = angles.len() as f32;
        let mut entropy = 0.0;
        for &count in &histogram {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }

        entropy / 5.269 // Normalize to [0, 1] range (log2(36) ≈ 5.17)
    }
}

impl Default for AttentionModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_noise_range() {
        let perlin = PerlinNoise::new();

        for i in 0..100 {
            let noise = perlin.noise2d(i as f32 * 0.1, i as f32 * 0.1);
            assert!(
                noise >= -1.0 && noise <= 1.0,
                "Noise out of range: {}",
                noise
            );
        }
    }

    #[test]
    fn test_mouse_path_generation() {
        let generator = MousePathGenerator::default();
        let start = Point::new(0.0, 0.0);
        let end = Point::new(100.0, 100.0);

        let path = generator.generate_path(start, end, 50);

        assert_eq!(path.len(), 50);
        assert_eq!(path[0], start);

        // End point should be close to target (within noise range)
        let final_point = path[path.len() - 1];
        assert!((final_point.x - end.x).abs() < 5.0);
        assert!((final_point.y - end.y).abs() < 5.0);
    }

    #[test]
    fn test_timing_generation() {
        let timing = TimingGenerator::new(1.0, 0.3);

        let mut delays = Vec::new();
        for _ in 0..100 {
            let delay = timing.generate_delay();
            assert!(delay > 0.0 && delay <= 10.0);
            delays.push(delay);
        }

        // Check mean is roughly correct
        let mean: f32 = delays.iter().sum::<f32>() / delays.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.3,
            "Mean too far from expected: {}",
            mean
        );
    }

    #[test]
    fn test_click_duration_realistic() {
        let timing = TimingGenerator::default();

        for _ in 0..100 {
            let duration = timing.generate_click_duration();
            assert!(duration >= 0.05 && duration <= 0.20);
        }
    }

    #[test]
    fn test_scroll_generation() {
        let generator = ScrollGenerator::new();
        let distance = 1000.0;

        let deltas = generator.generate_scroll(distance, 20);

        assert_eq!(deltas.len(), 20);

        let total: f32 = deltas.iter().sum();
        assert!(
            (total - distance).abs() < 1.0,
            "Total distance mismatch: {}",
            total
        );
    }

    #[test]
    fn test_attention_think_time() {
        let attention = AttentionModel::new();

        for _ in 0..100 {
            let think_time = attention.generate_think_time();
            assert!(think_time >= 0.3 && think_time <= 2.0);
        }
    }

    #[test]
    fn test_read_time_scales_with_words() {
        let attention = AttentionModel::new();

        let time_100 = attention.generate_read_time(100, 1.0);
        let time_200 = attention.generate_read_time(200, 1.0);

        // Should roughly double
        assert!(
            time_200 > time_100 * 1.5,
            "Read time doesn't scale with word count"
        );
    }

    #[test]
    fn test_movement_entropy_calculation() {
        let attention = AttentionModel::new();

        // Straight line = low entropy
        let straight_line = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, 20.0),
            Point::new(30.0, 30.0),
        ];
        let entropy_low = attention.calculate_movement_entropy(&straight_line);

        // Random path = higher entropy
        let random_path = vec![
            Point::new(0.0, 0.0),
            Point::new(5.0, 15.0),
            Point::new(20.0, 8.0),
            Point::new(15.0, 25.0),
        ];
        let entropy_high = attention.calculate_movement_entropy(&random_path);

        assert!(
            entropy_high > entropy_low,
            "Random path should have higher entropy"
        );
    }

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);

        assert_eq!(p1.distance(&p2), 5.0);
    }

    #[test]
    fn test_distraction_probability() {
        let attention = AttentionModel::new();

        let mut distractions = 0;
        for _ in 0..1000 {
            if let Some(duration) = attention.generate_distraction() {
                distractions += 1;
                assert!(duration >= 1.0 && duration <= 5.0);
            }
        }

        // Should be around 5% (50 out of 1000)
        assert!(
            distractions > 20 && distractions < 100,
            "Distraction rate: {}",
            distractions
        );
    }
}
