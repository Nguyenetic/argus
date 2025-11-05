/// Basic CAPTCHA solver usage examples
///
/// This example demonstrates how to:
/// 1. Create a CAPTCHA solver
/// 2. Solve different CAPTCHA types
/// 3. Track metrics
///
/// Run with: cargo run --example basic_usage
use anyhow::Result;
use argus_captcha::{CaptchaImage, CaptchaSolver};
use image::DynamicImage;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("=== Argus CAPTCHA Solver - Basic Usage ===\n");

    // 1. Create solver with default configuration
    let mut solver = CaptchaSolver::new()?;
    println!("✓ Solver initialized\n");

    // 2. Solve text CAPTCHA
    println!("--- Text CAPTCHA ---");
    if let Err(e) = solve_text_captcha(&mut solver).await {
        eprintln!("Text CAPTCHA failed: {}", e);
    }

    // 3. Solve reCAPTCHA v2 (image grid)
    println!("\n--- reCAPTCHA v2 (Image Grid) ---");
    if let Err(e) = solve_recaptcha(&mut solver).await {
        eprintln!("reCAPTCHA failed: {}", e);
    }

    // 4. Solve slider CAPTCHA
    println!("\n--- Slider CAPTCHA ---");
    if let Err(e) = solve_slider_captcha(&mut solver).await {
        eprintln!("Slider CAPTCHA failed: {}", e);
    }

    // 5. Solve rotation CAPTCHA
    println!("\n--- Rotation CAPTCHA ---");
    if let Err(e) = solve_rotation_captcha(&mut solver).await {
        eprintln!("Rotation CAPTCHA failed: {}", e);
    }

    // 6. Print metrics
    println!("\n=== Performance Metrics ===");
    print_metrics(&solver);

    Ok(())
}

/// Solve a text CAPTCHA (distorted characters)
async fn solve_text_captcha(solver: &mut CaptchaSolver) -> Result<()> {
    // Load image (replace with actual CAPTCHA image)
    let image = load_or_create_image("examples/data/text_captcha.png", 200, 60)?;

    // Solve
    let result = solver.solve_text(image)?;

    // Update metrics
    solver.update_metrics(&result, true);

    // Print result
    if let argus_captcha::Solution::Text(text) = &result.solution {
        println!("  Solution: \"{}\"", text);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Solve time: {:?}", result.solve_time);
        println!("  Used fallback: {}", result.used_fallback);
    }

    Ok(())
}

/// Solve a reCAPTCHA v2 (image grid)
async fn solve_recaptcha(solver: &mut CaptchaSolver) -> Result<()> {
    // Load 3x3 grid images (replace with actual CAPTCHA images)
    let mut images = Vec::new();
    for i in 0..9 {
        let path = format!("examples/data/recaptcha_grid_{}.png", i);
        images.push(load_or_create_image(&path, 100, 100)?);
    }

    // Query: "Select all images with traffic lights"
    let query = "traffic lights";

    // Solve
    let result = solver.solve_image_grid(images, query)?;

    // Update metrics
    solver.update_metrics(&result, true);

    // Print result
    if let argus_captcha::Solution::ImageIndices(indices) = &result.solution {
        println!("  Query: \"{}\"", query);
        println!("  Selected tiles: {:?}", indices);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Solve time: {:?}", result.solve_time);
    }

    Ok(())
}

/// Solve a slider CAPTCHA
async fn solve_slider_captcha(solver: &mut CaptchaSolver) -> Result<()> {
    // Load background and puzzle piece (replace with actual images)
    let background = load_or_create_image("examples/data/slider_bg.png", 400, 200)?;
    let puzzle_piece = load_or_create_image("examples/data/slider_piece.png", 60, 80)?;

    // Solve
    let result = solver.solve_slider(background, puzzle_piece)?;

    // Update metrics
    solver.update_metrics(&result, true);

    // Print result
    if let argus_captcha::Solution::SliderOffset(offset) = result.solution {
        println!("  X offset: {} pixels", offset);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Solve time: {:?}", result.solve_time);
    }

    Ok(())
}

/// Solve a rotation CAPTCHA
async fn solve_rotation_captcha(solver: &mut CaptchaSolver) -> Result<()> {
    // Load rotated image (replace with actual CAPTCHA image)
    let image = load_or_create_image("examples/data/rotation_captcha.png", 250, 250)?;

    // Solve
    let result = solver.solve_rotation(image)?;

    // Update metrics
    solver.update_metrics(&result, true);

    // Print result
    if let argus_captcha::Solution::RotationAngle(angle) = result.solution {
        println!("  Rotation angle: {:.1}°", angle);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);
        println!("  Solve time: {:?}", result.solve_time);
    }

    Ok(())
}

/// Load image from file or create dummy image
fn load_or_create_image(path: &str, width: u32, height: u32) -> Result<CaptchaImage> {
    let image = if Path::new(path).exists() {
        image::open(path)?
    } else {
        // Create dummy image for demo
        DynamicImage::new_rgb8(width, height)
    };

    Ok(CaptchaImage {
        original: image,
        preprocessed: None,
    })
}

/// Print solver metrics
fn print_metrics(solver: &CaptchaSolver) {
    let metrics = solver.metrics();

    println!("  Total attempts: {}", metrics.total_attempts);
    println!("  Successful: {}", metrics.successful_solves);
    println!("  Failed: {}", metrics.failed_solves);
    println!("  Success rate: {:.2}%", solver.success_rate() * 100.0);
    println!("  Avg solve time: {:?}", solver.avg_solve_time());

    if !metrics.by_type.is_empty() {
        println!("\n  By Type:");
        for (captcha_type, type_metrics) in &metrics.by_type {
            println!("    {}:", captcha_type);
            println!("      Attempts: {}", type_metrics.attempts);
            println!("      Successes: {}", type_metrics.successes);
            println!(
                "      Avg confidence: {:.2}%",
                type_metrics.avg_confidence * 100.0
            );
            println!("      Avg time: {:?}", type_metrics.avg_solve_time);
        }
    }
}
