/// Browser integration example
///
/// This example demonstrates how to integrate CAPTCHA solver with a browser
/// automation tool (chromiumoxide) to automatically solve CAPTCHAs during scraping.
///
/// Run with: cargo run --example browser_integration
use anyhow::{Context, Result};
use argus_captcha::{CaptchaImage, CaptchaSolver, CaptchaType};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::Page;
use futures::StreamExt;
use image::{DynamicImage, ImageFormat};
use std::io::Cursor;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("=== Browser + CAPTCHA Solver Integration ===\n");

    // 1. Create CAPTCHA solver
    let mut solver = CaptchaSolver::new()?;
    println!("✓ CAPTCHA solver initialized");

    // 2. Launch browser
    let (browser, mut handler) = Browser::launch(BrowserConfig::builder().build()?)
        .await
        .context("Failed to launch browser")?;

    // Spawn browser handler
    tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if let Err(e) = event {
                eprintln!("Browser handler error: {}", e);
            }
        }
    });

    println!("✓ Browser launched");

    // 3. Navigate to page with CAPTCHA
    let page = browser.new_page("about:blank").await?;
    println!("✓ Page created\n");

    // Example scenarios
    println!("--- Scenario 1: reCAPTCHA v2 ---");
    if let Err(e) = handle_recaptcha(&page, &mut solver).await {
        eprintln!("reCAPTCHA handling failed: {}", e);
    }

    println!("\n--- Scenario 2: Text CAPTCHA ---");
    if let Err(e) = handle_text_captcha(&page, &mut solver).await {
        eprintln!("Text CAPTCHA handling failed: {}", e);
    }

    println!("\n--- Scenario 3: Slider CAPTCHA ---");
    if let Err(e) = handle_slider_captcha(&page, &mut solver).await {
        eprintln!("Slider CAPTCHA handling failed: {}", e);
    }

    // Print final metrics
    println!("\n=== Session Metrics ===");
    println!("Success rate: {:.2}%", solver.success_rate() * 100.0);
    println!("Avg solve time: {:?}", solver.avg_solve_time());

    Ok(())
}

/// Handle reCAPTCHA v2 (image grid)
async fn handle_recaptcha(page: &Page, solver: &mut CaptchaSolver) -> Result<()> {
    // In real scenario, navigate to page with reCAPTCHA
    // page.goto("https://example.com/with-recaptcha").await?;

    println!("  Detecting CAPTCHA...");

    // Wait for reCAPTCHA iframe
    sleep(Duration::from_millis(500)).await;

    // Check if reCAPTCHA is present
    let has_recaptcha = page
        .evaluate("document.querySelector('iframe[src*=\"recaptcha\"]') !== null")
        .await?
        .into_value::<bool>()?;

    if !has_recaptcha {
        println!("  ℹ No reCAPTCHA detected (demo mode)");
        return Ok(());
    }

    println!("  ✓ reCAPTCHA detected");

    // Click on reCAPTCHA checkbox
    println!("  Clicking checkbox...");
    page.evaluate("document.querySelector('.recaptcha-checkbox-border').click()")
        .await?;

    sleep(Duration::from_secs(1)).await;

    // Check if challenge appears
    let has_challenge = page
        .evaluate("document.querySelector('.rc-imageselect-challenge') !== null")
        .await?
        .into_value::<bool>()?;

    if !has_challenge {
        println!("  ✓ Passed without challenge!");
        return Ok(());
    }

    println!("  Challenge appeared, solving...");

    // Extract query text
    let query: String = page
        .evaluate("document.querySelector('.rc-imageselect-desc-wrapper').innerText")
        .await?
        .into_value()?;

    println!("  Query: \"{}\"", query);

    // Extract grid images (3x3 or 4x4)
    let image_elements = page
        .evaluate("document.querySelectorAll('.rc-image-tile-wrapper img').length")
        .await?
        .into_value::<usize>()?;

    println!("  Grid size: {} tiles", image_elements);

    // Capture each tile image
    let mut images = Vec::new();
    for i in 0..image_elements {
        let script = format!(
            "document.querySelectorAll('.rc-image-tile-wrapper img')[{}].src",
            i
        );
        let img_src: String = page.evaluate(&script).await?.into_value()?;

        // Download image (in real scenario)
        // For demo, create placeholder
        images.push(CaptchaImage {
            original: DynamicImage::new_rgb8(100, 100),
            preprocessed: None,
        });
    }

    // Solve CAPTCHA
    let result = solver.solve_image_grid(images, &query)?;

    if let argus_captcha::Solution::ImageIndices(indices) = &result.solution {
        println!("  Solution: tiles {:?}", indices);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);

        // Click selected tiles
        for &idx in indices {
            let script = format!(
                "document.querySelectorAll('.rc-imageselect-tile')[{}].click()",
                idx
            );
            page.evaluate(&script).await?;
            sleep(Duration::from_millis(100)).await;
        }

        // Submit
        page.evaluate("document.querySelector('.rc-imageselect-verify-button').click()")
            .await?;

        sleep(Duration::from_secs(2)).await;

        println!("  ✓ Challenge submitted");
        solver.update_metrics(&result, true);
    }

    Ok(())
}

/// Handle text CAPTCHA
async fn handle_text_captcha(page: &Page, solver: &mut CaptchaSolver) -> Result<()> {
    println!("  Detecting text CAPTCHA...");

    // Find CAPTCHA image element
    let has_captcha = page
        .evaluate("document.querySelector('img.captcha-image') !== null")
        .await?
        .into_value::<bool>()?;

    if !has_captcha {
        println!("  ℹ No text CAPTCHA detected (demo mode)");
        return Ok(());
    }

    println!("  ✓ Text CAPTCHA detected");

    // Extract CAPTCHA image
    let img_data = page
        .evaluate("document.querySelector('img.captcha-image').src")
        .await?
        .into_value::<String>()?;

    // Decode base64 or download image
    let image = decode_captcha_image(&img_data)?;

    // Solve
    let result = solver.solve_text(image)?;

    if let argus_captcha::Solution::Text(text) = &result.solution {
        println!("  Solution: \"{}\"", text);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);

        // Fill input field
        let script = format!(
            "document.querySelector('input[name=\"captcha\"]').value = '{}'",
            text
        );
        page.evaluate(&script).await?;

        // Submit form
        page.evaluate("document.querySelector('form').submit()")
            .await?;

        println!("  ✓ Solution submitted");
        solver.update_metrics(&result, true);
    }

    Ok(())
}

/// Handle slider CAPTCHA (e.g., GeeTest, hCaptcha slider)
async fn handle_slider_captcha(page: &Page, solver: &mut CaptchaSolver) -> Result<()> {
    println!("  Detecting slider CAPTCHA...");

    let has_slider = page
        .evaluate("document.querySelector('.slider-captcha') !== null")
        .await?
        .into_value::<bool>()?;

    if !has_slider {
        println!("  ℹ No slider CAPTCHA detected (demo mode)");
        return Ok(());
    }

    println!("  ✓ Slider CAPTCHA detected");

    // Extract background and puzzle piece images
    let bg_img = extract_element_image(page, ".slider-bg").await?;
    let piece_img = extract_element_image(page, ".slider-piece").await?;

    // Solve
    let result = solver.solve_slider(bg_img, piece_img)?;

    if let argus_captcha::Solution::SliderOffset(offset) = result.solution {
        println!("  Offset: {} pixels", offset);
        println!("  Confidence: {:.2}%", result.confidence * 100.0);

        // Simulate slider drag
        let script = format!(
            r#"
            const slider = document.querySelector('.slider-handle');
            const event = new MouseEvent('mousedown', {{ clientX: 0 }});
            slider.dispatchEvent(event);

            const moveEvent = new MouseEvent('mousemove', {{ clientX: {} }});
            document.dispatchEvent(moveEvent);

            const upEvent = new MouseEvent('mouseup');
            document.dispatchEvent(upEvent);
            "#,
            offset
        );
        page.evaluate(&script).await?;

        println!("  ✓ Slider moved");
        solver.update_metrics(&result, true);

        sleep(Duration::from_secs(1)).await;
    }

    Ok(())
}

/// Decode CAPTCHA image from data URL or URL
fn decode_captcha_image(img_data: &str) -> Result<CaptchaImage> {
    // Check if it's a data URL
    if img_data.starts_with("data:image") {
        let parts: Vec<&str> = img_data.split(',').collect();
        if parts.len() == 2 {
            let bytes = base64::decode(parts[1])?;
            let img = image::load_from_memory(&bytes)?;
            return Ok(CaptchaImage {
                original: img,
                preprocessed: None,
            });
        }
    }

    // For demo, return dummy image
    Ok(CaptchaImage {
        original: DynamicImage::new_rgb8(200, 60),
        preprocessed: None,
    })
}

/// Extract image from element
async fn extract_element_image(page: &Page, selector: &str) -> Result<CaptchaImage> {
    // Take screenshot of element
    let script = format!(
        r#"
        const el = document.querySelector('{}');
        const rect = el.getBoundingClientRect();
        JSON.stringify({{
            x: rect.x,
            y: rect.y,
            width: rect.width,
            height: rect.height
        }})
        "#,
        selector
    );

    let _bounds = page.evaluate(&script).await?.into_value::<String>()?;

    // In real scenario, take screenshot with bounds
    // For demo, return dummy image
    Ok(CaptchaImage {
        original: DynamicImage::new_rgb8(400, 200),
        preprocessed: None,
    })
}
