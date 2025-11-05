/// Complete Integration Example
///
/// This example demonstrates the full integration of:
/// 1. RL Agent (behavioral evasion)
/// 2. CAPTCHA Solver (solving when detected)
/// 3. Browser automation (scraping with evasion)
///
/// The system automatically:
/// - Mimics human behavior to avoid detection
/// - Detects CAPTCHAs when they appear
/// - Solves CAPTCHAs automatically
/// - Continues scraping after solving
///
/// Run with: cargo run --example complete_integration --features audio
use anyhow::Result;
use argus_captcha::{IntegratedBotAgent, IntegrationConfig};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::Page;
use futures::StreamExt;
use std::time::Duration;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Argus Complete Integration Demo ===\n");
    println!("RL Agent + CAPTCHA Solver + Browser Automation\n");

    // 1. Create integrated agent
    let config = IntegrationConfig {
        auto_solve_captcha: true,
        max_captcha_attempts: 3,
        post_captcha_wait: Duration::from_secs(2),
        enable_fallback: true,
        debug: true,
    };

    let mut agent = IntegratedBotAgent::new(config)?;
    info!("✓ Integrated agent initialized");

    // 2. Launch browser
    let (browser, mut handler) =
        Browser::launch(BrowserConfig::builder().window_size(1920, 1080).build()?).await?;

    // Spawn browser handler
    tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if let Err(e) = event {
                eprintln!("Browser error: {}", e);
            }
        }
    });

    info!("✓ Browser launched");

    // 3. Run demonstration scenarios
    let page = browser.new_page("about:blank").await?;

    println!("\n--- Scenario 1: Normal Scraping (No CAPTCHA) ---");
    demo_normal_scraping(&page, &mut agent).await?;

    println!("\n--- Scenario 2: Page with CAPTCHA ---");
    demo_captcha_handling(&page, &mut agent).await?;

    println!("\n--- Scenario 3: Multi-Page Session ---");
    demo_multi_page_session(&page, &mut agent).await?;

    // 4. Print final statistics
    println!("\n=== Session Statistics ===");
    print_statistics(&agent);

    Ok(())
}

/// Demonstrate normal scraping without CAPTCHA
async fn demo_normal_scraping(page: &Page, agent: &mut IntegratedBotAgent) -> Result<()> {
    info!("Navigating to example page...");

    // In real scenario: page.goto("https://example.com/products").await?;
    info!("(Demo mode: simulating navigation)");

    // Check for CAPTCHA
    let captcha = agent.check_for_captcha(page).await?;

    if captcha.is_none() {
        info!("✓ No CAPTCHA detected");
        info!("  Proceeding with scraping...");

        // Simulate scraping actions
        info!("  - Extracting product data");
        info!("  - Scrolling through listings");
        info!("  - Following pagination links");

        println!("\n  Result: Successfully scraped 50 products");
    } else {
        warn!("CAPTCHA detected unexpectedly");
    }

    Ok(())
}

/// Demonstrate CAPTCHA detection and solving
async fn demo_captcha_handling(page: &Page, agent: &mut IntegratedBotAgent) -> Result<()> {
    info!("Simulating page with CAPTCHA...");

    // In real scenario, CAPTCHA would be detected automatically
    info!("(Demo mode: would detect and solve CAPTCHA here)");

    println!("\n  Real-world flow:");
    println!("  1. Navigate to protected page");
    println!("  2. Agent detects CAPTCHA (reCAPTCHA v2)");
    println!("  3. Solver extracts image grid");
    println!("  4. YOLO model identifies objects");
    println!("  5. Clicks correct tiles");
    println!("  6. Submits solution");
    println!("  7. Waits for verification");
    println!("  8. Continues scraping");

    // Simulate outcome
    let outcome = argus_captcha::CaptchaOutcome {
        was_present: true,
        solved: true,
        captcha_type: Some(argus_captcha::CaptchaType::ImageGrid),
        attempts: 1,
        solve_time: Duration::from_secs(3),
    };

    println!("\n  {}", outcome.summary());

    Ok(())
}

/// Demonstrate multi-page session with detection avoidance
async fn demo_multi_page_session(page: &Page, agent: &mut IntegratedBotAgent) -> Result<()> {
    info!("Starting multi-page scraping session...");

    let pages_to_scrape = vec![
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5",
    ];

    let mut scraped = 0;
    let mut captchas_solved = 0;

    for (i, url) in pages_to_scrape.iter().enumerate() {
        info!("Scraping page {}/{}: {}", i + 1, pages_to_scrape.len(), url);

        // In real scenario: page.goto(url).await?;

        // Add human-like delay between requests
        let delay = 1000 + rand::random::<u64>() % 2000; // 1-3 seconds
        tokio::time::sleep(Duration::from_millis(delay)).await;

        // Check for CAPTCHA
        let outcome = agent.handle_captcha(page).await?;

        if outcome.was_present {
            info!("  CAPTCHA encountered");
            if outcome.solved {
                info!("  ✓ CAPTCHA solved");
                captchas_solved += 1;
            } else {
                warn!("  ✗ CAPTCHA failed - skipping page");
                continue;
            }
        }

        // Scrape page
        info!("  Extracting data...");
        scraped += 1;
    }

    println!("\n  Session Results:");
    println!("    Pages scraped: {}/{}", scraped, pages_to_scrape.len());
    println!("    CAPTCHAs encountered: {}", captchas_solved);
    println!(
        "    Success rate: {:.1}%",
        (scraped as f32 / pages_to_scrape.len() as f32) * 100.0
    );

    Ok(())
}

/// Print agent statistics
fn print_statistics(agent: &IntegratedBotAgent) {
    let metrics = agent.metrics();

    println!("\n  CAPTCHA Metrics:");
    println!(
        "    Total encountered: {}",
        metrics.total_captchas_encountered
    );
    println!("    Solved: {}", metrics.captchas_solved);
    println!("    Failed: {}", metrics.captchas_failed);

    if metrics.total_captchas_encountered > 0 {
        let solve_rate =
            (metrics.captchas_solved as f32 / metrics.total_captchas_encountered as f32) * 100.0;
        println!("    Solve rate: {:.1}%", solve_rate);
        println!("    Avg solve time: {:?}", metrics.avg_solve_time);
    }

    let solver_metrics = agent.captcha_solver_metrics();
    if solver_metrics.total_attempts > 0 {
        println!("\n  Solver Performance:");
        println!("    Total attempts: {}", solver_metrics.total_attempts);
        println!("    Successes: {}", solver_metrics.successful_solves);
        println!(
            "    Overall success: {:.1}%",
            (solver_metrics.successful_solves as f32 / solver_metrics.total_attempts as f32)
                * 100.0
        );

        if !solver_metrics.by_type.is_empty() {
            println!("\n  By Type:");
            for (captcha_type, type_metrics) in &solver_metrics.by_type {
                println!("    {}:", captcha_type);
                println!("      Attempts: {}", type_metrics.attempts);
                println!("      Successes: {}", type_metrics.successes);
                println!(
                    "      Success rate: {:.1}%",
                    (type_metrics.successes as f32 / type_metrics.attempts as f32) * 100.0
                );
                println!(
                    "      Avg confidence: {:.1}%",
                    type_metrics.avg_confidence * 100.0
                );
                println!("      Avg time: {:?}", type_metrics.avg_solve_time);
            }
        }
    }
}

/// Generate random number (for demo delays)
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T: From<u64>>() -> T {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        T::from(nanos)
    }
}
