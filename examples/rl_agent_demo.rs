/// Complete RL Agent Demo
///
/// This example demonstrates the full RL agent system in action:
/// 1. Launch browser with chromiumoxide
/// 2. Load trained RL agent
/// 3. Execute intelligent bot evasion on real websites
/// 4. Track detection and behavior metrics
use anyhow::Result;
use argus_rl::{AgentConfig, IntegratedAgent};
use chromiumoxide::browser::{Browser, BrowserConfig};
use futures::StreamExt;
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Argus RL Agent Demo ===");

    // Launch browser
    info!("Launching browser...");
    let (browser, mut handler) = Browser::launch(
        BrowserConfig::builder()
            .window_size(1920, 1080)
            .build()
            .map_err(|e| anyhow::anyhow!("Browser config error: {}", e))?,
    )
    .await?;

    // Spawn browser handler
    tokio::spawn(async move {
        while let Some(event) = handler.next().await {
            if let Err(e) = event {
                eprintln!("Browser error: {}", e);
            }
        }
    });

    info!("Browser launched successfully");

    // Create page
    let page = browser.new_page("about:blank").await?;

    // Configure agent
    let config = AgentConfig {
        state_window_size: 50,
        stop_on_detection: false, // Continue even if detected
        max_duration: Duration::from_secs(300),
        debug: false,
    };

    // Create integrated agent
    info!("Loading RL agent...");
    let mut agent = IntegratedAgent::new("models/sdsac_bot_evasion", 1920.0, 1080.0, config)?;

    info!("Agent loaded successfully");

    // Test websites (from least to most protected)
    let test_sites = vec![
        ("Simple Test", "https://example.com"),
        ("E-commerce", "https://www.amazon.com"),
        ("Social Media", "https://www.reddit.com"),
        ("Protected", "https://www.cloudflare.com"),
    ];

    for (name, url) in test_sites {
        info!("\n=== Testing: {} ({}) ===", name, url);

        // Navigate to site
        page.goto(url).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Run agent for 20 steps
        info!("Running agent...");
        let result = agent.run(&page, 20).await?;

        // Print results
        info!("\n{}", result.summary());
        info!("Detections: {}", result.detections);
        info!(
            "Success: {}",
            if result.is_successful() {
                "YES ✓"
            } else {
                "NO ✗"
            }
        );

        // Print per-step details
        info!("\nStep-by-step breakdown:");
        for (i, step) in result.steps.iter().enumerate() {
            let detection_marker = if step.detection.detected {
                "⚠️"
            } else {
                "✓"
            };
            info!(
                "  Step {:2}: {:?} - {} ({}ms) {}",
                i + 1,
                step.action,
                step.result.action_type,
                step.result.duration_ms,
                detection_marker
            );

            if let Some(ref reason) = step.detection.reason {
                info!("           Detection: {}", reason);
            }
        }

        // Print statistics
        let stats = agent.statistics();
        info!("\nSession Statistics:");
        info!("  Total actions: {}", stats.total_actions);
        info!("  Requests: {}", stats.requests_in_session);
        info!("  Success rate: {:.1}%", stats.success_rate * 100.0);
        info!("  Avg interval: {:.2}s", stats.avg_request_interval);
        info!("  Behavior score: {:.2}", result.behavior_score);

        // Wait between tests
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    info!("\n=== Demo Complete ===");

    // Close browser
    browser.close().await?;

    Ok(())
}
