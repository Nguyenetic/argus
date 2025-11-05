/// Basic usage examples for intelligent browser automation
///
/// Run with: cargo run --example basic_usage
use anyhow::Result;
use argus_browser::{IntelligentBrowserBuilder, PoolStats};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Argus Browser - Basic Usage ===\n");

    // Example 1: Simple navigation
    println!("--- Example 1: Simple Navigation ---");
    simple_navigation().await?;

    // Example 2: Browser pool
    println!("\n--- Example 2: Browser Pool ---");
    browser_pool_demo().await?;

    // Example 3: Stealth mode
    println!("\n--- Example 3: Stealth Mode ---");
    stealth_mode_demo().await?;

    // Example 4: Session persistence
    println!("\n--- Example 4: Session Persistence ---");
    session_demo().await?;

    Ok(())
}

/// Example 1: Simple navigation
async fn simple_navigation() -> Result<()> {
    let browser = IntelligentBrowserBuilder::new()
        .pool_size(1, 3)
        .stealth(true)
        .headless(true)
        .build()
        .await?;

    println!("  Navigating to example.com...");
    let session = browser.navigate("https://example.com").await?;

    println!("  ✓ Page loaded");
    println!("  Session ID: {:?}", session.session_id);

    // Get page title
    let title = session.execute("document.title").await?;
    println!("  Page title: {}", title);

    // Cleanup
    browser.cleanup().await?;
    println!("  ✓ Browser closed");

    Ok(())
}

/// Example 2: Browser pool management
async fn browser_pool_demo() -> Result<()> {
    let browser = IntelligentBrowserBuilder::new()
        .pool_size(2, 5)
        .stealth(true)
        .build()
        .await?;

    println!("  Initial pool stats:");
    print_stats(&browser.stats().await);

    // Navigate to multiple pages
    println!("\n  Opening multiple pages...");
    let session1 = browser.navigate("https://example.com").await?;
    let session2 = browser.navigate("https://httpbin.org/html").await?;

    println!("  ✓ Opened 2 pages");

    println!("\n  Updated pool stats:");
    print_stats(&browser.stats().await);

    // Sessions automatically cleaned up when dropped
    drop(session1);
    drop(session2);

    browser.cleanup().await?;

    Ok(())
}

/// Example 3: Stealth mode demonstration
async fn stealth_mode_demo() -> Result<()> {
    let browser = IntelligentBrowserBuilder::new()
        .stealth(true)
        .pool_size(1, 2)
        .build()
        .await?;

    println!("  Navigating with stealth mode enabled...");
    let session = browser.navigate("https://httpbin.org/headers").await?;

    // Check if webdriver is detected
    let webdriver = session.execute("navigator.webdriver").await?;
    println!("  navigator.webdriver: {}", webdriver);

    // Check user agent
    let user_agent = session.execute("navigator.userAgent").await?;
    println!("  User Agent: {}", user_agent);

    // Check plugins
    let plugins_count = session.execute("navigator.plugins.length").await?;
    println!("  Plugins count: {}", plugins_count);

    println!("  ✓ Stealth mode applied");

    browser.cleanup().await?;

    Ok(())
}

/// Example 4: Session persistence
async fn session_demo() -> Result<()> {
    let browser = IntelligentBrowserBuilder::new()
        .session_storage("./data/sessions")
        .build()
        .await?;

    println!("  Creating new session...");
    let session = browser.navigate("https://example.com").await?;

    if let Some(session_id) = &session.session_id {
        println!("  ✓ Session created: {}", session_id);

        // Session is automatically saved
        drop(session);

        // Resume session
        println!("\n  Resuming session...");
        let resumed = browser
            .resume_session(session_id, "https://example.com")
            .await?;
        println!("  ✓ Session resumed");

        drop(resumed);
    }

    browser.cleanup().await?;

    Ok(())
}

/// Print pool statistics
fn print_stats(stats: &PoolStats) {
    println!("    Total instances: {}", stats.total_instances);
    println!("    Healthy instances: {}", stats.healthy_instances);
    println!("    Idle instances: {}", stats.idle_instances);
    println!("    Total pages: {}", stats.total_pages);
    println!("    Available slots: {}", stats.available_slots);
    println!("    Utilization: {:.1}%", stats.utilization() * 100.0);
}
