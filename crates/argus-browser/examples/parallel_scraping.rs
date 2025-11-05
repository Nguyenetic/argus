/// Parallel scraping example with browser pool
///
/// Run with: cargo run --example parallel_scraping
use anyhow::Result;
use argus_browser::IntelligentBrowserBuilder;
use chromiumoxide::Page;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
struct PageInfo {
    url: String,
    title: String,
    word_count: usize,
    load_time_ms: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Parallel Scraping Demo ===\n");

    // Create browser with larger pool for parallel scraping
    let browser = IntelligentBrowserBuilder::new()
        .pool_size(3, 10)
        .stealth(true)
        .request_delay(500, 1500)
        .headless(true)
        .build()
        .await?;

    // List of URLs to scrape
    let urls = vec![
        "https://example.com".to_string(),
        "https://httpbin.org/html".to_string(),
        "https://httpbin.org/headers".to_string(),
        "https://httpbin.org/ip".to_string(),
        "https://httpbin.org/user-agent".to_string(),
    ];

    println!("Scraping {} URLs in parallel...\n", urls.len());

    let start_time = Instant::now();

    // Scrape all URLs in parallel
    let results = browser
        .scrape_parallel(urls, |page| Box::pin(extract_page_info(page)))
        .await?;

    let total_time = start_time.elapsed();

    // Print results
    println!("\n=== Results ===\n");

    let mut successful = 0;
    let mut failed = 0;

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(info) => {
                println!("Page {}: ✓", i + 1);
                println!("  URL: {}", info.url);
                println!("  Title: {}", info.title);
                println!("  Words: {}", info.word_count);
                println!("  Load time: {}ms", info.load_time_ms);
                println!();
                successful += 1;
            }
            Err(e) => {
                println!("Page {}: ✗ Error: {}", i + 1, e);
                println!();
                failed += 1;
            }
        }
    }

    // Print summary
    println!("=== Summary ===");
    println!("Total pages: {}", results.len());
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!("Total time: {:.2}s", total_time.as_secs_f32());
    println!(
        "Avg time/page: {:.2}s",
        total_time.as_secs_f32() / results.len() as f32
    );

    // Print pool stats
    println!("\n=== Pool Statistics ===");
    let stats = browser.stats().await;
    println!("Total instances used: {}", stats.total_instances);
    println!("Total pages created: {}", stats.total_pages);
    println!("Pool utilization: {:.1}%", stats.utilization() * 100.0);

    // Cleanup
    browser.cleanup().await?;

    Ok(())
}

/// Extract page information
async fn extract_page_info(page: &Page) -> Result<PageInfo> {
    let start = Instant::now();

    // Get URL
    let url = page.url().await?.unwrap_or_default();

    // Get title
    let title_js = "document.title || 'No title'";
    let title: String = page.evaluate(title_js).await?.into_value()?;

    // Count words
    let word_count_js = r#"
        (document.body.innerText || '').split(/\s+/).filter(w => w.length > 0).length
    "#;
    let word_count: usize = page.evaluate(word_count_js).await?.into_value()?;

    let load_time_ms = start.elapsed().as_millis() as u64;

    Ok(PageInfo {
        url,
        title,
        word_count,
        load_time_ms,
    })
}
