/// Complete scraper example with all features
///
/// Demonstrates:
/// - Browser pool management
/// - Stealth mode
/// - Session persistence
/// - Error handling and retries
/// - Progress tracking
/// - Data extraction
///
/// Run with: cargo run --example complete_scraper
use anyhow::Result;
use argus_browser::{IntelligentBrowserBuilder, PageSession};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Product {
    name: String,
    price: f64,
    url: String,
}

#[derive(Debug)]
struct ScraperStats {
    total_pages: usize,
    successful: usize,
    failed: usize,
    total_time: Duration,
    products_found: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Complete E-Commerce Scraper Demo ===\n");

    // Configuration
    let config = ScraperConfig {
        base_url: "https://example-shop.com".to_string(),
        max_pages: 5,
        products_per_page: 20,
        enable_stealth: true,
        save_sessions: true,
    };

    // Run scraper
    let scraper = Scraper::new(config).await?;
    let results = scraper.run().await?;

    // Print results
    print_results(&results);

    Ok(())
}

struct ScraperConfig {
    base_url: String,
    max_pages: usize,
    products_per_page: usize,
    enable_stealth: bool,
    save_sessions: bool,
}

struct Scraper {
    config: ScraperConfig,
    browser: argus_browser::IntelligentBrowser,
}

impl Scraper {
    async fn new(config: ScraperConfig) -> Result<Self> {
        let browser = IntelligentBrowserBuilder::new()
            .pool_size(2, 5)
            .stealth(config.enable_stealth)
            .request_delay(1000, 3000)
            .max_retries(3)
            .session_storage("./data/sessions")
            .headless(true)
            .build()
            .await?;

        Ok(Self { config, browser })
    }

    async fn run(&self) -> Result<ScraperResults> {
        let start_time = Instant::now();
        let mut stats = ScraperStats {
            total_pages: 0,
            successful: 0,
            failed: 0,
            total_time: Duration::ZERO,
            products_found: 0,
        };

        let mut all_products = Vec::new();

        println!("Starting scrape of {} pages...\n", self.config.max_pages);

        for page_num in 1..=self.config.max_pages {
            println!("Scraping page {}/{}...", page_num, self.config.max_pages);
            stats.total_pages += 1;

            let url = format!("{}/products?page={}", self.config.base_url, page_num);

            match self.scrape_page(&url).await {
                Ok(products) => {
                    stats.successful += 1;
                    stats.products_found += products.len();
                    all_products.extend(products);
                    println!("  ✓ Found {} products", stats.products_found);
                }
                Err(e) => {
                    stats.failed += 1;
                    println!("  ✗ Error: {}", e);
                }
            }

            // Progress indicator
            let progress = (page_num as f32 / self.config.max_pages as f32) * 100.0;
            println!("  Progress: {:.1}%\n", progress);
        }

        stats.total_time = start_time.elapsed();

        Ok(ScraperResults {
            products: all_products,
            stats,
        })
    }

    async fn scrape_page(&self, url: &str) -> Result<Vec<Product>> {
        // Navigate with retry logic
        let session = self.browser.navigate_with_retry(url).await?;

        // Wait for products to load
        session.wait_for(".product-card").await.ok();

        // Extract products
        let products = extract_products(&session).await?;

        Ok(products)
    }
}

struct ScraperResults {
    products: Vec<Product>,
    stats: ScraperStats,
}

/// Extract products from page
async fn extract_products(session: &PageSession) -> Result<Vec<Product>> {
    let js = r#"
        Array.from(document.querySelectorAll('.product-card')).map(el => ({
            name: el.querySelector('.product-name')?.innerText || 'Unknown',
            price: parseFloat(el.querySelector('.product-price')?.innerText.replace(/[^0-9.]/g, '') || '0'),
            url: el.querySelector('a')?.href || ''
        }))
    "#;

    let result = session.execute(js).await?;
    let products: Vec<Product> = serde_json::from_value(result)?;

    Ok(products)
}

/// Print scraping results
fn print_results(results: &ScraperResults) {
    println!("\n=== Scraping Results ===\n");

    // Statistics
    println!(
        "Pages scraped: {}/{}",
        results.stats.successful, results.stats.total_pages
    );
    println!("Failed pages: {}", results.stats.failed);
    println!(
        "Success rate: {:.1}%",
        (results.stats.successful as f32 / results.stats.total_pages as f32) * 100.0
    );
    println!("Products found: {}", results.stats.products_found);
    println!("Total time: {:.2}s", results.stats.total_time.as_secs_f32());
    println!(
        "Avg time/page: {:.2}s",
        results.stats.total_time.as_secs_f32() / results.stats.total_pages as f32
    );

    // Sample products
    if !results.products.is_empty() {
        println!("\n=== Sample Products (first 5) ===\n");
        for (i, product) in results.products.iter().take(5).enumerate() {
            println!("{}. {} - ${:.2}", i + 1, product.name, product.price);
            println!("   URL: {}", product.url);
        }

        if results.products.len() > 5 {
            println!("\n   ... and {} more", results.products.len() - 5);
        }
    }

    // Price statistics
    if !results.products.is_empty() {
        let prices: Vec<f64> = results.products.iter().map(|p| p.price).collect();
        let avg_price = prices.iter().sum::<f64>() / prices.len() as f64;
        let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== Price Statistics ===");
        println!("Average: ${:.2}", avg_price);
        println!("Minimum: ${:.2}", min_price);
        println!("Maximum: ${:.2}", max_price);
    }
}
