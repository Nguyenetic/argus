//! Argus - Intelligent Web Intelligence System
//! A simple CLI tool for web scraping

use anyhow::{Context, Result};
use argus_browser::{BrowserPool, ChromeBrowser};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Parser)]
#[command(name = "argus")]
#[command(author = "Nguyenetic")]
#[command(version = "0.1.0")]
#[command(about = "🦅 Argus - Intelligent Web Intelligence System", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scrape a URL
    Scrape {
        /// URL to scrape
        url: String,

        /// Output file (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Extract links
        #[arg(short, long)]
        links: bool,

        /// Request timeout in seconds
        #[arg(short = 't', long, default_value = "30")]
        timeout: u64,

        /// Number of retry attempts
        #[arg(short = 'r', long, default_value = "3")]
        retries: u32,

        /// Use browser automation (JavaScript rendering)
        #[arg(short = 'b', long)]
        browser: bool,

        /// Wait for specific CSS selector before scraping
        #[arg(short = 'w', long)]
        wait_for: Option<String>,

        /// Take screenshot
        #[arg(short = 's', long)]
        screenshot: bool,
    },

    /// List scraped pages
    List {
        /// Output directory
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,

        /// Filter by URL pattern
        #[arg(short = 'f', long)]
        filter: Option<String>,
    },

    /// Show statistics
    Stats {
        /// Output directory
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },

    /// Delete scraped pages
    Delete {
        /// Delete all pages
        #[arg(long)]
        all: bool,

        /// Delete specific page by URL pattern
        #[arg(short, long)]
        url: Option<String>,

        /// Delete by file name
        #[arg(short, long)]
        file: Option<PathBuf>,
    },

    /// Export scraped pages to different formats
    Export {
        /// Output directory for scraped pages
        #[arg(short = 'd', long, default_value = "./data")]
        dir: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format
        #[arg(short, long, value_enum, default_value = "json")]
        format: ExportFormat,
    },

    /// Scrape multiple URLs in parallel
    Batch {
        /// File containing URLs (one per line)
        input: PathBuf,

        /// Number of concurrent scrapers
        #[arg(short, long, default_value = "5")]
        concurrency: usize,

        /// Output directory for scraped pages
        #[arg(short, long, default_value = "./data")]
        output: PathBuf,

        /// Extract links
        #[arg(short, long)]
        links: bool,

        /// Request timeout in seconds
        #[arg(short = 't', long, default_value = "30")]
        timeout: u64,

        /// Use browser automation
        #[arg(short = 'b', long)]
        browser: bool,

        /// Rate limit (requests per second, 0 = no limit)
        #[arg(short = 'r', long, default_value = "0")]
        rate_limit: u64,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum ExportFormat {
    Json,
    Csv,
    Markdown,
    Html,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScrapedPage {
    url: String,
    title: Option<String>,
    content: String,
    links: Vec<String>,
    scraped_at: String,
    content_length: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();

    // Ensure data directory exists
    fs::create_dir_all("./data")?;

    match cli.command {
        Commands::Scrape {
            url,
            output,
            links,
            timeout,
            retries,
            browser,
            wait_for,
            screenshot,
        } => {
            scrape_command(
                &url, output, links, timeout, retries, browser, wait_for, screenshot,
            )
            .await?;
        }
        Commands::List { dir, filter } => {
            list_command(&dir, filter)?;
        }
        Commands::Stats { dir } => {
            stats_command(&dir)?;
        }
        Commands::Delete { all, url, file } => {
            delete_command(all, url, file)?;
        }
        Commands::Export {
            dir,
            output,
            format,
        } => {
            export_command(&dir, &output, format)?;
        }
        Commands::Batch {
            input,
            concurrency,
            output,
            links,
            timeout,
            browser,
            rate_limit,
        } => {
            batch_command(
                &input,
                concurrency,
                &output,
                links,
                timeout,
                browser,
                rate_limit,
            )
            .await?;
        }
    }

    Ok(())
}

async fn scrape_command(
    url: &str,
    output: Option<PathBuf>,
    extract_links: bool,
    timeout_secs: u64,
    max_retries: u32,
    use_browser: bool,
    wait_for: Option<String>,
    take_screenshot: bool,
) -> Result<()> {
    println!("{}", "🦅 Argus - Starting scrape...".bright_blue().bold());
    println!("URL: {}", url.bright_white());
    if use_browser {
        println!(
            "{}",
            "🌐 Using browser automation (JavaScript enabled)".bright_cyan()
        );
    }
    println!();

    // Fetch HTML content
    let html = if use_browser {
        fetch_with_browser(url, wait_for.as_deref(), timeout_secs, take_screenshot).await?
    } else {
        fetch_with_http(url, timeout_secs, max_retries).await?
    };

    // Parse HTML
    println!("{}", "🔍 Parsing content...".yellow());
    let document = scraper::Html::parse_document(&html);

    // Extract title
    let title_selector = scraper::Selector::parse("title")
        .map_err(|_| anyhow::anyhow!("Failed to parse title selector"))?;
    let title = document
        .select(&title_selector)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string());

    // Extract text content
    let body_selector = scraper::Selector::parse("body")
        .map_err(|_| anyhow::anyhow!("Failed to parse body selector"))?;
    let content = document
        .select(&body_selector)
        .next()
        .map(|el| {
            el.text()
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    // Extract links if requested
    let mut links = Vec::new();
    if extract_links {
        let link_selector = scraper::Selector::parse("a[href]")
            .map_err(|_| anyhow::anyhow!("Failed to parse link selector"))?;
        links = document
            .select(&link_selector)
            .filter_map(|el| el.value().attr("href"))
            .map(|s| s.to_string())
            .collect();
    }

    // Create result
    let scraped = ScrapedPage {
        url: url.to_string(),
        title: title.clone(),
        content: content.trim().to_string(),
        links: links.clone(),
        scraped_at: Utc::now().to_rfc3339(),
        content_length: content.len(),
    };

    // Display results
    println!();
    println!("{}", "✅ Scrape complete!".bright_green().bold());
    println!();
    println!(
        "{}: {}",
        "Title".bright_cyan(),
        title.as_deref().unwrap_or("N/A")
    );
    println!(
        "{}: {} characters",
        "Content".bright_cyan(),
        scraped.content_length
    );
    if extract_links {
        println!("{}: {}", "Links found".bright_cyan(), links.len());
    }

    // Save to file
    let output_path = output.unwrap_or_else(|| {
        let filename = format!("page_{}.json", uuid::Uuid::new_v4());
        PathBuf::from("./data").join(filename)
    });

    fs::write(
        &output_path,
        serde_json::to_string_pretty(&scraped).context("Failed to serialize scraped data")?,
    )
    .context("Failed to write output file")?;

    println!();
    println!(
        "{} {}",
        "💾 Saved to:".bright_green(),
        output_path.display()
    );

    Ok(())
}

fn list_command(dir: &PathBuf, filter: Option<String>) -> Result<()> {
    println!("{}", "📚 Scraped Pages".bright_blue().bold());
    println!();

    let entries =
        fs::read_dir(dir).context(format!("Failed to read directory: {}", dir.display()))?;
    let mut count = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(page) = serde_json::from_str::<ScrapedPage>(&content) {
                // Apply filter if specified
                if let Some(ref pattern) = filter {
                    if !page.url.contains(pattern) {
                        continue;
                    }
                }

                count += 1;
                println!("{}", format!("{}. {}", count, page.url).bright_white());
                if let Some(title) = page.title {
                    println!("   📄 {}", title.bright_yellow());
                }
                println!("   🕒 {}", page.scraped_at);
                println!("   📁 {}", path.file_name().unwrap().to_string_lossy());
                println!();
            }
        }
    }

    if count == 0 {
        if filter.is_some() {
            println!("{}", "No pages found matching the filter.".yellow());
        } else {
            println!("{}", "No pages found. Try scraping a URL first!".yellow());
        }
    } else {
        println!("{} {}", "Total:".bright_green(), count);
    }

    Ok(())
}

fn stats_command(dir: &PathBuf) -> Result<()> {
    println!("{}", "📊 Statistics".bright_blue().bold());
    println!();

    let entries =
        fs::read_dir(dir).context(format!("Failed to read directory: {}", dir.display()))?;
    let mut total_pages = 0;
    let mut total_content = 0;
    let mut total_links = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(page) = serde_json::from_str::<ScrapedPage>(&content) {
                total_pages += 1;
                total_content += page.content_length;
                total_links += page.links.len();
            }
        }
    }

    println!("{}: {}", "Total pages scraped".bright_cyan(), total_pages);
    println!(
        "{}: {} characters",
        "Total content".bright_cyan(),
        total_content
    );
    println!("{}: {}", "Total links extracted".bright_cyan(), total_links);

    if total_pages > 0 {
        println!(
            "{}: {} characters",
            "Average content".bright_cyan(),
            total_content / total_pages
        );
    }

    Ok(())
}

fn delete_command(all: bool, url_pattern: Option<String>, file: Option<PathBuf>) -> Result<()> {
    if !all && url_pattern.is_none() && file.is_none() {
        return Err(anyhow::anyhow!(
            "Must specify one of: --all, --url <pattern>, or --file <path>"
        ));
    }

    let data_dir = PathBuf::from("./data");
    let mut deleted_count = 0;

    if let Some(file_path) = file {
        // Delete specific file
        let full_path = if file_path.is_absolute() {
            file_path
        } else {
            data_dir.join(&file_path)
        };

        if full_path.exists() {
            fs::remove_file(&full_path)
                .context(format!("Failed to delete file: {}", full_path.display()))?;
            println!("{} {}", "🗑️  Deleted:".bright_red(), full_path.display());
            deleted_count = 1;
        } else {
            return Err(anyhow::anyhow!("File not found: {}", full_path.display()));
        }
    } else {
        // Delete by pattern or all
        let entries = fs::read_dir(&data_dir)
            .context(format!("Failed to read directory: {}", data_dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let should_delete = if all {
                    true
                } else if let Some(ref pattern) = url_pattern {
                    // Check if URL matches pattern
                    let content = fs::read_to_string(&path)?;
                    if let Ok(page) = serde_json::from_str::<ScrapedPage>(&content) {
                        page.url.contains(pattern)
                    } else {
                        false
                    }
                } else {
                    false
                };

                if should_delete {
                    fs::remove_file(&path)
                        .context(format!("Failed to delete file: {}", path.display()))?;
                    println!("{} {}", "🗑️  Deleted:".bright_red(), path.display());
                    deleted_count += 1;
                }
            }
        }
    }

    println!();
    println!("{} {}", "Total deleted:".bright_green(), deleted_count);

    Ok(())
}

fn export_command(dir: &PathBuf, output: &PathBuf, format: ExportFormat) -> Result<()> {
    println!("{}", "📤 Exporting scraped pages...".bright_blue().bold());
    println!();

    // Collect all pages
    let entries =
        fs::read_dir(dir).context(format!("Failed to read directory: {}", dir.display()))?;
    let mut pages = Vec::new();

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(page) = serde_json::from_str::<ScrapedPage>(&content) {
                pages.push(page);
            }
        }
    }

    if pages.is_empty() {
        return Err(anyhow::anyhow!("No pages found to export"));
    }

    // Export based on format
    match format {
        ExportFormat::Json => export_json(&pages, output)?,
        ExportFormat::Csv => export_csv(&pages, output)?,
        ExportFormat::Markdown => export_markdown(&pages, output)?,
        ExportFormat::Html => export_html(&pages, output)?,
    }

    println!("{} {}", "✅ Exported to:".bright_green(), output.display());
    println!("{} {} pages", "📊 Total:".bright_cyan(), pages.len());

    Ok(())
}

fn export_json(pages: &[ScrapedPage], output: &PathBuf) -> Result<()> {
    let json = serde_json::to_string_pretty(pages).context("Failed to serialize pages to JSON")?;
    fs::write(output, json).context("Failed to write JSON file")?;
    Ok(())
}

fn export_csv(pages: &[ScrapedPage], output: &PathBuf) -> Result<()> {
    let mut writer = csv::Writer::from_path(output).context("Failed to create CSV writer")?;

    // Write header
    writer
        .write_record(&[
            "URL",
            "Title",
            "Content Length",
            "Links Count",
            "Scraped At",
        ])
        .context("Failed to write CSV header")?;

    // Write rows
    for page in pages {
        writer
            .write_record(&[
                &page.url,
                page.title.as_deref().unwrap_or("N/A"),
                &page.content_length.to_string(),
                &page.links.len().to_string(),
                &page.scraped_at,
            ])
            .context("Failed to write CSV row")?;
    }

    writer.flush().context("Failed to flush CSV writer")?;
    Ok(())
}

fn export_markdown(pages: &[ScrapedPage], output: &PathBuf) -> Result<()> {
    let mut md = String::new();
    md.push_str("# Scraped Pages\n\n");

    for (i, page) in pages.iter().enumerate() {
        md.push_str(&format!("## {}. {}\n\n", i + 1, page.url));
        if let Some(ref title) = page.title {
            md.push_str(&format!("**Title:** {}\n\n", title));
        }
        md.push_str(&format!("**Scraped at:** {}\n\n", page.scraped_at));
        md.push_str(&format!(
            "**Content length:** {} characters\n\n",
            page.content_length
        ));
        if !page.links.is_empty() {
            md.push_str(&format!("**Links found:** {}\n\n", page.links.len()));
        }
        md.push_str("---\n\n");
    }

    fs::write(output, md).context("Failed to write Markdown file")?;
    Ok(())
}

fn export_html(pages: &[ScrapedPage], output: &PathBuf) -> Result<()> {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("  <meta charset=\"UTF-8\">\n");
    html.push_str("  <title>Scraped Pages</title>\n");
    html.push_str("  <style>\n");
    html.push_str("    body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }\n");
    html.push_str("    .page { border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px; }\n");
    html.push_str("    .url { color: #0066cc; font-weight: bold; }\n");
    html.push_str("    .title { font-size: 1.2em; margin: 10px 0; }\n");
    html.push_str("    .meta { color: #666; font-size: 0.9em; }\n");
    html.push_str("  </style>\n");
    html.push_str("</head>\n<body>\n");
    html.push_str("  <h1>Scraped Pages</h1>\n");

    for (i, page) in pages.iter().enumerate() {
        html.push_str("  <div class=\"page\">\n");
        html.push_str(&format!(
            "    <div class=\"url\">{}. {}</div>\n",
            i + 1,
            escape_html(&page.url)
        ));
        if let Some(ref title) = page.title {
            html.push_str(&format!(
                "    <div class=\"title\">{}</div>\n",
                escape_html(title)
            ));
        }
        html.push_str(&format!(
            "    <div class=\"meta\">Scraped: {}</div>\n",
            page.scraped_at
        ));
        html.push_str(&format!(
            "    <div class=\"meta\">Content: {} characters</div>\n",
            page.content_length
        ));
        if !page.links.is_empty() {
            html.push_str(&format!(
                "    <div class=\"meta\">Links: {}</div>\n",
                page.links.len()
            ));
        }
        html.push_str("  </div>\n");
    }

    html.push_str("</body>\n</html>");

    fs::write(output, html).context("Failed to write HTML file")?;
    Ok(())
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Fetch page using HTTP client with retries
async fn fetch_with_http(url: &str, timeout_secs: u64, max_retries: u32) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .build()
        .context("Failed to create HTTP client")?;

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );

    let mut last_error = None;
    let mut html = String::new();

    for attempt in 1..=max_retries {
        spinner.set_message(format!(
            "📡 Fetching page (attempt {}/{})",
            attempt, max_retries
        ));
        spinner.tick();

        match client.get(url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    match response.text().await {
                        Ok(text) => {
                            html = text;
                            break;
                        }
                        Err(e) => {
                            last_error =
                                Some(anyhow::anyhow!("Failed to read response body: {}", e));
                        }
                    }
                } else {
                    last_error = Some(anyhow::anyhow!(
                        "HTTP error {}: {}",
                        response.status(),
                        response.status().canonical_reason().unwrap_or("Unknown")
                    ));
                }
            }
            Err(e) => {
                last_error = Some(anyhow::anyhow!("Network error: {}", e));
            }
        }

        if attempt < max_retries {
            let wait_time = Duration::from_secs(2u64.pow(attempt - 1));
            spinner.set_message(format!("⏳ Waiting {:?} before retry...", wait_time));
            tokio::time::sleep(wait_time).await;
        }
    }

    spinner.finish_and_clear();

    if html.is_empty() {
        return Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Failed to fetch page")));
    }

    Ok(html)
}

/// Batch scrape multiple URLs in parallel
async fn batch_command(
    input: &PathBuf,
    concurrency: usize,
    output_dir: &PathBuf,
    extract_links: bool,
    timeout_secs: u64,
    use_browser: bool,
    rate_limit: u64,
) -> Result<()> {
    println!("{}", "🦅 Argus - Batch Scraping".bright_blue().bold());
    println!();

    // Read URLs from file
    let urls_content = fs::read_to_string(input)
        .context(format!("Failed to read URLs from {}", input.display()))?;

    let urls: Vec<String> = urls_content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|s| s.to_string())
        .collect();

    if urls.is_empty() {
        return Err(anyhow::anyhow!("No URLs found in {}", input.display()));
    }

    println!("📋 Found {} URLs to scrape", urls.len());
    println!("⚡ Concurrency: {}", concurrency);
    if use_browser {
        println!("🌐 Using browser automation");
    }
    if rate_limit > 0 {
        println!("⏱️  Rate limit: {} req/sec", rate_limit);
    }
    println!();

    // Create output directory
    fs::create_dir_all(output_dir)?;

    // Setup progress tracking
    let multi_progress = MultiProgress::new();
    let main_progress = multi_progress.add(ProgressBar::new(urls.len() as u64));
    main_progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let start_time = std::time::Instant::now();

    // Initialize browser pool if using browser mode
    let browser_pool = if use_browser {
        Some(BrowserPool::new(concurrency).await?)
    } else {
        None
    };

    // Process URLs in parallel
    let results = if use_browser {
        // Browser mode with pool
        scrape_batch_browser(
            &urls,
            browser_pool.unwrap(),
            extract_links,
            timeout_secs,
            &main_progress,
            rate_limit,
        )
        .await
    } else {
        // HTTP mode with tokio tasks
        scrape_batch_http(
            &urls,
            concurrency,
            extract_links,
            timeout_secs,
            &main_progress,
            rate_limit,
        )
        .await
    };

    main_progress.finish_with_message("Complete!");

    // Summary
    let duration = start_time.elapsed();
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let error_count = results.iter().filter(|r| r.is_err()).count();

    println!();
    println!("{}", "✅ Batch scraping complete!".bright_green().bold());
    println!();
    println!("{}: {}", "Total URLs".bright_cyan(), urls.len());
    println!("{}: {}", "Successful".bright_green(), success_count);
    println!("{}: {}", "Failed".bright_red(), error_count);
    println!(
        "{}: {:.2}s",
        "Duration".bright_cyan(),
        duration.as_secs_f64()
    );
    println!(
        "{}: {:.2} pages/sec",
        "Throughput".bright_cyan(),
        urls.len() as f64 / duration.as_secs_f64()
    );

    // Save results
    for (i, result) in results.iter().enumerate() {
        if let Ok(page) = result {
            let filename = format!("batch_{}_{}.json", i, uuid::Uuid::new_v4());
            let path = output_dir.join(filename);
            fs::write(&path, serde_json::to_string_pretty(&page)?)?;
        }
    }

    Ok(())
}

/// Scrape URLs in parallel using HTTP client
async fn scrape_batch_http(
    urls: &[String],
    concurrency: usize,
    extract_links: bool,
    timeout_secs: u64,
    progress: &ProgressBar,
    rate_limit: u64,
) -> Vec<Result<ScrapedPage>> {
    use futures::stream::{self, StreamExt};

    let rate_limiter = if rate_limit > 0 {
        Some(Duration::from_millis(1000 / rate_limit))
    } else {
        None
    };

    stream::iter(urls)
        .map(|url| {
            let url = url.clone();
            let progress = progress.clone();
            async move {
                let result = scrape_url_http(&url, extract_links, timeout_secs).await;
                progress.inc(1);

                if let Some(delay) = rate_limiter {
                    sleep(delay).await;
                }

                result
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await
}

/// Scrape URLs in parallel using browser pool
async fn scrape_batch_browser(
    urls: &[String],
    pool: BrowserPool,
    extract_links: bool,
    _timeout_secs: u64,
    progress: &ProgressBar,
    rate_limit: u64,
) -> Vec<Result<ScrapedPage>> {
    use futures::stream::{self, StreamExt};

    let rate_limiter = if rate_limit > 0 {
        Some(Duration::from_millis(1000 / rate_limit))
    } else {
        None
    };

    stream::iter(urls)
        .map(|url| {
            let url = url.clone();
            let pool = &pool;
            let progress = progress.clone();
            async move {
                let result = scrape_url_browser(&url, pool, extract_links).await;
                progress.inc(1);

                if let Some(delay) = rate_limiter {
                    sleep(delay).await;
                }

                result
            }
        })
        .buffer_unordered(pool.size())
        .collect()
        .await
}

/// Scrape a single URL using HTTP
async fn scrape_url_http(url: &str, extract_links: bool, timeout_secs: u64) -> Result<ScrapedPage> {
    let html = fetch_with_http(url, timeout_secs, 3).await?;
    parse_html(url, &html, extract_links)
}

/// Scrape a single URL using browser pool
async fn scrape_url_browser(
    url: &str,
    pool: &BrowserPool,
    extract_links: bool,
) -> Result<ScrapedPage> {
    let guard = pool.acquire().await?;
    let browser = guard.browser();

    let html = browser.navigate(url, None).await?;
    parse_html(url, &html, extract_links)
}

/// Parse HTML and extract data
fn parse_html(url: &str, html: &str, extract_links: bool) -> Result<ScrapedPage> {
    let document = scraper::Html::parse_document(html);

    // Extract title
    let title_selector = scraper::Selector::parse("title")
        .map_err(|_| anyhow::anyhow!("Failed to parse title selector"))?;
    let title = document
        .select(&title_selector)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string());

    // Extract text content
    let body_selector = scraper::Selector::parse("body")
        .map_err(|_| anyhow::anyhow!("Failed to parse body selector"))?;
    let content = document
        .select(&body_selector)
        .next()
        .map(|el| {
            el.text()
                .collect::<Vec<_>>()
                .join(" ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();

    // Extract links if requested
    let mut links = Vec::new();
    if extract_links {
        let link_selector = scraper::Selector::parse("a[href]")
            .map_err(|_| anyhow::anyhow!("Failed to parse link selector"))?;
        links = document
            .select(&link_selector)
            .filter_map(|el| el.value().attr("href"))
            .map(|s| s.to_string())
            .collect();
    }

    Ok(ScrapedPage {
        url: url.to_string(),
        title,
        content: content.trim().to_string(),
        links,
        scraped_at: Utc::now().to_rfc3339(),
        content_length: content.len(),
    })
}

/// Fetch page using browser automation
async fn fetch_with_browser(
    url: &str,
    wait_for: Option<&str>,
    _timeout_secs: u64,
    take_screenshot: bool,
) -> Result<String> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );

    spinner.set_message("🌐 Launching browser...");
    let browser = ChromeBrowser::new(true)
        .await
        .context("Failed to launch browser")?;

    spinner.set_message("📡 Navigating to page...");
    let html = browser
        .navigate(url, wait_for)
        .await
        .context("Failed to navigate")?;

    if take_screenshot {
        spinner.set_message("📸 Taking screenshot...");
        let screenshot_data = browser
            .screenshot(url, true)
            .await
            .context("Failed to take screenshot")?;

        let screenshot_path =
            PathBuf::from("./data").join(format!("screenshot_{}.png", uuid::Uuid::new_v4()));
        fs::write(&screenshot_path, screenshot_data).context("Failed to save screenshot")?;

        println!("📸 Screenshot saved to: {}", screenshot_path.display());
    }

    spinner.finish_and_clear();
    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("Hello"), "Hello");
        assert_eq!(escape_html("<script>"), "&lt;script&gt;");
        assert_eq!(escape_html("A & B"), "A &amp; B");
        assert_eq!(escape_html("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_html("'single'"), "&#39;single&#39;");
    }

    #[test]
    fn test_scraped_page_serialization() {
        let page = ScrapedPage {
            url: "https://example.com".to_string(),
            title: Some("Example".to_string()),
            content: "Test content".to_string(),
            links: vec!["https://link1.com".to_string()],
            scraped_at: "2024-01-01T00:00:00Z".to_string(),
            content_length: 12,
        };

        let json = serde_json::to_string(&page).unwrap();
        let deserialized: ScrapedPage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.url, "https://example.com");
        assert_eq!(deserialized.title, Some("Example".to_string()));
        assert_eq!(deserialized.content_length, 12);
    }

    #[test]
    fn test_export_format_variants() {
        // Just ensure all variants exist
        let _json = ExportFormat::Json;
        let _csv = ExportFormat::Csv;
        let _markdown = ExportFormat::Markdown;
        let _html = ExportFormat::Html;
    }
}
