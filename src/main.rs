//! Argus - Intelligent Web Intelligence System
//! A simple CLI tool for web scraping

mod commands;

use anyhow::{Context, Result};
use argus_browser::{BrowserPool, ChromeBrowser};
use argus_extract::{ExtractionConfig, ExtractorEngine};
use argus_storage::{NexusQLConfig, RedisCacheConfig, ScrapedPage, Storage, StorageConfig};

use clap::{Parser, Subcommand, ValueEnum};
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;

#[cfg(feature = "rl")]
use argus_rl::RLAgent;

#[derive(Parser)]
#[command(name = "argus")]
#[command(author = "Nguyenetic")]
#[command(version = "0.1.0")]
#[command(about = "🦅 Argus - Intelligent Web Intelligence System", long_about = None)]
struct Cli {
    /// Database file path (use ":memory:" for in-memory)
    #[arg(long, default_value = "./data/argus.db", global = true)]
    db_path: String,

    /// Redis URL for caching (optional)
    #[arg(long, global = true)]
    redis_url: Option<String>,

    /// Disable caching
    #[arg(long, global = true)]
    no_cache: bool,

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

    /// Scrape Google Maps reviews
    Maps {
        /// Google Maps URL (e.g., https://maps.app.goo.gl/...)
        url: String,

        /// Maximum number of reviews to fetch
        #[arg(short, long, default_value = "50")]
        max_reviews: usize,

        /// Output JSON file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Quiet mode (minimal output, for cron jobs)
        #[arg(short, long)]
        quiet: bool,

        /// Disable headless mode (show browser window)
        #[arg(long)]
        no_headless: bool,
    },

    /// Extract structured data using a YAML configuration file
    Extract {
        /// URL to extract data from
        url: String,

        /// Path to YAML configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Output file path (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use browser automation (JavaScript rendering)
        #[arg(short = 'b', long)]
        browser: bool,

        /// Pretty print JSON output
        #[arg(long)]
        pretty: bool,

        /// Request timeout in seconds
        #[arg(short = 't', long, default_value = "30")]
        timeout: u64,
    },

    /// Validate a YAML extraction configuration file
    ValidateConfig {
        /// Path to YAML configuration file
        config: PathBuf,
    },

    /// Train the RL agent for anti-bot evasion
    #[cfg(feature = "rl")]
    Train {
        /// Number of training episodes
        #[arg(short, long, default_value = "1000")]
        episodes: usize,

        /// Output directory for model checkpoints
        #[arg(short, long, default_value = "./models")]
        output: PathBuf,

        /// Use CUDA for training (requires CUDA-enabled libtorch)
        #[arg(long)]
        cuda: bool,

        /// Learning rate
        #[arg(long, default_value = "0.0003")]
        learning_rate: f64,

        /// Batch size for experience replay
        #[arg(long, default_value = "64")]
        batch_size: usize,

        /// Discount factor (gamma)
        #[arg(long, default_value = "0.99")]
        gamma: f64,

        /// Save checkpoint every N episodes
        #[arg(long, default_value = "100")]
        checkpoint_interval: usize,
    },

    /// Scrape with RL-based anti-bot evasion
    #[cfg(feature = "rl")]
    ScrapeRl {
        /// URL to scrape
        url: String,

        /// Path to trained model (.pt file)
        #[arg(short, long)]
        model: PathBuf,

        /// Maximum steps for RL agent
        #[arg(long, default_value = "100")]
        max_steps: usize,

        /// Output file for scraped content (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Use CUDA for inference
        #[arg(long)]
        cuda: bool,

        /// Use deterministic action selection (no exploration)
        #[arg(long)]
        deterministic: bool,

        /// Verbose output (show agent decisions)
        #[arg(short, long)]
        verbose: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum ExportFormat {
    Json,
    Csv,
    Markdown,
    Html,
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

    // Initialize storage layer
    let storage_config = StorageConfig {
        nexus: NexusQLConfig {
            db_path: cli.db_path.clone(),
            enable_wal: true,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        },
        redis: if !cli.no_cache {
            cli.redis_url.as_ref().map(|url| RedisCacheConfig {
                url: url.clone(),
                default_ttl: 3600,
                key_prefix: "argus:".to_string(),
            })
        } else {
            None
        },
        enable_cache: !cli.no_cache,
    };

    let mut storage = Storage::new(storage_config)
        .await
        .context("Failed to initialize storage layer")?;

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
                &mut storage,
                &url,
                output,
                links,
                timeout,
                retries,
                browser,
                wait_for,
                screenshot,
            )
            .await?;
        }
        Commands::List { dir, filter } => {
            list_command(&mut storage, &dir, filter).await?;
        }
        Commands::Stats { dir } => {
            stats_command(&mut storage, &dir).await?;
        }
        Commands::Delete { all, url, file } => {
            delete_command(&mut storage, all, url, file).await?;
        }
        Commands::Export {
            dir,
            output,
            format,
        } => {
            export_command(&mut storage, &dir, &output, format).await?;
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
                &mut storage,
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
        Commands::Maps {
            url,
            max_reviews,
            output,
            quiet,
            no_headless,
        } => {
            maps_command(&url, max_reviews, output, quiet, no_headless).await?;
        }
        Commands::Extract {
            url,
            config,
            output,
            browser,
            pretty,
            timeout,
        } => {
            extract_command(&url, &config, output, browser, pretty, timeout).await?;
        }
        Commands::ValidateConfig { config } => {
            validate_config_command(&config)?;
        }

        #[cfg(feature = "rl")]
        Commands::Train {
            episodes,
            output,
            cuda,
            learning_rate,
            batch_size,
            gamma,
            checkpoint_interval,
        } => {
            commands::train::run_training(
                episodes,
                &output,
                cuda,
                learning_rate,
                batch_size,
                gamma,
                checkpoint_interval,
            )
            .await?;
        }

        #[cfg(feature = "rl")]
        Commands::ScrapeRl {
            url,
            model,
            max_steps,
            output,
            cuda,
            deterministic,
            verbose,
        } => {
            commands::scrape_rl::run_rl_scrape(
                &url,
                &model,
                max_steps,
                output,
                cuda,
                deterministic,
                verbose,
            )
            .await?;
        }
    }

    Ok(())
}

async fn scrape_command(
    storage: &mut Storage,
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

    // Create ScrapedPage using the storage model
    let scraped = ScrapedPage::new(
        url.to_string(),
        title.clone(),
        content.trim().to_string(),
        links.clone(),
    );

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

    // Save to database
    println!();
    println!("{}", "💾 Saving to database...".bright_green());
    storage
        .save_page(&scraped)
        .await
        .context("Failed to save page to database")?;

    println!("{}", "✅ Saved to database".bright_green());
    println!("{}: {}", "Page ID".bright_cyan(), scraped.id);

    // Optionally save to JSON file if output path specified
    if let Some(output_path) = output {
        fs::write(
            &output_path,
            serde_json::to_string_pretty(&scraped).context("Failed to serialize scraped data")?,
        )
        .context("Failed to write output file")?;

        println!(
            "{} {}",
            "📄 Also saved to file:".bright_green(),
            output_path.display()
        );
    }

    Ok(())
}

async fn list_command(storage: &mut Storage, _dir: &PathBuf, filter: Option<String>) -> Result<()> {
    println!("{}", "📚 Scraped Pages".bright_blue().bold());
    println!();

    // Fetch all pages from database
    let pages = storage
        .list_all(None)
        .await
        .context("Failed to list pages from database")?;

    let mut count = 0;

    for page in pages {
        // Apply filter if specified
        if let Some(ref pattern) = filter {
            if !page.url.contains(pattern) {
                continue;
            }
        }

        count += 1;
        println!("{}", format!("{}. {}", count, page.url).bright_white());
        if let Some(title) = &page.title {
            println!("   📄 {}", title.bright_yellow());
        }
        println!("   🕒 {}", page.scraped_at.to_rfc3339());
        println!("   🆔 {}", page.id.bright_cyan());
        println!(
            "   📊 {} characters, {} links",
            page.content_length,
            page.links.len()
        );
        println!();
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

async fn stats_command(storage: &mut Storage, _dir: &PathBuf) -> Result<()> {
    println!("{}", "📊 Statistics".bright_blue().bold());
    println!();

    // Get statistics from database
    let stats = storage
        .get_stats()
        .await
        .context("Failed to get statistics from database")?;

    println!(
        "{}: {}",
        "Total pages scraped".bright_cyan(),
        stats.total_pages
    );
    println!(
        "{}: {} characters ({:.2} MB)",
        "Total content".bright_cyan(),
        stats.total_content_bytes,
        stats.total_content_bytes as f64 / 1_048_576.0
    );
    println!(
        "{}: {}",
        "Total links extracted".bright_cyan(),
        stats.total_links
    );
    println!(
        "{}: {}",
        "Pages with embeddings".bright_cyan(),
        stats.pages_with_embeddings
    );

    if stats.total_pages > 0 {
        println!(
            "{}: {} characters",
            "Average content".bright_cyan(),
            stats.total_content_bytes / stats.total_pages
        );
    }

    if stats.cache_hit_rate > 0.0 {
        println!(
            "{}: {:.1}%",
            "Cache hit rate".bright_cyan(),
            stats.cache_hit_rate * 100.0
        );
    }

    Ok(())
}

async fn delete_command(
    storage: &mut Storage,
    all: bool,
    url_pattern: Option<String>,
    _file: Option<PathBuf>,
) -> Result<()> {
    if !all && url_pattern.is_none() {
        return Err(anyhow::anyhow!(
            "Must specify one of: --all or --url <pattern>"
        ));
    }

    println!("{}", "🗑️  Deleting pages...".bright_red().bold());
    println!();

    // Get list of pages to delete
    let pages = storage
        .list_all(None)
        .await
        .context("Failed to list pages from database")?;

    let mut deleted_count = 0;

    for page in pages {
        let should_delete = if all {
            true
        } else if let Some(ref pattern) = url_pattern {
            page.url.contains(pattern)
        } else {
            false
        };

        if should_delete {
            // Delete the page
            match storage.delete_page(&page.id).await {
                Ok(true) => {
                    println!(
                        "{} {} ({})",
                        "✅ Deleted:".bright_green(),
                        page.url,
                        page.id
                    );
                    deleted_count += 1;
                }
                Ok(false) => {
                    println!("{} {} ({})", "⚠️  Not found:".yellow(), page.url, page.id);
                }
                Err(e) => {
                    println!(
                        "{} {} - {}",
                        "❌ Failed to delete:".bright_red(),
                        page.url,
                        e
                    );
                }
            }
        }
    }

    println!();
    if deleted_count > 0 {
        println!(
            "{}",
            format!("✅ Successfully deleted {} page(s)", deleted_count).bright_green()
        );
    } else {
        println!("{}", "No pages found matching the criteria".yellow());
    }

    Ok(())
}

async fn export_command(
    storage: &mut Storage,
    _dir: &PathBuf,
    output: &PathBuf,
    format: ExportFormat,
) -> Result<()> {
    println!("{}", "📤 Exporting scraped pages...".bright_blue().bold());
    println!();

    // Fetch all pages from database
    let pages = storage
        .list_all(None)
        .await
        .context("Failed to list pages from database")?;

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
                &page.scraped_at.to_rfc3339(),
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
        md.push_str(&format!(
            "**Scraped at:** {}\n\n",
            page.scraped_at.to_rfc3339()
        ));
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
            page.scraped_at.to_rfc3339()
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
    storage: &mut Storage,
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
        use argus_browser::PoolConfig;
        let mut pool_config = PoolConfig::default();
        pool_config.max_instances = concurrency;
        pool_config.initial_instances = concurrency.min(2);
        Some(BrowserPool::new(pool_config).await?)
    } else {
        None
    };

    // Process URLs in parallel
    let results = if use_browser {
        // Browser mode with pool
        scrape_batch_browser(
            &urls,
            browser_pool.unwrap(),
            concurrency,
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

    // Save results to database
    println!();
    println!("{}", "💾 Saving to database...".bright_cyan());

    let mut saved_count = 0;
    for result in results.iter() {
        if let Ok(page) = result {
            match storage.save_page(page).await {
                Ok(_) => {
                    saved_count += 1;
                }
                Err(e) => {
                    println!("{} {} - {}", "❌ Failed to save:".bright_red(), page.url, e);
                }
            }
        }
    }

    println!("{}: {}", "Saved to database".bright_green(), saved_count);

    // Optionally export to JSON files
    if output_dir != &PathBuf::from("./data") {
        println!();
        println!("{}", "📁 Exporting to JSON files...".bright_cyan());

        for (i, result) in results.iter().enumerate() {
            if let Ok(page) = result {
                let filename = format!("batch_{}_{}.json", i, uuid::Uuid::new_v4());
                let path = output_dir.join(filename);
                fs::write(&path, serde_json::to_string_pretty(&page)?)?;
            }
        }

        println!("{}: {}", "Exported to".bright_green(), output_dir.display());
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
    concurrency: usize,
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
        .buffer_unordered(concurrency)
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
    let page = guard.new_page(url).await?;

    let html = page.content().await?;
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

    Ok(ScrapedPage::new(
        url.to_string(),
        title,
        content.trim().to_string(),
        links,
    ))
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

/// Scrape Google Maps reviews
async fn maps_command(
    url: &str,
    max_reviews: usize,
    output: Option<PathBuf>,
    quiet: bool,
    no_headless: bool,
) -> Result<()> {
    if !quiet {
        println!(
            "{}",
            "🗺️  Argus - Google Maps Review Scraper"
                .bright_blue()
                .bold()
        );
        println!("URL: {}", url.bright_white());
        println!("Max reviews: {}", max_reviews);
        if no_headless {
            println!("{}", "Mode: Visible browser (non-headless)".bright_yellow());
        }
        println!();
    }

    // Scrape Google Maps
    let result = commands::scrape_google_maps(url, max_reviews, !no_headless).await?;

    if !quiet {
        println!();
        println!("{}", "✅ Scrape complete!".bright_green().bold());
        println!();
        println!("{}: {}", "Business".bright_cyan(), result.business.name);
        if let Some(rating) = result.business.rating {
            println!("{}: ⭐ {:.1}", "Rating".bright_cyan(), rating);
        }
        if let Some(total) = result.business.total_reviews {
            println!("{}: {}", "Total reviews".bright_cyan(), total);
        }
        if let Some(addr) = &result.business.address {
            println!("{}: {}", "Address".bright_cyan(), addr);
        }
        if let Some(phone) = &result.business.phone {
            println!("{}: {}", "Phone".bright_cyan(), phone);
        }
        println!();
        println!(
            "{}: {}",
            "Reviews extracted".bright_green(),
            result.reviews.len()
        );

        // Show first few reviews as preview
        if !result.reviews.is_empty() {
            println!();
            println!("{}", "📝 Sample reviews:".bright_yellow());
            for (i, review) in result.reviews.iter().take(3).enumerate() {
                println!(
                    "  {}. {} (⭐{}) - {}",
                    i + 1,
                    review.author.bright_white(),
                    review.rating,
                    review.date
                );
                let preview: String = review.text.chars().take(80).collect();
                if review.text.len() > 80 {
                    println!("     \"{}...\"", preview);
                } else {
                    println!("     \"{}\"", preview);
                }
            }
        }
    }

    // Save to file if output path specified
    if let Some(output_path) = output {
        let json =
            serde_json::to_string_pretty(&result).context("Failed to serialize result to JSON")?;
        fs::write(&output_path, json).context("Failed to write output file")?;

        if !quiet {
            println!();
            println!(
                "{} {}",
                "📄 Saved to:".bright_green(),
                output_path.display()
            );
        } else {
            // In quiet mode, just print the path
            println!("{}", output_path.display());
        }
    } else if quiet {
        // In quiet mode without output file, print JSON to stdout
        let json = serde_json::to_string(&result)?;
        println!("{}", json);
    }

    Ok(())
}

/// Extract structured data using a configuration file
async fn extract_command(
    url: &str,
    config_path: &PathBuf,
    output: Option<PathBuf>,
    use_browser: bool,
    pretty: bool,
    timeout_secs: u64,
) -> Result<()> {
    println!(
        "{}",
        "🦅 Argus - Structured Data Extraction".bright_blue().bold()
    );
    println!("URL: {}", url.bright_white());
    println!(
        "Config: {}",
        config_path.display().to_string().bright_cyan()
    );
    if use_browser {
        println!(
            "{}",
            "🌐 Using browser automation (JavaScript enabled)".bright_cyan()
        );
    }
    println!();

    // Load and validate configuration
    println!("{}", "📋 Loading configuration...".yellow());
    let engine = ExtractorEngine::from_file(config_path)
        .context("Failed to load extraction configuration")?;

    println!(
        "   {} extractors configured",
        engine.config().extractors.len()
    );

    // Check if URL matches the config's URL patterns (if any)
    if !engine.matches_url(url) {
        println!(
            "{}",
            "⚠️  Warning: URL does not match config's url_patterns".yellow()
        );
    }

    // Fetch HTML content
    let html = if use_browser {
        fetch_with_browser(url, None, timeout_secs, false).await?
    } else {
        fetch_with_http(url, timeout_secs, 3).await?
    };

    // Run extraction
    println!("{}", "🔍 Extracting structured data...".yellow());
    let result = engine.extract(url, &html).context("Extraction failed")?;

    // Display results summary
    println!();
    println!("{}", "✅ Extraction complete!".bright_green().bold());
    println!();
    println!("{}: {}", "Config name".bright_cyan(), result.config_name);
    println!("{}: {}", "URL".bright_cyan(), result.url);
    println!(
        "{}: {}ms",
        "Duration".bright_cyan(),
        result.metadata.duration_ms
    );
    println!(
        "{}: {}",
        "Extractors matched".bright_green(),
        result.metadata.extractors_matched.len()
    );
    if !result.metadata.extractors_failed.is_empty() {
        println!(
            "{}: {}",
            "Extractors failed".bright_red(),
            result.metadata.extractors_failed.len()
        );
    }

    // Show data summary
    println!();
    println!("{}", "📊 Extracted data:".bright_yellow());
    for (name, data) in &result.data {
        let count = match data {
            argus_extract::ExtractedData::Single(_) => "1 item".to_string(),
            argus_extract::ExtractedData::Array(arr) => format!("{} items", arr.len()),
            argus_extract::ExtractedData::Sections(secs) => format!("{} sections", secs.len()),
        };
        println!("   {}: {}", name.bright_white(), count);
    }

    // Show validation errors if any
    if !result.validation_errors.is_empty() {
        println!();
        println!("{}", "⚠️  Validation warnings:".yellow());
        for error in &result.validation_errors {
            println!("   - {}", error);
        }
    }

    // Serialize result
    let json = if pretty {
        serde_json::to_string_pretty(&result).context("Failed to serialize result")?
    } else {
        serde_json::to_string(&result).context("Failed to serialize result")?
    };

    // Output result
    if let Some(output_path) = output {
        fs::write(&output_path, &json).context("Failed to write output file")?;
        println!();
        println!(
            "{} {}",
            "📄 Saved to:".bright_green(),
            output_path.display()
        );
    } else {
        // Print to stdout
        println!();
        println!("{}", "📄 Result:".bright_cyan());
        println!("{}", json);
    }

    Ok(())
}

/// Validate an extraction configuration file
fn validate_config_command(config_path: &PathBuf) -> Result<()> {
    println!(
        "{}",
        "🔍 Validating extraction configuration..."
            .bright_blue()
            .bold()
    );
    println!("File: {}", config_path.display().to_string().bright_white());
    println!();

    // Try to load the configuration
    let yaml_content =
        fs::read_to_string(config_path).context("Failed to read configuration file")?;

    let config: ExtractionConfig =
        serde_yaml::from_str(&yaml_content).context("Failed to parse YAML configuration")?;

    // Validate configuration
    println!("{}", "✅ Configuration is valid!".bright_green().bold());
    println!();
    println!("{}: {}", "Version".bright_cyan(), config.version);
    println!("{}: {}", "Name".bright_cyan(), config.name);
    if let Some(ref desc) = config.description {
        println!("{}: {}", "Description".bright_cyan(), desc);
    }
    println!(
        "{}: {}",
        "Extractors".bright_cyan(),
        config.extractors.len()
    );

    // List extractors
    println!();
    println!("{}", "📋 Configured extractors:".bright_yellow());
    for extractor in &config.extractors {
        let type_name = match &extractor.config {
            argus_extract::config::ExtractorType::Table(_) => "table",
            argus_extract::config::ExtractorType::Paired(_) => "paired",
            argus_extract::config::ExtractorType::Sections(_) => "sections",
            argus_extract::config::ExtractorType::List(_) => "list",
            argus_extract::config::ExtractorType::KeyValue(_) => "keyvalue",
            argus_extract::config::ExtractorType::Repeating(_) => "repeating",
        };
        println!(
            "   {} [{}]",
            extractor.name.bright_white(),
            type_name.bright_cyan()
        );
    }

    // Show URL patterns if any
    if !config.url_patterns.is_empty() {
        println!();
        println!("{}", "🔗 URL patterns:".bright_yellow());
        for pattern in &config.url_patterns {
            println!("   {}", pattern);
        }
    }

    Ok(())
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
        let page = ScrapedPage::new(
            "https://example.com".to_string(),
            Some("Example".to_string()),
            "Test content".to_string(),
            vec!["https://link1.com".to_string()],
        );

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
