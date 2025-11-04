//! Argus - Intelligent Web Intelligence System
//! A simple CLI tool for web scraping

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use chrono::Utc;

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
    },

    /// List scraped pages
    List {
        /// Output directory
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },

    /// Show statistics
    Stats {
        /// Output directory
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },
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
        Commands::Scrape { url, output, links } => {
            scrape_command(&url, output, links).await?;
        }
        Commands::List { dir } => {
            list_command(&dir)?;
        }
        Commands::Stats { dir } => {
            stats_command(&dir)?;
        }
    }

    Ok(())
}

async fn scrape_command(url: &str, output: Option<PathBuf>, extract_links: bool) -> Result<()> {
    println!("{}", "🦅 Argus - Starting scrape...".bright_blue().bold());
    println!("URL: {}", url.bright_white());
    println!();

    // Fetch the page
    println!("{}", "📡 Fetching page...".yellow());
    let response = reqwest::get(url).await?;
    let html = response.text().await?;

    // Parse HTML
    println!("{}", "🔍 Parsing content...".yellow());
    let document = scraper::Html::parse_document(&html);

    // Extract title
    let title_selector = scraper::Selector::parse("title").unwrap();
    let title = document
        .select(&title_selector)
        .next()
        .map(|el| el.text().collect::<String>());

    // Extract text content
    let body_selector = scraper::Selector::parse("body").unwrap();
    let content = document
        .select(&body_selector)
        .next()
        .map(|el| el.text().collect::<String>())
        .unwrap_or_default();

    // Extract links if requested
    let mut links = Vec::new();
    if extract_links {
        let link_selector = scraper::Selector::parse("a[href]").unwrap();
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
    println!("{}: {}", "Title".bright_cyan(), title.as_deref().unwrap_or("N/A"));
    println!("{}: {} characters", "Content".bright_cyan(), scraped.content_length);
    if extract_links {
        println!("{}: {}", "Links found".bright_cyan(), links.len());
    }

    // Save to file
    let output_path = output.unwrap_or_else(|| {
        let filename = format!("page_{}.json", uuid::Uuid::new_v4());
        PathBuf::from("./data").join(filename)
    });

    fs::write(&output_path, serde_json::to_string_pretty(&scraped)?)?;

    println!();
    println!("{} {}", "💾 Saved to:".bright_green(), output_path.display());

    Ok(())
}

fn list_command(dir: &PathBuf) -> Result<()> {
    println!("{}", "📚 Scraped Pages".bright_blue().bold());
    println!();

    let entries = fs::read_dir(dir)?;
    let mut count = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(page) = serde_json::from_str::<ScrapedPage>(&content) {
                count += 1;
                println!("{}", format!("{}. {}", count, page.url).bright_white());
                if let Some(title) = page.title {
                    println!("   📄 {}", title.bright_yellow());
                }
                println!("   🕒 {}", page.scraped_at);
                println!();
            }
        }
    }

    if count == 0 {
        println!("{}", "No pages found. Try scraping a URL first!".yellow());
    } else {
        println!("{} {}", "Total:".bright_green(), count);
    }

    Ok(())
}

fn stats_command(dir: &PathBuf) -> Result<()> {
    println!("{}", "📊 Statistics".bright_blue().bold());
    println!();

    let entries = fs::read_dir(dir)?;
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
    println!("{}: {} characters", "Total content".bright_cyan(), total_content);
    println!("{}: {}", "Total links extracted".bright_cyan(), total_links);

    if total_pages > 0 {
        println!("{}: {} characters", "Average content".bright_cyan(), total_content / total_pages);
    }

    Ok(())
}
