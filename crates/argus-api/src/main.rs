//! Argus API Server

use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("🦅 Starting Argus API Server...");

    // Build application state
    let app_state = Arc::new(AppState {
        version: argus_core::VERSION.to_string(),
    });

    // Build router
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/api/v1/scrape", post(scrape_url))
        .with_state(app_state);

    // Start server
    let addr = "0.0.0.0:3000";
    info!("🚀 Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// Application state
struct AppState {
    version: String,
}

// Root handler
async fn root(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "name": "Argus",
        "version": state.version,
        "description": "Intelligent Web Intelligence System",
        "status": "running"
    }))
}

// Health check handler
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

// Scrape request
#[derive(Debug, Deserialize)]
struct ScrapeRequest {
    url: String,
    #[serde(default)]
    use_browser: bool,
}

// Scrape response
#[derive(Debug, Serialize)]
struct ScrapeResponse {
    success: bool,
    url: String,
    title: Option<String>,
    content_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// Scrape handler
async fn scrape_url(
    Json(req): Json<ScrapeRequest>,
) -> Result<Json<ScrapeResponse>, StatusCode> {
    info!("Scraping URL: {}", req.url);

    // Simple scraping using reqwest for now
    match reqwest::get(&req.url).await {
        Ok(response) => {
            match response.text().await {
                Ok(html) => {
                    // Basic title extraction
                    let title = extract_title(&html);

                    Ok(Json(ScrapeResponse {
                        success: true,
                        url: req.url,
                        title,
                        content_length: html.len(),
                        error: None,
                    }))
                }
                Err(e) => {
                    Ok(Json(ScrapeResponse {
                        success: false,
                        url: req.url,
                        title: None,
                        content_length: 0,
                        error: Some(format!("Failed to read response: {}", e)),
                    }))
                }
            }
        }
        Err(e) => {
            Ok(Json(ScrapeResponse {
                success: false,
                url: req.url,
                title: None,
                content_length: 0,
                error: Some(format!("HTTP request failed: {}", e)),
            }))
        }
    }
}

fn extract_title(html: &str) -> Option<String> {
    // Very basic title extraction
    if let Some(start) = html.find("<title>") {
        if let Some(end) = html[start..].find("</title>") {
            let title = &html[start + 7..start + end];
            return Some(title.trim().to_string());
        }
    }
    None
}
