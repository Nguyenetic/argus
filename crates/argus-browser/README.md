# Argus Browser - Intelligent Browser Automation

Production-ready browser automation with advanced anti-detection capabilities, browser pool management, and intelligent scraping features.

## Features

### 🎭 Stealth Mode
- **Hide automation flags**: Disable `navigator.webdriver` and other bot detection signals
- **User agent rotation**: Randomize user agents across requests
- **Canvas fingerprint randomization**: Add noise to canvas fingerprints
- **WebGL fingerprint randomization**: Randomize WebGL vendor/renderer
- **Navigator spoofing**: Emulate real browser properties (plugins, languages, etc.)
- **Screen dimension randomization**: Vary screen sizes to avoid fingerprinting

### 🏊 Browser Pool Management
- **Dynamic scaling**: Auto-scale from min to max instances based on load
- **Health checking**: Automatic health checks and recovery
- **Load balancing**: Distribute requests across healthy instances
- **Resource cleanup**: Automatic cleanup of idle browsers
- **RAII pattern**: Automatic resource release with `BrowserGuard`

### 💾 Session Persistence
- **Cookie management**: Save and restore cookies across sessions
- **Local storage**: Persist local storage data
- **Session storage**: Persist session storage data
- **Multi-session**: Manage multiple isolated sessions
- **Session restoration**: Resume sessions with full state

### 🧠 Intelligent Features
- **RL integration**: Behavioral evasion with reinforcement learning (optional)
- **CAPTCHA solving**: Automatic CAPTCHA detection and solving (optional)
- **Retry logic**: Exponential backoff on failures
- **Human-like delays**: Variable delays between requests
- **Parallel scraping**: Concurrent scraping with pool management

## Quick Start

### Basic Usage

```rust
use argus_browser::IntelligentBrowserBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create browser with default settings
    let browser = IntelligentBrowserBuilder::new()
        .pool_size(2, 10)
        .stealth(true)
        .build()
        .await?;

    // Navigate to page
    let session = browser.navigate("https://example.com").await?;

    // Extract data
    let title = session.execute("document.title").await?;
    println!("Title: {}", title);

    // Cleanup
    browser.cleanup().await?;

    Ok(())
}
```

### Advanced Configuration

```rust
let browser = IntelligentBrowserBuilder::new()
    .pool_size(3, 15)              // Min 3, max 15 instances
    .stealth(true)                 // Enable stealth mode
    .rl_evasion(true)              // Enable RL behavioral evasion
    .captcha_solving(true)         // Enable CAPTCHA solving
    .request_delay(1000, 3000)     // 1-3 second delays
    .max_retries(3)                // Retry failed requests 3 times
    .session_storage("./sessions") // Save sessions
    .window_size(1920, 1080)       // Browser window size
    .headless(true)                // Headless mode
    .build()
    .await?;
```

## Examples

### 1. Simple Scraping

```rust
let browser = IntelligentBrowserBuilder::new()
    .stealth(true)
    .build()
    .await?;

let session = browser.navigate("https://example.com").await?;

// Get page content
let content = session.content().await?;
println!("Content: {}", content);

// Execute JavaScript
let links_count = session.execute(
    "document.querySelectorAll('a').length"
).await?;
println!("Links: {}", links_count);
```

### 2. Parallel Scraping

```rust
let browser = IntelligentBrowserBuilder::new()
    .pool_size(5, 20)
    .stealth(true)
    .build()
    .await?;

let urls = vec![
    "https://example.com/page1".to_string(),
    "https://example.com/page2".to_string(),
    "https://example.com/page3".to_string(),
];

let results = browser.scrape_parallel(urls, |page| {
    Box::pin(async move {
        let title = page.evaluate("document.title").await?;
        Ok(title.into_value::<String>()?)
    })
}).await?;

for result in results {
    match result {
        Ok(title) => println!("✓ {}", title),
        Err(e) => println!("✗ {}", e),
    }
}
```

### 3. Session Management

```rust
let browser = IntelligentBrowserBuilder::new()
    .session_storage("./data/sessions")
    .build()
    .await?;

// Create session
let session = browser.navigate("https://example.com").await?;
let session_id = session.session_id.clone().unwrap();

// Do some work...
drop(session);

// Resume session later
let resumed = browser.resume_session(&session_id, "https://example.com").await?;
// Session state (cookies, storage) is restored
```

### 4. E-Commerce Scraping

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Product {
    name: String,
    price: f64,
    url: String,
}

async fn scrape_products(browser: &IntelligentBrowser) -> Result<Vec<Product>> {
    let session = browser.navigate("https://shop.example.com/products").await?;

    // Wait for products to load
    session.wait_for(".product-card").await?;

    // Extract products
    let js = r#"
        Array.from(document.querySelectorAll('.product-card')).map(el => ({
            name: el.querySelector('.product-name').innerText,
            price: parseFloat(el.querySelector('.product-price').innerText.replace(/[^0-9.]/g, '')),
            url: el.querySelector('a').href
        }))
    "#;

    let result = session.execute(js).await?;
    let products: Vec<Product> = serde_json::from_value(result)?;

    Ok(products)
}
```

### 5. With Retry Logic

```rust
// Automatically retries on failure with exponential backoff
let session = browser.navigate_with_retry("https://example.com").await?;

// Or use scrape_with_retry for extraction
let data = browser.scrape("https://example.com", |page| {
    Box::pin(async move {
        // Extract data
        let result = page.evaluate("document.body.innerText").await?;
        Ok(result.into_value::<String>()?)
    })
}).await?;
```

## Architecture

### Component Overview

```
IntelligentBrowser
├── BrowserPool (pool.rs)
│   ├── Instance management
│   ├── Health checking
│   ├── Load balancing
│   └── Auto-scaling
├── StealthMode (stealth.rs)
│   ├── Webdriver hiding
│   ├── User agent rotation
│   ├── Fingerprint randomization
│   └── Navigator spoofing
├── SessionManager (session.rs)
│   ├── Cookie persistence
│   ├── Storage persistence
│   └── Session restoration
└── Intelligent Layer (intelligent.rs)
    ├── High-level API
    ├── Retry logic
    ├── Parallel scraping
    └── RL/CAPTCHA integration
```

### Stealth Techniques

1. **Navigator.webdriver**: Deleted and overridden to return `undefined`
2. **User Agent**: Randomized from pool of real browser agents
3. **Canvas Fingerprint**: Slight noise added to rendering
4. **WebGL Fingerprint**: Vendor/renderer randomized
5. **Screen Dimensions**: Randomized from common resolutions
6. **Plugins**: Emulated PDF and Chrome plugins
7. **Languages**: Set to realistic values (`en-US`, `en`)
8. **Hardware Concurrency**: Set to realistic values (4-8 cores)

### Browser Pool

The pool maintains a collection of browser instances with:

- **Min instances**: Always kept alive for fast access
- **Max instances**: Upper limit to prevent resource exhaustion
- **Health checks**: Periodic checks (default: 30s) to ensure instances are responsive
- **Idle cleanup**: Removes idle instances after timeout (default: 5 minutes)
- **Auto-scaling**: Creates new instances on demand up to max limit

### Session Persistence

Sessions include:
- **Cookies**: All cookies from the current domain
- **Local Storage**: Key-value pairs from localStorage
- **Session Storage**: Key-value pairs from sessionStorage
- **User Agent**: The user agent used for the session
- **Viewport**: Window dimensions
- **Metadata**: Custom key-value pairs

## Configuration

### Pool Configuration

```rust
use argus_browser::PoolConfig;
use std::time::Duration;

let pool_config = PoolConfig {
    min_instances: 2,
    max_instances: 10,
    initial_instances: 2,
    max_idle_time: Duration::from_secs(300),     // 5 minutes
    health_check_interval: Duration::from_secs(30), // 30 seconds
    enable_stealth: true,
    window_size: (1920, 1080),
    headless: true,
    incognito: true,
    ..Default::default()
};
```

### Stealth Configuration

```rust
use argus_browser::StealthConfig;

let stealth_config = StealthConfig {
    randomize_user_agent: true,
    hide_webdriver: true,
    randomize_canvas: true,
    randomize_webgl: true,
    spoof_navigator: true,
    randomize_screen: true,
    emulate_plugins: true,
};
```

## Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Browser startup | 500-800ms | First instance |
| Page navigation | 200-500ms | Cached instance |
| Pool acquisition | <10ms | Available instance |
| Stealth application | 50-100ms | Per page |
| Session save | 10-50ms | To disk |
| Session restore | 20-100ms | From disk |

### Throughput

- **Single browser**: 100-200 pages/minute
- **Pool (5 instances)**: 500-1000 pages/minute
- **Pool (10 instances)**: 1000-2000 pages/minute

### Resource Usage

Per browser instance:
- **Memory**: 100-200 MB
- **CPU**: 5-10% (idle), 20-40% (active)
- **Disk**: 10-20 MB (cache)

## Error Handling

The library uses `anyhow::Result` for error handling:

```rust
use anyhow::{Result, Context};

async fn scrape_safe(browser: &IntelligentBrowser) -> Result<String> {
    let session = browser.navigate("https://example.com")
        .await
        .context("Failed to navigate")?;

    let content = session.content()
        .await
        .context("Failed to get content")?;

    Ok(content)
}
```

Common errors:
- `Browser launch failed`: Chrome not installed or path incorrect
- `Navigation timeout`: Page took too long to load
- `Pool exhausted`: All browser instances in use
- `Session not found`: Session ID doesn't exist

## Logging

Enable logging with `tracing`:

```rust
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::INFO)
    .init();
```

Log levels:
- **ERROR**: Critical failures
- **WARN**: Non-critical issues (health check failures, retries)
- **INFO**: Important events (browser creation, navigation)
- **DEBUG**: Detailed operations (stealth application, session save/load)
- **TRACE**: Very detailed (pool operations, CDP commands)

## Integration

### With RL Agent (argus-rl)

```rust
#[cfg(feature = "rl")]
{
    let browser = IntelligentBrowserBuilder::new()
        .rl_evasion(true)
        .build()
        .await?;

    // RL agent automatically applied during navigation
    let session = browser.navigate("https://example.com").await?;
}
```

### With CAPTCHA Solver (argus-captcha)

```rust
#[cfg(feature = "captcha")]
{
    let browser = IntelligentBrowserBuilder::new()
        .captcha_solving(true)
        .build()
        .await?;

    // CAPTCHA automatically detected and solved
    let session = browser.navigate("https://example.com").await?;
}
```

## Testing

Run examples:

```bash
# Basic usage
cargo run --example basic_usage

# Parallel scraping
cargo run --example parallel_scraping

# Complete scraper
cargo run --example complete_scraper
```

Run tests:

```bash
cargo test
```

## Troubleshooting

### Chrome not found

```
Error: Browser launch failed
```

**Solution**: Install Chrome or set `CHROME_PATH` environment variable:

```bash
export CHROME_PATH=/path/to/chrome
```

### Pool exhausted

```
Error: All browser instances in use
```

**Solution**: Increase pool size or add delay between requests:

```rust
.pool_size(5, 20)  // Increase max instances
.request_delay(2000, 5000)  // Longer delays
```

### Memory issues

If running out of memory with large pools:

1. Reduce max instances
2. Decrease idle timeout
3. Enable cleanup more aggressively
4. Use headless mode

## Best Practices

### 1. Resource Management

```rust
// Always cleanup when done
let browser = IntelligentBrowserBuilder::new().build().await?;
// ... use browser ...
browser.cleanup().await?;
```

### 2. Error Handling

```rust
// Use retry logic for flaky operations
let session = browser.navigate_with_retry(url).await?;

// Or implement custom retry
for attempt in 0..3 {
    match browser.navigate(url).await {
        Ok(session) => break,
        Err(e) if attempt < 2 => continue,
        Err(e) => return Err(e),
    }
}
```

### 3. Rate Limiting

```rust
// Add delays to avoid overwhelming servers
let browser = IntelligentBrowserBuilder::new()
    .request_delay(1000, 3000)  // 1-3 seconds
    .build()
    .await?;
```

### 4. Pool Sizing

- Start with small pool (2-5 instances)
- Monitor resource usage
- Scale up if needed
- Keep max < 20 for typical workloads

### 5. Session Management

- Save sessions for authenticated workflows
- Clear old sessions periodically
- Use unique session IDs
- Handle session expiration gracefully

## License

MIT License - See LICENSE file for details.

## See Also

- `argus-rl` - RL-based behavioral evasion
- `argus-captcha` - CAPTCHA detection and solving
- `argus-core` - Core types and utilities
- `chromiumoxide` - Chrome DevTools Protocol wrapper
