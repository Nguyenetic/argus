# 🦅 Argus - Simple CLI Web Scraper

**No server, no Docker - just a simple command-line tool!**

## Installation

```bash
# Build Argus
cargo build --release

# The binary will be at: target/release/argus.exe
```

## Usage

### Scrape a URL

```bash
# Basic scraping
argus scrape https://example.com

# Save to specific file
argus scrape https://example.com -o output.json

# Extract links too
argus scrape https://example.com --links
```

### List scraped pages

```bash
argus list
```

### Show statistics

```bash
argus stats
```

## Examples

```bash
# Scrape a documentation page
argus scrape https://docs.rust-lang.org/book/

# Scrape with link extraction
argus scrape https://news.ycombinator.com --links

# View all scraped pages
argus list

# See statistics
argus stats
```

## Output

All scraped pages are saved to `./data/` directory as JSON files:

```json
{
  "url": "https://example.com",
  "title": "Example Domain",
  "content": "This domain is for use in illustrative examples...",
  "links": ["https://www.iana.org/domains/example"],
  "scraped_at": "2025-11-04T12:00:00Z",
  "content_length": 1234
}
```

## What's Next?

This is the MVP! Future enhancements:
- Browser automation with chromiumoxide
- Anti-bot RL agent
- Parallel scraping
- Export to different formats
- Content analysis with ML

## Documentation

- Full roadmap: `docs/DETAILED_ROADMAP_RUST.md`
- Research: `docs/RESEARCH_COMPENDIUM.md`

---

**Simple, fast, and ready to use!** 🚀
