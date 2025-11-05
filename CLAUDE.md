# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Argus** is an intelligent web intelligence system that combines web scraping with advanced AI features including:
- Reinforcement Learning for anti-bot evasion
- Graph Neural Networks for content understanding
- Integration with NexusQL (brain-inspired database)
- Distributed edge computing capabilities

**Current Phase:** Phase 1 - Foundation (MVP)
- Simple CLI tool with basic HTTP scraping
- Workspace structure prepared for future ML/RL components
- NexusQL integration planned but pending API clarification

## Commands

### Building and Running

```bash
# Build the project
cargo build

# Build optimized release version
cargo build --release

# Run the CLI tool
cargo run -- <subcommand>

# Run with release optimizations
cargo run --release -- <subcommand>
```

### CLI Usage

```bash
# Scrape a URL and save to ./data/
cargo run -- scrape https://example.com

# Scrape with link extraction
cargo run -- scrape https://example.com --links

# Scrape and specify output file
cargo run -- scrape https://example.com --output custom.json

# List all scraped pages
cargo run -- list

# Show scraping statistics
cargo run -- stats
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p argus-core

# Run a specific test
cargo test test_version

# Run tests with output visible
cargo test -- --nocapture

# Run tests in a specific file/module
cargo test types::
```

### Code Quality

```bash
# Format code
cargo fmt

# Check formatting without modifying
cargo fmt --all -- --check

# Run linter
cargo clippy

# Run clippy with strict warnings
cargo clippy -- -D warnings
```

### Development

```bash
# Check code without building
cargo check

# Build documentation
cargo doc --no-deps --open

# Clean build artifacts
cargo clean
```

## Architecture

### Workspace Structure

Argus uses a **Rust workspace** with multiple crates for modularity:

```
argus/
├── src/                    # Main CLI binary
│   ├── main.rs            # CLI entry point (scrape/list/stats commands)
│   └── storage.rs         # NexusQL integration layer (WIP)
├── crates/
│   ├── argus-core/        # Core types, errors, traits
│   ├── argus-browser/     # Browser automation (chromiumoxide) - planned
│   ├── argus-rl/          # Reinforcement Learning agent (DQN) - planned
│   ├── argus-storage/     # Database & cache layer (PostgreSQL/Redis) - planned
│   └── argus-api/         # REST API server (Axum) - planned
└── data/                  # JSON storage for scraped pages
```

**Key Point:** Most crates (`argus-browser`, `argus-rl`, `argus-storage`, `argus-api`) are scaffolded but not yet implemented. The working code is in `src/main.rs` (CLI) and `argus-core` (types).

### Current Implementation

**Main Binary (`src/main.rs`):**
- CLI using `clap` with derive macros
- Three subcommands: `scrape`, `list`, `stats`
- HTTP scraping with `reqwest` (async)
- HTML parsing with `scraper` crate
- JSON file storage in `./data/`

**Core Types (`argus-core`):**
- `Page`: Scraped page representation with UUID, URL, content, metadata
- `PageStatus`: Enum for tracking scrape status (Pending/InProgress/Success/Failed)
- `ScrapeConfig`: Configuration for scraping operations
- `ScrapeResult`: Result wrapper with success/error/duration

**Current Limitations:**
- No browser automation (uses simple HTTP only)
- No JavaScript rendering
- No database persistence (JSON files only)
- No ML/RL features yet

### Integration with NexusQL

**Status:** Designed but blocked on API clarification (GitHub Issues #1 and #2)

NexusQL is a brain-inspired database with:
- **HDC (Hyperdimensional Computing)**: One-shot learning for website patterns
- **ColBERT**: Semantic search across scraped content
- **HNSW**: Fast vector similarity search
- **PostgreSQL-compatible** interface

**Integration Plan:**
1. Store scraped pages with vector embeddings
2. Use HDC for rapid pattern learning (5 examples to understand new sites)
3. Semantic search with ColBERT for content queries
4. HNSW indexes for fast retrieval at scale

See `NEXUSQL_INTEGRATION.md` for full details.

**Blocked Items:**
- Parameterized query API (SQL injection prevention)
- Vector insertion API (inserting `Vec<f32>` into VECTOR columns)
- HDC/ColBERT public APIs

## Development Workflow

### Adding New Features

1. **Determine the right crate:**
   - Core types/traits → `argus-core`
   - Browser automation → `argus-browser`
   - ML/RL logic → `argus-rl`
   - Database/cache → `argus-storage`
   - API endpoints → `argus-api`
   - CLI commands → `src/main.rs`

2. **Update Cargo.toml** if adding dependencies (use workspace inheritance when possible)

3. **Write tests** in the same file or `tests/` directory

4. **Update TODO.md** to track progress

### Workspace Dependencies

Common dependencies are defined in root `Cargo.toml` `[workspace.dependencies]` section. Crates inherit with:

```toml
[dependencies]
tokio.workspace = true
anyhow.workspace = true
# etc.
```

### Adding a CLI Command

Commands are defined in `src/main.rs` using `clap`:

```rust
#[derive(Subcommand)]
enum Commands {
    Scrape { /* fields */ },
    List { /* fields */ },
    // Add new command here
    NewCommand {
        #[arg(short, long)]
        some_option: String,
    },
}

// Then implement in main():
match cli.command {
    Commands::NewCommand { some_option } => {
        new_command_handler(&some_option).await?;
    }
    // ...
}
```

## Roadmap Context

The project follows a **28-week roadmap** divided into 4 phases:

1. **Phase 1 (Weeks 1-6):** Foundation - Basic scraping, browser automation, simple RL agent, REST API
2. **Phase 2 (Weeks 7-14):** Intelligence - GNN, transformers, few-shot learning, LLM integration
3. **Phase 3 (Weeks 15-20):** Distribution - Kafka queue, edge computing, monitoring
4. **Phase 4 (Weeks 21-28):** Advanced - Quantum-safe crypto, advanced RL, federated learning

**Current Status:** Week 1 completed (basic CLI), Week 2-3 pending (browser automation)

See `TODO.md` for detailed task tracking and `docs/DETAILED_ROADMAP_RUST.md` for complete roadmap.

## Important Notes

### NexusQL Dependency

The project depends on a **local path** to NexusQL:
```toml
nexusql = { path = "../NexusQL" }
```

**If building fails** with "cannot find crate `nexusql`", either:
1. Clone NexusQL alongside this repo: `git clone https://github.com/Nguyenetic/NexusQL.git`
2. Comment out the NexusQL dependency in `Cargo.toml` (removes `src/storage.rs` compilation)

### Performance Targets

When implementing features, keep these targets in mind:

- **Throughput:** 100K+ pages/hour (MVP), 15M+ pages/day (final)
- **API Latency:** <100ms p95 (MVP), <45ms p95 (final)
- **Extraction Accuracy:** >90% (with GNN)
- **Bot Evasion Rate:** >80% (DQN), >98% (Rainbow + PPO)

### Python Components

Legacy Python code exists in `api/`, `scrapers/`, `tasks/` but is **not actively used**. The current implementation is Rust-only. Python may be used later for:
- ML model training (`pyproject.toml` has PyTorch, spaCy, etc.)
- Jupyter notebooks for research
- Python API client

### Data Storage

**Current:** JSON files in `./data/` with UUIDs as filenames
**Future:** PostgreSQL + pgvector + Redis caching

The `ScrapedPage` struct in `src/main.rs` should eventually migrate to using `argus_core::Page` type for consistency.

## Research & Documentation

Key documents to understand the project vision:

- `README.md`: Project overview and features
- `TODO.md`: Immediate priorities and detailed task breakdown
- `NEXUSQL_INTEGRATION.md`: Database integration plan
- `docs/ADVANCED_RESEARCH_PRD_RUST.md`: Complete PRD with research
- `docs/DETAILED_ROADMAP_RUST.md`: 28-week implementation plan
- `docs/RESEARCH_COMPENDIUM.md`: 35,000+ word knowledge base
- `DOCUMENTATION_INDEX.md`: Guide to all documentation

The project is research-driven with 45+ academic/industry sources informing the design.
