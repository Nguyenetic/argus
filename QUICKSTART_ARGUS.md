# 🦅 Argus Quick Start Guide

## Current Progress ✅

We've successfully set up:
- ✅ Cargo workspace with 5 crates
- ✅ Basic API server structure
- ✅ Database and storage layers
- ✅ Docker Compose configuration
- ✅ Environment configuration
- ✅ Database migrations

## Next Steps

### 1. Install Docker Desktop (Required)

**Download:** https://www.docker.com/products/docker-desktop/

After installation:
```bash
# Verify Docker is installed
docker --version
docker compose version
```

### 2. Start Database Services

```bash
# Start PostgreSQL and Redis
cd "C:\Users\jnguyen\Documents\Project\web-scraper"
docker compose -f docker-compose.dev.yml up -d

# Check status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs -f
```

### 3. Run Database Migrations

```bash
# Install sqlx-cli
cargo install sqlx-cli --no-default-features --features postgres

# Run migrations
sqlx migrate run --database-url "postgresql://argus:argus_dev_password@localhost:5432/argus_db"
```

### 4. Build and Run Argus

The first build is currently running in the background and may take 5-10 minutes.

Once complete:

```bash
# Build (if not already done)
cargo build --release

# Run the API server
cargo run --bin argus-server

# Or use the release build
./target/release/argus-server
```

### 5. Test the API

```bash
# Health check
curl http://localhost:3000/health

# Root endpoint
curl http://localhost:3000/

# Scrape a URL
curl -X POST http://localhost:3000/api/v1/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## Project Structure

```
argus/
├── crates/
│   ├── argus-core/        # Core types and utilities
│   ├── argus-browser/     # Browser automation
│   ├── argus-rl/          # Reinforcement learning
│   ├── argus-storage/     # Database and cache
│   └── argus-api/         # REST API server
├── migrations/            # Database migrations
├── docker-compose.dev.yml # Development services
├── .env                   # Environment config
└── Cargo.toml            # Workspace config
```

## What's Working Now

### API Server (Basic)
- ✅ GET `/` - Server info
- ✅ GET `/health` - Health check
- ✅ POST `/api/v1/scrape` - Basic URL scraping

### Storage Layer
- ✅ PostgreSQL connection with SQLx
- ✅ Redis caching layer
- ✅ Page storage operations

### Browser
- ✅ Basic HTTP client scraping
- 🚧 Chrome automation (coming next)

## What's Next (Phase 1 Roadmap)

### Week 2: Browser Automation
- [ ] Full chromiumoxide integration
- [ ] Screenshot capture
- [ ] JavaScript rendering
- [ ] Content extraction with scraper crate

### Week 3: RL Agent
- [ ] DQN implementation
- [ ] State/action spaces
- [ ] Training loop
- [ ] Anti-bot evasion

### Week 4-5: API Enhancement
- [ ] Authentication (JWT)
- [ ] Rate limiting
- [ ] Job queue system
- [ ] WebSocket for real-time updates

### Week 6: Testing & MVP
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Documentation

## Troubleshooting

### Build Issues

If cargo build fails:
```bash
# Clean and rebuild
cargo clean
cargo build
```

### Database Connection Issues

```bash
# Check if containers are running
docker compose -f docker-compose.dev.yml ps

# Restart containers
docker compose -f docker-compose.dev.yml restart

# Check logs
docker compose -f docker-compose.dev.yml logs postgres
```

### Port Conflicts

If ports 3000, 5432, or 6379 are in use:
1. Stop conflicting services
2. Or modify ports in `docker-compose.dev.yml` and `.env`

## Development Commands

```bash
# Format code
cargo fmt --all

# Run linters
cargo clippy --all

# Run tests
cargo test --all

# Build docs
cargo doc --no-deps --open

# Watch for changes (install cargo-watch first)
cargo install cargo-watch
cargo watch -x run
```

## Environment Variables

Edit `.env` to configure:
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `PORT` - API server port
- `RUST_LOG` - Logging level

## Resources

- **Documentation:** See `docs/` folder
- **Roadmap:** `docs/DETAILED_ROADMAP_RUST.md`
- **Research:** `docs/RESEARCH_COMPENDIUM.md`
- **GitHub:** https://github.com/Nguyenetic/argus

---

**Status:** MVP in progress (Week 1 of 28)
**Next Milestone:** Working browser automation (Week 2)

🦅 **Welcome to Argus!**
