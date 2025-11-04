# Quick Start Guide

**Get your web scraper running in 5 minutes!**

> 💡 **New!** We now support [UV](https://github.com/astral-sh/uv) - 10-100x faster than pip!
> See [UV_SETUP.md](UV_SETUP.md) for the modern setup guide.

Choose your setup based on what you want to do:

## 🚀 For Quick Testing (Recommended First)

**Zero installation, just Python:**

### Option A: With UV (10x faster) ⚡
```bash
# 1. Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Mac/Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Install and run
uv sync
uv run python simple_scraper.py https://example.com
```

### Option B: With pip (traditional)
```bash
# 1. Install minimal dependencies
pip install beautifulsoup4 httpx

# 2. Run test scrape
python simple_scraper.py https://example.com

# Done! Check scraped_data folder
```

**What you get:**
- ✅ Working web scraper
- ✅ HTML parsing
- ✅ JSON output
- ❌ No database
- ❌ No vector search

---

## 🧪 For Local Development (Most Common)

**Python + SQLite + ChromaDB (no Docker):**

### Option A: With UV (Recommended) ⚡
```bash
# One command!
uv sync

# Start developing
uv run python simple_scraper.py https://example.com
```

### Option B: With pip
**Windows:**
```bash
# Run setup script
run_local.bat
# Or PowerShell:
.\run_local.ps1
```

**Mac/Linux:**
```bash
# Install dependencies
pip install -r requirements-minimal.txt

# Test it
python simple_scraper.py https://example.com
```

**What you get:**
- ✅ Working web scraper
- ✅ SQLite database (local file)
- ✅ Vector search (ChromaDB)
- ✅ Embeddings (local model)
- ❌ No distributed workers
- ❌ No monitoring

---

## 🏢 For Production Features (Full Stack)

**PostgreSQL + Redis + Full Features:**

### Option A: Databases in Docker, Python Local (Hybrid)

```bash
# 1. Start just databases
docker-compose -f docker-compose.minimal.yml up -d

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start services locally
make dev
```

### Option B: Everything in Docker

```bash
# One command
docker-compose up -d

# Access services
# API: http://localhost:8000
# Flower: http://localhost:5555
```

**What you get:**
- ✅ Full production features
- ✅ PostgreSQL + pgvector
- ✅ Distributed workers (Celery)
- ✅ Redis caching
- ✅ Monitoring (Prometheus/Grafana)
- ✅ S3 storage (MinIO)

---

## 📊 Comparison Table

| Feature | Simple | Minimal | Hybrid | Full Docker |
|---------|--------|---------|--------|-------------|
| **Setup Time** | 1 min | 5 min (30s with UV) | 10 min | 15 min |
| **Installation** | pip/uv | pip/uv | Docker+pip/uv | Docker |
| **Database** | None | SQLite | PostgreSQL | PostgreSQL |
| **Vector Search** | ❌ | ChromaDB | pgvector | pgvector |
| **Distributed** | ❌ | ❌ | ✅ | ✅ |
| **Monitoring** | ❌ | ❌ | ❌ | ✅ |
| **Best For** | Testing | Development | Production Dev | Demo/Staging |

---

## 🎯 Recommended Path

### Day 1: Start Simple
```bash
python simple_scraper.py https://example.com
```
**Goal:** Understand how scraping works

### Day 2: Add Local Features
```bash
run_local.bat  # or .ps1 on Windows
```
**Goal:** Test with database and vector search

### Day 3: Add Production Features
```bash
docker-compose up -d
```
**Goal:** Try distributed workers and monitoring

---

## 🛠️ Installation by Operating System

### Windows

**Simplest (No Docker):**
```powershell
# PowerShell
.\run_local.ps1

# Or Command Prompt
run_local.bat
```

**With Docker:**
```powershell
# Install Docker Desktop first
# Then:
docker-compose up -d
```

### macOS

**With Homebrew:**
```bash
# Install services
brew install postgresql@15 redis
brew services start postgresql@15 redis

# Install Python deps
pip install -r requirements.txt

# Start scraper
make dev
```

**With Docker:**
```bash
docker-compose up -d
```

### Linux (Ubuntu/Debian)

**Native:**
```bash
# Install services
sudo apt update
sudo apt install postgresql-15 postgresql-15-pgvector redis-server
sudo systemctl start postgresql redis

# Install Python deps
pip install -r requirements.txt

# Start scraper
make dev
```

**With Docker:**
```bash
docker-compose up -d
```

---

## 🔥 Common Commands

### Testing
```bash
# Simple test
python simple_scraper.py https://example.com

# With output file
python simple_scraper.py https://example.com --output result.json

# Print to console
python simple_scraper.py https://example.com --print
```

### API Mode
```bash
# Start API server
uvicorn api.app:app --reload

# Test API
curl http://localhost:8000/docs
```

### With Docker
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Reset everything
docker-compose down -v
```

---

## 📖 Next Steps

After you have it running:

1. **Read the docs:**
   - `README.md` - Full documentation
   - `docs/ARCHITECTURE.md` - System design
   - `docs/LOCAL_SETUP.md` - Detailed local setup
   - `docs/PRD.md` - Product requirements

2. **Customize configuration:**
   - Copy `.env.example` to `.env`
   - Edit settings for your needs

3. **Add your scrapers:**
   - Check `scrapers/` folder
   - Implement custom scraping logic

4. **Scale up:**
   - Add more Celery workers
   - Configure proxy rotation
   - Enable monitoring

---

## ❓ Troubleshooting

### "Python not found"
```bash
# Download from python.org
# Or use package manager:
# Windows: winget install Python.Python.3.11
# Mac: brew install python@3.11
# Linux: sudo apt install python3.11
```

### "pip install fails"
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Use minimal requirements
pip install -r requirements-minimal.txt
```

### "Docker not working"
```bash
# Check Docker is running
docker --version

# Restart Docker Desktop (Windows/Mac)
# Or: sudo systemctl restart docker (Linux)
```

### "Port already in use"
```bash
# Change ports in .env:
API_PORT=8001
POSTGRES_PORT=5433
REDIS_PORT=6380
```

### "ChromaDB fails on Windows"
```bash
# Install Visual C++ Build Tools
# Or use alternative: pip install faiss-cpu
```

---

## 🎓 Learning Resources

### Understand the Stack

**Web Scraping:**
- Crawlee: https://crawlee.dev/python/
- Playwright: https://playwright.dev/python/
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/

**Vector Search:**
- ChromaDB: https://docs.trychroma.com/
- pgvector: https://github.com/pgvector/pgvector
- Embeddings: https://huggingface.co/sentence-transformers

**Infrastructure:**
- FastAPI: https://fastapi.tiangolo.com/
- Celery: https://docs.celeryq.dev/
- Redis: https://redis.io/docs/

### Example Use Cases

**1. Content Aggregation:**
```python
from simple_scraper import SimpleScraper

scraper = SimpleScraper()
result = await scraper.scrape("https://news.ycombinator.com")
print(f"Found {result['links_found']} articles")
```

**2. Price Monitoring:**
```python
# Schedule with Celery
@celery_app.task
def monitor_prices(url):
    result = scrape_page(url)
    if result['price'] < threshold:
        send_alert(result)
```

**3. Knowledge Base:**
```python
# Store with vector search
from storage.hybrid_search import hybrid_search_pages

results = await hybrid_search_pages(
    session,
    query="machine learning tutorials",
    top_k=10
)
```

---

## 💡 Pro Tips

### Development
- Use `--reload` flag for auto-restart during development
- Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs
- Use Flower (http://localhost:5555) to monitor Celery tasks

### Performance
- Enable Redis caching to avoid re-scraping
- Use hybrid search for better relevance
- Configure proxy rotation for rate limiting

### Production
- Use environment variables for secrets
- Enable monitoring (Prometheus/Grafana)
- Set up backup strategy for PostgreSQL
- Use S3/MinIO for HTML archival

---

## 🚨 Need Help?

1. **Check logs:**
   ```bash
   # Docker
   docker-compose logs -f api

   # Local
   tail -f logs/scraper.log
   ```

2. **Common issues:** See `docs/LOCAL_SETUP.md` troubleshooting section

3. **Reset everything:**
   ```bash
   # Docker
   docker-compose down -v
   docker-compose up -d

   # Local
   rm -rf venv scraped_data *.db
   run_local.bat
   ```

---

## ✨ What's Next?

Once you're comfortable:

- **Add authentication** (JWT tokens)
- **Implement rate limiting** per domain
- **Create custom scrapers** for specific sites
- **Build a dashboard** with Grafana
- **Scale horizontally** with multiple workers
- **Deploy to cloud** (AWS, GCP, Azure)

---

## 📝 Summary

**Fastest path to working scraper:**

```bash
# 1. Clone/download project
cd web-scraper

# 2. Install dependencies
pip install beautifulsoup4 httpx

# 3. Run test
python simple_scraper.py https://example.com

# 4. Check results
cat scraped_data/*.json
```

**That's it!** You now have a working web scraper. Explore the docs to add more features.

---

## 📚 Documentation Index

- `QUICKSTART.md` - You are here!
- `README.md` - Complete documentation
- `PROJECT_SUMMARY.md` - High-level overview
- `docs/LOCAL_SETUP.md` - Detailed local setup
- `docs/ARCHITECTURE.md` - System architecture
- `docs/PRD.md` - Product requirements

**Happy scraping! 🎉**
