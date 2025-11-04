# Local Development Setup (No Docker Required)

## Quick Answer: Do I Need Docker?

**No!** You have 3 options:

| Option | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Local (No Docker)** | Development, testing | Simple, fast, debug-friendly | Manual setup |
| **Docker Compose** | Quick demo, staging | One command, isolated | Needs Docker installed |
| **Cloud** | Production | Scalable, managed | Costs money |

---

## Option 1: Pure Local Setup (Recommended for Development)

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 15+** (with pgvector)
- **Redis 7+**
- **MinIO** (optional - can use local filesystem)

### Step 1: Install PostgreSQL with pgvector

#### Windows:
```bash
# Download PostgreSQL installer from postgresql.org
# Then install pgvector:
# 1. Download from: https://github.com/pgvector/pgvector/releases
# 2. Extract and follow Windows installation instructions
```

#### macOS (Homebrew):
```bash
brew install postgresql@15
brew services start postgresql@15
brew install pgvector
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install postgresql-15 postgresql-15-pgvector
sudo systemctl start postgresql
```

### Step 2: Install Redis

#### Windows:
```bash
# Download Redis from: https://github.com/microsoftarchive/redis/releases
# Or use WSL:
wsl --install
wsl
sudo apt install redis-server
redis-server
```

#### macOS:
```bash
brew install redis
brew services start redis
```

#### Linux:
```bash
sudo apt install redis-server
sudo systemctl start redis
```

### Step 3: Set Up Python Environment

```bash
cd C:\Users\jnguyen\Documents\Project\web-scraper

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Configure Environment

```bash
# Copy and edit .env
cp .env.example .env
```

Edit `.env` for local setup:
```bash
# Local PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/webscraper

# Local Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Skip MinIO (use local filesystem for now)
# S3_ENDPOINT_URL=http://localhost:9000
```

### Step 5: Initialize Database

```bash
# Create database
psql -U postgres -c "CREATE DATABASE webscraper;"

# Enable pgvector extension
psql -U postgres -d webscraper -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations (if using Alembic)
# alembic upgrade head

# Or initialize directly with Python
python -c "
from storage.database import init_db
import asyncio
asyncio.run(init_db())
"
```

### Step 6: Start Services

**Terminal 1 - API Server:**
```bash
cd C:\Users\jnguyen\Documents\Project\web-scraper
venv\Scripts\activate
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Celery Worker:**
```bash
cd C:\Users\jnguyen\Documents\Project\web-scraper
venv\Scripts\activate
celery -A orchestration.celery_app worker --loglevel=info --pool=solo
```

**Terminal 3 - Celery Beat (Optional - for scheduled jobs):**
```bash
cd C:\Users\jnguyen\Documents\Project\web-scraper
venv\Scripts\activate
celery -A orchestration.celery_app beat --loglevel=info
```

**Terminal 4 - Flower (Optional - monitoring):**
```bash
cd C:\Users\jnguyen\Documents\Project\web-scraper
venv\Scripts\activate
celery -A orchestration.celery_app flower --port=5555
```

### Step 7: Access Services

- **API:** http://localhost:8000/docs
- **Flower:** http://localhost:5555 (if started)

---

## Option 2: Hybrid Setup (Minimal Docker)

**Use Docker only for PostgreSQL and Redis, run Python locally:**

### docker-compose.minimal.yml
```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: scraper
      POSTGRES_PASSWORD: scraper123
      POSTGRES_DB: webscraper
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**Start just databases:**
```bash
docker-compose -f docker-compose.minimal.yml up -d
```

**Then run Python services locally as in Option 1, Step 6**

---

## Option 3: Lightweight Alternative Stack

If you want something even simpler:

### Use SQLite Instead of PostgreSQL

**Pros:**
- Zero installation (built into Python)
- Single file database
- Perfect for development/testing

**Cons:**
- No pgvector (use ChromaDB or FAISS instead)
- Single-writer (can't handle distributed workers well)
- Limited to ~100K records realistically

### Alternative Configuration:

```python
# config/settings.py (modify for SQLite)

DATABASE_URL = "sqlite+aiosqlite:///./webscraper.db"

# For vectors, add ChromaDB
VECTOR_DB_TYPE = "chromadb"  # Instead of pgvector
CHROMADB_PATH = "./chroma_data"
```

### Install Alternative Dependencies:

```bash
pip install chromadb  # Instead of pgvector
# or
pip install faiss-cpu  # Facebook's vector search
```

---

## Comparison: Docker vs Local

### Docker Compose (Full Stack)

**Pros:**
- ✅ One command setup (`docker-compose up`)
- ✅ Isolated environment
- ✅ Same environment as production
- ✅ Easy to reset (delete volumes)
- ✅ Includes monitoring (Prometheus, Grafana)

**Cons:**
- ❌ Needs Docker Desktop (4GB+ RAM)
- ❌ Slower on Windows (WSL2 overhead)
- ❌ Harder to debug (inside containers)
- ❌ Uses more disk space

### Local Development

**Pros:**
- ✅ Fast iteration (no container rebuilds)
- ✅ Easy debugging (native IDE support)
- ✅ Direct file access
- ✅ Lower resource usage
- ✅ Works without Docker

**Cons:**
- ❌ Manual installation of services
- ❌ "Works on my machine" problems
- ❌ Need to manage multiple terminals
- ❌ Harder to match production environment

---

## Recommended Setup by Use Case

### 1. Learning / Prototyping
**Use:** Local setup with SQLite + ChromaDB
```bash
pip install chromadb
python simple_scraper.py  # No database needed!
```

### 2. Active Development
**Use:** Hybrid (Docker for DB, local Python)
```bash
docker-compose -f docker-compose.minimal.yml up -d
python run.py
```

### 3. Testing / Demo
**Use:** Full Docker Compose
```bash
docker-compose up -d
```

### 4. Production
**Use:** Cloud (AWS, GCP) or Kubernetes
```bash
kubectl apply -f k8s/
```

---

## Simplified Local Setup Script

Create `run_local.bat` (Windows) or `run_local.sh` (Unix):

```bash
#!/bin/bash
# run_local.sh - Start everything locally

echo "🚀 Starting Web Scraper (Local Mode)"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -q -r requirements.txt

# Check PostgreSQL
if ! psql -U postgres -lqt | cut -d \| -f 1 | grep -qw webscraper; then
    echo "Creating database..."
    psql -U postgres -c "CREATE DATABASE webscraper;"
    psql -U postgres -d webscraper -c "CREATE EXTENSION vector;"
fi

# Initialize database
python -c "from storage.database import init_db; import asyncio; asyncio.run(init_db())"

echo "✅ Setup complete!"
echo ""
echo "Now start services in separate terminals:"
echo "  Terminal 1: uvicorn api.app:app --reload"
echo "  Terminal 2: celery -A orchestration.celery_app worker"
echo ""
echo "Or use: make dev"
```

---

## Windows-Specific Quick Start

### PowerShell Script: `start_local.ps1`

```powershell
# Check Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found. Install from python.org"
    exit
}

# Create venv
if (!(Test-Path venv)) {
    python -m venv venv
}

# Activate
.\venv\Scripts\Activate.ps1

# Install
pip install -r requirements.txt

Write-Host "✅ Ready!"
Write-Host "Run: uvicorn api.app:app --reload"
```

---

## Troubleshooting Local Setup

### "PostgreSQL not found"
```bash
# Windows: Add to PATH or specify full path
"C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres

# Or use online PostgreSQL (free tier)
# DATABASE_URL=postgresql://user:pass@db.provider.com/dbname
```

### "Redis connection refused"
```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# Start Redis
redis-server  # or: brew services start redis
```

### "ModuleNotFoundError"
```bash
# Make sure venv is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### "pgvector extension not found"
```bash
# Check PostgreSQL version (need 15+)
psql --version

# Install pgvector manually
# See: https://github.com/pgvector/pgvector#installation
```

---

## Minimal Working Example (No Docker, No Database)

If you just want to test scraping without setting up infrastructure:

```python
# minimal_scraper.py - Zero dependencies scraper

import asyncio
from scrapers.crawlee_scraper import CrawleeScraper

async def main():
    scraper = CrawleeScraper()
    result = await scraper.scrape("https://example.com")

    print(f"✅ Scraped: {result['title']}")
    print(f"Content: {result['content'][:200]}...")

    # Save to file instead of database
    with open('scraped.json', 'w') as f:
        import json
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
```

```bash
# Run it
python minimal_scraper.py
```

---

## My Recommendation

**For you specifically:**

### **Phase 1: Start Simple**
```bash
# Just Python + SQLite + local filesystem
pip install -r requirements-minimal.txt
python simple_scraper.py
```

### **Phase 2: Add Search**
```bash
# Add ChromaDB for vector search
pip install chromadb
# Now you have semantic search!
```

### **Phase 3: Scale Up**
```bash
# When ready, add PostgreSQL + Redis
# Use docker-compose.minimal.yml for just databases
docker-compose -f docker-compose.minimal.yml up -d
```

### **Phase 4: Production**
```bash
# Use full Docker Compose or cloud deployment
docker-compose up -d
```

---

## Bottom Line

**You absolutely do NOT need Docker to use this project.**

**Three ways to run without Docker:**

1. **Simplest:** Python + SQLite + ChromaDB (no external services)
2. **Middle:** Python + cloud-hosted PostgreSQL + Redis (e.g., Supabase free tier)
3. **Full local:** Install PostgreSQL + Redis locally, run Python normally

**Docker is just a convenience for:**
- Quick demos
- Consistent environments
- Easy reset/cleanup
- Production-like setup

**For development, local setup is often better** - faster, easier to debug, and you have full control.

---

## Next Steps

Want me to create:
- [ ] `requirements-minimal.txt` (lightweight dependencies)
- [ ] `simple_scraper.py` (no database needed)
- [ ] `run_local.bat` (Windows startup script)
- [ ] SQLite + ChromaDB configuration

Just let me know what you need!
