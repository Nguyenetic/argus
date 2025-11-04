# 🚀 UV Setup Guide

**Modern Python package management with `uv` - 10-100x faster than pip!**

---

## What is UV?

[UV](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver, written in Rust. It's a drop-in replacement for pip that's **10-100x faster**.

**Benefits:**
- ⚡ **10-100x faster** than pip
- 🔒 **Better dependency resolution**
- 📦 **Built-in virtual environment management**
- 🎯 **Compatible with pyproject.toml**
- 🔄 **Automatic lockfile generation**

---

## Quick Start with UV

### 1. Install UV

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Mac/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative (pip):**
```bash
pip install uv
```

### 2. Create Project Environment

```bash
cd web-scraper

# Create venv and install dependencies (one command!)
uv sync

# That's it! 🎉
```

---

## Installation Options

### Option 1: Minimal Setup (No Docker, SQLite)

**Perfect for:** Testing, learning, local development

```bash
# Install only core dependencies
uv sync

# Or explicitly:
uv pip install -e .
```

**What you get:**
- FastAPI + Uvicorn
- BeautifulSoup4 (HTML parsing)
- SQLite (built-in database)
- ChromaDB (vector search)
- sentence-transformers (embeddings)

**Test it:**
```bash
python simple_scraper.py https://example.com
```

---

### Option 2: Add Scraping Features

**Perfect for:** Advanced scraping needs

```bash
# Install with scraping extras
uv sync --extra scraping
```

**What you get:**
- Crawlee + Playwright
- SeleniumBase (Cloudflare bypass)
- Scrapling (adaptive selectors)

**Test it:**
```bash
uv run python -c "from scrapers.crawlee_scraper import CrawleeScraper; print('✅ Scraping extras loaded!')"
```

---

### Option 3: Full Stack (PostgreSQL + Redis)

**Perfect for:** Production features, distributed processing

```bash
# Install all production dependencies
uv sync --extra full
```

**What you get:**
- PostgreSQL + pgvector
- Redis + Celery
- All scraping features
- Knowledge graph (spaCy)
- S3/MinIO storage
- Monitoring (Prometheus)

**Start services:**
```bash
# PostgreSQL + Redis (Docker)
docker-compose -f docker-compose.minimal.yml up -d

# Start API
uv run uvicorn api.app:app --reload
```

---

### Option 4: Development Setup

**Perfect for:** Contributing to the project

```bash
# Install everything including dev tools
uv sync --extra dev --extra full
```

**What you get:**
- All production dependencies
- pytest + coverage
- black + ruff (formatting)
- mypy (type checking)
- pre-commit hooks

**Run tests:**
```bash
uv run pytest
```

---

## Common UV Commands

### Installation

```bash
# Install dependencies from pyproject.toml
uv sync

# Install with specific extras
uv sync --extra full
uv sync --extra scraping --extra vector

# Install in editable mode
uv pip install -e .

# Install specific package
uv pip install requests
```

### Running Commands

```bash
# Run Python with uv environment
uv run python simple_scraper.py

# Run pytest
uv run pytest

# Run uvicorn
uv run uvicorn api.app:app --reload

# Run celery worker
uv run celery -A orchestration.celery_app worker
```

### Environment Management

```bash
# Create new venv
uv venv

# Activate venv (Windows)
.venv\Scripts\activate

# Activate venv (Mac/Linux)
source .venv/bin/activate

# Remove venv
rm -rf .venv
```

### Dependency Management

```bash
# Add new dependency
uv pip install package-name
# Then update pyproject.toml manually

# Upgrade all packages
uv pip install --upgrade -r pyproject.toml

# Show installed packages
uv pip list

# Generate requirements.txt (for compatibility)
uv pip freeze > requirements.txt
```

---

## Migration from requirements.txt

If you're coming from the old setup:

### Before (pip):
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

### After (uv):
```bash
uv sync
# That's it! ✨
```

**Speed comparison:**
- **pip**: ~2-5 minutes
- **uv**: ~10-30 seconds

---

## pyproject.toml Overview

Our `pyproject.toml` defines dependency groups:

```toml
[project]
dependencies = [...]  # Core/minimal

[project.optional-dependencies]
full = [...]           # All production features
scraping = [...]       # Advanced scraping
vector = [...]         # PostgreSQL + pgvector
distributed = [...]    # Celery + Redis
knowledge = [...]      # spaCy NER
storage = [...]        # S3/MinIO
monitoring = [...]     # Prometheus + Grafana
dev = [...]           # Development tools
all = [...]           # Everything
```

---

## Installation by Use Case

### 1. Quick Test (2 minutes)

```bash
uv sync
uv run python simple_scraper.py https://example.com
```

### 2. Local Development (5 minutes)

```bash
uv sync --extra scraping
uv run uvicorn api.app:app --reload
```

### 3. Production Setup (10 minutes)

```bash
# Install everything
uv sync --extra full

# Start databases (Docker)
docker-compose -f docker-compose.minimal.yml up -d

# Start API
uv run uvicorn api.app:app

# Start worker (separate terminal)
uv run celery -A orchestration.celery_app worker
```

### 4. Development (15 minutes)

```bash
# Install with dev tools
uv sync --extra dev --extra full

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check . --fix
```

---

## Troubleshooting UV

### "uv: command not found"

```bash
# Add to PATH (Windows)
$env:Path += ";$HOME\.cargo\bin"

# Add to PATH (Mac/Linux)
export PATH="$HOME/.cargo/bin:$PATH"

# Or reinstall
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "No module named 'X'"

```bash
# Ensure you're using uv's environment
uv run python your_script.py

# Or activate venv first
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

### "Dependency conflict"

```bash
# UV has better resolution than pip, but if issues:
uv sync --refresh

# Or start fresh
rm -rf .venv
uv sync
```

### "Package X not found"

```bash
# Check if it's in optional dependencies
uv sync --extra full

# Or install manually
uv pip install package-name
```

---

## UV vs pip vs poetry

| Feature | pip | poetry | **uv** |
|---------|-----|--------|--------|
| **Speed** | Slow (5 min) | Medium (2 min) | **Fast (30s)** ⚡ |
| **Lock file** | ❌ | ✅ | ✅ |
| **Dependency resolution** | Basic | Good | **Excellent** |
| **Virtual env** | Manual | Built-in | **Built-in** |
| **pyproject.toml** | Partial | ✅ | ✅ |
| **Written in** | Python | Python | **Rust** 🦀 |

---

## Best Practices with UV

### 1. Always Use `uv sync`

```bash
# After cloning repo
uv sync

# After pulling changes
git pull
uv sync

# After changing pyproject.toml
uv sync
```

### 2. Use `uv run` for Scripts

```bash
# Instead of:
python script.py

# Use:
uv run python script.py
```

### 3. Pin Dependencies for Production

```bash
# Generate lockfile
uv pip freeze > requirements.lock

# Install from lockfile
uv pip install -r requirements.lock
```

### 4. Use Extras for Optional Features

```bash
# Development
uv sync --extra dev

# Production
uv sync --extra full

# Custom combination
uv sync --extra scraping --extra vector
```

---

## Integration with IDEs

### VS Code

1. Install Python extension
2. Select interpreter: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Mac/Linux)
3. UV will auto-detect the environment

### PyCharm

1. **Settings** → **Project** → **Python Interpreter**
2. Add **Existing Environment**
3. Select `.venv/bin/python`

### Cursor / Claude Code

UV environments work automatically! Just run:
```bash
uv sync
```

---

## CI/CD with UV

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Run tests
        run: uv run pytest
```

### Docker with UV

```dockerfile
FROM python:3.11-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml .
COPY . .

# Install dependencies with UV
RUN uv sync --extra full

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0"]
```

---

## Comparison: Old vs New Setup

### Old Way (requirements.txt)

```bash
# Step 1: Create venv
python -m venv venv

# Step 2: Activate
source venv/bin/activate

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Install deps (slow!)
pip install -r requirements.txt  # ⏰ 2-5 minutes

# Step 5: Install optional deps
pip install -r requirements-minimal.txt
```

### New Way (uv + pyproject.toml)

```bash
# One command!
uv sync  # ⚡ 10-30 seconds

# Choose your extras
uv sync --extra full
uv sync --extra scraping --extra dev
```

**Benefits:**
- 🚀 **90% faster installation**
- 🔒 **Better dependency resolution**
- 🎯 **Organized dependency groups**
- 📦 **Modern Python packaging**

---

## Converting Other Projects to UV

Have another project with requirements.txt?

```bash
cd your-project

# Create pyproject.toml from requirements.txt
uv pip compile requirements.txt -o pyproject.toml

# Or manually create pyproject.toml
# Then:
uv sync
```

---

## FAQ

### Do I still need requirements.txt?

No! But you can keep it for compatibility:

```bash
# Generate requirements.txt from pyproject.toml
uv pip freeze > requirements.txt
```

### Can I use pip with pyproject.toml?

Yes! But it's slower:

```bash
pip install -e .
pip install -e .[full]
```

### How do I add a new dependency?

```bash
# Option 1: Install with uv
uv pip install package-name

# Option 2: Edit pyproject.toml
# Add to [project.dependencies] or [project.optional-dependencies]
# Then: uv sync
```

### Does UV work with Docker?

Yes! See the Docker example above or use our updated Dockerfile.

---

## Next Steps

1. **Install UV:** Follow installation steps above
2. **Sync dependencies:** `uv sync`
3. **Test it:** `uv run python simple_scraper.py https://example.com`
4. **Choose your extras:** See "Installation Options" above
5. **Start scraping!** 🚀

---

## Resources

- **UV Docs:** https://github.com/astral-sh/uv
- **pyproject.toml Spec:** https://packaging.python.org/en/latest/specifications/pyproject-toml/
- **Our pyproject.toml:** [pyproject.toml](pyproject.toml)

---

**Welcome to modern Python packaging! 🎉**

Need help? Check our other guides:
- [QUICKSTART.md](QUICKSTART.md) - Get started fast
- [docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md) - Detailed setup
- [README.md](README.md) - Complete documentation
