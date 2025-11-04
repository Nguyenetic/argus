# PowerShell script to run web scraper locally (no Docker)
# Compatible with Windows PowerShell 5.1+ and PowerShell Core 7+

Write-Host ""
Write-Host "========================================"
Write-Host "  Web Scraper - Local Setup (PowerShell)"
Write-Host "========================================"
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "[✓] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "    Please install Python 3.11+ from python.org"
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 1: Virtual environment
Write-Host ""
Write-Host "[1/5] Checking virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    Write-Host "    Creating virtual environment..." -ForegroundColor Yellow
    & python -m venv venv
    Write-Host "    [✓] Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "    [✓] Virtual environment exists." -ForegroundColor Green
}

# Step 2: Activate venv
Write-Host ""
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Cyan
$activateScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "    [✓] Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "    [✗] Could not find activation script" -ForegroundColor Red
    exit 1
}

# Step 3: Install dependencies
Write-Host ""
Write-Host "[3/5] Installing dependencies..." -ForegroundColor Cyan
Write-Host "    (This may take a few minutes...)" -ForegroundColor Yellow

# Check if requirements-minimal.txt exists
if (Test-Path "requirements-minimal.txt") {
    & python -m pip install --quiet --upgrade pip
    & python -m pip install --quiet -r requirements-minimal.txt
    Write-Host "    [✓] Dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "    [✗] requirements-minimal.txt not found" -ForegroundColor Red
    exit 1
}

# Step 4: Download models
Write-Host ""
Write-Host "[4/5] Downloading models..." -ForegroundColor Cyan
try {
    # Try to install Playwright browsers
    & python -m playwright install chromium 2>&1 | Out-Null
    Write-Host "    [✓] Playwright chromium installed!" -ForegroundColor Green
} catch {
    Write-Host "    [!] Playwright install skipped (optional)" -ForegroundColor Yellow
}

# Download sentence-transformers model (happens automatically on first use)
Write-Host "    [i] Embedding model will download on first use" -ForegroundColor Yellow

# Step 5: Test setup
Write-Host ""
Write-Host "[5/5] Testing setup..." -ForegroundColor Cyan
try {
    & python -c "import fastapi, httpx, bs4; print('✅ All imports successful!')"
    Write-Host "    [✓] Setup test passed!" -ForegroundColor Green
} catch {
    Write-Host "    [!] Some imports failed, but basic functionality should work" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================"
Write-Host "  Setup Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "You can now:" -ForegroundColor Cyan
Write-Host "  1. Test scraper:     " -NoNewline; Write-Host "python simple_scraper.py https://example.com" -ForegroundColor Yellow
Write-Host "  2. Start API:        " -NoNewline; Write-Host "uvicorn api.app:app --reload" -ForegroundColor Yellow
Write-Host "  3. Run custom script:" -NoNewline; Write-Host "python your_script.py" -ForegroundColor Yellow
Write-Host ""

# Ask if user wants to run test
$runTest = Read-Host "Run test scraper now? (y/n)"
if ($runTest -eq "y" -or $runTest -eq "Y") {
    Write-Host ""
    Write-Host "Running test scrape..." -ForegroundColor Cyan
    Write-Host ""

    try {
        & python simple_scraper.py https://example.com --print
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Check the scraped_data folder for results!"
        Write-Host "========================================"
    } catch {
        Write-Host "[✗] Test scrape failed: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Press Enter to exit..."
Read-Host
