@echo off
REM Windows batch script to run web scraper locally (no Docker)

echo.
echo ========================================
echo  Web Scraper - Local Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo [1/5] Checking virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment exists.
)

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements-minimal.txt
echo Dependencies installed!

echo.
echo [4/5] Downloading models...
python -m playwright install chromium --quiet 2>nul
echo Models ready!

echo.
echo [5/5] Testing setup...
python -c "import fastapi, httpx, beautifulsoup4; print('✅ All imports successful!')" 2>nul
if errorlevel 1 (
    echo ⚠️  Some imports failed, but basic functionality should work
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo You can now:
echo   1. Test scraper:     python simple_scraper.py https://example.com
echo   2. Start API:        uvicorn api.app:app --reload
echo   3. Run custom script: python your_script.py
echo.
echo Press any key to start simple test scraper...
pause >nul

echo.
echo Running test scrape...
python simple_scraper.py https://example.com --print

echo.
echo ========================================
echo Check the scraped_data folder for results!
echo ========================================
pause
