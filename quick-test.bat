@echo off
REM Quick test runner for NaiveHub
REM This script runs the comprehensive test suite

echo.
echo ================================
echo  NaiveHub Quick Test Runner
echo ================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Check if Docker is available
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: Docker not found. Will run standalone tests only.
    echo.
)

REM Install dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements.txt --quiet

REM Run the test suite
echo.
echo 🚀 Running NaiveHub Integration Tests...
echo.
python test_naivehub.py

echo.
echo ✅ Test completed!
pause
