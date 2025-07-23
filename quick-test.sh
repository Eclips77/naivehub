#!/bin/bash
# Quick test runner for NaiveHub
# This script runs the comprehensive test suite

echo ""
echo "================================"
echo "  NaiveHub Quick Test Runner"
echo "================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install Python 3.11+"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "⚠️  Warning: Docker not found. Will run standalone tests only."
    echo ""
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt --quiet

# Run the test suite
echo ""
echo "🚀 Running NaiveHub Integration Tests..."
echo ""
python3 test_naivehub.py

echo ""
echo "✅ Test completed!"
