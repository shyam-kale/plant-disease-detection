#!/bin/bash

echo "========================================"
echo "  CropHealth AI - Starting Server"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Check if MySQL is running
echo "Checking MySQL connection..."
mysql -u root -pRoot@1234 -e "SELECT 1;" &> /dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Cannot connect to MySQL"
    echo "Please ensure MySQL is running and credentials are correct"
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the application
echo ""
echo "Starting CropHealth AI..."
echo ""
python3 app.py
