#!/bin/bash

# BioBatchNet PyPI Publishing Script
# Usage: ./publish.sh [test|prod]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default to test mode
MODE=${1:-test}

echo -e "${GREEN}BioBatchNet Publishing Script${NC}"
echo "Mode: $MODE"
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info/

# Install/upgrade build tools
echo -e "${YELLOW}Installing build tools...${NC}"
pip install --upgrade pip setuptools wheel twine build

# Build the distribution
echo -e "${YELLOW}Building distribution...${NC}"
python -m build

# Show built files
echo -e "${GREEN}Built files:${NC}"
ls -la dist/

# Twine check (README rendering, metadata)
echo -e "${YELLOW}Validating distributions with twine...${NC}"
python -m twine check dist/*

if [ "$MODE" == "test" ]; then
    echo -e "${YELLOW}Uploading to TestPyPI...${NC}"
    python -m twine upload --repository testpypi dist/*
    
    echo -e "${GREEN}Success! Package uploaded to TestPyPI${NC}"
    echo ""
    echo "To test installation:"
    echo "pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ biobatchnet"
    
elif [ "$MODE" == "prod" ]; then
    echo -e "${RED}WARNING: About to upload to production PyPI${NC}"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Uploading to PyPI...${NC}"
        python -m twine upload dist/*
        
        echo -e "${GREEN}Success! Package uploaded to PyPI${NC}"
        echo ""
        echo "Users can now install with:"
        echo "pip install biobatchnet"
    else
        echo "Upload cancelled"
        exit 1
    fi
else
    echo -e "${RED}Invalid mode: $MODE${NC}"
    exit 1
fi

echo -e "${GREEN}Done!${NC}"
