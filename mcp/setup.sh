#!/bin/bash

# Setup script for Penn Electric Racing MCP Server

echo "🏎️  Penn Electric Racing MCP Server Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -e .

# Check for .env file
echo ""
if [ -f ".env" ]; then
    echo "✓ .env file exists"
    
    # Check if GEMINI_API_KEY is set in .env
    if grep -q "GEMINI_API_KEY=.*[a-zA-Z0-9]" .env; then
        echo "✓ GEMINI_API_KEY is configured in .env"
    else
        echo "⚠️  GEMINI_API_KEY not configured in .env"
        echo "   Edit .env and add your API key from https://makersuite.google.com/app/apikey"
    fi
else
    echo "⚠️  .env file not found"
    echo ""
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
    echo "To enable AI-powered CAN search:"
    echo "1. Get an API key from https://makersuite.google.com/app/apikey"
    echo "2. Edit .env and add your GEMINI_API_KEY"
fi

# Check for CanDefines.xml
echo ""
can_defines_path="/Users/ericcao/Documents/GitHub/Penn-Electric-Racing/codegen/CanDefines.xml"
if [ -f "$can_defines_path" ]; then
    echo "✓ CanDefines.xml found at $can_defines_path"
else
    echo "⚠️  CanDefines.xml not found at default location"
    echo "   Set CAN_DEFINES_PATH environment variable if it's elsewhere"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Try the example:"
echo "  python example_usage.py"
echo ""
echo "Or run the MCP server:"
echo "  python -m per_mcp_server"
