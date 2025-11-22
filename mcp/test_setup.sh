#!/bin/bash
# Quick test script for the MCP server

echo "==================================="
echo "PER MCP Server - Quick Test"
echo "==================================="
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "   Please create .env file with your S3 credentials"
    echo "   Run: cp .env.example .env"
    exit 1
fi

echo "✓ .env file found"

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
python -c "import mcp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ MCP library not installed"
    echo "   Run: pip install -e ."
    exit 1
fi
echo "✓ MCP library installed"

python -c "import rapidfuzz" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ rapidfuzz not installed"
    echo "   Run: pip install -e ."
    exit 1
fi
echo "✓ rapidfuzz installed"

python -c "import boto3" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ boto3 not installed"
    echo "   Run: pip install -e ."
    exit 1
fi
echo "✓ boto3 installed"

# Test the server starts
echo ""
echo "Testing server startup..."
timeout 5 python -m per_mcp_server 2>&1 | head -n 20 &
SERVER_PID=$!
sleep 2

if ps -p $SERVER_PID > /dev/null; then
    echo "✓ Server started successfully"
    kill $SERVER_PID 2>/dev/null
else
    echo "❌ Server failed to start"
    echo "   Check logs above for errors"
    exit 1
fi

echo ""
echo "==================================="
echo "✅ All tests passed!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Configure Claude Desktop (see GEMINI_SETUP.md)"
echo "2. Or test with: python example_usage.py"
echo "3. Or run inspector: npx @modelcontextprotocol/inspector python -m per_mcp_server"
echo ""
