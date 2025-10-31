#!/bin/bash
# Start MCP Server Script
# =======================

echo "🚀 Starting MCP Server for Web Task Automator"
echo "=============================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed!"
    echo "Please install Node.js from https://nodejs.org/"
    echo "Required version: Node.js 20 or later"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 20 ]; then
    echo "❌ Node.js version $NODE_VERSION is too old!"
    echo "Required version: Node.js 20 or later"
    exit 1
fi

echo "✅ Node.js version: $(node --version)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed!"
    echo "Please install npm (comes with Node.js)"
    exit 1
fi

echo "✅ npm version: $(npm --version)"

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ package.json not found!"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies!"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Dependencies already installed"
fi

# Check if simple_mcp_server.js exists
if [ ! -f "simple_mcp_server.js" ]; then
    echo "❌ simple_mcp_server.js not found!"
    echo "Please ensure the MCP server file exists"
    exit 1
fi

# Check if port 3000 is available
if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 3000 is already in use!"
    echo "Please stop the service using port 3000 or use a different port"
    echo "To find what's using port 3000: lsof -i :3000"
    exit 1
fi

echo "✅ Port 3000 is available"

# Start the MCP server
echo "🚀 Starting MCP Server..."
echo "📡 Server will be available at: http://localhost:3000"
echo "🔧 Health check: http://localhost:3000/health"
echo "📚 API documentation: http://localhost:3000/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
npm start

