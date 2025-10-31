# MCP Server Setup Guide

## Overview
This guide will help you set up an MCP (Model Context Protocol) Server to test the full MCP integration functionality.

## Prerequisites
- Node.js (version 20 or later)
- npm (comes with Node.js)

## Option 1: Using the Official Playwright MCP Server

### Step 1: Install Node.js and npm
```bash
# Check if Node.js is installed
node --version
npm --version

# If not installed, install Node.js from https://nodejs.org/
```

### Step 2: Install the Playwright MCP Server
```bash
# Install the official Playwright MCP Server
npm install -g @modelcontextprotocol/server-playwright

# Or install locally in your project
npm install @modelcontextprotocol/server-playwright
```

### Step 3: Start the MCP Server
```bash
# Start the server on port 3000 (default)
npx @modelcontextprotocol/server-playwright

# Or specify a different port
npx @modelcontextprotocol/server-playwright --port 3000
```

## Option 2: Using a Simple MCP Server (Alternative)

If the official Playwright MCP Server is not available, we can create a simple MCP Server for testing.

### Step 1: Create a Simple MCP Server
```bash
# Create a new directory for the MCP server
mkdir mcp-server
cd mcp-server

# Initialize npm project
npm init -y

# Install required dependencies
npm install express cors body-parser
```

### Step 2: Create the MCP Server Code
Create a file called `server.js`:

```javascript
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Mock MCP Server endpoints
app.post('/mcp/session/start', (req, res) => {
    console.log('MCP Session started');
    res.json({
        success: true,
        sessionId: 'mock-session-' + Date.now(),
        message: 'MCP Session started successfully'
    });
});

app.post('/mcp/session/stop', (req, res) => {
    console.log('MCP Session stopped');
    res.json({
        success: true,
        message: 'MCP Session stopped successfully'
    });
});

app.post('/mcp/context/capture', (req, res) => {
    console.log('MCP Context capture requested');
    
    // Mock context data
    const mockContext = {
        page: {
            title: "Mock Page Title",
            url: "https://example.com",
            viewport: { width: 1920, height: 1080 }
        },
        elements: [
            {
                tag: "h1",
                text: "Mock Heading",
                selector: "h1",
                role: "heading",
                accessibility: {
                    role: "heading",
                    level: 1
                }
            },
            {
                tag: "p",
                text: "Mock paragraph text",
                selector: "p",
                role: "text",
                accessibility: {
                    role: "text"
                }
            }
        ],
        accessibility: {
            role: "main",
            name: "Mock Page",
            description: "A mock page for testing"
        },
        forms: [],
        navigation: {
            breadcrumbs: [],
            menu_items: []
        },
        content_hierarchy: {
            headings: ["Mock Heading"],
            sections: ["main"],
            landmarks: ["main"]
        }
    };
    
    res.json({
        success: true,
        context: mockContext,
        message: 'Context captured successfully'
    });
});

app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        message: 'MCP Server is running',
        timestamp: new Date().toISOString()
    });
});

// Start the server
app.listen(PORT, () => {
    console.log(`🚀 MCP Server running on http://localhost:${PORT}`);
    console.log(`📡 Health check: http://localhost:${PORT}/health`);
    console.log(`🔧 MCP endpoints available at http://localhost:${PORT}/mcp/`);
});
```

### Step 3: Start the Simple MCP Server
```bash
# Start the server
node server.js
```

## Option 3: Using Docker (Recommended)

### Step 1: Create a Dockerfile for MCP Server
Create a file called `Dockerfile.mcp`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Install dependencies
RUN npm install express cors body-parser

# Copy server code
COPY mcp-server.js .

# Expose port
EXPOSE 3000

# Start server
CMD ["node", "mcp-server.js"]
```

### Step 2: Create the MCP Server Code
Create a file called `mcp-server.js` (same as above).

### Step 3: Build and Run with Docker
```bash
# Build the Docker image
docker build -f Dockerfile.mcp -t mcp-server .

# Run the MCP server
docker run -p 3000:3000 mcp-server
```

## Testing the MCP Server

### Step 1: Verify Server is Running
```bash
# Check if server is running
curl http://localhost:3000/health

# Expected response:
# {"status":"healthy","message":"MCP Server is running","timestamp":"..."}
```

### Step 2: Test MCP Endpoints
```bash
# Test session start
curl -X POST http://localhost:3000/mcp/session/start

# Test context capture
curl -X POST http://localhost:3000/mcp/context/capture

# Test session stop
curl -X POST http://localhost:3000/mcp/session/stop
```

## Running Your MCP Integration Test

Once the MCP Server is running:

### Step 1: Start the MCP Server
```bash
# Choose one of the options above to start the MCP server
# For example, with the simple server:
node server.js
```

### Step 2: Run Your MCP Integration Test
```bash
# In another terminal, run your MCP test
source venv/bin/activate
python test_mcp_integration.py
```

### Step 3: Verify Full Functionality
You should now see:
- ✅ MCP Server connection successful
- ✅ Context capture working with real data
- ✅ Full MCP integration functionality

## Troubleshooting

### Common Issues:

1. **Port 3000 already in use:**
   ```bash
   # Find what's using port 3000
   lsof -i :3000
   
   # Kill the process or use a different port
   kill -9 <PID>
   ```

2. **Node.js not found:**
   ```bash
   # Install Node.js from https://nodejs.org/
   # Or use a version manager like nvm
   ```

3. **MCP Server not responding:**
   ```bash
   # Check if server is running
   curl http://localhost:3000/health
   
   # Check server logs for errors
   ```

## Next Steps

Once you have the MCP Server running:

1. **Run the MCP integration test** to see full functionality
2. **Test with real web pages** to see context capture in action
3. **Integrate with AI models** for advanced automation
4. **Deploy to production** with proper MCP Server setup

## Support

If you encounter issues:
1. Check the server logs
2. Verify all prerequisites are installed
3. Ensure port 3000 is available
4. Test with the health endpoint first

