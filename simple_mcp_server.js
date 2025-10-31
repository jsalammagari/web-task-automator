#!/usr/bin/env node
/**
 * Simple MCP Server for Testing
 * =============================
 * 
 * This is a simple MCP (Model Context Protocol) Server that provides
 * mock endpoints for testing the MCP integration.
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Logging middleware
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Mock MCP Server endpoints
app.post('/mcp/session/start', (req, res) => {
    console.log('🚀 MCP Session started');
    res.json({
        success: true,
        sessionId: 'mock-session-' + Date.now(),
        message: 'MCP Session started successfully',
        timestamp: new Date().toISOString()
    });
});

app.post('/mcp/session/stop', (req, res) => {
    console.log('🛑 MCP Session stopped');
    res.json({
        success: true,
        message: 'MCP Session stopped successfully',
        timestamp: new Date().toISOString()
    });
});

// Browser session endpoints (for MCP integration compatibility)
app.post('/browser/start', (req, res) => {
    console.log('🚀 Browser session started');
    res.json({
        success: true,
        sessionId: 'browser-session-' + Date.now(),
        message: 'Browser session started successfully',
        timestamp: new Date().toISOString()
    });
});

app.post('/browser/close', (req, res) => {
    console.log('🛑 Browser session closed');
    res.json({
        success: true,
        message: 'Browser session closed successfully',
        timestamp: new Date().toISOString()
    });
});

app.post('/browser/navigate', (req, res) => {
    const { url } = req.body;
    console.log(`🌐 Browser navigating to: ${url}`);
    res.json({
        success: true,
        message: `Successfully navigated to ${url}`,
        url: url,
        timestamp: new Date().toISOString()
    });
});

app.post('/mcp/context/capture', (req, res) => {
    console.log('🧠 MCP Context capture requested');
    
    // Mock context data that simulates what a real MCP Server would return
    const mockContext = {
        page: {
            title: "Example Domain",
            url: "https://example.com",
            viewport: { width: 1920, height: 1080 },
            timestamp: new Date().toISOString()
        },
        elements: [
            {
                tag: "h1",
                text: "Example Domain",
                selector: "h1",
                role: "heading",
                level: 1,
                accessibility: {
                    role: "heading",
                    level: 1,
                    name: "Example Domain"
                }
            },
            {
                tag: "p",
                text: "This domain is for use in illustrative examples in documents.",
                selector: "p",
                role: "text",
                accessibility: {
                    role: "text"
                }
            },
            {
                tag: "a",
                text: "More information...",
                selector: "a",
                role: "link",
                href: "https://www.iana.org/domains/example",
                accessibility: {
                    role: "link",
                    name: "More information"
                }
            }
        ],
        accessibility: {
            role: "main",
            name: "Example Domain",
            description: "A simple example page for testing"
        },
        forms: [],
        navigation: {
            breadcrumbs: [],
            menu_items: []
        },
        content_hierarchy: {
            headings: ["Example Domain"],
            sections: ["main"],
            landmarks: ["main"]
        },
        metadata: {
            capture_time: new Date().toISOString(),
            mcp_server: "simple-mock-server",
            version: "1.0.0"
        }
    };
    
    res.json({
        success: true,
        context: mockContext,
        message: 'Context captured successfully',
        timestamp: new Date().toISOString()
    });
});

app.post('/mcp/context/capture-form', (req, res) => {
    console.log('📝 MCP Form context capture requested');
    
    // Mock form context data
    const mockFormContext = {
        page: {
            title: "HTML form",
            url: "https://httpbin.org/forms/post",
            viewport: { width: 1920, height: 1080 },
            timestamp: new Date().toISOString()
        },
        elements: [
            {
                tag: "input",
                name: "custname",
                type: "text",
                placeholder: "Customer name",
                selector: "input[name='custname']",
                role: "textbox",
                accessibility: {
                    role: "textbox",
                    name: "Customer name"
                }
            },
            {
                tag: "input",
                name: "custtel",
                type: "tel",
                placeholder: "Telephone number",
                selector: "input[name='custtel']",
                role: "textbox",
                accessibility: {
                    role: "textbox",
                    name: "Telephone number"
                }
            },
            {
                tag: "input",
                name: "custemail",
                type: "email",
                placeholder: "Email address",
                selector: "input[name='custemail']",
                role: "textbox",
                accessibility: {
                    role: "textbox",
                    name: "Email address"
                }
            },
            {
                tag: "textarea",
                name: "comments",
                placeholder: "Comments",
                selector: "textarea[name='comments']",
                role: "textbox",
                accessibility: {
                    role: "textbox",
                    name: "Comments"
                }
            }
        ],
        forms: [
            {
                form_id: "form1",
                action: "/post",
                method: "POST",
                fields: ["custname", "custtel", "custemail", "comments"],
                submit_button: "input[type='submit']"
            }
        ],
        accessibility: {
            role: "main",
            name: "HTML form",
            description: "A form for customer information"
        },
        navigation: {
            breadcrumbs: [],
            menu_items: []
        },
        content_hierarchy: {
            headings: ["HTML form"],
            sections: ["form"],
            landmarks: ["main", "form"]
        },
        metadata: {
            capture_time: new Date().toISOString(),
            mcp_server: "simple-mock-server",
            version: "1.0.0"
        }
    };
    
    res.json({
        success: true,
        context: mockFormContext,
        message: 'Form context captured successfully',
        timestamp: new Date().toISOString()
    });
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        message: 'MCP Server is running',
        timestamp: new Date().toISOString(),
        endpoints: [
            'POST /mcp/session/start',
            'POST /mcp/session/stop',
            'POST /mcp/context/capture',
            'POST /mcp/context/capture-form',
            'GET /health'
        ]
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Simple MCP Server for Testing',
        version: '1.0.0',
        endpoints: {
            health: 'GET /health',
            session_start: 'POST /mcp/session/start',
            session_stop: 'POST /mcp/session/stop',
            context_capture: 'POST /mcp/context/capture',
            form_context_capture: 'POST /mcp/context/capture-form'
        },
        documentation: 'See setup_mcp_server.md for usage instructions'
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        success: false,
        error: err.message,
        timestamp: new Date().toISOString()
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.path,
        method: req.method,
        timestamp: new Date().toISOString()
    });
});

// Start the server
app.listen(PORT, () => {
    console.log('🚀 Simple MCP Server started!');
    console.log(`📡 Server running on http://localhost:${PORT}`);
    console.log(`🔧 Health check: http://localhost:${PORT}/health`);
    console.log(`📚 API documentation: http://localhost:${PORT}/`);
    console.log('');
    console.log('Available endpoints:');
    console.log('  POST /mcp/session/start     - Start MCP session');
    console.log('  POST /mcp/session/stop      - Stop MCP session');
    console.log('  POST /mcp/context/capture   - Capture page context');
    console.log('  POST /mcp/context/capture-form - Capture form context');
    console.log('  GET  /health                - Health check');
    console.log('');
    console.log('Press Ctrl+C to stop the server');
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down MCP Server...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Shutting down MCP Server...');
    process.exit(0);
});
