# Web Task Automator

A Python program that can automatically perform tasks on websites using Playwright for browser automation.

## Features

- **Basic Browser Automation**: Control web browsers using Playwright
- **Core Browser Actions**: Click, type, navigate, and interact with web elements
- **Fixed Task Automation**: Complete automated workflows for specific tasks
- **E-commerce Automation**: Real-world shopping and product search automation
- **Reliability & Error Handling**: Comprehensive error handling and retry mechanisms
- **MCP Server Integration**: AI context integration with Playwright MCP Server
- **Element Interaction**: Get text, attributes, and handle multiple elements
- **Page Navigation**: Back, forward, refresh, and scroll functionality
- **Element Waiting**: Wait for elements to be visible, clickable, or in specific states
- **Timeout Management**: Advanced timeout handling for slow pages and operations
- **Retry Mechanisms**: Exponential backoff retry for failed operations
- **Exception Handling**: Custom exceptions for different error types
- **Enhanced Logging**: Detailed logging for debugging and monitoring
- **Performance Tracking**: Operation timing and performance metrics
- **Accessibility Data**: Extract accessibility tree and element roles
- **AI Context**: Generate AI-friendly webpage context data
- **Edge Case Handling**: Handle slow loading, missing elements, and network issues
- **Async Support**: Built with async/await for better performance
- **Context Management**: Proper resource cleanup with context managers

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup Steps

#### Quick Setup (Recommended)

1. **Clone or download this project**
   ```bash
   cd web-task-automator
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Run the automated setup script**
   ```bash
   python setup.py
   ```

#### Manual Setup

1. **Clone or download this project**
   ```bash
   cd web-task-automator
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**
   ```bash
   playwright install chromium
   ```

## Usage

### Basic Browser Automation

Run the basic browser automation demo:

```bash
python browser_automation.py
```

This will:
- Launch a browser (visible by default for demo)
- Navigate to example.com
- Display the page title
- Close the browser automatically

### Testing the Installation

To verify everything is working correctly, you can run the basic demo:

```bash
# Test basic functionality
python browser_automation.py

# The output should show:
# ✅ Success! Page loaded: Example Domain
```

### Using the BrowserAutomation Class

```python
import asyncio
from browser_automation import BrowserAutomation

async def my_automation():
    async with BrowserAutomation(headless=True) as automation:
        # Navigate to a website
        await automation.navigate_to("https://example.com")
        
        # Get page title
        title = await automation.get_page_title()
        print(f"Page title: {title}")
        
        # Click on elements
        await automation.click_element("a")
        
        # Type text into forms
        await automation.type_text("input[name='search']", "search query")
        
        # Navigate pages
        await automation.page_back()
        await automation.page_forward()
        await automation.page_refresh()
        
        # Wait for elements
        await automation.wait_for_element("h1", state='visible')
        
        # Get element information
        text = await automation.get_element_text("h1")
        attribute = await automation.get_element_attribute("a", "href")

# Run the automation
asyncio.run(my_automation())
```

### Using Fixed Task Automation

```python
import asyncio
from fixed_task_automation import FixedTaskAutomation

async def run_fixed_task():
    # Create fixed task automation
    task_automation = FixedTaskAutomation(headless=True)
    
    # Run complete task
    results = await task_automation.run_complete_task(
        search_term="laptop",
        url="https://httpbin.org/forms/post"
    )
    
    print(f"Task completed: {results['navigation_success']}")
    print(f"Results found: {len(results['search_results'])}")

# Run the fixed task
asyncio.run(run_fixed_task())
```

### Using E-commerce Task Automation

```python
import asyncio
from ecommerce_task_automation import EcommerceTaskAutomation

async def run_ecommerce_task():
    # Create e-commerce task automation
    task_automation = EcommerceTaskAutomation(headless=True)
    
    # Run complete e-commerce task
    results = await task_automation.run_complete_task(
        search_term="python",
        url="https://books.toscrape.com/"
    )
    
    print(f"Products found: {results['products_found']}")
    print(f"Prices found: {len(results['prices_found'])}")

# Run the e-commerce task
asyncio.run(run_ecommerce_task())
```

## Project Structure

```
web-task-automator/
├── browser_automation.py           # Main automation module
├── reliable_browser_automation.py  # Enhanced reliability automation
├── mcp_server_integration.py       # MCP Server integration for AI context
├── fixed_task_automation.py        # Fixed task automation
├── ecommerce_task_automation.py    # E-commerce task automation
├── requirements.txt                # Python dependencies
├── setup.py                        # Automated setup script
├── README.md                       # This file
├── .gitignore                      # Git ignore file
└── venv/                           # Virtual environment (created during setup)
```

## Dependencies

- **playwright**: Browser automation library
- **asyncio**: Asynchronous programming support
- **aiohttp**: HTTP client for MCP Server communication
- **requests**: HTTP library for server setup
- **typing-extensions**: Type hints support

## Error Handling

The automation includes comprehensive error handling for:
- Browser launch failures
- Navigation timeouts
- Missing elements
- Network issues
- Resource cleanup

## Logging

The application includes detailed logging to help with debugging:
- Browser startup/shutdown events
- Navigation attempts
- Error messages
- Success confirmations

## Development

### Running in Development Mode

To see the browser in action (non-headless mode):

```python
async with BrowserAutomation(headless=False) as automation:
    # Your automation code here
```

### Adding New Features

The `BrowserAutomation` class is designed to be extensible. You can add new methods for:
- Clicking elements
- Filling forms
- Taking screenshots
- Waiting for specific conditions

## Troubleshooting

### Common Issues

1. **Playwright not installed**: Run `playwright install chromium`
2. **Permission errors**: Ensure you have proper permissions for browser installation
3. **Network timeouts**: Check your internet connection and firewall settings

### Getting Help

Check the logs for detailed error messages. The application provides comprehensive logging to help identify issues.

## Core Browser Actions

The BrowserAutomation class now includes comprehensive browser interaction capabilities:

### Click Actions
- `click_element(selector, timeout=10000)` - Click buttons, links, and other elements
- Supports CSS selectors, XPath, and text content

### Text Input
- `type_text(selector, text, clear_first=True, timeout=10000)` - Type into form fields
- `clear_first` option to clear existing text before typing

### Page Navigation
- `page_back()` - Navigate to previous page
- `page_forward()` - Navigate to next page  
- `page_refresh()` - Refresh current page

### Element Interaction
- `get_element_text(selector, timeout=10000)` - Get text content
- `get_element_attribute(selector, attribute, timeout=10000)` - Get attribute values
- `get_all_elements(selector, timeout=10000)` - Get multiple elements
- `scroll_to_element(selector, timeout=10000)` - Scroll to make element visible

### Element Waiting
- `wait_for_element(selector, state='visible', timeout=10000)` - Wait for element states
- States: 'visible', 'hidden', 'attached', 'detached'

## Fixed Task Automation

The project now includes complete fixed task automation capabilities:

### Basic Fixed Task
- **Task**: Navigate to a website, search for a product, and report results
- **Features**: Form interaction, search functionality, result extraction
- **Error Handling**: Graceful handling of missing elements and network issues

### E-commerce Task
- **Task**: Navigate to an e-commerce site, search for products, extract prices
- **Features**: Product search, price extraction, result analysis
- **Real-world Testing**: Works with actual e-commerce websites

### Task Results
Both automation types provide comprehensive results:
- Navigation success/failure status
- Search results with product information
- Price extraction and analysis
- Error messages and edge case handling
- Execution time and performance metrics

## MCP Server Integration

The project includes integration with Playwright MCP Server for AI context:

### AI Context Features
- **Webpage Context**: Extract comprehensive webpage information for AI processing
- **Accessibility Data**: Capture accessibility tree and element roles
- **Element Structure**: Detailed element information including roles, states, and properties
- **AI-Friendly Format**: Structured data optimized for AI consumption

### MCP Server Components
- **MCPServerManager**: Handles server lifecycle and connection management
- **MCPBrowserAutomation**: Enhanced browser automation with AI context
- **Data Exchange**: Bidirectional communication with MCP Server
- **Context Capture**: Real-time webpage context extraction

### Usage Example
```python
from mcp_server_integration import MCPServerManager, MCPBrowserAutomation

async def ai_automation():
    mcp_manager = MCPServerManager()
    automation = MCPBrowserAutomation(mcp_manager)
    
    # Start server and session
    await mcp_manager.start_server()
    await automation.start_session()
    
    # Navigate and capture context
    await automation.navigate_to("https://example.com")
    context = await automation.get_ai_context()
    
    # AI can now use context for intelligent decisions
    print(f"Page has {context['summary']['total_elements']} elements")
    
    await automation.close_session()
    await mcp_manager.stop_server()
```

## Reliability and Error Handling

The project now includes comprehensive reliability features:

### Enhanced Error Handling
- **Custom Exceptions**: `AutomationError`, `TimeoutError`, `ElementNotFoundError`, `NetworkError`
- **Retry Mechanisms**: Exponential backoff retry for failed operations
- **Timeout Management**: Advanced timeout handling for slow pages and operations
- **Graceful Degradation**: Continue operation when possible, fail gracefully when not

### Advanced Logging
- **Debug Logging**: Detailed logging for debugging and monitoring
- **Performance Tracking**: Operation timing and performance metrics
- **Error Tracking**: Comprehensive error logging and reporting
- **Session Monitoring**: Track browser sessions and resource usage

### Reliability Features
- **Network Error Detection**: Detect and handle network issues
- **Element State Management**: Wait for elements in various states
- **Resource Cleanup**: Proper cleanup of browser resources
- **Error Recovery**: Automatic recovery from common errors

## Next Steps

This is the foundation for more advanced features:
- AI-driven task planning
- Web API endpoints
- Advanced browser interactions
- Task scheduling and monitoring
