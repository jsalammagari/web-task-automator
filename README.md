# Web Task Automator

A Python program that can automatically perform tasks on websites using Playwright for browser automation.

## Features

- **Basic Browser Automation**: Control web browsers using Playwright
- **Core Browser Actions**: Click, type, navigate, and interact with web elements
- **Fixed Task Automation**: Complete automated workflows for specific tasks
- **E-commerce Automation**: Real-world shopping and product search automation
- **Reliability & Error Handling**: Comprehensive error handling and retry mechanisms
- **MCP Server Integration**: AI context integration with Playwright MCP Server
- **AI Language Model Integration**: LLM-powered dynamic task planning
- **Dynamic Task Execution Engine**: AI-driven adaptive task execution
- **Natural Language Goal Processing**: Process user goals in plain English
- **Web API Foundation**: RESTful API for remote automation control
- **Remote Task Execution**: Advanced task queuing and remote execution
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
├── ai_task_planner.py              # AI Language Model integration for task planning
├── dynamic_execution_engine.py      # Dynamic task execution engine
├── natural_language_processor.py    # Natural language goal processing
├── web_api.py                       # FastAPI web API server
├── remote_execution_api.py          # Remote task execution API
├── start_api.py                     # API server startup script
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
- **openai**: OpenAI API client for GPT models
- **anthropic**: Anthropic API client for Claude models
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

## AI Language Model Integration

The project includes AI-powered task planning using Language Models:

### LLM Features
- **Multi-Provider Support**: OpenAI GPT, Anthropic Claude, and local models
- **Dynamic Task Planning**: Convert natural language goals to automation steps
- **Context-Aware Planning**: Use webpage context for intelligent task generation
- **Error Recovery**: AI-powered recovery suggestions for failed tasks
- **Structured Output**: JSON command generation with validation

### Supported LLM Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet, Claude-3-haiku
- **Local Models**: Ollama, local LLM servers

### Usage Example
```python
from ai_task_planner import LLMClient, LLMProvider, AITaskPlanner

# Initialize LLM client
llm_client = LLMClient(LLMProvider.OPENAI, api_key="your-api-key")

# Create task planner
task_planner = AITaskPlanner(llm_client)

# Create task plan from natural language
goal = "Fill out the contact form and submit it"
context = {
    "page": {"title": "Contact Form", "url": "https://example.com/contact"},
    "elements": [
        {"tag": "input", "selector": "input[name='email']", "role": "textbox"},
        {"tag": "button", "selector": "button[type='submit']", "role": "button"}
    ]
}

task_plan = await task_planner.create_task_plan(goal, context)
print(f"Created plan with {len(task_plan.steps)} steps")

# Execute the plan
executor = AITaskExecutor(task_planner, automation_client)
results = await executor.execute_task_plan(task_plan)
```

### Task Planning Features
- **Natural Language Input**: Describe goals in plain English
- **Context Integration**: Use webpage structure for better planning
- **Step Validation**: Validate generated automation steps
- **Error Recovery**: AI suggestions for failed operations
- **Confidence Scoring**: Estimate task success probability

## Dynamic Task Execution Engine

The project includes an AI-driven dynamic task execution system:

### Execution Engine Features
- **Plan Parser**: Parse AI-generated JSON commands with validation
- **Command Executor**: Translate AI commands to browser actions
- **Command Validation**: Validate AI-generated commands before execution
- **Feedback Loop**: Dynamic plan adjustment based on execution results
- **Monitoring & Logging**: Comprehensive AI decision-making monitoring
- **Adaptive Execution**: Real-time plan adaptation for better success rates

### Core Components
- **CommandValidator**: Validates AI commands for correctness
- **PlanParser**: Parses and structures AI-generated plans
- **CommandExecutor**: Executes commands on browser automation
- **FeedbackLoop**: Analyzes results and suggests improvements
- **ExecutionMonitor**: Tracks AI decisions and performance
- **DynamicExecutionEngine**: Main orchestration engine

### Usage Example
```python
from dynamic_execution_engine import DynamicExecutionEngine

# Initialize execution engine
engine = DynamicExecutionEngine(automation_client, ai_task_planner)

# Execute AI plan with dynamic adaptation
plan_data = {
    "goal": "Fill out contact form",
    "steps": [
        {"action": "navigate", "url": "https://example.com/contact"},
        {"action": "type", "selector": "input[name='name']", "text": "John Doe"},
        {"action": "click", "selector": "button[type='submit']"}
    ]
}

# Execute with adaptation
result = await engine.execute_with_adaptation(plan_data, context)
print(f"Execution success: {result.success}")
print(f"Success rate: {result.metrics.success_rate:.2%}")

# Get execution summary
summary = engine.get_execution_summary()
print(f"Total executions: {summary['total_executions']}")
```

### Execution Features
- **Command Validation**: Pre-execution validation of AI commands
- **Retry Logic**: Automatic retry for failed commands
- **Performance Tracking**: Execution time and success rate monitoring
- **Feedback Analysis**: AI-powered execution result analysis
- **Plan Adaptation**: Dynamic plan improvement based on feedback
- **Comprehensive Logging**: Detailed execution and decision logs

## Natural Language Goal Processing

The project includes comprehensive natural language processing for user goals:

### NLP Features
- **Intent Recognition**: Automatically identify user intent (shopping, form filling, data extraction, etc.)
- **Entity Extraction**: Extract products, colors, prices, websites, and other entities from goals
- **Complexity Detection**: Classify goals as simple, moderate, complex, or ambiguous
- **Context Extraction**: Extract requirements, constraints, and preferences from goals
- **Goal Translation**: Convert natural language goals to structured automation plans
- **Ambiguous Goal Handling**: Provide suggestions for unclear or ambiguous goals

### Supported Intents
- **Shopping**: Buy products, compare prices, add to cart
- **Form Filling**: Complete forms, enter information
- **Data Extraction**: Extract text, prices, information from pages
- **Navigation**: Go to websites, click links, browse pages
- **Search**: Search for products, information, content
- **Click Actions**: Click buttons, links, menu items

### Usage Example
```python
from natural_language_processor import NaturalLanguageGoalProcessor

# Initialize processor
processor = NaturalLanguageGoalProcessor(ai_task_planner)

# Process user goal
goal = "Buy the cheapest blue shirt on Amazon under $30"
webpage_context = {
    "page": {"title": "Amazon", "url": "https://amazon.com"},
    "elements": [{"tag": "input", "selector": "input[name='search']"}]
}

# Get automation plan
plan = await processor.process_user_goal(goal, webpage_context)
print(f"Generated plan with {len(plan['steps'])} steps")
print(f"Intent: {plan['intent']}")
print(f"Confidence: {plan['processing_info']['confidence']}")
```

### Goal Processing Features
- **Natural Language Input**: Describe goals in plain English
- **Intent Classification**: Automatic intent recognition with confidence scoring
- **Entity Recognition**: Extract products, colors, sizes, prices, websites
- **Complexity Analysis**: Determine goal complexity and processing requirements
- **Context Understanding**: Extract requirements, constraints, and preferences
- **Plan Generation**: Convert goals to structured automation plans
- **Suggestion System**: Provide feedback for ambiguous or unclear goals

## Web API Foundation

The project includes a comprehensive FastAPI-based web API for remote automation control:

### API Features
- **RESTful Endpoints**: Complete REST API for automation control
- **Task Management**: Create, monitor, and cancel automation tasks
- **AI-Powered Automation**: Natural language goal processing via API
- **Basic Automation**: Direct browser action execution
- **Real-time Status**: Task progress monitoring and status updates
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Security**: API key authentication and CORS support

### API Endpoints
- **GET /health** - Health check and service status
- **POST /tasks** - Create new automation tasks
- **GET /tasks/{task_id}** - Get task status and results
- **GET /tasks** - List all tasks
- **DELETE /tasks/{task_id}** - Cancel running tasks
- **POST /automation/basic** - Execute basic automation
- **POST /automation/ai-powered** - AI-powered automation
- **GET /automation/capabilities** - Get available features
- **POST /automation/validate** - Validate automation requests

### Usage Example
```bash
# Start the API server
python start_api.py

# Create a task
curl -X POST "http://localhost:8000/tasks" \
  -H "Authorization: Bearer demo-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Buy the cheapest blue shirt on Amazon",
    "url": "https://amazon.com",
    "headless": true
  }'

# Check task status
curl -X GET "http://localhost:8000/tasks/task_123" \
  -H "Authorization: Bearer demo-api-key"

# Execute basic automation
curl -X POST "http://localhost:8000/automation/basic" \
  -H "Authorization: Bearer demo-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "actions": [
      {"type": "click", "selector": "button"},
      {"type": "type", "selector": "input", "text": "test"}
    ]
  }'
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Security Features
- **API Key Authentication**: Bearer token authentication
- **CORS Support**: Configurable cross-origin resource sharing
- **Request Validation**: Pydantic model validation
- **Error Handling**: Comprehensive error responses
- **Rate Limiting**: Built-in request throttling (configurable)

## Remote Task Execution

The project includes an advanced remote task execution system with queuing and status tracking:

### Remote Execution Features
- **Task Queuing**: Priority-based task queue with worker threads
- **Status Tracking**: Real-time task progress and status monitoring
- **Result Delivery**: Automatic result delivery with callback support
- **Batch Processing**: Submit multiple tasks simultaneously
- **Priority Handling**: Task priority levels (low, normal, high, urgent)
- **Timeout Management**: Configurable task timeouts and cancellation
- **Error Handling**: Comprehensive error handling and recovery

### Enhanced API Endpoints
- **POST /tasks/submit** - Submit new task for remote execution
- **GET /tasks/{task_id}/status** - Get detailed task status
- **GET /tasks/{task_id}/result** - Get task execution results
- **DELETE /tasks/{task_id}** - Cancel running task
- **GET /queue/status** - Get queue status and statistics
- **POST /tasks/batch** - Submit multiple tasks at once
- **POST /tasks/validate** - Validate task request before submission

### Usage Example
```bash
# Start the remote execution API server
python remote_execution_api.py

# Submit a high-priority task
curl -X POST "http://localhost:8001/tasks/submit" \
  -H "Authorization: Bearer demo-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Buy the cheapest blue shirt on Amazon",
    "target_url": "https://amazon.com",
    "priority": "high",
    "timeout_seconds": 300,
    "callback_url": "https://your-app.com/webhook"
  }'

# Check task status
curl -X GET "http://localhost:8001/tasks/task_abc123/status" \
  -H "Authorization: Bearer demo-api-key"

# Get task results
curl -X GET "http://localhost:8001/tasks/task_abc123/result" \
  -H "Authorization: Bearer demo-api-key"

# Submit batch tasks
curl -X POST "http://localhost:8001/tasks/batch" \
  -H "Authorization: Bearer demo-api-key" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "task_description": "First task",
      "priority": "normal"
    },
    {
      "task_description": "Second task", 
      "priority": "high"
    }
  ]'
```

### Task Queue Management
- **Priority Queue**: Tasks executed based on priority level
- **Worker Threads**: Configurable number of concurrent workers
- **Queue Status**: Real-time queue statistics and monitoring
- **Task Cancellation**: Cancel running or queued tasks
- **Result Storage**: Persistent storage of task results
- **Callback Notifications**: Automatic result delivery via webhooks

### Advanced Features
- **Task Validation**: Pre-submission validation of task requests
- **Error Recovery**: Automatic retry and error handling
- **Performance Monitoring**: Task execution metrics and statistics
- **Resource Management**: Efficient resource usage and cleanup
- **Scalability**: Horizontal scaling with multiple worker threads

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

## Deployment Options

### Docker Deployment (Recommended)

#### Quick Start with Docker
```bash
# 1. Clone the repository
git clone <repository-url>
cd web-task-automator

# 2. Set up environment variables
cp env.template .env
# Edit .env with your GROQ_API_KEY

# 3. Run with Docker Compose
docker-compose up -d

# 4. Check status
docker-compose ps
```

#### Production Deployment
```bash
# Run with production profile (includes Nginx)
docker-compose --profile production up -d
```

### Manual Deployment

#### System Requirements
- Python 3.8+
- 2GB RAM minimum
- 1GB disk space
- Internet connection for API calls

#### Production Setup
```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2. Create application user
sudo useradd -m -s /bin/bash automator
sudo usermod -aG sudo automator

# 3. Set up application directory
sudo mkdir -p /opt/web-task-automator
sudo chown automator:automator /opt/web-task-automator

# 4. Deploy application
cd /opt/web-task-automator
git clone <repository-url> .
sudo chown -R automator:automator .

# 5. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 6. Configure environment
cp env.template .env
# Edit .env with production values

# 7. Set up systemd service
sudo cp web-task-automator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable web-task-automator
sudo systemctl start web-task-automator
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Browser Installation Issues

**Problem:** `playwright install` fails
```bash
# Solution: Install system dependencies first
sudo apt-get update
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2
playwright install chromium
```

**Problem:** Browser crashes or won't start
```bash
# Solution: Run in headless mode with additional flags
export PLAYWRIGHT_BROWSERS_PATH=/usr/bin
playwright install chromium --with-deps
```

#### 2. API Key Issues

**Problem:** `GROQ_API_KEY not set` error
```bash
# Solution: Check environment variables
echo $GROQ_API_KEY
# If empty, set it:
export GROQ_API_KEY="your-actual-api-key"
# Or add to .env file:
echo "GROQ_API_KEY=your-actual-api-key" >> .env
```

**Problem:** API rate limiting
```bash
# Solution: Add delays between requests
export GROQ_RATE_LIMIT_DELAY=1.0
```

#### 3. Memory and Performance Issues

**Problem:** Out of memory errors
```bash
# Solution: Increase memory limits
export PLAYWRIGHT_BROWSER_MEMORY_LIMIT=512
# Or run with fewer concurrent tasks
export MAX_CONCURRENT_TASKS=2
```

**Problem:** Slow performance
```bash
# Solution: Optimize browser settings
export PLAYWRIGHT_HEADLESS=true
export PLAYWRIGHT_SLOW_MO=0
```

#### 4. Network and Connectivity Issues

**Problem:** Timeout errors
```bash
# Solution: Increase timeout values
export DEFAULT_TIMEOUT=30000
export NETWORK_TIMEOUT=60000
```

**Problem:** SSL certificate errors
```bash
# Solution: Disable SSL verification (not recommended for production)
export PLAYWRIGHT_IGNORE_HTTPS_ERRORS=true
```

#### 5. Docker-Specific Issues

**Problem:** Docker container won't start
```bash
# Solution: Check logs
docker-compose logs web-task-automator

# Check if ports are available
netstat -tulpn | grep :8000
```

**Problem:** Permission denied in Docker
```bash
# Solution: Fix file permissions
sudo chown -R 1000:1000 .
docker-compose down
docker-compose up -d
```

#### 6. API Endpoint Issues

**Problem:** 404 errors on API endpoints
```bash
# Solution: Check if API is running
curl http://localhost:8000/health

# Check API documentation
curl http://localhost:8000/docs
```

**Problem:** CORS errors
```bash
# Solution: Configure CORS in web_api.py
# Add your domain to allowed origins
```

### Debugging Commands

#### Check System Status
```bash
# Check if all services are running
docker-compose ps

# Check logs
docker-compose logs -f web-task-automator

# Check API health
curl -X GET http://localhost:8000/health
```

#### Test Individual Components
```bash
# Test browser automation
python -c "from browser_automation import BrowserAutomation; print('Browser automation OK')"

# Test AI integration
python -c "from ai_task_planner import LLMClient, LLMProvider; print('AI integration OK')"

# Test API endpoints
python -c "from web_api import app; print('API OK')"
```

#### Performance Monitoring
```bash
# Monitor resource usage
docker stats

# Check disk usage
df -h

# Monitor logs
tail -f logs/*.log
```

### Getting Help

1. **Check the logs** in the `logs/` directory
2. **Verify environment variables** are set correctly
3. **Test individual components** using the debugging commands above
4. **Check system resources** (memory, disk space, network)
5. **Review the API documentation** at `http://localhost:8000/docs`

### Support

For additional help:
- Check the [Issues](https://github.com/your-repo/issues) page
- Review the [API Documentation](http://localhost:8000/docs)
- Contact the development team

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
