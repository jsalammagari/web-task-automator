# Web Task Automator

A Python program that can automatically perform tasks on websites using Playwright for browser automation.

## Features

- **Basic Browser Automation**: Control web browsers using Playwright
- **Error Handling**: Robust error handling for reliable automation
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

# Run the automation
asyncio.run(my_automation())
```

## Project Structure

```
web-task-automator/
├── browser_automation.py    # Main automation module
├── requirements.txt         # Python dependencies
├── setup.py                # Automated setup script
├── README.md                # This file
└── venv/                    # Virtual environment (created during setup)
```

## Dependencies

- **playwright**: Browser automation library
- **asyncio**: Asynchronous programming support
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

## Next Steps

This is the foundation for more advanced features:
- AI-driven task planning
- Web API endpoints
- Advanced browser interactions
- Task scheduling and monitoring
