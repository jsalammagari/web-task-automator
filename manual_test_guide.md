# Manual Testing Guide for Web Task Automator

This guide provides step-by-step commands to manually test all the implemented functionality.

## Prerequisites

Make sure you're in the project directory and have the virtual environment activated:

```bash
cd /Users/jahnavi/Desktop/projects/web-task-automator
source venv/bin/activate
```

## Test Commands

### 1. Basic Functionality Test

**Test the main browser automation demo:**
```bash
python browser_automation.py
```

**Expected Output:**
- Should show browser launching (visible mode)
- Navigate to example.com
- Display "✅ Success! Page loaded: Example Domain"
- Browser should close automatically

### 2. Test Headless Mode

**Create a simple headless test:**
```bash
cat > test_headless.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from browser_automation import BrowserAutomation

async def test():
    async with BrowserAutomation(headless=True) as automation:
        success = await automation.navigate_to("https://example.com")
        title = await automation.get_page_title()
        print(f"Headless test: {'✅ SUCCESS' if success and title else '❌ FAILED'}")
        print(f"Page title: {title}")

asyncio.run(test())
EOF

python test_headless.py
```

**Expected Output:**
- Should run without opening visible browser
- Display "Headless test: ✅ SUCCESS"
- Show "Page title: Example Domain"

### 3. Test Error Handling

**Create an error handling test:**
```bash
cat > test_errors.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from browser_automation import BrowserAutomation

async def test_errors():
    # Test 1: Navigation without starting browser
    print("Test 1: Operations without browser...")
    automation = BrowserAutomation(headless=True)
    nav_result = await automation.navigate_to("https://example.com")
    title_result = await automation.get_page_title()
    print(f"Navigation result: {nav_result} (should be False)")
    print(f"Title result: {title_result} (should be None)")
    
    # Test 2: Invalid URL
    print("\nTest 2: Invalid URL...")
    async with BrowserAutomation(headless=True) as automation:
        success = await automation.navigate_to("https://this-domain-does-not-exist-12345.com")
        print(f"Invalid URL result: {success} (should be False)")

asyncio.run(test_errors())
EOF

python test_errors.py
```

**Expected Output:**
- Should show navigation and title operations failing without browser
- Should show invalid URL navigation failing gracefully

### 4. Test Multiple Navigations

**Create a multiple navigation test:**
```bash
cat > test_multiple.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from browser_automation import BrowserAutomation

async def test_multiple():
    async with BrowserAutomation(headless=True) as automation:
        urls = [
            "https://example.com",
            "https://httpbin.org/get",
            "https://jsonplaceholder.typicode.com/posts/1"
        ]
        
        for i, url in enumerate(urls, 1):
            print(f"Navigation {i}: {url}")
            success = await automation.navigate_to(url)
            title = await automation.get_page_title()
            print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
            print(f"  Title: {title or 'No title'}")
            print()

asyncio.run(test_multiple())
EOF

python test_multiple.py
```

**Expected Output:**
- Should successfully navigate to all three URLs
- Show success/failure status for each
- Display page titles (some may be empty for API endpoints)

### 5. Test Performance

**Create a performance test:**
```bash
cat > test_performance.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import time
from browser_automation import BrowserAutomation

async def test_performance():
    start_time = time.time()
    
    async with BrowserAutomation(headless=True) as automation:
        urls = [
            "https://example.com",
            "https://httpbin.org/get",
            "https://jsonplaceholder.typicode.com/posts/1"
        ]
        
        for i, url in enumerate(urls, 1):
            print(f"Loading page {i}/{len(urls)}: {url}")
            success = await automation.navigate_to(url)
            print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n⏱️  Total time: {duration:.2f} seconds")
    print(f"📊 Average per page: {duration/len(urls):.2f} seconds")

asyncio.run(test_performance())
EOF

python test_performance.py
```

**Expected Output:**
- Should show timing information
- Total time should be reasonable (2-5 seconds)
- Average per page should be under 2 seconds

### 6. Test Concurrent Sessions

**Create a concurrent session test:**
```bash
cat > test_concurrent.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from browser_automation import BrowserAutomation

async def single_session(session_id, url):
    async with BrowserAutomation(headless=True) as automation:
        success = await automation.navigate_to(url)
        title = await automation.get_page_title()
        return f"Session {session_id}: {'✅' if success else '❌'} {title or 'No title'}"

async def test_concurrent():
    tasks = [
        single_session(1, "https://example.com"),
        single_session(2, "https://httpbin.org/get"),
        single_session(3, "https://jsonplaceholder.typicode.com/posts/1")
    ]
    
    print("Running 3 concurrent browser sessions...")
    results = await asyncio.gather(*tasks)
    
    for result in results:
        print(f"  {result}")

asyncio.run(test_concurrent())
EOF

python test_concurrent.py
```

**Expected Output:**
- Should run 3 browser sessions simultaneously
- Show results for all sessions
- Demonstrate concurrent browser management

### 7. Test Different Websites

**Create a comprehensive website test:**
```bash
cat > test_websites.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from browser_automation import BrowserAutomation

async def test_websites():
    websites = [
        ("Example Domain", "https://example.com"),
        ("HTTPBin API", "https://httpbin.org/get"),
        ("JSONPlaceholder", "https://jsonplaceholder.typicode.com/posts/1"),
        ("GitHub", "https://github.com"),
        ("Stack Overflow", "https://stackoverflow.com")
    ]
    
    async with BrowserAutomation(headless=True) as automation:
        for name, url in websites:
            print(f"Testing {name}...")
            try:
                success = await automation.navigate_to(url)
                title = await automation.get_page_title()
                print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                print(f"  Title: {title or 'No title'}")
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
            print()

asyncio.run(test_websites())
EOF

python test_websites.py
```

**Expected Output:**
- Should test various types of websites
- Show success/failure for each
- Handle different page structures and loading times

## Cleanup Commands

After testing, clean up the test files:

```bash
rm test_headless.py test_errors.py test_multiple.py test_performance.py test_concurrent.py test_websites.py
```

## Expected Behavior Summary

All tests should demonstrate:

1. **✅ Browser Launch**: Both headless and visible modes work
2. **✅ Navigation**: Successfully navigates to various websites
3. **✅ Error Handling**: Gracefully handles errors and invalid URLs
4. **✅ Performance**: Fast navigation (sub-second for most pages)
5. **✅ Resource Management**: Proper cleanup and no memory leaks
6. **✅ Concurrent Operations**: Multiple sessions work simultaneously
7. **✅ Logging**: Detailed logging for debugging

## Troubleshooting

If any test fails:

1. **Check virtual environment**: Make sure `source venv/bin/activate` is run
2. **Check dependencies**: Run `pip list | grep playwright`
3. **Check browser installation**: Run `playwright install chromium`
4. **Check network**: Ensure internet connection is working
5. **Check logs**: Look for error messages in the output

## Success Indicators

- All commands should run without errors
- Browser should launch and close properly
- Navigation should work to valid URLs
- Error handling should work for invalid URLs
- Performance should be reasonable (under 5 seconds total)
- No memory leaks or hanging processes
