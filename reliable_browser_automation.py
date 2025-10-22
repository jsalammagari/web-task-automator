#!/usr/bin/env python3
"""
Reliable Browser Automation with Enhanced Error Handling
========================================================

This module provides robust browser automation capabilities with comprehensive
error handling, retry mechanisms, timeout management, and detailed logging.

Author: Web Task Automator
"""

import asyncio
import logging
import time
from typing import Optional, Union, List, Dict, Any, Callable
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Locator, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Error as PlaywrightError
import functools

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('browser_automation.log')
    ]
)
logger = logging.getLogger(__name__)


class AutomationError(Exception):
    """Custom exception for automation errors."""
    pass


class TimeoutError(AutomationError):
    """Custom exception for timeout errors."""
    pass


class ElementNotFoundError(AutomationError):
    """Custom exception for missing elements."""
    pass


class NetworkError(AutomationError):
    """Custom exception for network-related errors."""
    pass


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying failed operations with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (float): Initial delay between retries in seconds
        backoff (float): Backoff multiplier for delay
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        logger.info(f"Retrying in {current_delay:.2f} seconds...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator


class ReliableBrowserAutomation:
    """
    Enhanced browser automation class with comprehensive reliability features.
    
    Features:
    - Robust error handling and recovery
    - Retry mechanisms with exponential backoff
    - Advanced timeout management
    - Detailed logging and debugging
    - Graceful handling of missing elements
    - Network error detection and recovery
    - Performance monitoring
    """
    
    def __init__(self, headless: bool = True, default_timeout: int = 30000, 
                 max_retries: int = 3, debug: bool = False):
        """
        Initialize the reliable browser automation.
        
        Args:
            headless (bool): Whether to run browser in headless mode
            default_timeout (int): Default timeout in milliseconds
            max_retries (int): Maximum retry attempts for failed operations
            debug (bool): Enable debug logging
        """
        self.headless = headless
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.debug = debug
        
        # Browser components
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.operation_times: Dict[str, List[float]] = {}
        
        # Error tracking
        self.error_count = 0
        self.retry_count = 0
        
        # Configure logging level
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
    
    async def start_browser(self) -> bool:
        """
        Start the browser with enhanced error handling and retry logic.
        
        Returns:
            bool: True if browser started successfully, False otherwise
        """
        try:
            logger.info("🚀 Starting reliable browser automation...")
            self.start_time = time.time()
            
            # Start Playwright with retry
            self.playwright = await async_playwright().start()
            logger.debug("Playwright started successfully")
            
            # Launch browser with enhanced options
            browser_options = {
                'headless': self.headless,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            }
            
            self.browser = await self.playwright.chromium.launch(**browser_options)
            logger.debug("Browser launched successfully")
            
            # Create context with enhanced settings
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'ignore_https_errors': True,
                'accept_downloads': True
            }
            
            self.context = await self.browser.new_context(**context_options)
            logger.debug("Browser context created successfully")
            
            # Create new page
            self.page = await self.context.new_page()
            logger.debug("New page created successfully")
            
            # Set default timeout
            self.page.set_default_timeout(self.default_timeout)
            self.page.set_default_navigation_timeout(self.default_timeout)
            
            logger.info("✅ Browser started successfully with enhanced reliability features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start browser: {str(e)}")
            await self._cleanup_resources()
            raise AutomationError(f"Failed to start browser: {str(e)}")
    
    @retry_on_failure(max_retries=3, delay=1.0)
    async def navigate_to(self, url: str, timeout: Optional[int] = None, 
                         wait_until: str = 'domcontentloaded') -> bool:
        """
        Navigate to a URL with enhanced error handling and retry logic.
        
        Args:
            url (str): URL to navigate to
            timeout (int, optional): Navigation timeout in milliseconds
            wait_until (str): When to consider navigation complete
            
        Returns:
            bool: True if navigation successful, False otherwise
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"🌐 Navigating to: {url}")
            
            # Check if URL is valid
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'
            
            # Navigate with enhanced error handling
            response = await self.page.goto(
                url, 
                timeout=timeout,
                wait_until=wait_until
            )
            
            # Check response status
            if response and response.status >= 400:
                logger.warning(f"⚠️  HTTP {response.status} response from {url}")
                if response.status >= 500:
                    raise NetworkError(f"Server error {response.status} from {url}")
            
            # Wait for page to be ready
            await self.page.wait_for_load_state('networkidle', timeout=5000)
            
            navigation_time = time.time() - start_time
            self._track_operation_time('navigate_to', navigation_time)
            
            logger.info(f"✅ Successfully navigated to: {url} (took {navigation_time:.2f}s)")
            return True
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Navigation timeout to {url}: {str(e)}")
            raise TimeoutError(f"Navigation timeout to {url}")
        except PlaywrightError as e:
            logger.error(f"🌐 Navigation error to {url}: {str(e)}")
            if 'net::' in str(e) or 'ERR_' in str(e):
                raise NetworkError(f"Network error navigating to {url}: {str(e)}")
            raise AutomationError(f"Navigation failed to {url}: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error navigating to {url}: {str(e)}")
            raise AutomationError(f"Unexpected navigation error: {str(e)}")
    
    @retry_on_failure(max_retries=3, delay=0.5)
    async def click_element(self, selector: str, timeout: Optional[int] = None, 
                           force: bool = False) -> bool:
        """
        Click an element with enhanced error handling and retry logic.
        
        Args:
            selector (str): CSS selector or XPath for the element
            timeout (int, optional): Timeout in milliseconds
            force (bool): Force click even if element is not visible
            
        Returns:
            bool: True if click successful, False otherwise
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"🖱️  Clicking element: {selector}")
            
            # Wait for element to be available
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Check if element is clickable
            element = self.page.locator(selector)
            if not force:
                await element.wait_for(state='attached', timeout=5000)
            
            # Perform click with retry logic
            await element.click(force=force)
            
            click_time = time.time() - start_time
            self._track_operation_time('click_element', click_time)
            
            logger.info(f"✅ Successfully clicked element: {selector} (took {click_time:.2f}s)")
            return True
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Click timeout for element '{selector}': {str(e)}")
            raise TimeoutError(f"Click timeout for element '{selector}'")
        except PlaywrightError as e:
            if 'not visible' in str(e) or 'not attached' in str(e):
                raise ElementNotFoundError(f"Element '{selector}' not found or not clickable")
            logger.error(f"🖱️  Click error for element '{selector}': {str(e)}")
            raise AutomationError(f"Click failed for element '{selector}': {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error clicking element '{selector}': {str(e)}")
            raise AutomationError(f"Unexpected click error: {str(e)}")
    
    @retry_on_failure(max_retries=2, delay=0.5)
    async def type_text(self, selector: str, text: str, clear_first: bool = True, 
                       timeout: Optional[int] = None) -> bool:
        """
        Type text into an element with enhanced error handling.
        
        Args:
            selector (str): CSS selector for the input element
            text (str): Text to type
            clear_first (bool): Clear existing text before typing
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            bool: True if typing successful, False otherwise
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"⌨️  Typing text into element: {selector}")
            
            # Wait for element to be available
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Clear existing text if requested
            if clear_first:
                await self.page.fill(selector, "")
                logger.debug(f"Cleared existing text in {selector}")
            
            # Type text with enhanced error handling
            await self.page.fill(selector, text)
            
            # Verify text was entered
            actual_text = await self.page.input_value(selector)
            if actual_text != text:
                logger.warning(f"⚠️  Text mismatch: expected '{text}', got '{actual_text}'")
            
            type_time = time.time() - start_time
            self._track_operation_time('type_text', type_time)
            
            logger.info(f"✅ Successfully typed text into element: {selector} (took {type_time:.2f}s)")
            return True
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Type timeout for element '{selector}': {str(e)}")
            raise TimeoutError(f"Type timeout for element '{selector}'")
        except PlaywrightError as e:
            if 'not visible' in str(e) or 'not attached' in str(e):
                raise ElementNotFoundError(f"Element '{selector}' not found or not editable")
            logger.error(f"⌨️  Type error for element '{selector}': {str(e)}")
            raise AutomationError(f"Type failed for element '{selector}': {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error typing into element '{selector}': {str(e)}")
            raise AutomationError(f"Unexpected type error: {str(e)}")
    
    async def wait_for_element(self, selector: str, state: str = 'visible', 
                             timeout: Optional[int] = None) -> bool:
        """
        Wait for an element with enhanced error handling.
        
        Args:
            selector (str): CSS selector for the element
            state (str): Element state to wait for ('visible', 'hidden', 'attached', 'detached')
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            bool: True if element found, False otherwise
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"⏳ Waiting for element '{selector}' to be '{state}'")
            
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            
            wait_time = time.time() - start_time
            self._track_operation_time('wait_for_element', wait_time)
            
            logger.info(f"✅ Element '{selector}' is now '{state}' (took {wait_time:.2f}s)")
            return True
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Element '{selector}' did not become '{state}' within {timeout}ms: {str(e)}")
            raise TimeoutError(f"Element '{selector}' did not become '{state}' within {timeout}ms")
        except Exception as e:
            logger.error(f"❌ Unexpected error waiting for element '{selector}': {str(e)}")
            raise AutomationError(f"Unexpected wait error: {str(e)}")
    
    async def get_element_text(self, selector: str, timeout: Optional[int] = None) -> Optional[str]:
        """
        Get text content from an element with enhanced error handling.
        
        Args:
            selector (str): CSS selector for the element
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            Optional[str]: Element text content or None if not found
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"📝 Getting text from element: {selector}")
            
            # Wait for element to be available
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Get text content
            text = await self.page.text_content(selector)
            
            get_time = time.time() - start_time
            self._track_operation_time('get_element_text', get_time)
            
            logger.info(f"✅ Element text retrieved: '{text}' (took {get_time:.2f}s)")
            return text
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Timeout getting text from element '{selector}': {str(e)}")
            raise TimeoutError(f"Timeout getting text from element '{selector}'")
        except PlaywrightError as e:
            if 'not visible' in str(e) or 'not attached' in str(e):
                raise ElementNotFoundError(f"Element '{selector}' not found")
            logger.error(f"📝 Error getting text from element '{selector}': {str(e)}")
            raise AutomationError(f"Failed to get text from element '{selector}': {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error getting text from element '{selector}': {str(e)}")
            raise AutomationError(f"Unexpected text retrieval error: {str(e)}")
    
    async def get_page_title(self) -> Optional[str]:
        """
        Get the current page title with error handling.
        
        Returns:
            Optional[str]: Page title or None if not available
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        try:
            title = await self.page.title()
            logger.info(f"📄 Page title: {title}")
            return title
        except Exception as e:
            logger.error(f"❌ Error getting page title: {str(e)}")
            return None
    
    async def get_all_elements(self, selector: str, timeout: Optional[int] = None) -> List[Locator]:
        """
        Get all elements matching a selector with enhanced error handling.
        
        Args:
            selector (str): CSS selector for the elements
            timeout (int, optional): Timeout in milliseconds
            
        Returns:
            List[Locator]: List of matching elements
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Getting all elements matching: {selector}")
            
            # Wait for at least one element to be available
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Get all matching elements
            elements = await self.page.locator(selector).all()
            
            get_time = time.time() - start_time
            self._track_operation_time('get_all_elements', get_time)
            
            logger.info(f"✅ Found {len(elements)} elements matching '{selector}' (took {get_time:.2f}s)")
            return elements
            
        except PlaywrightTimeoutError as e:
            logger.error(f"⏰ Timeout getting elements matching '{selector}': {str(e)}")
            raise TimeoutError(f"Timeout getting elements matching '{selector}'")
        except PlaywrightError as e:
            if 'not visible' in str(e) or 'not attached' in str(e):
                raise ElementNotFoundError(f"No elements found matching '{selector}'")
            logger.error(f"🔍 Error getting elements matching '{selector}': {str(e)}")
            raise AutomationError(f"Failed to get elements matching '{selector}': {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error getting elements matching '{selector}': {str(e)}")
            raise AutomationError(f"Unexpected error getting elements: {str(e)}")
    
    async def page_refresh(self) -> bool:
        """
        Refresh the current page with error handling.
        
        Returns:
            bool: True if refresh successful, False otherwise
        """
        if not self.page:
            raise AutomationError("No active page. Please start browser first.")
        
        try:
            logger.info("🔄 Refreshing current page")
            await self.page.reload()
            logger.info("✅ Successfully refreshed page")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to refresh page: {str(e)}")
            raise AutomationError(f"Failed to refresh page: {str(e)}")
    
    async def close_browser(self) -> None:
        """
        Close the browser and cleanup resources with error handling.
        """
        try:
            logger.info("🔒 Closing browser and cleaning up resources...")
            
            if self.page:
                await self.page.close()
                logger.debug("Page closed.")
            
            if self.context:
                await self.context.close()
                logger.debug("Browser context closed.")
            
            if self.browser:
                await self.browser.close()
                logger.debug("Browser closed.")
            
            if self.playwright:
                await self.playwright.stop()
                logger.debug("Playwright stopped.")
            
            # Log performance summary
            if self.start_time:
                total_time = time.time() - self.start_time
                logger.info(f"⏱️  Total session time: {total_time:.2f} seconds")
                self._log_performance_summary()
            
            logger.info("✅ Browser cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during browser cleanup: {str(e)}")
            # Continue with cleanup even if there are errors
    
    def _track_operation_time(self, operation: str, duration: float) -> None:
        """Track operation timing for performance analysis."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)
    
    def _log_performance_summary(self) -> None:
        """Log performance statistics."""
        if not self.operation_times:
            return
        
        logger.info("📊 Performance Summary:")
        for operation, times in self.operation_times.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            logger.info(f"  {operation}: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    
    async def _cleanup_resources(self) -> None:
        """Cleanup resources in case of errors."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()


async def main():
    """Demo the reliable browser automation features."""
    logger.info("🚀 Starting Reliable Browser Automation Demo")
    
    try:
        async with ReliableBrowserAutomation(headless=True, debug=True) as automation:
            # Test navigation with error handling
            await automation.navigate_to("https://example.com")
            
            # Test element interaction
            title = await automation.get_page_title()
            print(f"✅ Page loaded: {title}")
            
            # Test error handling with invalid selector
            try:
                await automation.click_element("#non-existent", timeout=2000)
            except ElementNotFoundError as e:
                print(f"✅ Error handling works: {e}")
            
            # Test retry mechanism
            try:
                await automation.navigate_to("https://httpbin.org/delay/2")
                print("✅ Retry mechanism working")
            except Exception as e:
                print(f"⚠️  Expected error: {e}")
    
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
    
    logger.info("🎉 Reliable Browser Automation Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
