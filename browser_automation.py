#!/usr/bin/env python3
"""
Basic Browser Automation Foundation
====================================

This module provides the foundational browser automation capabilities using Playwright.
It includes basic browser launching, navigation, and error handling.

Author: Web Task Automator
"""

import asyncio
import logging
from typing import Optional, Union, List
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Locator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrowserAutomation:
    """
    Basic browser automation class using Playwright.
    
    This class provides comprehensive browser automation capabilities including:
    - Browser launching and management
    - Basic navigation and page interactions
    - Click actions on buttons, links, and elements
    - Text input for form fields
    - Page navigation (back, forward, refresh)
    - Element waiting and synchronization
    - Error handling and recovery
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the browser automation.
        
        Args:
            headless (bool): Whether to run browser in headless mode. Defaults to True.
        """
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
    
    async def start_browser(self) -> bool:
        """
        Start the browser and create a new context.
        
        Returns:
            bool: True if browser started successfully, False otherwise.
        """
        try:
            logger.info("Starting Playwright...")
            self.playwright = await async_playwright().start()
            
            logger.info("Launching browser...")
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            logger.info("Creating browser context...")
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            logger.info("Creating new page...")
            self.page = await self.context.new_page()
            
            logger.info("Browser started successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start browser: {str(e)}")
            return False
    
    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a specific URL.
        
        Args:
            url (str): The URL to navigate to.
            
        Returns:
            bool: True if navigation successful, False otherwise.
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info(f"Navigating to: {url}")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            logger.info(f"Successfully navigated to: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {str(e)}")
            return False
    
    async def get_page_title(self) -> Optional[str]:
        """
        Get the current page title.
        
        Returns:
            Optional[str]: Page title if available, None otherwise.
        """
        if not self.page:
            logger.error("No active page.")
            return None
        
        try:
            title = await self.page.title()
            logger.info(f"Page title: {title}")
            return title
        except Exception as e:
            logger.error(f"Failed to get page title: {str(e)}")
            return None
    
    async def click_element(self, selector: str, timeout: int = 10000) -> bool:
        """
        Click on an element using a CSS selector, XPath, or text content.
        
        Args:
            selector (str): CSS selector, XPath, or text content to click
            timeout (int): Maximum time to wait for element in milliseconds
            
        Returns:
            bool: True if click successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info(f"Clicking element: {selector}")
            
            # Wait for element to be visible and clickable
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Click the element
            await self.page.click(selector)
            logger.info(f"Successfully clicked element: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to click element '{selector}': {str(e)}")
            return False
    
    async def type_text(self, selector: str, text: str, clear_first: bool = True, timeout: int = 10000) -> bool:
        """
        Type text into an input field or textarea.
        
        Args:
            selector (str): CSS selector, XPath, or text content of the input field
            text (str): Text to type
            clear_first (bool): Whether to clear the field before typing
            timeout (int): Maximum time to wait for element in milliseconds
            
        Returns:
            bool: True if typing successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info(f"Typing text into element: {selector}")
            
            # Wait for element to be visible and enabled
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Clear the field if requested
            if clear_first:
                await self.page.fill(selector, "")
            
            # Type the text
            await self.page.fill(selector, text)
            logger.info(f"Successfully typed text into element: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to type text into element '{selector}': {str(e)}")
            return False
    
    async def wait_for_element(self, selector: str, state: str = 'visible', timeout: int = 10000) -> bool:
        """
        Wait for an element to be in a specific state.
        
        Args:
            selector (str): CSS selector, XPath, or text content
            state (str): Element state to wait for ('visible', 'hidden', 'attached', 'detached')
            timeout (int): Maximum time to wait in milliseconds
            
        Returns:
            bool: True if element found in specified state, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info(f"Waiting for element '{selector}' to be '{state}'")
            await self.page.wait_for_selector(selector, state=state, timeout=timeout)
            logger.info(f"Element '{selector}' is now '{state}'")
            return True
            
        except Exception as e:
            logger.error(f"Element '{selector}' did not become '{state}' within {timeout}ms: {str(e)}")
            return False
    
    async def get_element_text(self, selector: str, timeout: int = 10000) -> Optional[str]:
        """
        Get the text content of an element.
        
        Args:
            selector (str): CSS selector, XPath, or text content
            timeout (int): Maximum time to wait for element in milliseconds
            
        Returns:
            Optional[str]: Element text if found, None otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return None
        
        try:
            logger.info(f"Getting text from element: {selector}")
            
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Get the text content
            text = await self.page.text_content(selector)
            logger.info(f"Element text: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to get text from element '{selector}': {str(e)}")
            return None
    
    async def get_element_attribute(self, selector: str, attribute: str, timeout: int = 10000) -> Optional[str]:
        """
        Get an attribute value of an element.
        
        Args:
            selector (str): CSS selector, XPath, or text content
            attribute (str): Attribute name to get
            timeout (int): Maximum time to wait for element in milliseconds
            
        Returns:
            Optional[str]: Attribute value if found, None otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return None
        
        try:
            logger.info(f"Getting attribute '{attribute}' from element: {selector}")
            
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Get the attribute value
            value = await self.page.get_attribute(selector, attribute)
            logger.info(f"Attribute '{attribute}' value: {value}")
            return value
            
        except Exception as e:
            logger.error(f"Failed to get attribute '{attribute}' from element '{selector}': {str(e)}")
            return None
    
    async def page_back(self) -> bool:
        """
        Navigate back to the previous page.
        
        Returns:
            bool: True if navigation successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info("Navigating back to previous page")
            await self.page.go_back()
            logger.info("Successfully navigated back")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate back: {str(e)}")
            return False
    
    async def page_forward(self) -> bool:
        """
        Navigate forward to the next page.
        
        Returns:
            bool: True if navigation successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info("Navigating forward to next page")
            await self.page.go_forward()
            logger.info("Successfully navigated forward")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate forward: {str(e)}")
            return False
    
    async def page_refresh(self) -> bool:
        """
        Refresh the current page.
        
        Returns:
            bool: True if refresh successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info("Refreshing current page")
            await self.page.reload()
            logger.info("Successfully refreshed page")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh page: {str(e)}")
            return False
    
    async def scroll_to_element(self, selector: str, timeout: int = 10000) -> bool:
        """
        Scroll to make an element visible.
        
        Args:
            selector (str): CSS selector, XPath, or text content
            timeout (int): Maximum time to wait for element in milliseconds
            
        Returns:
            bool: True if scroll successful, False otherwise
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return False
        
        try:
            logger.info(f"Scrolling to element: {selector}")
            
            # Wait for element to be attached to DOM
            await self.page.wait_for_selector(selector, state='attached', timeout=timeout)
            
            # Scroll element into view
            await self.page.locator(selector).scroll_into_view_if_needed()
            logger.info(f"Successfully scrolled to element: {selector}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scroll to element '{selector}': {str(e)}")
            return False
    
    async def get_all_elements(self, selector: str, timeout: int = 10000) -> List[Locator]:
        """
        Get all elements matching a selector.
        
        Args:
            selector (str): CSS selector, XPath, or text content
            timeout (int): Maximum time to wait for at least one element in milliseconds
            
        Returns:
            List[Locator]: List of element locators
        """
        if not self.page:
            logger.error("No active page. Please start browser first.")
            return []
        
        try:
            logger.info(f"Getting all elements matching: {selector}")
            
            # Wait for at least one element to be visible
            await self.page.wait_for_selector(selector, state='visible', timeout=timeout)
            
            # Get all matching elements
            elements = await self.page.locator(selector).all()
            logger.info(f"Found {len(elements)} elements matching '{selector}'")
            return elements
            
        except Exception as e:
            logger.error(f"Failed to get elements matching '{selector}': {str(e)}")
            return []
    
    async def close_browser(self) -> None:
        """
        Close the browser and cleanup resources.
        """
        try:
            if self.page:
                await self.page.close()
                logger.info("Page closed.")
            
            if self.context:
                await self.context.close()
                logger.info("Browser context closed.")
            
            if self.browser:
                await self.browser.close()
                logger.info("Browser closed.")
            
            if self.playwright:
                await self.playwright.stop()
                logger.info("Playwright stopped.")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()


async def main():
    """
    Main function to demonstrate core browser automation capabilities.
    """
    logger.info("Starting core browser automation demo...")
    
    # Example usage with context manager
    async with BrowserAutomation(headless=False) as automation:
        # Navigate to a test website
        success = await automation.navigate_to("https://example.com")
        
        if success:
            title = await automation.get_page_title()
            print(f"✅ Success! Page loaded: {title}")
            
            # Demonstrate core browser actions
            print("\n🧪 Testing core browser actions...")
            
            # Test element interaction (if elements exist)
            try:
                # Try to get text from the main heading
                heading_text = await automation.get_element_text("h1")
                if heading_text:
                    print(f"📝 Found heading: {heading_text}")
                
                # Test page navigation
                print("🔄 Testing page refresh...")
                refresh_success = await automation.page_refresh()
                print(f"   Refresh result: {'✅ SUCCESS' if refresh_success else '❌ FAILED'}")
                
            except Exception as e:
                print(f"⚠️  Some actions not applicable on this page: {str(e)}")
            
        else:
            print("❌ Failed to load page")
    
    logger.info("Core browser automation demo completed.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
