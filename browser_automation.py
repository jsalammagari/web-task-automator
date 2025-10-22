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
from typing import Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrowserAutomation:
    """
    Basic browser automation class using Playwright.
    
    This class provides foundational browser automation capabilities including:
    - Browser launching and management
    - Basic navigation
    - Error handling
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
    Main function to demonstrate basic browser automation.
    """
    logger.info("Starting basic browser automation demo...")
    
    # Example usage with context manager
    async with BrowserAutomation(headless=False) as automation:
        # Navigate to a test website
        success = await automation.navigate_to("https://example.com")
        
        if success:
            title = await automation.get_page_title()
            print(f"✅ Success! Page loaded: {title}")
        else:
            print("❌ Failed to load page")
    
    logger.info("Browser automation demo completed.")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
