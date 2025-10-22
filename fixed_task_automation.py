#!/usr/bin/env python3
"""
Fixed Task Automation
=================

This module implements a specific automated task that demonstrates
the full capabilities of the browser automation system.

Task: "Navigate to an e-commerce site, search for a product, and report results"

Author: Web Task Automator
"""

import asyncio
import logging
import time
from typing import Optional, Dict, List, Tuple
from browser_automation import BrowserAutomation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedTaskAutomation:
    """
    Fixed task automation class that implements a specific automated workflow.
    
    This class demonstrates a complete e-commerce search task:
    1. Navigate to an e-commerce website
    2. Search for a specific product
    3. Extract and report search results
    4. Handle errors and edge cases gracefully
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the fixed task automation.
        
        Args:
            headless (bool): Whether to run browser in headless mode. Defaults to True.
        """
        self.headless = headless
        self.automation = None
        self.task_results = {
            'navigation_success': False,
            'search_success': False,
            'results_found': False,
            'error_messages': [],
            'search_results': [],
            'total_time': 0
        }
    
    async def start_automation(self) -> bool:
        """
        Start the browser automation.
        
        Returns:
            bool: True if automation started successfully, False otherwise.
        """
        try:
            self.automation = BrowserAutomation(headless=self.headless)
            success = await self.automation.start_browser()
            if success:
                logger.info("Fixed task automation started successfully")
                return True
            else:
                logger.error("Failed to start browser automation")
                return False
        except Exception as e:
            logger.error(f"Error starting automation: {str(e)}")
            return False
    
    async def close_automation(self) -> None:
        """
        Close the browser automation and cleanup resources.
        """
        try:
            if self.automation:
                await self.automation.close_browser()
                logger.info("Fixed task automation closed")
        except Exception as e:
            logger.error(f"Error closing automation: {str(e)}")
    
    async def navigate_to_ecommerce_site(self, url: str = "https://httpbin.org/forms/post") -> bool:
        """
        Navigate to an e-commerce-like website for testing.
        
        Args:
            url (str): URL to navigate to. Defaults to a form page for testing.
            
        Returns:
            bool: True if navigation successful, False otherwise.
        """
        try:
            logger.info(f"Step 1: Navigating to e-commerce site: {url}")
            success = await self.automation.navigate_to(url)
            
            if success:
                # Wait for page to load completely
                await self.automation.wait_for_element("body", state='visible', timeout=10000)
                self.task_results['navigation_success'] = True
                logger.info("✅ Navigation successful")
                return True
            else:
                self.task_results['error_messages'].append("Failed to navigate to e-commerce site")
                logger.error("❌ Navigation failed")
                return False
                
        except Exception as e:
            error_msg = f"Navigation error: {str(e)}"
            self.task_results['error_messages'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def search_for_product(self, search_term: str = "laptop") -> bool:
        """
        Search for a product on the website.
        
        Args:
            search_term (str): Product to search for. Defaults to "laptop".
            
        Returns:
            bool: True if search successful, False otherwise.
        """
        try:
            logger.info(f"Step 2: Searching for product: '{search_term}'")
            
            # Try to find and fill search input
            search_selectors = [
                "input[name='search']",
                "input[type='search']",
                "input[placeholder*='search']",
                "input[placeholder*='Search']",
                "input[name='q']",
                "input[name='query']",
                "input[name='custname']",  # Fallback for test form
                "input[type='text']"
            ]
            
            search_success = False
            for selector in search_selectors:
                try:
                    # Wait for search input to be visible
                    if await self.automation.wait_for_element(selector, state='visible', timeout=3000):
                        # Clear and type search term
                        await self.automation.type_text(selector, search_term, clear_first=True)
                        logger.info(f"✅ Found search input: {selector}")
                        search_success = True
                        break
                except Exception as e:
                    logger.debug(f"Search input {selector} not found: {str(e)}")
                    continue
            
            if not search_success:
                # If no search input found, try to click on a search button or link
                search_buttons = ["button[type='submit']", "input[type='submit']", "button"]
                for button in search_buttons:
                    try:
                        if await self.automation.wait_for_element(button, state='visible', timeout=2000):
                            await self.automation.click_element(button)
                            logger.info(f"✅ Clicked search button: {button}")
                            search_success = True
                            break
                    except Exception as e:
                        logger.debug(f"Search button {button} not found: {str(e)}")
                        continue
            
            if search_success:
                self.task_results['search_success'] = True
                logger.info("✅ Search action completed")
                return True
            else:
                error_msg = "No search input or button found"
                self.task_results['error_messages'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            self.task_results['error_messages'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def extract_search_results(self) -> bool:
        """
        Extract and report search results.
        
        Returns:
            bool: True if results extracted successfully, False otherwise.
        """
        try:
            logger.info("Step 3: Extracting search results")
            
            # Wait a moment for results to load
            await asyncio.sleep(2)
            
            # Try to find result elements
            result_selectors = [
                ".search-result",
                ".product-item",
                ".result-item",
                "li",
                "div[class*='result']",
                "div[class*='item']",
                "p",
                "h1", "h2", "h3"
            ]
            
            results = []
            for selector in result_selectors:
                try:
                    elements = await self.automation.get_all_elements(selector, timeout=3000)
                    if elements:
                        for i, element in enumerate(elements[:5]):  # Limit to first 5 results
                            try:
                                text = await element.text_content()
                                if text and len(text.strip()) > 10:  # Only meaningful text
                                    results.append({
                                        'selector': selector,
                                        'index': i,
                                        'text': text.strip()[:100]  # Limit text length
                                    })
                            except Exception as e:
                                logger.debug(f"Error getting text from element {i}: {str(e)}")
                                continue
                except Exception as e:
                    logger.debug(f"Error with selector {selector}: {str(e)}")
                    continue
            
            if results:
                self.task_results['search_results'] = results
                self.task_results['results_found'] = True
                logger.info(f"✅ Found {len(results)} search results")
                return True
            else:
                # If no specific results found, get page content
                page_title = await self.automation.get_page_title()
                page_text = await self.automation.get_element_text("body")
                
                if page_title or page_text:
                    self.task_results['search_results'] = [{
                        'selector': 'page_content',
                        'index': 0,
                        'text': f"Page: {page_title or 'No title'} - {page_text[:100] if page_text else 'No content'}"
                    }]
                    self.task_results['results_found'] = True
                    logger.info("✅ Extracted page content as results")
                    return True
                else:
                    error_msg = "No search results or page content found"
                    self.task_results['error_messages'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
                    return False
                    
        except Exception as e:
            error_msg = f"Results extraction error: {str(e)}"
            self.task_results['error_messages'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    async def handle_edge_cases(self) -> None:
        """
        Handle edge cases like slow loading, missing elements, and network issues.
        """
        try:
            logger.info("Step 4: Handling edge cases")
            
            # Check for slow loading
            start_time = time.time()
            await self.automation.wait_for_element("body", state='visible', timeout=15000)
            load_time = time.time() - start_time
            
            if load_time > 10:
                logger.warning(f"⚠️  Slow loading detected: {load_time:.2f} seconds")
                self.task_results['error_messages'].append(f"Slow loading: {load_time:.2f}s")
            
            # Check for network issues by trying to refresh
            try:
                refresh_success = await self.automation.page_refresh()
                if not refresh_success:
                    logger.warning("⚠️  Network issues detected - refresh failed")
                    self.task_results['error_messages'].append("Network issues detected")
            except Exception as e:
                logger.warning(f"⚠️  Network check failed: {str(e)}")
                self.task_results['error_messages'].append(f"Network check failed: {str(e)}")
            
            # Check for missing critical elements
            critical_elements = ["body", "html"]
            for element in critical_elements:
                try:
                    if not await self.automation.wait_for_element(element, state='visible', timeout=5000):
                        logger.warning(f"⚠️  Missing critical element: {element}")
                        self.task_results['error_messages'].append(f"Missing element: {element}")
                except Exception as e:
                    logger.warning(f"⚠️  Element check failed for {element}: {str(e)}")
                    self.task_results['error_messages'].append(f"Element check failed: {element}")
            
            logger.info("✅ Edge case handling completed")
            
        except Exception as e:
            error_msg = f"Edge case handling error: {str(e)}"
            self.task_results['error_messages'].append(error_msg)
            logger.error(f"❌ {error_msg}")
    
    async def run_complete_task(self, search_term: str = "laptop", url: str = "https://httpbin.org/forms/post") -> Dict:
        """
        Run the complete fixed task automation.
        
        Args:
            search_term (str): Product to search for. Defaults to "laptop".
            url (str): URL to navigate to. Defaults to a test form page.
            
        Returns:
            Dict: Task results with success/failure status and details.
        """
        start_time = time.time()
        logger.info("🚀 Starting Fixed Task Automation")
        logger.info("=" * 50)
        logger.info(f"Task: Navigate to {url}, search for '{search_term}', and report results")
        logger.info("=" * 50)
        
        try:
            # Step 1: Start automation
            if not await self.start_automation():
                self.task_results['error_messages'].append("Failed to start automation")
                return self.task_results
            
            # Step 2: Navigate to e-commerce site
            nav_success = await self.navigate_to_ecommerce_site(url)
            
            # Step 3: Search for product
            search_success = await self.search_for_product(search_term)
            
            # Step 4: Extract results
            results_success = await self.extract_search_results()
            
            # Step 5: Handle edge cases
            await self.handle_edge_cases()
            
            # Calculate total time
            self.task_results['total_time'] = time.time() - start_time
            
            # Determine overall success
            overall_success = nav_success and (search_success or results_success)
            
            # Print results
            self.print_task_results(overall_success)
            
            return self.task_results
            
        except Exception as e:
            error_msg = f"Task execution error: {str(e)}"
            self.task_results['error_messages'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            self.print_task_results(False)
            return self.task_results
        
        finally:
            # Always cleanup
            await self.close_automation()
    
    def print_task_results(self, overall_success: bool) -> None:
        """
        Print clear success/failure messages to console.
        
        Args:
            overall_success (bool): Whether the overall task was successful.
        """
        print("\n" + "=" * 60)
        print("📊 FIXED TASK AUTOMATION RESULTS")
        print("=" * 60)
        
        if overall_success:
            print("🎉 TASK COMPLETED SUCCESSFULLY!")
        else:
            print("❌ TASK FAILED")
        
        print(f"\n⏱️  Total Execution Time: {self.task_results['total_time']:.2f} seconds")
        
        print(f"\n📋 Step Results:")
        print(f"   Navigation: {'✅ SUCCESS' if self.task_results['navigation_success'] else '❌ FAILED'}")
        print(f"   Search: {'✅ SUCCESS' if self.task_results['search_success'] else '❌ FAILED'}")
        print(f"   Results: {'✅ SUCCESS' if self.task_results['results_found'] else '❌ FAILED'}")
        
        if self.task_results['search_results']:
            print(f"\n🔍 Search Results Found: {len(self.task_results['search_results'])}")
            for i, result in enumerate(self.task_results['search_results'][:3], 1):
                print(f"   {i}. {result['text']}")
        
        if self.task_results['error_messages']:
            print(f"\n⚠️  Issues Encountered: {len(self.task_results['error_messages'])}")
            for error in self.task_results['error_messages']:
                print(f"   - {error}")
        
        print("\n" + "=" * 60)


async def main():
    """
    Main function to demonstrate the fixed task automation.
    """
    # Create and run the fixed task automation
    task_automation = FixedTaskAutomation(headless=False)  # Set to True for headless mode
    
    # Run the complete task
    results = await task_automation.run_complete_task(
        search_term="laptop",
        url="https://httpbin.org/forms/post"
    )
    
    return results


if __name__ == "__main__":
    # Run the fixed task automation
    asyncio.run(main())
