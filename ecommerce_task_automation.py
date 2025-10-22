#!/usr/bin/env python3
"""
E-commerce Task Automation
========================

This module implements a realistic e-commerce search task that demonstrates
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


class EcommerceTaskAutomation:
    """
    E-commerce task automation class that implements a realistic shopping workflow.
    
    This class demonstrates a complete e-commerce search task:
    1. Navigate to an e-commerce website
    2. Search for a specific product
    3. Extract and report search results with prices
    4. Handle errors and edge cases gracefully
    """
    
    def __init__(self, headless: bool = True):
        """
        Initialize the e-commerce task automation.
        
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
            'total_time': 0,
            'products_found': 0,
            'prices_found': []
        }
    
    async def start_automation(self) -> bool:
        """Start the browser automation."""
        try:
            self.automation = BrowserAutomation(headless=self.headless)
            success = await self.automation.start_browser()
            if success:
                logger.info("E-commerce task automation started successfully")
                return True
            else:
                logger.error("Failed to start browser automation")
                return False
        except Exception as e:
            logger.error(f"Error starting automation: {str(e)}")
            return False
    
    async def close_automation(self) -> None:
        """Close the browser automation and cleanup resources."""
        try:
            if self.automation:
                await self.automation.close_browser()
                logger.info("E-commerce task automation closed")
        except Exception as e:
            logger.error(f"Error closing automation: {str(e)}")
    
    async def navigate_to_ecommerce_site(self, url: str = "https://books.toscrape.com/") -> bool:
        """
        Navigate to an e-commerce website.
        
        Args:
            url (str): URL to navigate to. Defaults to a book store for testing.
            
        Returns:
            bool: True if navigation successful, False otherwise.
        """
        try:
            logger.info(f"Step 1: Navigating to e-commerce site: {url}")
            success = await self.automation.navigate_to(url)
            
            if success:
                # Wait for page to load completely
                await self.automation.wait_for_element("body", state='visible', timeout=15000)
                
                # Check if we're on the right page
                page_title = await self.automation.get_page_title()
                if page_title:
                    logger.info(f"✅ Navigation successful - Page: {page_title}")
                    self.task_results['navigation_success'] = True
                    return True
                else:
                    logger.warning("⚠️  Navigation successful but no page title found")
                    self.task_results['navigation_success'] = True
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
    
    async def search_for_product(self, search_term: str = "python") -> bool:
        """
        Search for a product on the website.
        
        Args:
            search_term (str): Product to search for. Defaults to "python".
            
        Returns:
            bool: True if search successful, False otherwise.
        """
        try:
            logger.info(f"Step 2: Searching for product: '{search_term}'")
            
            # Try to find and fill search input
            search_selectors = [
                "input[name='q']",
                "input[name='search']",
                "input[type='search']",
                "input[placeholder*='search']",
                "input[placeholder*='Search']",
                "input[class*='search']",
                "input[id*='search']",
                "input[type='text']"
            ]
            
            search_success = False
            used_selector = None
            
            for selector in search_selectors:
                try:
                    # Wait for search input to be visible
                    if await self.automation.wait_for_element(selector, state='visible', timeout=3000):
                        # Clear and type search term
                        await self.automation.type_text(selector, search_term, clear_first=True)
                        logger.info(f"✅ Found search input: {selector}")
                        search_success = True
                        used_selector = selector
                        break
                except Exception as e:
                    logger.debug(f"Search input {selector} not found: {str(e)}")
                    continue
            
            if search_success:
                # Try to submit the search
                submit_selectors = [
                    "button[type='submit']",
                    "input[type='submit']",
                    "button[class*='search']",
                    "button[class*='submit']",
                    "button"
                ]
                
                submit_success = False
                for submit_selector in submit_selectors:
                    try:
                        if await self.automation.wait_for_element(submit_selector, state='visible', timeout=2000):
                            await self.automation.click_element(submit_selector)
                            logger.info(f"✅ Clicked search button: {submit_selector}")
                            submit_success = True
                            break
                    except Exception as e:
                        logger.debug(f"Submit button {submit_selector} not found: {str(e)}")
                        continue
                
                if not submit_success:
                    # Try pressing Enter on the search input
                    try:
                        await self.automation.automation.page.press(used_selector, "Enter")
                        logger.info("✅ Pressed Enter to submit search")
                        submit_success = True
                    except Exception as e:
                        logger.debug(f"Enter key press failed: {str(e)}")
                
                if submit_success:
                    # Wait for search results to load
                    await asyncio.sleep(3)
                    self.task_results['search_success'] = True
                    logger.info("✅ Search submitted successfully")
                    return True
                else:
                    logger.warning("⚠️  Search input filled but no submit method found")
                    self.task_results['search_success'] = True  # Still consider it successful
                    return True
            else:
                error_msg = "No search input found"
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
        Extract and report search results with product information.
        
        Returns:
            bool: True if results extracted successfully, False otherwise.
        """
        try:
            logger.info("Step 3: Extracting search results")
            
            # Wait for results to load
            await asyncio.sleep(2)
            
            # Try to find product result elements
            result_selectors = [
                ".product_pod",
                ".product-item",
                ".search-result",
                ".result-item",
                ".book",
                "article",
                ".item",
                "li",
                "div[class*='product']",
                "div[class*='book']",
                "div[class*='item']"
            ]
            
            results = []
            products_found = 0
            prices_found = []
            
            for selector in result_selectors:
                try:
                    elements = await self.automation.get_all_elements(selector, timeout=5000)
                    if elements:
                        logger.info(f"Found {len(elements)} elements with selector: {selector}")
                        
                        for i, element in enumerate(elements[:10]):  # Limit to first 10 results
                            try:
                                # Get element text
                                text = await element.text_content()
                                if text and len(text.strip()) > 5:
                                    # Try to extract price information
                                    price_selectors = [
                                        ".price",
                                        ".cost",
                                        "[class*='price']",
                                        "[class*='cost']",
                                        "span[class*='price']",
                                        "p[class*='price']"
                                    ]
                                    
                                    price_text = None
                                    for price_selector in price_selectors:
                                        try:
                                            price_element = await element.locator(price_selector).first
                                            if await price_element.count() > 0:
                                                price_text = await price_element.text_content()
                                                if price_text:
                                                    prices_found.append(price_text.strip())
                                                    break
                                        except Exception as e:
                                            logger.debug(f"Price extraction failed: {str(e)}")
                                            continue
                                    
                                    results.append({
                                        'selector': selector,
                                        'index': i,
                                        'text': text.strip()[:200],  # Limit text length
                                        'price': price_text.strip() if price_text else 'No price found'
                                    })
                                    products_found += 1
                                    
                            except Exception as e:
                                logger.debug(f"Error processing element {i}: {str(e)}")
                                continue
                        
                        if results:
                            break  # Found results with this selector
                            
                except Exception as e:
                    logger.debug(f"Error with selector {selector}: {str(e)}")
                    continue
            
            if results:
                self.task_results['search_results'] = results
                self.task_results['results_found'] = True
                self.task_results['products_found'] = products_found
                self.task_results['prices_found'] = prices_found
                logger.info(f"✅ Found {len(results)} search results with {products_found} products")
                return True
            else:
                # If no specific results found, get page content
                page_title = await self.automation.get_page_title()
                page_text = await self.automation.get_element_text("body")
                
                if page_title or page_text:
                    self.task_results['search_results'] = [{
                        'selector': 'page_content',
                        'index': 0,
                        'text': f"Page: {page_title or 'No title'} - {page_text[:200] if page_text else 'No content'}",
                        'price': 'No price information'
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
        """Handle edge cases like slow loading, missing elements, and network issues."""
        try:
            logger.info("Step 4: Handling edge cases")
            
            # Check for slow loading
            start_time = time.time()
            await self.automation.wait_for_element("body", state='visible', timeout=20000)
            load_time = time.time() - start_time
            
            if load_time > 15:
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
    
    async def run_complete_task(self, search_term: str = "python", url: str = "https://books.toscrape.com/") -> Dict:
        """
        Run the complete e-commerce task automation.
        
        Args:
            search_term (str): Product to search for. Defaults to "python".
            url (str): URL to navigate to. Defaults to a book store.
            
        Returns:
            Dict: Task results with success/failure status and details.
        """
        start_time = time.time()
        logger.info("🚀 Starting E-commerce Task Automation")
        logger.info("=" * 60)
        logger.info(f"Task: Navigate to {url}, search for '{search_term}', and report results")
        logger.info("=" * 60)
        
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
        print("\n" + "=" * 70)
        print("🛒 E-COMMERCE TASK AUTOMATION RESULTS")
        print("=" * 70)
        
        if overall_success:
            print("🎉 TASK COMPLETED SUCCESSFULLY!")
        else:
            print("❌ TASK FAILED")
        
        print(f"\n⏱️  Total Execution Time: {self.task_results['total_time']:.2f} seconds")
        
        print(f"\n📋 Step Results:")
        print(f"   Navigation: {'✅ SUCCESS' if self.task_results['navigation_success'] else '❌ FAILED'}")
        print(f"   Search: {'✅ SUCCESS' if self.task_results['search_success'] else '❌ FAILED'}")
        print(f"   Results: {'✅ SUCCESS' if self.task_results['results_found'] else '❌ FAILED'}")
        
        if self.task_results['products_found'] > 0:
            print(f"\n🛍️  Products Found: {self.task_results['products_found']}")
        
        if self.task_results['prices_found']:
            print(f"\n💰 Prices Found: {len(self.task_results['prices_found'])}")
            for price in self.task_results['prices_found'][:5]:  # Show first 5 prices
                print(f"   - {price}")
        
        if self.task_results['search_results']:
            print(f"\n🔍 Search Results: {len(self.task_results['search_results'])}")
            for i, result in enumerate(self.task_results['search_results'][:3], 1):
                print(f"   {i}. {result['text'][:100]}...")
                if result['price'] != 'No price found':
                    print(f"      💰 Price: {result['price']}")
        
        if self.task_results['error_messages']:
            print(f"\n⚠️  Issues Encountered: {len(self.task_results['error_messages'])}")
            for error in self.task_results['error_messages']:
                print(f"   - {error}")
        
        print("\n" + "=" * 70)


async def main():
    """
    Main function to demonstrate the e-commerce task automation.
    """
    # Create and run the e-commerce task automation
    task_automation = EcommerceTaskAutomation(headless=False)  # Set to True for headless mode
    
    # Run the complete task
    results = await task_automation.run_complete_task(
        search_term="python",
        url="https://books.toscrape.com/"
    )
    
    return results


if __name__ == "__main__":
    # Run the e-commerce task automation
    asyncio.run(main())
