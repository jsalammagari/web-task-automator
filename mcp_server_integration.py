#!/usr/bin/env python3
"""
MCP Server Integration for AI Context
=====================================

This module provides integration with Playwright MCP Server to enable
AI-driven browser automation with detailed webpage context and accessibility data.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import subprocess
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import aiohttp
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_integration.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AccessibilityData:
    """Data class for accessibility information."""
    role: str
    name: str
    description: Optional[str] = None
    state: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    children: Optional[List['AccessibilityData']] = None


@dataclass
class ElementContext:
    """Data class for element context information."""
    selector: str
    tag_name: str
    text_content: str
    attributes: Dict[str, str]
    bounding_box: Dict[str, float]
    is_visible: bool
    is_enabled: bool
    accessibility: Optional[AccessibilityData] = None


@dataclass
class PageContext:
    """Data class for complete page context."""
    url: str
    title: str
    viewport: Dict[str, int]
    elements: List[ElementContext]
    accessibility_tree: Optional[AccessibilityData] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPServerManager:
    """
    Manager for Playwright MCP Server integration.
    
    Handles server lifecycle, connection management, and data exchange.
    """
    
    def __init__(self, server_port: int = 3000, server_host: str = "localhost"):
        """
        Initialize MCP Server Manager.
        
        Args:
            server_port (int): Port for MCP Server
            server_host (str): Host for MCP Server
        """
        self.server_port = server_port
        self.server_host = server_host
        self.server_url = f"http://{server_host}:{server_port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        
        # MCP Server configuration
        self.mcp_server_path = Path("mcp-server-playwright")
        self.server_timeout = 30
        self.connection_retries = 3
        
    async def start_server(self) -> bool:
        """
        Start the Playwright MCP Server.
        
        Returns:
            bool: True if server started successfully, False otherwise
        """
        try:
            logger.info("🚀 Starting Playwright MCP Server...")
            
            # Check if MCP Server directory exists
            if not self.mcp_server_path.exists():
                logger.error(f"❌ MCP Server directory not found: {self.mcp_server_path}")
                return False
            
            # Start the server process
            server_cmd = [
                "node", 
                str(self.mcp_server_path / "dist" / "index.js"),
                "--port", str(self.server_port)
            ]
            
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.mcp_server_path
            )
            
            # Wait for server to start
            await asyncio.sleep(3)
            
            # Test connection
            if await self._test_connection():
                logger.info(f"✅ MCP Server started successfully on {self.server_url}")
                return True
            else:
                logger.error("❌ Failed to connect to MCP Server")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start MCP Server: {str(e)}")
            return False
    
    async def _test_connection(self) -> bool:
        """Test connection to MCP Server."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health", timeout=5) as response:
                    if response.status == 200:
                        self.is_connected = True
                        return True
        except Exception as e:
            logger.debug(f"Connection test failed: {str(e)}")
        
        return False
    
    async def stop_server(self) -> None:
        """Stop the MCP Server."""
        try:
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                logger.info("✅ MCP Server stopped")
            
            if self.session:
                await self.session.close()
                
            self.is_connected = False
            
        except Exception as e:
            logger.error(f"❌ Error stopping MCP Server: {str(e)}")
    
    async def ensure_connection(self) -> bool:
        """
        Ensure connection to MCP Server is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.is_connected:
            for attempt in range(self.connection_retries):
                logger.info(f"🔄 Attempting to connect to MCP Server (attempt {attempt + 1})")
                if await self._test_connection():
                    return True
                await asyncio.sleep(2)
            
            logger.error("❌ Failed to establish connection to MCP Server")
            return False
        
        return True


class MCPBrowserAutomation:
    """
    Browser automation with MCP Server integration for AI context.
    
    Provides enhanced browser automation with detailed webpage context,
    accessibility data, and AI-friendly data structures.
    """
    
    def __init__(self, mcp_manager: Optional[MCPServerManager] = None, headless: bool = True):
        """
        Initialize MCP Browser Automation.
        
        Args:
            mcp_manager (MCPServerManager, optional): MCP Server manager instance. If None, creates a default one.
            headless (bool): Whether to run browser in headless mode
        """
        self.mcp_manager = mcp_manager or MCPServerManager()
        self.headless = headless
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_page_context: Optional[PageContext] = None
        
    async def start_session(self) -> bool:
        """
        Start a new browser session via MCP Server.
        
        Returns:
            bool: True if session started successfully, False otherwise
        """
        try:
            if not await self.mcp_manager.ensure_connection():
                return False
            
            self.session = aiohttp.ClientSession()
            
            # Start browser session
            session_data = {
                "headless": self.headless,
                "viewport": {"width": 1920, "height": 1080}
            }
            
            async with self.session.post(
                f"{self.mcp_manager.server_url}/browser/start",
                json=session_data
            ) as response:
                if response.status == 200:
                    logger.info("✅ Browser session started via MCP Server")
                    return True
                else:
                    logger.error(f"❌ Failed to start browser session: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error starting browser session: {str(e)}")
            return False
    
    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to a URL and capture page context.
        
        Args:
            url (str): URL to navigate to
            
        Returns:
            bool: True if navigation successful, False otherwise
        """
        try:
            if not self.session:
                logger.error("❌ No active session. Please start session first.")
                return False
            
            # Navigate to URL
            nav_data = {"url": url}
            async with self.session.post(
                f"{self.mcp_manager.server_url}/browser/navigate",
                json=nav_data
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ Navigation failed: {response.status}")
                    return False
            
            # Wait for page to load
            await asyncio.sleep(2)
            
            # Capture page context
            await self._capture_page_context(url)
            
            logger.info(f"✅ Successfully navigated to: {url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Navigation error: {str(e)}")
            return False
    
    async def _capture_page_context(self, url: str) -> None:
        """Capture comprehensive page context including accessibility data."""
        try:
            # Get basic page information
            page_info = await self._get_page_info()
            
            # Get all elements with context
            elements = await self._get_elements_context()
            
            # Get accessibility tree
            accessibility_tree = await self._get_accessibility_tree()
            
            # Create page context
            self.current_page_context = PageContext(
                url=url,
                title=page_info.get("title", ""),
                viewport=page_info.get("viewport", {}),
                elements=elements,
                accessibility_tree=accessibility_tree,
                metadata=page_info.get("metadata", {})
            )
            
            logger.info(f"📊 Captured context for {len(elements)} elements")
            
        except Exception as e:
            logger.error(f"❌ Error capturing page context: {str(e)}")
    
    async def _get_page_info(self) -> Dict[str, Any]:
        """Get basic page information."""
        try:
            async with self.session.get(
                f"{self.mcp_manager.server_url}/browser/page-info"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"❌ Error getting page info: {str(e)}")
            return {}
    
    async def _get_elements_context(self) -> List[ElementContext]:
        """Get detailed context for all page elements."""
        try:
            async with self.session.get(
                f"{self.mcp_manager.server_url}/browser/elements"
            ) as response:
                if response.status == 200:
                    elements_data = await response.json()
                    return [self._parse_element_data(elem) for elem in elements_data]
                else:
                    return []
        except Exception as e:
            logger.error(f"❌ Error getting elements context: {str(e)}")
            return []
    
    async def _get_accessibility_tree(self) -> Optional[AccessibilityData]:
        """Get accessibility tree for the page."""
        try:
            async with self.session.get(
                f"{self.mcp_manager.server_url}/browser/accessibility"
            ) as response:
                if response.status == 200:
                    tree_data = await response.json()
                    return self._parse_accessibility_data(tree_data)
                else:
                    return None
        except Exception as e:
            logger.error(f"❌ Error getting accessibility tree: {str(e)}")
            return None
    
    def _parse_element_data(self, element_data: Dict[str, Any]) -> ElementContext:
        """Parse element data into ElementContext."""
        return ElementContext(
            selector=element_data.get("selector", ""),
            tag_name=element_data.get("tagName", ""),
            text_content=element_data.get("textContent", ""),
            attributes=element_data.get("attributes", {}),
            bounding_box=element_data.get("boundingBox", {}),
            is_visible=element_data.get("isVisible", False),
            is_enabled=element_data.get("isEnabled", False),
            accessibility=self._parse_accessibility_data(element_data.get("accessibility"))
        )
    
    def _parse_accessibility_data(self, data: Optional[Dict[str, Any]]) -> Optional[AccessibilityData]:
        """Parse accessibility data."""
        if not data:
            return None
        
        return AccessibilityData(
            role=data.get("role", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            state=data.get("state"),
            properties=data.get("properties"),
            children=[self._parse_accessibility_data(child) for child in data.get("children", [])]
        )
    
    async def get_ai_context(self) -> Dict[str, Any]:
        """
        Get AI-friendly context data for the current page.
        
        Returns:
            Dict[str, Any]: Comprehensive context data for AI processing
        """
        if not self.current_page_context:
            return {"error": "No page context available"}
        
        # Convert to AI-friendly format
        context = {
            "page": {
                "url": self.current_page_context.url,
                "title": self.current_page_context.title,
                "viewport": self.current_page_context.viewport
            },
            "elements": [
                {
                    "selector": elem.selector,
                    "tag": elem.tag_name,
                    "text": elem.text_content,
                    "visible": elem.is_visible,
                    "enabled": elem.is_enabled,
                    "role": elem.accessibility.role if elem.accessibility else None,
                    "name": elem.accessibility.name if elem.accessibility else None
                }
                for elem in self.current_page_context.elements
            ],
            "accessibility": self._accessibility_to_dict(self.current_page_context.accessibility_tree),
            "summary": {
                "total_elements": len(self.current_page_context.elements),
                "visible_elements": len([e for e in self.current_page_context.elements if e.is_visible]),
                "interactive_elements": len([e for e in self.current_page_context.elements if e.is_enabled])
            }
        }
        
        return context
    
    def _accessibility_to_dict(self, accessibility: Optional[AccessibilityData]) -> Optional[Dict[str, Any]]:
        """Convert accessibility data to dictionary."""
        if not accessibility:
            return None
        
        return {
            "role": accessibility.role,
            "name": accessibility.name,
            "description": accessibility.description,
            "state": accessibility.state,
            "properties": accessibility.properties,
            "children": [self._accessibility_to_dict(child) for child in (accessibility.children or [])]
        }
    
    async def perform_action(self, action: str, selector: str, **kwargs) -> bool:
        """
        Perform an action on an element via MCP Server.
        
        Args:
            action (str): Action to perform (click, type, etc.)
            selector (str): Element selector
            **kwargs: Additional action parameters
            
        Returns:
            bool: True if action successful, False otherwise
        """
        try:
            if not self.session:
                logger.error("❌ No active session")
                return False
            
            action_data = {
                "action": action,
                "selector": selector,
                **kwargs
            }
            
            async with self.session.post(
                f"{self.mcp_manager.server_url}/browser/action",
                json=action_data
            ) as response:
                if response.status == 200:
                    logger.info(f"✅ Action '{action}' performed on '{selector}'")
                    return True
                else:
                    logger.error(f"❌ Action failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Action error: {str(e)}")
            return False
    
    async def close_session(self) -> None:
        """Close the browser session."""
        try:
            if self.session:
                async with self.session.post(
                    f"{self.mcp_manager.server_url}/browser/close"
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Browser session closed")
                
                await self.session.close()
                self.session = None
                
        except Exception as e:
            logger.error(f"❌ Error closing session: {str(e)}")


class MCPIntegrationDemo:
    """Demo class for MCP Server integration."""
    
    def __init__(self):
        self.mcp_manager = MCPServerManager()
        self.automation = MCPBrowserAutomation(self.mcp_manager)
    
    async def run_demo(self):
        """Run MCP integration demo."""
        try:
            logger.info("🚀 Starting MCP Server Integration Demo")
            
            # Start MCP Server
            if not await self.mcp_manager.start_server():
                logger.error("❌ Failed to start MCP Server")
                return
            
            # Start browser session
            if not await self.automation.start_session():
                logger.error("❌ Failed to start browser session")
                return
            
            # Navigate to a page
            if await self.automation.navigate_to("https://example.com"):
                # Get AI context
                context = await self.automation.get_ai_context()
                logger.info(f"📊 AI Context captured: {context['summary']}")
                
                # Save context to file
                with open("ai_context.json", "w") as f:
                    json.dump(context, f, indent=2)
                logger.info("💾 AI context saved to ai_context.json")
            
            # Close session
            await self.automation.close_session()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
        finally:
            await self.mcp_manager.stop_server()


async def main():
    """Main function for MCP integration demo."""
    demo = MCPIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
