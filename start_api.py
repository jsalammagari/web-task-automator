#!/usr/bin/env python3
"""
Start Web API Server
====================

Simple script to start the web API server.

Author: Web Task Automator
"""

import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server."""
    logger.info("🚀 Starting Web Task Automator API Server...")
    logger.info("📖 API Documentation will be available at: http://localhost:8000/docs")
    logger.info("🔧 ReDoc documentation at: http://localhost:8000/redoc")
    logger.info("❤️  Health check at: http://localhost:8000/health")
    
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
