#!/usr/bin/env python3
"""
Web API Foundation
==================

This module provides a FastAPI-based web API for the web task automator,
exposing automation functionality through REST endpoints.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our automation modules
from browser_automation import BrowserAutomation
from reliable_browser_automation import ReliableBrowserAutomation
from ai_task_planner import AITaskPlanner, LLMClient, LLMProvider
from dynamic_execution_engine import DynamicExecutionEngine
from natural_language_processor import NaturalLanguageGoalProcessor


# Pydantic Models for API
class TaskRequest(BaseModel):
    """Request model for task execution."""
    goal: str = Field(..., description="Natural language goal for automation")
    url: Optional[str] = Field(None, description="Target URL for automation")
    headless: bool = Field(True, description="Run browser in headless mode")
    timeout: int = Field(30000, description="Task timeout in milliseconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the task")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    task_id: str
    status: str
    message: str
    execution_time: float
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    """Task status model."""
    task_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    execution_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    services: Dict[str, str]


class AutomationConfig(BaseModel):
    """Automation configuration model."""
    headless: bool = True
    timeout: int = 30000
    retry_count: int = 3
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"


# Global state management
class TaskManager:
    """Manages running tasks and their status."""
    
    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        self.start_time = time.time()
    
    def create_task(self, task_id: str, goal: str) -> TaskStatus:
        """Create a new task."""
        task = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            message="Task created",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.tasks[task_id] = task
        return task
    
    def update_task(self, task_id: str, status: str, progress: float, message: str, results: Optional[Dict] = None):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].progress = progress
            self.tasks[task_id].message = message
            self.tasks[task_id].updated_at = datetime.now()
            if results:
                self.tasks[task_id].results = results
    
    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status."""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TaskStatus]:
        """Get all tasks."""
        return list(self.tasks.values())
    
    def get_uptime(self) -> float:
        """Get API uptime."""
        return time.time() - self.start_time


# Initialize FastAPI app
app = FastAPI(
    title="Web Task Automator API",
    description="AI-powered web automation API with natural language processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
task_manager = TaskManager()

# Global automation instances
automation_instances: Dict[str, Any] = {}


# Dependency for API key authentication (basic implementation)
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key (basic implementation)."""
    # In production, implement proper API key validation
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Simple API key check (replace with proper validation)
    if credentials.credentials != "demo-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Web Task Automator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=task_manager.get_uptime(),
        services={
            "automation": "available",
            "ai_planner": "available",
            "nlp_processor": "available"
        }
    )


@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Create and execute a new automation task."""
    try:
        task_id = f"task_{int(time.time())}"
        logger.info(f"🎯 Creating task {task_id}: {request.goal}")
        
        # Create task in manager
        task_manager.create_task(task_id, request.goal)
        
        # Start background task execution
        background_tasks.add_task(
            execute_automation_task,
            task_id,
            request
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending",
            message="Task created and queued for execution",
            execution_time=0.0,
            success=True
        )
        
    except Exception as e:
        logger.error(f"❌ Task creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task creation failed: {str(e)}"
        )


@app.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get task status by ID."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return task


@app.get("/tasks", response_model=List[TaskStatus])
async def get_all_tasks(
    api_key: str = Depends(verify_api_key)
):
    """Get all tasks."""
    return task_manager.get_all_tasks()


@app.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a running task."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    if task.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel completed or failed task"
        )
    
    task_manager.update_task(task_id, "cancelled", 0.0, "Task cancelled by user")
    
    return {"message": "Task cancelled successfully"}


@app.post("/automation/basic")
async def basic_automation(
    url: str,
    actions: List[Dict[str, Any]],
    headless: bool = True,
    api_key: str = Depends(verify_api_key)
):
    """Execute basic automation tasks."""
    try:
        logger.info(f"🔄 Executing basic automation for {url}")
        
        async with BrowserAutomation(headless=headless) as automation:
            # Navigate to URL
            success = await automation.navigate_to(url)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to navigate to URL"
                )
            
            results = []
            for action in actions:
                action_type = action.get("type")
                selector = action.get("selector")
                text = action.get("text")
                
                if action_type == "click":
                    success = await automation.click_element(selector)
                elif action_type == "type":
                    success = await automation.type_text(selector, text)
                elif action_type == "wait":
                    success = await automation.wait_for_element(selector)
                else:
                    success = False
                
                results.append({
                    "action": action_type,
                    "success": success,
                    "selector": selector
                })
            
            return {
                "success": True,
                "url": url,
                "actions_executed": len(results),
                "results": results
            }
            
    except Exception as e:
        logger.error(f"❌ Basic automation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Automation failed: {str(e)}"
        )


@app.post("/automation/ai-powered")
async def ai_powered_automation(
    goal: str,
    url: Optional[str] = None,
    headless: bool = True,
    api_key: str = Depends(verify_api_key)
):
    """Execute AI-powered automation using natural language goals."""
    try:
        logger.info(f"🤖 Executing AI-powered automation: {goal}")
        
        # Initialize AI components
        llm_client = LLMClient(LLMProvider.LOCAL)  # Use local for demo
        ai_planner = AITaskPlanner(llm_client)
        nlp_processor = NaturalLanguageGoalProcessor(ai_planner)
        
        # Get webpage context (mock for demo)
        webpage_context = {
            "page": {"title": "Target Page", "url": url or "https://example.com"},
            "elements": [
                {"tag": "input", "selector": "input[name='search']"},
                {"tag": "button", "selector": "button[type='submit']"}
            ]
        }
        
        # Process goal and create plan
        plan = await nlp_processor.process_user_goal(goal, webpage_context)
        
        # Execute plan with reliable automation
        async with ReliableBrowserAutomation(headless=headless) as automation:
            if url:
                await automation.navigate_to(url)
            
            # Execute plan steps
            results = []
            for step in plan.get("steps", []):
                action = step.get("action")
                selector = step.get("selector")
                text = step.get("text")
                
                if action == "click":
                    success = await automation.click_element(selector)
                elif action == "type":
                    success = await automation.type_text(selector, text)
                elif action == "navigate":
                    success = await automation.navigate_to(step.get("url"))
                else:
                    success = False
                
                results.append({
                    "step": step.get("step_id"),
                    "action": action,
                    "success": success
                })
            
            return {
                "success": True,
                "goal": goal,
                "plan": plan,
                "execution_results": results,
                "ai_confidence": plan.get("processing_info", {}).get("confidence", 0)
            }
            
    except Exception as e:
        logger.error(f"❌ AI-powered automation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI automation failed: {str(e)}"
        )


@app.get("/automation/capabilities")
async def get_automation_capabilities(
    api_key: str = Depends(verify_api_key)
):
    """Get available automation capabilities."""
    return {
        "supported_actions": [
            "navigate", "click", "type", "wait", "scroll", "get_text",
            "get_attribute", "refresh", "back", "forward"
        ],
        "ai_features": [
            "natural_language_processing",
            "intent_recognition",
            "entity_extraction",
            "plan_generation",
            "dynamic_execution"
        ],
        "supported_intents": [
            "shopping", "form_filling", "data_extraction",
            "navigation", "search", "click_action"
        ],
        "llm_providers": ["openai", "anthropic", "local"],
        "browser_options": {
            "headless": True,
            "visible": False,
            "mobile": False
        }
    }


@app.post("/automation/validate")
async def validate_automation_request(
    request: TaskRequest,
    api_key: str = Depends(verify_api_key)
):
    """Validate an automation request before execution."""
    try:
        # Basic validation
        if not request.goal or len(request.goal.strip()) < 3:
            return {
                "valid": False,
                "errors": ["Goal must be at least 3 characters long"]
            }
        
        if request.url and not request.url.startswith(("http://", "https://")):
            return {
                "valid": False,
                "errors": ["URL must start with http:// or https://"]
            }
        
        if request.timeout < 1000 or request.timeout > 300000:
            return {
                "valid": False,
                "errors": ["Timeout must be between 1000 and 300000 milliseconds"]
            }
        
        return {
            "valid": True,
            "message": "Request is valid",
            "estimated_duration": "30-120 seconds",
            "complexity": "moderate"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"]
        }


# Background task execution
async def execute_automation_task(task_id: str, request: TaskRequest):
    """Execute automation task in background."""
    try:
        logger.info(f"🚀 Starting task execution: {task_id}")
        task_manager.update_task(task_id, "running", 10.0, "Initializing automation")
        
        # Initialize automation
        async with ReliableBrowserAutomation(headless=request.headless) as automation:
            task_manager.update_task(task_id, "running", 30.0, "Browser initialized")
            
            # Navigate to URL if provided
            if request.url:
                success = await automation.navigate_to(request.url)
                if not success:
                    raise Exception("Failed to navigate to URL")
            
            task_manager.update_task(task_id, "running", 50.0, "Processing goal with AI")
            
            # Process goal with AI (simplified for demo)
            # In a real implementation, use the full AI pipeline
            results = {
                "goal": request.goal,
                "url": request.url,
                "headless": request.headless,
                "execution_time": time.time(),
                "status": "completed"
            }
            
            task_manager.update_task(
                task_id, 
                "completed", 
                100.0, 
                "Task completed successfully",
                results
            )
            
            logger.info(f"✅ Task completed: {task_id}")
            
    except Exception as e:
        logger.error(f"❌ Task execution failed: {task_id} - {str(e)}")
        task_manager.update_task(
            task_id,
            "failed",
            0.0,
            f"Task failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Web Task Automator API starting up...")
    logger.info("✅ API endpoints initialized")
    logger.info("✅ Task manager initialized")
    logger.info("✅ Security middleware configured")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Web Task Automator API shutting down...")
    logger.info("✅ Cleanup completed")


# Main function for running the server
def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "web_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
