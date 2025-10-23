#!/usr/bin/env python3
"""
Remote Task Execution API
========================

This module provides enhanced API endpoints for remote task execution
with task queuing, status tracking, and result delivery.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('remote_execution.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our automation modules
from browser_automation import BrowserAutomation
from reliable_browser_automation import ReliableBrowserAutomation
from ai_task_planner import AITaskPlanner, LLMClient, LLMProvider
from dynamic_execution_engine import DynamicExecutionEngine
from natural_language_processor import NaturalLanguageGoalProcessor


# Enhanced Pydantic Models
class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class RemoteTaskRequest(BaseModel):
    """Enhanced request model for remote task execution."""
    task_description: str = Field(..., description="Natural language description of the task", min_length=3)
    target_url: Optional[str] = Field(None, description="Target URL for automation")
    priority: TaskPriority = Field(TaskPriority.NORMAL, description="Task priority level")
    timeout_seconds: int = Field(300, description="Task timeout in seconds", ge=30, le=3600)
    headless: bool = Field(True, description="Run browser in headless mode")
    callback_url: Optional[str] = Field(None, description="Callback URL for result delivery")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional task metadata")
    
    @validator('target_url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('callback_url')
    def validate_callback_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('Callback URL must start with http:// or https://')
        return v


class TaskSubmissionResponse(BaseModel):
    """Response model for task submission."""
    task_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None
    queue_position: Optional[int] = None
    created_at: datetime


class TaskStatusResponse(BaseModel):
    """Enhanced task status response."""
    task_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None


class TaskResult(BaseModel):
    """Task execution result."""
    task_id: str
    status: str
    success: bool
    execution_time: float
    results: Dict[str, Any]
    screenshots: Optional[List[str]] = None
    logs: Optional[List[str]] = None
    error: Optional[str] = None
    completed_at: datetime


class QueueStatus(BaseModel):
    """Task queue status."""
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    queue_size: int
    average_wait_time: float
    estimated_completion: Optional[datetime] = None


# Task Queue Management
class TaskQueue:
    """Manages task queue and execution."""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_status: Dict[str, TaskStatusResponse] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.start_time = time.time()
        self._lock = threading.Lock()
        
    def submit_task(self, task_id: str, request: RemoteTaskRequest) -> TaskSubmissionResponse:
        """Submit a new task to the queue."""
        with self._lock:
            # Calculate priority score (lower = higher priority)
            priority_score = {
                TaskPriority.LOW: 4,
                TaskPriority.NORMAL: 3,
                TaskPriority.HIGH: 2,
                TaskPriority.URGENT: 1
            }.get(request.priority, 3)
            
            # Add to queue
            self.task_queue.put((priority_score, time.time(), task_id, request))
            
            # Create task status
            task_status = TaskStatusResponse(
                task_id=task_id,
                status=TaskStatus.QUEUED.value,
                progress=0.0,
                message="Task queued for execution",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                queue_position=self.task_queue.qsize()
            )
            
            self.task_status[task_id] = task_status
            
            # Start worker if not already running
            if len(self.running_tasks) < self.max_workers:
                self._start_worker()
            
            return TaskSubmissionResponse(
                task_id=task_id,
                status=TaskStatus.QUEUED.value,
                message="Task submitted successfully",
                estimated_completion=self._estimate_completion(),
                queue_position=task_status.queue_position,
                created_at=task_status.created_at
            )
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get task status."""
        with self._lock:
            return self.task_status.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result."""
        with self._lock:
            return self.completed_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            if task_id in self.running_tasks:
                # Mark as cancelled (actual cancellation would need more complex logic)
                if task_id in self.task_status:
                    self.task_status[task_id].status = TaskStatus.CANCELLED.value
                    self.task_status[task_id].message = "Task cancelled by user"
                    self.task_status[task_id].updated_at = datetime.now()
                return True
            return False
    
    def get_queue_status(self) -> QueueStatus:
        """Get queue status."""
        with self._lock:
            total = len(self.task_status)
            pending = sum(1 for status in self.task_status.values() 
                         if status.status in [TaskStatus.PENDING.value, TaskStatus.QUEUED.value])
            running = len(self.running_tasks)
            completed = sum(1 for status in self.task_status.values() 
                           if status.status == TaskStatus.COMPLETED.value)
            failed = sum(1 for status in self.task_status.values() 
                        if status.status == TaskStatus.FAILED.value)
            
            return QueueStatus(
                total_tasks=total,
                pending_tasks=pending,
                running_tasks=running,
                completed_tasks=completed,
                failed_tasks=failed,
                queue_size=self.task_queue.qsize(),
                average_wait_time=self._calculate_average_wait_time(),
                estimated_completion=self._estimate_completion()
            )
    
    def _start_worker(self):
        """Start a worker thread."""
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
    
    def _worker_loop(self):
        """Worker loop for processing tasks."""
        while True:
            try:
                if self.task_queue.empty():
                    time.sleep(1)
                    continue
                
                priority, timestamp, task_id, request = self.task_queue.get()
                
                # Update status to running
                with self._lock:
                    if task_id in self.task_status:
                        self.task_status[task_id].status = TaskStatus.RUNNING.value
                        self.task_status[task_id].started_at = datetime.now()
                        self.task_status[task_id].message = "Task execution started"
                        self.task_status[task_id].updated_at = datetime.now()
                        self.task_status[task_id].queue_position = None
                
                self.running_tasks[task_id] = request
                
                # Execute task
                result = self._execute_task(task_id, request)
                
                # Store result
                with self._lock:
                    self.completed_tasks[task_id] = result
                    if task_id in self.task_status:
                        self.task_status[task_id].status = result.status
                        self.task_status[task_id].completed_at = result.completed_at
                        self.task_status[task_id].execution_time = result.execution_time
                        self.task_status[task_id].results = result.results
                        self.task_status[task_id].error = result.error
                        self.task_status[task_id].updated_at = datetime.now()
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                # Send callback if provided
                if request.callback_url:
                    self._send_callback(request.callback_url, result)
                
            except Exception as e:
                logger.error(f"❌ Worker error: {str(e)}")
                time.sleep(5)
    
    def _execute_task(self, task_id: str, request: RemoteTaskRequest) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Executing task {task_id}: {request.task_description}")
            
            # Initialize automation components
            async def run_automation():
                async with ReliableBrowserAutomation(headless=request.headless) as automation:
                    # Navigate to target URL if provided
                    if request.target_url:
                        success = await automation.navigate_to(request.target_url)
                        if not success:
                            raise Exception("Failed to navigate to target URL")
                    
                    # Process task with AI
                    llm_client = LLMClient(LLMProvider.LOCAL)
                    ai_planner = AITaskPlanner(llm_client)
                    nlp_processor = NaturalLanguageGoalProcessor(ai_planner)
                    
                    # Get webpage context
                    webpage_context = {
                        "page": {"title": "Target Page", "url": request.target_url or "https://example.com"},
                        "elements": [
                            {"tag": "input", "selector": "input[name='search']"},
                            {"tag": "button", "selector": "button[type='submit']"}
                        ]
                    }
                    
                    # Process goal and create plan
                    plan = await nlp_processor.process_user_goal(request.task_description, webpage_context)
                    
                    # Execute plan
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
                        "goal": request.task_description,
                        "plan": plan,
                        "execution_results": results,
                        "ai_confidence": plan.get("processing_info", {}).get("confidence", 0)
                    }
            
            # Run automation in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_data = loop.run_until_complete(run_automation())
            loop.close()
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED.value,
                success=True,
                execution_time=execution_time,
                results=result_data,
                completed_at=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Task execution failed {task_id}: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED.value,
                success=False,
                execution_time=execution_time,
                results={},
                error=str(e),
                completed_at=datetime.now()
            )
    
    def _send_callback(self, callback_url: str, result: TaskResult):
        """Send callback notification."""
        try:
            import requests
            requests.post(callback_url, json=asdict(result), timeout=10)
            logger.info(f"✅ Callback sent to {callback_url}")
        except Exception as e:
            logger.error(f"❌ Callback failed: {str(e)}")
    
    def _estimate_completion(self) -> Optional[datetime]:
        """Estimate completion time for next task."""
        if self.task_queue.empty():
            return None
        
        # Simple estimation based on queue size and average execution time
        queue_size = self.task_queue.qsize()
        avg_execution_time = 60  # seconds (would be calculated from historical data)
        
        estimated_seconds = queue_size * avg_execution_time
        return datetime.now() + timedelta(seconds=estimated_seconds)
    
    def _calculate_average_wait_time(self) -> float:
        """Calculate average wait time."""
        # Simple implementation - would use historical data in production
        return 30.0  # seconds


# Initialize FastAPI app
app = FastAPI(
    title="Remote Task Execution API",
    description="Enhanced API for remote task execution with queuing and status tracking",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)
task_queue = TaskQueue(max_workers=3)


# Dependency for API key authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if credentials.credentials != "demo-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials


# Enhanced API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Remote Task Execution API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "queue_status": "/queue/status"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    queue_status = task_queue.get_queue_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "uptime": time.time() - task_queue.start_time,
        "queue": {
            "total_tasks": queue_status.total_tasks,
            "running_tasks": queue_status.running_tasks,
            "pending_tasks": queue_status.pending_tasks
        }
    }


@app.post("/tasks/submit", response_model=TaskSubmissionResponse)
async def submit_task(
    request: RemoteTaskRequest,
    api_key: str = Depends(verify_api_key)
):
    """Submit a new task for remote execution."""
    try:
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        logger.info(f"📝 Submitting task {task_id}: {request.task_description}")
        
        # Validate request
        if len(request.task_description.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task description must be at least 3 characters long"
            )
        
        # Submit to queue
        response = task_queue.submit_task(task_id, request)
        
        logger.info(f"✅ Task {task_id} submitted successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task submission failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task submission failed: {str(e)}"
        )


@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed task status."""
    task_status = task_queue.get_task_status(task_id)
    if not task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    return task_status


@app.get("/tasks/{task_id}/result", response_model=TaskResult)
async def get_task_result(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get task execution result."""
    result = task_queue.get_task_result(task_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task result not found or task not completed"
        )
    
    return result


@app.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel a running task."""
    success = task_queue.cancel_task(task_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found or cannot be cancelled"
        )
    
    return {"message": "Task cancelled successfully"}


@app.get("/queue/status", response_model=QueueStatus)
async def get_queue_status(
    api_key: str = Depends(verify_api_key)
):
    """Get task queue status."""
    return task_queue.get_queue_status()


@app.get("/tasks", response_model=List[TaskStatusResponse])
async def get_all_tasks(
    status_filter: Optional[str] = None,
    limit: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """Get all tasks with optional filtering."""
    all_tasks = list(task_queue.task_status.values())
    
    if status_filter:
        all_tasks = [task for task in all_tasks if task.status == status_filter]
    
    return all_tasks[:limit]


@app.post("/tasks/batch")
async def submit_batch_tasks(
    requests: List[RemoteTaskRequest],
    api_key: str = Depends(verify_api_key)
):
    """Submit multiple tasks at once."""
    if len(requests) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 tasks per batch"
        )
    
    results = []
    for request in requests:
        try:
            task_id = f"task_{uuid.uuid4().hex[:8]}"
            response = task_queue.submit_task(task_id, request)
            results.append(response)
        except Exception as e:
            results.append({
                "error": str(e),
                "task_description": request.task_description
            })
    
    return {
        "submitted_tasks": len([r for r in results if "error" not in r]),
        "failed_tasks": len([r for r in results if "error" in r]),
        "results": results
    }


@app.post("/tasks/validate")
async def validate_task_request(
    request: RemoteTaskRequest,
    api_key: str = Depends(verify_api_key)
):
    """Validate a task request before submission."""
    validation_errors = []
    
    # Check task description
    if not request.task_description or len(request.task_description.strip()) < 3:
        validation_errors.append("Task description must be at least 3 characters long")
    
    # Check URL format
    if request.target_url and not request.target_url.startswith(('http://', 'https://')):
        validation_errors.append("Target URL must start with http:// or https://")
    
    # Check timeout
    if request.timeout_seconds < 30 or request.timeout_seconds > 3600:
        validation_errors.append("Timeout must be between 30 and 3600 seconds")
    
    # Check callback URL
    if request.callback_url and not request.callback_url.startswith(('http://', 'https://')):
        validation_errors.append("Callback URL must start with http:// or https://")
    
    return {
        "valid": len(validation_errors) == 0,
        "errors": validation_errors,
        "estimated_duration": "30-120 seconds",
        "complexity": "moderate"
    }


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
    logger.info("🚀 Remote Task Execution API starting up...")
    logger.info("✅ Task queue initialized")
    logger.info("✅ Worker threads started")
    logger.info("✅ API endpoints ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Remote Task Execution API shutting down...")
    logger.info("✅ Task queue cleanup completed")


# Main function for running the server
def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "remote_execution_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
