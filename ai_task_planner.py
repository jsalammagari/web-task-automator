#!/usr/bin/env python3
"""
AI Task Planner Integration
==========================

This module provides AI-driven task planning using Language Models (LLMs)
to convert natural language goals into structured automation commands.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import aiohttp
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_task_planner.log')
    ]
)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AutomationCommand:
    """Represents a single automation command."""
    action: str
    selector: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    timeout: Optional[int] = None
    wait_for: Optional[str] = None
    description: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskStep:
    """Represents a step in a task plan."""
    step_id: int
    command: AutomationCommand
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class TaskPlan:
    """Represents a complete task plan."""
    goal: str
    steps: List[TaskStep]
    estimated_duration: Optional[float] = None
    confidence_score: Optional[float] = None
    created_at: Optional[str] = None


class PromptTemplates:
    """Templates for LLM prompts."""
    
    TASK_PLANNING_PROMPT = """
You are an expert web automation specialist. Given a user's goal and webpage context, create a detailed step-by-step automation plan.

User Goal: {goal}

Webpage Context:
{context}

Available Actions:
- navigate: Go to a URL
- click: Click an element
- type: Type text into an input field
- wait: Wait for an element to appear
- scroll: Scroll to an element
- get_text: Extract text from an element
- get_attribute: Get an attribute value
- refresh: Refresh the page
- back: Go back in browser history
- forward: Go forward in browser history

Response Format (JSON):
{{
    "goal": "User's original goal",
    "steps": [
        {{
            "step_id": 1,
            "action": "navigate",
            "url": "https://example.com",
            "description": "Navigate to the website",
            "timeout": 10000
        }},
        {{
            "step_id": 2,
            "action": "click",
            "selector": "button#submit",
            "description": "Click the submit button",
            "timeout": 5000
        }},
        {{
            "step_id": 3,
            "action": "type",
            "selector": "input[name='email']",
            "text": "user@example.com",
            "description": "Enter email address",
            "timeout": 5000
        }}
    ],
    "estimated_duration": 30.5,
    "confidence_score": 0.95
}}

Instructions:
1. Analyze the webpage context to understand available elements
2. Create logical steps to achieve the user's goal
3. Use appropriate selectors based on the context
4. Include timeouts and error handling
5. Provide clear descriptions for each step
6. Estimate execution time and confidence score
7. Return ONLY valid JSON, no additional text
"""

    CONTEXT_ENHANCEMENT_PROMPT = """
Enhance the following webpage context for better task planning:

Original Context:
{context}

Enhancement Instructions:
1. Identify interactive elements (buttons, links, forms)
2. Highlight important text content
3. Note element relationships and hierarchy
4. Identify potential navigation paths
5. Flag any accessibility features
6. Suggest optimal selectors for automation

Return enhanced context in JSON format:
{{
    "enhanced_elements": [
        {{
            "selector": "button#submit",
            "type": "interactive",
            "description": "Submit button for form",
            "priority": "high",
            "suggested_actions": ["click"]
        }}
    ],
    "navigation_suggestions": ["Next page", "Previous page"],
    "form_elements": ["email", "password", "name"],
    "accessibility_features": ["aria-labels", "roles"]
}}
"""

    ERROR_RECOVERY_PROMPT = """
Task execution failed. Analyze the error and suggest recovery steps.

Original Goal: {goal}
Failed Step: {failed_step}
Error: {error}
Current Context: {context}

Provide recovery options in JSON format:
{{
    "recovery_options": [
        {{
            "option": "retry_with_different_selector",
            "new_selector": "button[type='submit']",
            "description": "Try alternative selector"
        }},
        {{
            "option": "wait_and_retry",
            "wait_time": 3000,
            "description": "Wait for element to load"
        }},
        {{
            "option": "alternative_approach",
            "steps": [
                {{
                    "action": "scroll",
                    "selector": "button#submit",
                    "description": "Scroll to button first"
                }}
            ]
        }}
    ],
    "recommended_option": "retry_with_different_selector"
}}
"""


class LLMClient:
    """Client for interacting with Language Models."""
    
    def __init__(self, provider: LLMProvider, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider (LLMProvider): LLM provider to use
            api_key (str, optional): API key for the provider
            model (str, optional): Model name to use
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY")
        self.model = model or self._get_default_model()
        self.client = None
        
        self._initialize_client()
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            LLMProvider.LOCAL: "llama-2-7b"
        }
        return defaults.get(self.provider, "gpt-4")
    
    def _initialize_client(self) -> None:
        """Initialize the LLM client."""
        try:
            if self.provider == LLMProvider.OPENAI:
                if not self.api_key:
                    raise ValueError("OpenAI API key required")
                openai.api_key = self.api_key
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
                
            elif self.provider == LLMProvider.ANTHROPIC:
                if not self.api_key:
                    raise ValueError("Anthropic API key required")
                self.client = Anthropic(api_key=self.api_key)
                
            elif self.provider == LLMProvider.LOCAL:
                # For local models, we'll use a mock client
                self.client = LocalLLMClient()
                
            logger.info(f"✅ LLM client initialized: {self.provider.value} - {self.model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {str(e)}")
            raise
    
    async def generate_response(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: LLM response
        """
        try:
            if self.provider == LLMProvider.OPENAI:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content
                
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.provider == LLMProvider.LOCAL:
                return await self.client.generate_response(prompt, max_tokens)
                
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {str(e)}")
            raise


class LocalLLMClient:
    """Mock client for local LLM models."""
    
    async def generate_response(self, prompt: str, max_tokens: int) -> str:
        """Generate mock response for local models."""
        # This is a mock implementation
        # In a real implementation, you would integrate with local models like Ollama, etc.
        return json.dumps({
            "goal": "Mock task",
            "steps": [
                {
                    "step_id": 1,
                    "action": "navigate",
                    "url": "https://example.com",
                    "description": "Mock navigation step"
                }
            ],
            "estimated_duration": 10.0,
            "confidence_score": 0.8
        })


class AITaskPlanner:
    """AI-powered task planner for web automation."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize AI Task Planner.
        
        Args:
            llm_client (LLMClient): LLM client instance
        """
        self.llm_client = llm_client
        self.templates = PromptTemplates()
        self.task_history: List[TaskPlan] = []
    
    async def create_task_plan(self, goal: str, context: Dict[str, Any]) -> TaskPlan:
        """
        Create a task plan from a natural language goal.
        
        Args:
            goal (str): User's goal in natural language
            context (Dict[str, Any]): Webpage context from MCP
            
        Returns:
            TaskPlan: Generated task plan
        """
        try:
            logger.info(f"🎯 Creating task plan for goal: {goal}")
            
            # Format context for LLM
            context_str = self._format_context_for_llm(context)
            
            # Create prompt
            prompt = self.templates.TASK_PLANNING_PROMPT.format(
                goal=goal,
                context=context_str
            )
            
            # Get LLM response
            response = await self.llm_client.generate_response(prompt)
            
            # Parse and validate response
            plan_data = self._parse_llm_response(response)
            
            # Create task plan
            task_plan = self._create_task_plan_from_data(goal, plan_data)
            
            # Store in history
            self.task_history.append(task_plan)
            
            logger.info(f"✅ Task plan created with {len(task_plan.steps)} steps")
            return task_plan
            
        except Exception as e:
            logger.error(f"❌ Failed to create task plan: {str(e)}")
            raise
    
    def _format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """Format webpage context for LLM consumption."""
        try:
            elements = context.get("elements", [])
            page_info = context.get("page", {})
            
            context_str = f"Page: {page_info.get('title', 'Unknown')} ({page_info.get('url', 'Unknown')})\n"
            context_str += f"Viewport: {page_info.get('viewport', {})}\n\n"
            
            context_str += "Available Elements:\n"
            for elem in elements[:20]:  # Limit to first 20 elements
                context_str += f"- {elem.get('tag', 'unknown')}: {elem.get('text', '')[:50]}...\n"
                if elem.get('role'):
                    context_str += f"  Role: {elem['role']}\n"
                if elem.get('selector'):
                    context_str += f"  Selector: {elem['selector']}\n"
            
            if context.get("accessibility"):
                context_str += f"\nAccessibility: {context['accessibility'].get('role', 'unknown')}\n"
            
            return context_str
            
        except Exception as e:
            logger.error(f"❌ Error formatting context: {str(e)}")
            return "Context formatting failed"
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            # Clean response (remove markdown formatting if present)
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON
            plan_data = json.loads(cleaned_response)
            
            # Validate required fields
            if "steps" not in plan_data:
                raise ValueError("Missing 'steps' in LLM response")
            
            if not isinstance(plan_data["steps"], list):
                raise ValueError("'steps' must be a list")
            
            # Validate each step
            for i, step in enumerate(plan_data["steps"]):
                if "action" not in step:
                    raise ValueError(f"Step {i} missing 'action'")
                if "step_id" not in step:
                    step["step_id"] = i + 1
            
            return plan_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {str(e)}")
            raise ValueError(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Response parsing failed: {str(e)}")
            raise
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response to extract valid JSON."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON object boundaries
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON object found in response")
        
        return response[start_idx:end_idx + 1]
    
    def _create_task_plan_from_data(self, goal: str, plan_data: Dict[str, Any]) -> TaskPlan:
        """Create TaskPlan object from parsed data."""
        steps = []
        
        for step_data in plan_data["steps"]:
            command = AutomationCommand(
                action=step_data["action"],
                selector=step_data.get("selector"),
                text=step_data.get("text"),
                url=step_data.get("url"),
                timeout=step_data.get("timeout", 10000),
                wait_for=step_data.get("wait_for"),
                description=step_data.get("description"),
                max_retries=step_data.get("max_retries", 3)
            )
            
            task_step = TaskStep(
                step_id=step_data["step_id"],
                command=command,
                status=TaskStatus.PENDING
            )
            
            steps.append(task_step)
        
        return TaskPlan(
            goal=goal,
            steps=steps,
            estimated_duration=plan_data.get("estimated_duration"),
            confidence_score=plan_data.get("confidence_score"),
            created_at=str(asyncio.get_event_loop().time())
        )
    
    async def enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance webpage context using LLM.
        
        Args:
            context (Dict[str, Any]): Original context
            
        Returns:
            Dict[str, Any]: Enhanced context
        """
        try:
            context_str = self._format_context_for_llm(context)
            
            prompt = self.templates.CONTEXT_ENHANCEMENT_PROMPT.format(
                context=context_str
            )
            
            response = await self.llm_client.generate_response(prompt)
            enhanced_data = json.loads(self._clean_json_response(response))
            
            # Merge with original context
            enhanced_context = context.copy()
            enhanced_context.update(enhanced_data)
            
            logger.info("✅ Context enhanced with LLM insights")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"❌ Context enhancement failed: {str(e)}")
            return context  # Return original context if enhancement fails
    
    async def suggest_recovery(self, goal: str, failed_step: TaskStep, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest recovery options for failed tasks.
        
        Args:
            goal (str): Original goal
            failed_step (TaskStep): The step that failed
            error (str): Error message
            context (Dict[str, Any]): Current context
            
        Returns:
            Dict[str, Any]: Recovery suggestions
        """
        try:
            context_str = self._format_context_for_llm(context)
            
            prompt = self.templates.ERROR_RECOVERY_PROMPT.format(
                goal=goal,
                failed_step=json.dumps(asdict(failed_step), indent=2),
                error=error,
                context=context_str
            )
            
            response = await self.llm_client.generate_response(prompt)
            recovery_data = json.loads(self._clean_json_response(response))
            
            logger.info("✅ Recovery options generated")
            return recovery_data
            
        except Exception as e:
            logger.error(f"❌ Recovery suggestion failed: {str(e)}")
            return {"recovery_options": [], "recommended_option": "manual_intervention"}
    
    def get_task_history(self) -> List[TaskPlan]:
        """Get task execution history."""
        return self.task_history.copy()
    
    def save_task_plan(self, task_plan: TaskPlan, filename: str) -> None:
        """Save task plan to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(asdict(task_plan), f, indent=2, default=str)
            logger.info(f"💾 Task plan saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save task plan: {str(e)}")


class AITaskExecutor:
    """Execute AI-generated task plans."""
    
    def __init__(self, task_planner: AITaskPlanner, automation_client):
        """
        Initialize AI Task Executor.
        
        Args:
            task_planner (AITaskPlanner): Task planner instance
            automation_client: Browser automation client
        """
        self.task_planner = task_planner
        self.automation_client = automation_client
        self.current_task: Optional[TaskPlan] = None
    
    async def execute_task_plan(self, task_plan: TaskPlan) -> Dict[str, Any]:
        """
        Execute a complete task plan.
        
        Args:
            task_plan (TaskPlan): Task plan to execute
            
        Returns:
            Dict[str, Any]: Execution results
        """
        try:
            self.current_task = task_plan
            results = {
                "task_id": id(task_plan),
                "goal": task_plan.goal,
                "total_steps": len(task_plan.steps),
                "completed_steps": 0,
                "failed_steps": 0,
                "execution_time": 0,
                "steps_results": []
            }
            
            start_time = asyncio.get_event_loop().time()
            
            for step in task_plan.steps:
                step_result = await self._execute_step(step)
                results["steps_results"].append(step_result)
                
                if step_result["success"]:
                    results["completed_steps"] += 1
                else:
                    results["failed_steps"] += 1
                    # Try to get recovery suggestions
                    if self.current_task:
                        recovery = await self.task_planner.suggest_recovery(
                            task_plan.goal, step, step_result["error"], {}
                        )
                        step_result["recovery_options"] = recovery
            
            results["execution_time"] = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"✅ Task execution completed: {results['completed_steps']}/{results['total_steps']} steps")
            return results
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {str(e)}")
            return {"error": str(e), "success": False}
    
    async def _execute_step(self, step: TaskStep) -> Dict[str, Any]:
        """Execute a single task step."""
        try:
            step.status = TaskStatus.IN_PROGRESS
            start_time = asyncio.get_event_loop().time()
            
            # Execute the command
            success = await self._execute_command(step.command)
            
            step.execution_time = asyncio.get_event_loop().time() - start_time
            
            if success:
                step.status = TaskStatus.COMPLETED
                return {
                    "step_id": step.step_id,
                    "success": True,
                    "execution_time": step.execution_time,
                    "description": step.command.description
                }
            else:
                step.status = TaskStatus.FAILED
                return {
                    "step_id": step.step_id,
                    "success": False,
                    "error": "Command execution failed",
                    "execution_time": step.execution_time,
                    "description": step.command.description
                }
                
        except Exception as e:
            step.status = TaskStatus.FAILED
            step.error_message = str(e)
            return {
                "step_id": step.step_id,
                "success": False,
                "error": str(e),
                "execution_time": step.execution_time or 0,
                "description": step.command.description
            }
    
    async def _execute_command(self, command: AutomationCommand) -> bool:
        """Execute a single automation command."""
        try:
            if command.action == "navigate":
                return await self.automation_client.navigate_to(command.url)
            elif command.action == "click":
                return await self.automation_client.click_element(command.selector, command.timeout)
            elif command.action == "type":
                return await self.automation_client.type_text(command.selector, command.text, timeout=command.timeout)
            elif command.action == "wait":
                return await self.automation_client.wait_for_element(command.wait_for, timeout=command.timeout)
            elif command.action == "scroll":
                return await self.automation_client.scroll_to_element(command.selector, command.timeout)
            elif command.action == "get_text":
                text = await self.automation_client.get_element_text(command.selector, command.timeout)
                return text is not None
            elif command.action == "refresh":
                return await self.automation_client.page_refresh()
            elif command.action == "back":
                return await self.automation_client.page_back()
            elif command.action == "forward":
                return await self.automation_client.page_forward()
            else:
                logger.warning(f"⚠️  Unknown action: {command.action}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Command execution failed: {str(e)}")
            return False


async def main():
    """Demo function for AI Task Planner."""
    try:
        # Initialize LLM client (using OpenAI as example)
        llm_client = LLMClient(LLMProvider.OPENAI)
        
        # Initialize task planner
        task_planner = AITaskPlanner(llm_client)
        
        # Mock context
        context = {
            "page": {"title": "Example Page", "url": "https://example.com"},
            "elements": [
                {"tag": "button", "text": "Submit", "selector": "button#submit", "role": "button"},
                {"tag": "input", "text": "", "selector": "input[name='email']", "role": "textbox"}
            ],
            "accessibility": {"role": "main", "name": "Main Content"}
        }
        
        # Create task plan
        goal = "Fill out the contact form with my email and submit it"
        task_plan = await task_planner.create_task_plan(goal, context)
        
        print(f"✅ Task plan created with {len(task_plan.steps)} steps")
        for step in task_plan.steps:
            print(f"  Step {step.step_id}: {step.command.action} - {step.command.description}")
        
        # Save task plan
        task_planner.save_task_plan(task_plan, "ai_task_plan.json")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
