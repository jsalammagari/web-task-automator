#!/usr/bin/env python3
"""
Dynamic Task Execution Engine
============================

This module provides AI-driven dynamic task execution with plan parsing,
command execution, validation, feedback loops, and comprehensive monitoring.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dynamic_execution.log')
    ]
)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status for commands and plans."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ValidationResult(Enum):
    """Validation result for commands."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNKNOWN = "unknown"


@dataclass
class ExecutionMetrics:
    """Metrics for execution monitoring."""
    total_commands: int = 0
    successful_commands: int = 0
    failed_commands: int = 0
    retry_attempts: int = 0
    total_execution_time: float = 0.0
    average_command_time: float = 0.0
    success_rate: float = 0.0
    ai_decisions_made: int = 0
    plan_adjustments: int = 0


@dataclass
class CommandResult:
    """Result of command execution."""
    command_id: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    output_data: Optional[Any] = None
    retry_count: int = 0
    validation_result: Optional[ValidationResult] = None


@dataclass
class PlanExecutionResult:
    """Result of plan execution."""
    plan_id: str
    goal: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    execution_time: float
    success: bool
    metrics: ExecutionMetrics
    command_results: List[CommandResult]
    feedback_data: Dict[str, Any]


class CommandValidator:
    """Validates AI-generated commands."""
    
    SUPPORTED_ACTIONS = {
        "navigate", "click", "type", "wait", "scroll", "get_text", 
        "get_attribute", "refresh", "back", "forward", "hover", "double_click"
    }
    
    REQUIRED_FIELDS = {
        "navigate": ["url"],
        "click": ["selector"],
        "type": ["selector", "text"],
        "wait": ["selector"],
        "scroll": ["selector"],
        "get_text": ["selector"],
        "get_attribute": ["selector", "attribute"],
        "refresh": [],
        "back": [],
        "forward": [],
        "hover": ["selector"],
        "double_click": ["selector"]
    }
    
    def validate_command(self, command: Dict[str, Any]) -> Tuple[ValidationResult, List[str]]:
        """
        Validate an AI-generated command.
        
        Args:
            command (Dict[str, Any]): Command to validate
            
        Returns:
            Tuple[ValidationResult, List[str]]: Validation result and issues
        """
        issues = []
        
        # Check if action is supported
        action = command.get("action")
        if not action:
            issues.append("Missing 'action' field")
            return ValidationResult.INVALID, issues
        
        if action not in self.SUPPORTED_ACTIONS:
            issues.append(f"Unsupported action: {action}")
            return ValidationResult.INVALID, issues
        
        # Check required fields
        required_fields = self.REQUIRED_FIELDS.get(action, [])
        for field in required_fields:
            if field not in command or not command[field]:
                issues.append(f"Missing required field '{field}' for action '{action}'")
        
        # Validate selector format
        if "selector" in command:
            selector = command["selector"]
            if not isinstance(selector, str) or not selector.strip():
                issues.append("Invalid selector format")
        
        # Validate timeout
        if "timeout" in command:
            timeout = command["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                issues.append("Invalid timeout value")
        
        # Validate URL format
        if "url" in command:
            url = command["url"]
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                issues.append("Invalid URL format")
        
        # Determine validation result
        if not issues:
            return ValidationResult.VALID, []
        elif any("Missing required field" in issue for issue in issues):
            return ValidationResult.INVALID, issues
        else:
            return ValidationResult.WARNING, issues


class PlanParser:
    """Parses AI-generated task plans."""
    
    def __init__(self):
        self.validator = CommandValidator()
    
    def parse_plan(self, plan_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse AI-generated plan into executable commands.
        
        Args:
            plan_data (Dict[str, Any]): Plan data from AI
            
        Returns:
            List[Dict[str, Any]]: Parsed commands
        """
        try:
            commands = []
            
            if "steps" not in plan_data:
                raise ValueError("Plan missing 'steps' field")
            
            for i, step in enumerate(plan_data["steps"]):
                # Add step ID if missing
                if "step_id" not in step:
                    step["step_id"] = i + 1
                
                # Validate command
                validation_result, issues = self.validator.validate_command(step)
                
                if validation_result == ValidationResult.INVALID:
                    logger.warning(f"⚠️  Invalid command {i+1}: {', '.join(issues)}")
                    continue
                
                if validation_result == ValidationResult.WARNING:
                    logger.warning(f"⚠️  Command {i+1} warnings: {', '.join(issues)}")
                
                # Add metadata
                step["command_id"] = f"cmd_{i+1}_{int(time.time())}"
                step["validation_result"] = validation_result.value
                
                commands.append(step)
            
            logger.info(f"✅ Parsed {len(commands)} valid commands from plan")
            return commands
            
        except Exception as e:
            logger.error(f"❌ Plan parsing failed: {str(e)}")
            raise
    
    def extract_plan_metadata(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from plan."""
        return {
            "goal": plan_data.get("goal", "Unknown"),
            "estimated_duration": plan_data.get("estimated_duration", 0),
            "confidence_score": plan_data.get("confidence_score", 0),
            "total_steps": len(plan_data.get("steps", [])),
            "created_at": datetime.now().isoformat()
        }


class CommandExecutor:
    """Executes AI-generated commands on browser automation."""
    
    def __init__(self, automation_client):
        """
        Initialize command executor.
        
        Args:
            automation_client: Browser automation client
        """
        self.automation_client = automation_client
        self.execution_history: List[CommandResult] = []
    
    async def execute_command(self, command: Dict[str, Any]) -> CommandResult:
        """
        Execute a single command.
        
        Args:
            command (Dict[str, Any]): Command to execute
            
        Returns:
            CommandResult: Execution result
        """
        command_id = command.get("command_id", f"cmd_{int(time.time())}")
        action = command.get("action")
        timeout = command.get("timeout", 10000)
        
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Executing command: {action}")
            
            # Execute based on action type
            success = await self._execute_action(command, timeout)
            
            execution_time = time.time() - start_time
            
            result = CommandResult(
                command_id=command_id,
                success=success,
                execution_time=execution_time,
                validation_result=ValidationResult(command.get("validation_result", "unknown"))
            )
            
            if success:
                logger.info(f"✅ Command executed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"❌ Command execution failed")
                result.error_message = "Command execution failed"
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"❌ Command execution error: {error_msg}")
            
            result = CommandResult(
                command_id=command_id,
                success=False,
                execution_time=execution_time,
                error_message=error_msg,
                validation_result=ValidationResult(command.get("validation_result", "unknown"))
            )
            
            self.execution_history.append(result)
            return result
    
    async def _execute_action(self, command: Dict[str, Any], timeout: int) -> bool:
        """Execute specific action based on command."""
        action = command.get("action")
        
        try:
            if action == "navigate":
                return await self.automation_client.navigate_to(command["url"])
            
            elif action == "click":
                return await self.automation_client.click_element(
                    command["selector"], timeout
                )
            
            elif action == "type":
                clear_first = command.get("clear_first", True)
                return await self.automation_client.type_text(
                    command["selector"], command["text"], clear_first, timeout
                )
            
            elif action == "wait":
                state = command.get("state", "visible")
                return await self.automation_client.wait_for_element(
                    command["selector"], state, timeout
                )
            
            elif action == "scroll":
                return await self.automation_client.scroll_to_element(
                    command["selector"], timeout
                )
            
            elif action == "get_text":
                text = await self.automation_client.get_element_text(
                    command["selector"], timeout
                )
                return text is not None
            
            elif action == "get_attribute":
                value = await self.automation_client.get_element_attribute(
                    command["selector"], command["attribute"], timeout
                )
                return value is not None
            
            elif action == "refresh":
                return await self.automation_client.page_refresh()
            
            elif action == "back":
                return await self.automation_client.page_back()
            
            elif action == "forward":
                return await self.automation_client.page_forward()
            
            elif action == "hover":
                # Implement hover if supported by automation client
                return await self._hover_element(command["selector"], timeout)
            
            elif action == "double_click":
                # Implement double click if supported by automation client
                return await self._double_click_element(command["selector"], timeout)
            
            else:
                logger.warning(f"⚠️  Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Action execution failed: {str(e)}")
            return False
    
    async def _hover_element(self, selector: str, timeout: int) -> bool:
        """Hover over an element."""
        try:
            # This would need to be implemented in the automation client
            # For now, we'll simulate with a click
            return await self.automation_client.click_element(selector, timeout)
        except Exception:
            return False
    
    async def _double_click_element(self, selector: str, timeout: int) -> bool:
        """Double click an element."""
        try:
            # This would need to be implemented in the automation client
            # For now, we'll simulate with two clicks
            success1 = await self.automation_client.click_element(selector, timeout)
            await asyncio.sleep(0.1)
            success2 = await self.automation_client.click_element(selector, timeout)
            return success1 and success2
        except Exception:
            return False


class FeedbackLoop:
    """Implements feedback loop for plan adjustment."""
    
    def __init__(self, ai_task_planner):
        """
        Initialize feedback loop.
        
        Args:
            ai_task_planner: AI task planner instance
        """
        self.ai_task_planner = ai_task_planner
        self.feedback_history: List[Dict[str, Any]] = []
    
    async def analyze_execution_results(self, execution_result: PlanExecutionResult) -> Dict[str, Any]:
        """
        Analyze execution results and provide feedback.
        
        Args:
            execution_result (PlanExecutionResult): Execution results
            
        Returns:
            Dict[str, Any]: Feedback analysis
        """
        try:
            feedback = {
                "execution_id": execution_result.plan_id,
                "success_rate": execution_result.metrics.success_rate,
                "failed_commands": execution_result.failed_steps,
                "recommendations": [],
                "plan_adjustments": []
            }
            
            # Analyze failed commands
            failed_commands = [
                result for result in execution_result.command_results
                if not result.success
            ]
            
            if failed_commands:
                feedback["recommendations"].append("Review failed commands for selector issues")
                
                # Suggest plan adjustments
                for failed_cmd in failed_commands:
                    if "selector" in failed_cmd.command_id:
                        feedback["plan_adjustments"].append({
                            "type": "selector_update",
                            "command_id": failed_cmd.command_id,
                            "suggestion": "Update selector or add wait conditions"
                        })
            
            # Analyze execution time
            if execution_result.metrics.average_command_time > 5.0:
                feedback["recommendations"].append("Consider increasing timeouts for slow operations")
            
            # Analyze success patterns
            if execution_result.metrics.success_rate > 0.8:
                feedback["recommendations"].append("Plan execution was mostly successful")
            else:
                feedback["recommendations"].append("Plan needs significant adjustments")
            
            self.feedback_history.append(feedback)
            logger.info(f"📊 Feedback analysis completed: {len(feedback['recommendations'])} recommendations")
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Feedback analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def suggest_plan_improvements(self, original_plan: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest improvements to the original plan.
        
        Args:
            original_plan (Dict[str, Any]): Original plan
            feedback (Dict[str, Any]): Feedback analysis
            
        Returns:
            Dict[str, Any]: Improved plan suggestions
        """
        try:
            improvements = {
                "original_plan": original_plan,
                "feedback": feedback,
                "suggested_changes": [],
                "improved_plan": original_plan.copy()
            }
            
            # Add timeout improvements
            for step in improvements["improved_plan"].get("steps", []):
                if "timeout" not in step or step["timeout"] < 5000:
                    step["timeout"] = 10000
                    improvements["suggested_changes"].append({
                        "type": "timeout_increase",
                        "step_id": step.get("step_id"),
                        "change": "Increased timeout to 10 seconds"
                    })
            
            # Add wait conditions for critical steps
            for step in improvements["improved_plan"].get("steps", []):
                if step.get("action") in ["click", "type"] and "wait" not in step:
                    step["wait_for"] = step.get("selector")
                    improvements["suggested_changes"].append({
                        "type": "wait_condition",
                        "step_id": step.get("step_id"),
                        "change": "Added wait condition before action"
                    })
            
            logger.info(f"🔧 Plan improvements suggested: {len(improvements['suggested_changes'])} changes")
            return improvements
            
        except Exception as e:
            logger.error(f"❌ Plan improvement suggestion failed: {str(e)}")
            return {"error": str(e)}


class ExecutionMonitor:
    """Monitors and logs AI decision-making and execution."""
    
    def __init__(self):
        self.metrics = ExecutionMetrics()
        self.decision_log: List[Dict[str, Any]] = []
        self.performance_log: List[Dict[str, Any]] = []
    
    def log_ai_decision(self, decision_type: str, context: Dict[str, Any], decision: Any) -> None:
        """Log AI decision-making process."""
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "context": context,
            "decision": decision,
            "metrics": asdict(self.metrics)
        }
        
        self.decision_log.append(decision_entry)
        self.metrics.ai_decisions_made += 1
        
        logger.info(f"🧠 AI Decision: {decision_type} - {decision}")
    
    def log_execution_performance(self, command_result: CommandResult) -> None:
        """Log execution performance metrics."""
        performance_entry = {
            "timestamp": datetime.now().isoformat(),
            "command_id": command_result.command_id,
            "success": command_result.success,
            "execution_time": command_result.execution_time,
            "retry_count": command_result.retry_count
        }
        
        self.performance_log.append(performance_entry)
        
        # Update metrics
        self.metrics.total_commands += 1
        self.metrics.total_execution_time += command_result.execution_time
        
        if command_result.success:
            self.metrics.successful_commands += 1
        else:
            self.metrics.failed_commands += 1
        
        if command_result.retry_count > 0:
            self.metrics.retry_attempts += command_result.retry_count
        
        # Calculate success rate
        if self.metrics.total_commands > 0:
            self.metrics.success_rate = self.metrics.successful_commands / self.metrics.total_commands
        
        # Calculate average command time
        if self.metrics.total_commands > 0:
            self.metrics.average_command_time = self.metrics.total_execution_time / self.metrics.total_commands
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with metrics."""
        return {
            "metrics": asdict(self.metrics),
            "total_decisions": len(self.decision_log),
            "total_performance_entries": len(self.performance_log),
            "recent_decisions": self.decision_log[-5:] if self.decision_log else [],
            "recent_performance": self.performance_log[-5:] if self.performance_log else []
        }
    
    def save_monitoring_data(self, filename: str) -> None:
        """Save monitoring data to file."""
        try:
            monitoring_data = {
                "execution_summary": self.get_execution_summary(),
                "decision_log": self.decision_log,
                "performance_log": self.performance_log,
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(monitoring_data, f, indent=2, default=str)
            
            logger.info(f"💾 Monitoring data saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save monitoring data: {str(e)}")


class DynamicExecutionEngine:
    """Main dynamic task execution engine."""
    
    def __init__(self, automation_client=None, ai_task_planner=None):
        """
        Initialize dynamic execution engine.
        
        Args:
            automation_client: Browser automation client (optional)
            ai_task_planner: AI task planner instance (optional)
        """
        self.automation_client = automation_client
        self.ai_task_planner = ai_task_planner
        
        # Initialize components
        self.plan_parser = PlanParser()
        self.command_executor = CommandExecutor(automation_client)
        self.feedback_loop = FeedbackLoop(ai_task_planner)
        self.monitor = ExecutionMonitor()
        
        # Execution state
        self.current_execution: Optional[PlanExecutionResult] = None
        self.execution_history: List[PlanExecutionResult] = []
    
    async def execute_ai_plan(self, plan_data: Dict[str, Any], context: Dict[str, Any]) -> PlanExecutionResult:
        """
        Execute an AI-generated plan dynamically.
        
        Args:
            plan_data (Dict[str, Any]): AI-generated plan
            context (Dict[str, Any]): Webpage context
            
        Returns:
            PlanExecutionResult: Execution results
        """
        try:
            plan_id = f"plan_{int(time.time())}"
            logger.info(f"🚀 Starting dynamic execution of plan: {plan_id}")
            
            # Parse plan
            commands = self.plan_parser.parse_plan(plan_data)
            plan_metadata = self.plan_parser.extract_plan_metadata(plan_data)
            
            # Log AI decision
            self.monitor.log_ai_decision(
                "plan_execution",
                {"plan_id": plan_id, "total_commands": len(commands)},
                f"Executing {len(commands)} commands"
            )
            
            # Execute commands
            command_results = []
            start_time = time.time()
            
            for i, command in enumerate(commands):
                logger.info(f"🔄 Executing command {i+1}/{len(commands)}: {command.get('action')}")
                
                # Execute command
                result = await self.command_executor.execute_command(command)
                command_results.append(result)
                
                # Log performance
                self.monitor.log_execution_performance(result)
                
                # Handle failures with retry logic
                if not result.success and command.get("retry_count", 0) < 3:
                    logger.warning(f"⚠️  Command failed, retrying...")
                    command["retry_count"] = command.get("retry_count", 0) + 1
                    result.retry_count = command["retry_count"]
                    
                    # Wait before retry
                    await asyncio.sleep(1)
                    
                    # Retry execution
                    retry_result = await self.command_executor.execute_command(command)
                    command_results[-1] = retry_result  # Replace the failed result
                    self.monitor.log_execution_performance(retry_result)
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            successful_commands = sum(1 for r in command_results if r.success)
            failed_commands = len(command_results) - successful_commands
            
            metrics = ExecutionMetrics(
                total_commands=len(command_results),
                successful_commands=successful_commands,
                failed_commands=failed_commands,
                total_execution_time=execution_time,
                average_command_time=execution_time / len(command_results) if command_results else 0,
                success_rate=successful_commands / len(command_results) if command_results else 0
            )
            
            # Create execution result
            execution_result = PlanExecutionResult(
                plan_id=plan_id,
                goal=plan_metadata.get("goal", "Unknown"),
                total_steps=len(commands),
                completed_steps=successful_commands,
                failed_steps=failed_commands,
                execution_time=execution_time,
                success=failed_commands == 0,
                metrics=metrics,
                command_results=command_results,
                feedback_data={}
            )
            
            # Analyze results and provide feedback
            feedback = await self.feedback_loop.analyze_execution_results(execution_result)
            execution_result.feedback_data = feedback
            
            # Log AI decision for feedback
            self.monitor.log_ai_decision(
                "feedback_analysis",
                {"success_rate": metrics.success_rate, "failed_commands": failed_commands},
                f"Feedback: {len(feedback.get('recommendations', []))} recommendations"
            )
            
            # Store execution result
            self.current_execution = execution_result
            self.execution_history.append(execution_result)
            
            logger.info(f"✅ Dynamic execution completed: {successful_commands}/{len(commands)} commands successful")
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Dynamic execution failed: {str(e)}")
            # Create failed execution result
            return PlanExecutionResult(
                plan_id=plan_id if 'plan_id' in locals() else f"plan_{int(time.time())}",
                goal="Unknown",
                total_steps=0,
                completed_steps=0,
                failed_steps=1,
                execution_time=0,
                success=False,
                metrics=ExecutionMetrics(),
                command_results=[],
                feedback_data={"error": str(e)}
            )
    
    async def execute_with_adaptation(self, plan_data: Dict[str, Any], context: Dict[str, Any]) -> PlanExecutionResult:
        """
        Execute plan with dynamic adaptation based on feedback.
        
        Args:
            plan_data (Dict[str, Any]): AI-generated plan
            context (Dict[str, Any]): Webpage context
            
        Returns:
            PlanExecutionResult: Execution results with adaptations
        """
        try:
            # Initial execution
            initial_result = await self.execute_ai_plan(plan_data, context)
            
            # If execution failed significantly, try to adapt
            if initial_result.metrics.success_rate < 0.5:
                logger.info("🔄 Low success rate, attempting plan adaptation...")
                
                # Get feedback and suggestions
                feedback = initial_result.feedback_data
                improvements = await self.feedback_loop.suggest_plan_improvements(plan_data, feedback)
                
                # Log adaptation decision
                self.monitor.log_ai_decision(
                    "plan_adaptation",
                    {"original_success_rate": initial_result.metrics.success_rate},
                    f"Adapting plan with {len(improvements.get('suggested_changes', []))} changes"
                )
                
                # Execute improved plan if available
                if improvements.get("improved_plan"):
                    adapted_result = await self.execute_ai_plan(improvements["improved_plan"], context)
                    adapted_result.feedback_data["adaptation_applied"] = True
                    adapted_result.feedback_data["original_result"] = asdict(initial_result)
                    
                    return adapted_result
            
            return initial_result
            
        except Exception as e:
            logger.error(f"❌ Adaptive execution failed: {str(e)}")
            return initial_result if 'initial_result' in locals() else await self.execute_ai_plan(plan_data, context)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        return {
            "current_execution": asdict(self.current_execution) if self.current_execution else None,
            "execution_history": [asdict(result) for result in self.execution_history],
            "monitoring_summary": self.monitor.get_execution_summary(),
            "total_executions": len(self.execution_history)
        }
    
    def save_execution_report(self, filename: str) -> None:
        """Save comprehensive execution report."""
        try:
            report = {
                "execution_summary": self.get_execution_summary(),
                "monitoring_data": {
                    "decision_log": self.monitor.decision_log,
                    "performance_log": self.monitor.performance_log
                },
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Execution report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save execution report: {str(e)}")


async def main():
    """Demo function for Dynamic Execution Engine."""
    try:
        # This would be used with actual automation client and AI task planner
        logger.info("🚀 Dynamic Execution Engine Demo")
        logger.info("This module provides AI-driven dynamic task execution")
        logger.info("with plan parsing, command execution, validation,")
        logger.info("feedback loops, and comprehensive monitoring.")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
