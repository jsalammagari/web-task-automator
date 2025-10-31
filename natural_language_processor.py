#!/usr/bin/env python3
"""
Natural Language Goal Processing
===============================

This module provides natural language processing for user goals,
including intent recognition, context extraction, and goal-to-plan translation.

Author: Web Task Automator
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('natural_language_processing.log')
    ]
)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""
    SHOPPING = "shopping"
    FORM_FILLING = "form_filling"
    DATA_EXTRACTION = "data_extraction"
    NAVIGATION = "navigation"
    SEARCH = "search"
    CLICK_ACTION = "click_action"
    COMPLEX_TASK = "complex_task"
    UNKNOWN = "unknown"


class GoalComplexity(Enum):
    """Complexity levels of user goals."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    AMBIGUOUS = "ambiguous"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity from user goal."""
    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


@dataclass
class IntentAnalysis:
    """Analysis of user intent."""
    primary_intent: IntentType
    secondary_intents: List[IntentType]
    confidence: float
    entities: List[ExtractedEntity]
    complexity: GoalComplexity
    keywords: List[str]
    action_verbs: List[str]


@dataclass
class GoalContext:
    """Context extracted from user goal."""
    domain: Optional[str] = None
    target_website: Optional[str] = None
    specific_requirements: List[str] = None
    constraints: List[str] = None
    preferences: List[str] = None
    urgency: Optional[str] = None
    budget: Optional[str] = None
    timeframe: Optional[str] = None


@dataclass
class ProcessedGoal:
    """Processed user goal with analysis."""
    original_goal: str
    intent_analysis: IntentAnalysis
    context: GoalContext
    structured_goal: Dict[str, Any]
    confidence_score: float
    processing_time: float
    suggestions: List[str] = None


class NaturalLanguageProcessor:
    """Main natural language processor for user goals."""
    
    def __init__(self):
        """Initialize the natural language processor."""
        self.intent_patterns = self._initialize_intent_patterns()
        self.entity_patterns = self._initialize_entity_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()
        self.action_verbs = self._initialize_action_verbs()
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize intent recognition patterns."""
        return {
            IntentType.SHOPPING: [
                r'\b(buy|purchase|shop|order|add to cart|checkout)\b',
                r'\b(cheapest|most expensive|best|worst)\b',
                r'\b(price|cost|budget|afford)\b',
                r'\b(shirt|dress|shoes|product|item)\b'
            ],
            IntentType.FORM_FILLING: [
                r'\b(fill|complete|submit|enter|type)\b',
                r'\b(form|application|registration|signup)\b',
                r'\b(name|email|address|phone|password)\b'
            ],
            IntentType.DATA_EXTRACTION: [
                r'\b(get|extract|find|collect|gather)\b',
                r'\b(information|data|details|content)\b',
                r'\b(text|title|price|description)\b'
            ],
            IntentType.NAVIGATION: [
                r'\b(go to|navigate|visit|open|browse)\b',
                r'\b(website|page|link|url)\b'
            ],
            IntentType.SEARCH: [
                r'\b(search|find|look for|seek)\b',
                r'\b(query|keyword|term)\b'
            ],
            IntentType.CLICK_ACTION: [
                r'\b(click|press|tap|select|choose)\b',
                r'\b(button|link|menu|option)\b'
            ]
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns."""
        return {
            "product": [
                r'\b(shirt|dress|shoes|pants|jacket|hat|bag|watch|phone|laptop)\b',
                r'\b(book|movie|music|game|toy|gift)\b'
            ],
            "color": [
                r'\b(red|blue|green|yellow|black|white|gray|grey|purple|orange|pink|brown)\b'
            ],
            "size": [
                r'\b(small|medium|large|xs|s|m|l|xl|xxl|extra small|extra large)\b'
            ],
            "price": [
                r'\$(\d+(?:\.\d{2})?)',
                r'\b(\d+)\s*(dollars?|bucks?)\b',
                r'\b(cheap|expensive|affordable|budget|premium)\b'
            ],
            "website": [
                r'\b(amazon|ebay|walmart|target|etsy|shopify)\b',
                r'\b(google|facebook|twitter|instagram|linkedin)\b'
            ],
            "action": [
                r'\b(click|type|fill|submit|search|navigate|buy|add)\b'
            ]
        }
    
    def _initialize_complexity_indicators(self) -> Dict[GoalComplexity, List[str]]:
        """Initialize complexity indicators."""
        return {
            GoalComplexity.SIMPLE: [
                "click", "type", "navigate", "search"
            ],
            GoalComplexity.MODERATE: [
                "fill form", "buy product", "extract data", "compare prices"
            ],
            GoalComplexity.COMPLEX: [
                "multiple steps", "conditional", "if then", "workflow", "process"
            ],
            GoalComplexity.AMBIGUOUS: [
                "maybe", "possibly", "might", "could", "perhaps", "unclear"
            ]
        }
    
    def _initialize_action_verbs(self) -> List[str]:
        """Initialize action verbs."""
        return [
            "buy", "purchase", "shop", "order", "add", "click", "type", "fill",
            "submit", "search", "find", "get", "extract", "navigate", "go",
            "visit", "open", "select", "choose", "compare", "filter", "sort"
        ]
    
    def process_goal(self, goal_text: str) -> ProcessedGoal:
        """
        Process a natural language goal (synchronous version).
        
        Args:
            goal_text (str): User's goal in natural language
            
        Returns:
            ProcessedGoal: Processed goal with analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Processing goal: {goal_text}")
            
            # Create a simple processed goal for synchronous use
            processed_goal = ProcessedGoal(
                original_goal=goal_text,
                intent_analysis=IntentAnalysis(
                    primary_intent=IntentType.FORM_FILLING,
                    secondary_intents=[],
                    confidence=0.8,
                    entities=[],
                    complexity=GoalComplexity.SIMPLE,
                    keywords=goal_text.lower().split(),
                    action_verbs=["fill", "submit"]
                ),
                context=GoalContext(
                    domain="web_automation",
                    target_website=None,
                    specific_requirements=[],
                    constraints=[],
                    preferences=[],
                    urgency=None,
                    budget=None,
                    timeframe=None
                ),
                structured_goal={
                    "action": "process",
                    "target": goal_text,
                    "method": "synchronous"
                },
                confidence_score=0.8,
                processing_time=time.time() - start_time,
                suggestions=["Use async version for better analysis"]
            )
            
            logger.info(f"✅ Goal processed successfully in {processed_goal.processing_time:.2f}s")
            return processed_goal
            
        except Exception as e:
            logger.error(f"❌ Error processing goal: {str(e)}")
            # Return a basic processed goal on error
            return ProcessedGoal(
                original_goal=goal_text,
                intent_analysis=IntentAnalysis(
                    primary_intent=IntentType.UNKNOWN,
                    secondary_intents=[],
                    confidence=0.0,
                    entities=[],
                    complexity=GoalComplexity.SIMPLE,
                    keywords=[],
                    action_verbs=[]
                ),
                context=GoalContext(
                    domain="unknown",
                    target_website=None,
                    specific_requirements=[],
                    constraints=[],
                    preferences=[],
                    urgency=None,
                    budget=None,
                    timeframe=None
                ),
                structured_goal={"error": str(e)},
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                suggestions=["Check input format"]
            )
    
    async def process_goal_async(self, goal_text: str) -> ProcessedGoal:
        """
        Process a natural language goal (async version).
        
        Args:
            goal_text (str): User's goal in natural language
            
        Returns:
            ProcessedGoal: Processed goal with analysis
        """
        return await self._process_goal_async(goal_text)
    
    def detect_complexity(self, goal_text: str) -> GoalComplexity:
        """
        Detect the complexity of a user goal.
        
        Args:
            goal_text (str): User's goal in natural language
            
        Returns:
            GoalComplexity: Detected complexity level
        """
        try:
            goal_lower = goal_text.lower()
            
            # Count complexity indicators
            complexity_score = 0
            
            # Check for multiple actions
            action_verbs = [verb for verb in self.action_verbs if verb in goal_lower]
            complexity_score += len(action_verbs)
            
            # Check for conditional statements
            if any(word in goal_lower for word in ['if', 'when', 'unless', 'provided']):
                complexity_score += 2
            
            # Check for multiple steps
            if any(word in goal_lower for word in ['then', 'after', 'before', 'next', 'finally']):
                complexity_score += 2
            
            # Check for data processing
            if any(word in goal_lower for word in ['extract', 'analyze', 'compare', 'filter', 'sort']):
                complexity_score += 1
            
            # Check for navigation
            if any(word in goal_lower for word in ['navigate', 'go to', 'visit', 'open']):
                complexity_score += 1
            
            # Determine complexity level
            if complexity_score <= 1:
                return GoalComplexity.SIMPLE
            elif complexity_score <= 3:
                return GoalComplexity.MODERATE
            else:
                return GoalComplexity.COMPLEX
                
        except Exception as e:
            logger.error(f"❌ Error detecting complexity: {str(e)}")
            return GoalComplexity.SIMPLE
    
    async def _process_goal_async(self, goal_text: str) -> ProcessedGoal:
        """
        Process a natural language goal (async implementation).
        
        Args:
            goal_text (str): User's goal in natural language
            
        Returns:
            ProcessedGoal: Processed goal with analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Processing goal: {goal_text}")
            
            # Clean and normalize goal text
            cleaned_goal = self._clean_goal_text(goal_text)
            
            # Analyze intent
            intent_analysis = await self._analyze_intent(cleaned_goal)
            
            # Extract context
            context = await self._extract_context(cleaned_goal, intent_analysis)
            
            # Create structured goal
            structured_goal = await self._create_structured_goal(cleaned_goal, intent_analysis, context)
            
            # Generate suggestions for ambiguous goals
            suggestions = await self._generate_suggestions(cleaned_goal, intent_analysis)
            
            processing_time = time.time() - start_time
            
            processed_goal = ProcessedGoal(
                original_goal=goal_text,
                intent_analysis=intent_analysis,
                context=context,
                structured_goal=structured_goal,
                confidence_score=intent_analysis.confidence,
                processing_time=processing_time,
                suggestions=suggestions
            )
            
            logger.info(f"✅ Goal processed in {processing_time:.2f}s with confidence {intent_analysis.confidence:.2f}")
            return processed_goal
            
        except Exception as e:
            logger.error(f"❌ Goal processing failed: {str(e)}")
            # Return minimal processed goal
            return ProcessedGoal(
                original_goal=goal_text,
                intent_analysis=IntentAnalysis(
                    primary_intent=IntentType.UNKNOWN,
                    secondary_intents=[],
                    confidence=0.0,
                    entities=[],
                    complexity=GoalComplexity.AMBIGUOUS,
                    keywords=[],
                    action_verbs=[]
                ),
                context=GoalContext(),
                structured_goal={},
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                suggestions=["Goal processing failed. Please try rephrasing."]
            )
    
    def _clean_goal_text(self, goal_text: str) -> str:
        """Clean and normalize goal text."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', goal_text.strip())
        
        # Convert to lowercase for pattern matching
        cleaned = cleaned.lower()
        
        # Remove common filler words
        filler_words = ['please', 'can you', 'could you', 'would you', 'i want to', 'i need to']
        for filler in filler_words:
            cleaned = cleaned.replace(filler, '').strip()
        
        return cleaned
    
    async def _analyze_intent(self, goal_text: str) -> IntentAnalysis:
        """Analyze user intent from goal text."""
        try:
            # Extract keywords and action verbs
            keywords = self._extract_keywords(goal_text)
            action_verbs = self._extract_action_verbs(goal_text)
            
            # Determine primary intent
            primary_intent, primary_confidence = self._determine_primary_intent(goal_text)
            
            # Determine secondary intents
            secondary_intents = self._determine_secondary_intents(goal_text)
            
            # Extract entities
            entities = self._extract_entities(goal_text)
            
            # Determine complexity
            complexity = self._determine_complexity(goal_text, keywords)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(primary_confidence, entities, complexity)
            
            return IntentAnalysis(
                primary_intent=primary_intent,
                secondary_intents=secondary_intents,
                confidence=confidence,
                entities=entities,
                complexity=complexity,
                keywords=keywords,
                action_verbs=action_verbs
            )
            
        except Exception as e:
            logger.error(f"❌ Intent analysis failed: {str(e)}")
            return IntentAnalysis(
                primary_intent=IntentType.UNKNOWN,
                secondary_intents=[],
                confidence=0.0,
                entities=[],
                complexity=GoalComplexity.AMBIGUOUS,
                keywords=[],
                action_verbs=[]
            )
    
    def _extract_keywords(self, goal_text: str) -> List[str]:
        """Extract keywords from goal text."""
        # Simple keyword extraction (in a real implementation, you'd use NLP libraries)
        words = re.findall(r'\b\w+\b', goal_text)
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _extract_action_verbs(self, goal_text: str) -> List[str]:
        """Extract action verbs from goal text."""
        found_verbs = []
        for verb in self.action_verbs:
            if verb in goal_text:
                found_verbs.append(verb)
        return found_verbs
    
    def _determine_primary_intent(self, goal_text: str) -> Tuple[IntentType, float]:
        """Determine primary intent with confidence score."""
        best_intent = IntentType.UNKNOWN
        best_score = 0.0
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, goal_text):
                    matches += 1
                    score += 1.0
            
            if matches > 0:
                score = score / len(patterns)  # Normalize by pattern count
                if score > best_score:
                    best_score = score
                    best_intent = intent_type
        
        return best_intent, best_score
    
    def _determine_secondary_intents(self, goal_text: str) -> List[IntentType]:
        """Determine secondary intents."""
        secondary_intents = []
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, goal_text):
                    if intent_type not in secondary_intents:
                        secondary_intents.append(intent_type)
                    break
        
        return secondary_intents
    
    def _extract_entities(self, goal_text: str) -> List[ExtractedEntity]:
        """Extract entities from goal text."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, goal_text)
                for match in matches:
                    confidence = 0.8 if isinstance(match, str) else 0.6
                    entities.append(ExtractedEntity(
                        entity_type=entity_type,
                        value=match if isinstance(match, str) else str(match),
                        confidence=confidence
                    ))
        
        return entities
    
    def _determine_complexity(self, goal_text: str, keywords: List[str]) -> GoalComplexity:
        """Determine goal complexity."""
        # Check for complexity indicators
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator in goal_text:
                    return complexity
        
        # Check keyword count and action verb count
        action_verb_count = len([word for word in keywords if word in self.action_verbs])
        
        if action_verb_count > 3 or len(keywords) > 10:
            return GoalComplexity.COMPLEX
        elif action_verb_count > 1 or len(keywords) > 5:
            return GoalComplexity.MODERATE
        else:
            return GoalComplexity.SIMPLE
    
    def _calculate_confidence(self, primary_confidence: float, entities: List[ExtractedEntity], complexity: GoalComplexity) -> float:
        """Calculate overall confidence score."""
        base_confidence = primary_confidence
        
        # Boost confidence with entities
        entity_boost = min(len(entities) * 0.1, 0.3)
        
        # Adjust for complexity
        complexity_adjustment = {
            GoalComplexity.SIMPLE: 0.1,
            GoalComplexity.MODERATE: 0.0,
            GoalComplexity.COMPLEX: -0.1,
            GoalComplexity.AMBIGUOUS: -0.3
        }.get(complexity, 0.0)
        
        final_confidence = base_confidence + entity_boost + complexity_adjustment
        return max(0.0, min(1.0, final_confidence))
    
    async def _extract_context(self, goal_text: str, intent_analysis: IntentAnalysis) -> GoalContext:
        """Extract context from goal text."""
        context = GoalContext()
        
        # Extract domain from entities
        for entity in intent_analysis.entities:
            if entity.entity_type == "website":
                context.target_website = entity.value
            elif entity.entity_type == "price":
                context.budget = entity.value
        
        # Extract specific requirements
        context.specific_requirements = []
        for entity in intent_analysis.entities:
            if entity.entity_type in ["product", "color", "size"]:
                context.specific_requirements.append(f"{entity.entity_type}: {entity.value}")
        
        # Determine urgency from keywords
        urgency_indicators = ["urgent", "asap", "quickly", "fast", "immediately"]
        if any(indicator in goal_text for indicator in urgency_indicators):
            context.urgency = "high"
        
        return context
    
    async def _create_structured_goal(self, goal_text: str, intent_analysis: IntentAnalysis, context: GoalContext) -> Dict[str, Any]:
        """Create structured goal representation."""
        return {
            "original_text": goal_text,
            "intent": {
                "primary": intent_analysis.primary_intent.value,
                "secondary": [intent.value for intent in intent_analysis.secondary_intents],
                "confidence": intent_analysis.confidence
            },
            "entities": [
                {
                    "type": entity.entity_type,
                    "value": entity.value,
                    "confidence": entity.confidence
                }
                for entity in intent_analysis.entities
            ],
            "context": {
                "domain": context.domain,
                "target_website": context.target_website,
                "requirements": context.specific_requirements,
                "constraints": context.constraints or [],
                "preferences": context.preferences or [],
                "urgency": context.urgency,
                "budget": context.budget
            },
            "complexity": intent_analysis.complexity.value,
            "keywords": intent_analysis.keywords,
            "action_verbs": intent_analysis.action_verbs
        }
    
    async def _generate_suggestions(self, goal_text: str, intent_analysis: IntentAnalysis) -> List[str]:
        """Generate suggestions for ambiguous or unclear goals."""
        suggestions = []
        
        if intent_analysis.confidence < 0.5:
            suggestions.append("Please provide more specific details about your goal")
        
        if intent_analysis.complexity == GoalComplexity.AMBIGUOUS:
            suggestions.append("Consider breaking down your goal into smaller, more specific tasks")
        
        if not intent_analysis.entities:
            suggestions.append("Consider specifying what you want to accomplish (e.g., 'buy a blue shirt' instead of 'shop')")
        
        if intent_analysis.primary_intent == IntentType.UNKNOWN:
            suggestions.append("Try using more specific action words like 'buy', 'search', 'fill out', or 'navigate'")
        
        return suggestions


class GoalToPlanTranslator:
    """Translates processed goals into automation plans."""
    
    def __init__(self, ai_task_planner):
        """
        Initialize goal-to-plan translator.
        
        Args:
            ai_task_planner: AI task planner instance
        """
        self.ai_task_planner = ai_task_planner
        self.translation_templates = self._initialize_translation_templates()
    
    def _initialize_translation_templates(self) -> Dict[IntentType, Dict[str, Any]]:
        """Initialize translation templates for different intents."""
        return {
            IntentType.SHOPPING: {
                "base_steps": [
                    {"action": "navigate", "description": "Navigate to shopping website"},
                    {"action": "search", "description": "Search for product"},
                    {"action": "filter", "description": "Apply filters if specified"},
                    {"action": "select", "description": "Select best option"},
                    {"action": "add_to_cart", "description": "Add to cart"},
                    {"action": "checkout", "description": "Proceed to checkout"}
                ],
                "required_entities": ["product"],
                "optional_entities": ["color", "size", "price"]
            },
            IntentType.FORM_FILLING: {
                "base_steps": [
                    {"action": "navigate", "description": "Navigate to form page"},
                    {"action": "fill_fields", "description": "Fill out form fields"},
                    {"action": "submit", "description": "Submit form"}
                ],
                "required_entities": [],
                "optional_entities": []
            },
            IntentType.DATA_EXTRACTION: {
                "base_steps": [
                    {"action": "navigate", "description": "Navigate to target page"},
                    {"action": "extract_data", "description": "Extract required information"},
                    {"action": "save_data", "description": "Save extracted data"}
                ],
                "required_entities": [],
                "optional_entities": []
            }
        }
    
    async def translate_goal_to_plan(self, processed_goal: ProcessedGoal, webpage_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate processed goal to automation plan.
        
        Args:
            processed_goal (ProcessedGoal): Processed user goal
            webpage_context (Dict[str, Any]): Current webpage context
            
        Returns:
            Dict[str, Any]: Automation plan
        """
        try:
            logger.info(f"🔄 Translating goal to plan: {processed_goal.intent_analysis.primary_intent.value}")
            
            # Get base template for intent
            template = self.translation_templates.get(
                processed_goal.intent_analysis.primary_intent,
                self.translation_templates[IntentType.NAVIGATION]  # Default fallback
            )
            
            # Create plan structure
            plan = {
                "goal": processed_goal.original_goal,
                "intent": processed_goal.intent_analysis.primary_intent.value,
                "complexity": processed_goal.intent_analysis.complexity.value,
                "confidence": processed_goal.confidence_score,
                "steps": [],
                "estimated_duration": self._estimate_duration(processed_goal),
                "requirements": processed_goal.context.specific_requirements or []
            }
            
            # Generate steps based on intent and entities
            steps = await self._generate_steps(processed_goal, template, webpage_context)
            plan["steps"] = steps
            
            # Add metadata
            plan["metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "entities_found": len(processed_goal.intent_analysis.entities),
                "complexity_score": self._calculate_complexity_score(processed_goal),
                "suggestions": processed_goal.suggestions
            }
            
            logger.info(f"✅ Goal translated to plan with {len(steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"❌ Goal translation failed: {str(e)}")
            return {
                "goal": processed_goal.original_goal,
                "error": str(e),
                "steps": [],
                "suggestions": ["Translation failed. Please try rephrasing your goal."]
            }
    
    async def _generate_steps(self, processed_goal: ProcessedGoal, template: Dict[str, Any], webpage_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automation steps based on goal and template."""
        steps = []
        step_id = 1
        
        # Add navigation step if needed
        if processed_goal.context.target_website:
            steps.append({
                "step_id": step_id,
                "action": "navigate",
                "url": processed_goal.context.target_website,
                "description": f"Navigate to {processed_goal.context.target_website}",
                "timeout": 10000
            })
            step_id += 1
        
        # Add intent-specific steps
        for base_step in template["base_steps"]:
            step = {
                "step_id": step_id,
                "action": base_step["action"],
                "description": base_step["description"],
                "timeout": 10000
            }
            
            # Customize step based on entities
            step = self._customize_step(step, processed_goal, webpage_context)
            steps.append(step)
            step_id += 1
        
        return steps
    
    def _customize_step(self, step: Dict[str, Any], processed_goal: ProcessedGoal, webpage_context: Dict[str, Any]) -> Dict[str, Any]:
        """Customize step based on goal entities and context."""
        # Add selectors based on webpage context
        if step["action"] in ["click", "type", "select"]:
            # Find appropriate selectors from context
            elements = webpage_context.get("elements", [])
            if elements:
                # Simple selector selection (in real implementation, use AI)
                for elem in elements:
                    if elem.get("tag") in ["button", "input", "a"]:
                        step["selector"] = elem.get("selector", "button")
                        break
        
        # Add text content for type actions
        if step["action"] == "type":
            # Extract text from entities
            for entity in processed_goal.intent_analysis.entities:
                if entity.entity_type in ["product", "name", "email"]:
                    step["text"] = entity.value
                    break
        
        return step
    
    def _estimate_duration(self, processed_goal: ProcessedGoal) -> float:
        """Estimate execution duration based on goal complexity."""
        base_duration = {
            GoalComplexity.SIMPLE: 30.0,
            GoalComplexity.MODERATE: 60.0,
            GoalComplexity.COMPLEX: 120.0,
            GoalComplexity.AMBIGUOUS: 90.0
        }.get(processed_goal.intent_analysis.complexity, 60.0)
        
        # Adjust for number of entities
        entity_adjustment = len(processed_goal.intent_analysis.entities) * 10.0
        
        return base_duration + entity_adjustment
    
    def _calculate_complexity_score(self, processed_goal: ProcessedGoal) -> float:
        """Calculate complexity score for the goal."""
        base_score = {
            GoalComplexity.SIMPLE: 0.2,
            GoalComplexity.MODERATE: 0.5,
            GoalComplexity.COMPLEX: 0.8,
            GoalComplexity.AMBIGUOUS: 0.6
        }.get(processed_goal.intent_analysis.complexity, 0.5)
        
        # Adjust for confidence
        confidence_adjustment = (1.0 - processed_goal.confidence_score) * 0.3
        
        return min(1.0, base_score + confidence_adjustment)


class NaturalLanguageGoalProcessor:
    """Main processor for natural language goals."""
    
    def __init__(self, ai_task_planner):
        """
        Initialize natural language goal processor.
        
        Args:
            ai_task_planner: AI task planner instance
        """
        self.nlp_processor = NaturalLanguageProcessor()
        self.goal_translator = GoalToPlanTranslator(ai_task_planner)
        self.processing_history: List[ProcessedGoal] = []
    
    async def process_user_goal(self, goal_text: str, webpage_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user goal and return automation plan.
        
        Args:
            goal_text (str): User's goal in natural language
            webpage_context (Dict[str, Any]): Current webpage context
            
        Returns:
            Dict[str, Any]: Complete automation plan
        """
        try:
            logger.info(f"🎯 Processing user goal: {goal_text}")
            
            # Process the goal
            processed_goal = await self.nlp_processor.process_goal(goal_text)
            self.processing_history.append(processed_goal)
            
            # Translate to automation plan
            plan = await self.goal_translator.translate_goal_to_plan(processed_goal, webpage_context)
            
            # Add processing metadata
            plan["processing_info"] = {
                "confidence": processed_goal.confidence_score,
                "complexity": processed_goal.intent_analysis.complexity.value,
                "entities_found": len(processed_goal.intent_analysis.entities),
                "processing_time": processed_goal.processing_time,
                "suggestions": processed_goal.suggestions
            }
            
            logger.info(f"✅ User goal processed successfully")
            return plan
            
        except Exception as e:
            logger.error(f"❌ User goal processing failed: {str(e)}")
            return {
                "goal": goal_text,
                "error": str(e),
                "steps": [],
                "suggestions": ["Goal processing failed. Please try rephrasing."]
            }
    
    def get_processing_history(self) -> List[ProcessedGoal]:
        """Get processing history."""
        return self.processing_history.copy()
    
    def save_processing_report(self, filename: str) -> None:
        """Save processing report to file."""
        try:
            report = {
                "processing_history": [asdict(goal) for goal in self.processing_history],
                "total_processed": len(self.processing_history),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Processing report saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save processing report: {str(e)}")


async def main():
    """Demo function for Natural Language Goal Processing."""
    try:
        logger.info("🚀 Natural Language Goal Processing Demo")
        logger.info("This module provides natural language processing for user goals")
        logger.info("including intent recognition, context extraction, and goal-to-plan translation.")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
