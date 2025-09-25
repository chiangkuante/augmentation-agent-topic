"""
LLM Agent module for intelligent topic analysis and optimization.

This module implements the core intelligence of the framework using OpenAI's LLM models
to analyze topics, evaluate quality, and generate optimization commands with cost control.
"""

import json
import logging
import time
import tiktoken
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
from pathlib import Path
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Configuration imports
import sys
sys.path.append(str(project_root))
from config.settings import DEFAULT_LLM_MODELS, OPTIMIZER_CONFIG
from config.prompts import get_prompt_template, get_system_prompt

logger = logging.getLogger(__name__)

# JSON Schemas for Structured Outputs
TOPIC_NAME_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "A concise, descriptive name for this topic (max 5 words)"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary explaining what this topic represents (max 2 sentences)"
        }
    },
    "required": ["name", "summary"],
    "additionalProperties": False
}

TOPIC_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "How well do the keywords relate to each other and the topic name (1-10)"
        },
        "uniqueness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "How distinct is this topic from other potential topics (1-10)"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of your scoring (max 2 sentences)"
        }
    },
    "required": ["coherence", "uniqueness", "reason"],
    "additionalProperties": False
}

OPTIMIZATION_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "similar_topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic_ids": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "similarity_reason": {"type": "string"}
                        },
                        "required": ["topic_ids", "similarity_reason"],
                        "additionalProperties": False
                    }
                },
                "low_quality_topics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic_id": {"type": "integer"},
                            "quality_issues": {"type": "string"}
                        },
                        "required": ["topic_id", "quality_issues"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["similar_topics", "low_quality_topics"],
            "additionalProperties": False
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["MERGE", "SPLIT"]
                    },
                    "targets": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "reason": {"type": "string"},
                    "expected_benefit": {"type": "string"}
                },
                "required": ["action", "targets", "reason", "expected_benefit"],
                "additionalProperties": False
            }
        }
    },
    "required": ["analysis", "recommendations"],
    "additionalProperties": False
}

@dataclass
class CostTracker:
    """Track API costs and usage."""
    total_cost: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    start_time: Optional[datetime] = None

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float):
        """Add usage statistics."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost
        self.api_calls += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "api_calls": self.api_calls,
            "elapsed_time": elapsed_time,
            "avg_cost_per_call": self.total_cost / self.api_calls if self.api_calls > 0 else 0
        }

class LLMAgent:
    """
    LLM Agent for topic analysis and optimization with cost control.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5",
        temperature: float = 0.1,
        max_retries: int = 3,
        batch_size: int = 10
    ):
        """
        Initialize the LLM Agent.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model_name: LLM model to use
            temperature: Temperature for generation (0.1 for stability)
            max_retries: Maximum retries for API calls
            batch_size: Topics per batch for analysis
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Validate that only GPT-5 series models are supported
        if not self._is_gpt5_model(model_name):
            supported_models = list(DEFAULT_LLM_MODELS.keys())
            raise ValueError(f"Only GPT-5 series models are supported. Got '{model_name}'. Supported models: {supported_models}")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Model configuration
        if model_name not in DEFAULT_LLM_MODELS:
            logger.warning(f"Model {model_name} not in default configs, using gpt-5 pricing")
            self.model_config = DEFAULT_LLM_MODELS["gpt-5"]
        else:
            self.model_config = DEFAULT_LLM_MODELS[model_name]

        # Initialize cost tracking
        self.cost_tracker = CostTracker(start_time=datetime.now())

        # Initialize tokenizer for cost estimation
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            # Fallback tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"LLM Agent initialized with model: {model_name}")
        logger.info("Budget tracking disabled")

    def _is_gpt5_model(self, model_name: str) -> bool:
        """Check if the model is a GPT-5 series model."""
        return model_name.startswith("gpt-5") or model_name in DEFAULT_LLM_MODELS

    def get_topic_name_and_summary(self, keywords: List[str]) -> Dict[str, str]:
        """
        Generate topic name and summary from keywords.

        Args:
            keywords: List of topic keywords

        Returns:
            Dictionary with 'name' and 'summary' keys

        Raises:
            Exception: If API call fails or budget exceeded
        """
        # Budget checking removed

        keywords_str = ", ".join(keywords[:10])  # Limit to top 10 keywords
        prompt = get_prompt_template("naming", keywords=keywords_str)

        try:
            response = self._make_api_call(prompt, max_tokens=200,
                                         json_schema=TOPIC_NAME_SUMMARY_SCHEMA,
                                         schema_name="topic_name_summary")
            result = self._parse_json_response(response)

            # Validate response format
            if not isinstance(result, dict) or 'name' not in result or 'summary' not in result:
                logger.warning("Invalid response format or empty LLM response, using fallback")
                if not keywords:
                    # Handle empty keywords list
                    return {
                        "name": "Unidentified Topic",
                        "summary": "This topic could not be properly identified due to lack of keywords."
                    }
                return {
                    "name": f"Topic: {keywords[0]} related",
                    "summary": f"This topic focuses on {', '.join(keywords[:3])} and related concepts."
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get topic name and summary: {e}")
            # Return fallback response
            if not keywords:
                return {
                    "name": "Unidentified Topic",
                    "summary": "This topic could not be properly identified due to an error."
                }
            return {
                "name": f"Topic: {keywords[0]} related",
                "summary": f"This topic focuses on {', '.join(keywords[:3])} and related concepts."
            }

    def evaluate_topic_quality(self, topic_name: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate the quality of a topic.

        Args:
            topic_name: Name of the topic
            keywords: List of topic keywords

        Returns:
            Dictionary with 'coherence', 'uniqueness', and 'reason' keys
        """
        # Budget checking removed

        keywords_str = ", ".join(keywords[:10])
        prompt = get_prompt_template("quality", topic_name=topic_name, keywords=keywords_str)

        try:
            response = self._make_api_call(prompt, max_tokens=300,
                                         json_schema=TOPIC_QUALITY_SCHEMA,
                                         schema_name="topic_quality")
            result = self._parse_json_response(response)

            # Validate and normalize scores
            if isinstance(result, dict):
                coherence = max(1, min(10, result.get('coherence', 5)))
                uniqueness = max(1, min(10, result.get('uniqueness', 5)))
                reason = result.get('reason', 'No specific reason provided.')

                return {
                    "coherence": coherence,
                    "uniqueness": uniqueness,
                    "reason": reason
                }
            else:
                raise ValueError("Invalid response format")

        except Exception as e:
            logger.error(f"Failed to evaluate topic quality: {e}")
            # Return moderate scores as fallback
            return {
                "coherence": 5,
                "uniqueness": 5,
                "reason": "Quality evaluation failed, using default scores."
            }

    def generate_optimization_commands(self, topics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate optimization commands for topics using batch processing.

        Args:
            topics_data: List of topic dictionaries with 'id', 'name', 'keywords', etc.

        Returns:
            List of action dictionaries with 'action', 'targets', 'reason'
        """
        # Budget checking removed

        logger.info(f"Generating optimization commands for {len(topics_data)} topics")

        all_commands = []

        # Process topics in batches to reduce API calls
        for i in range(0, len(topics_data), self.batch_size):
            batch = topics_data[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch)} topics")

            try:
                batch_commands = self._process_topic_batch(batch)
                all_commands.extend(batch_commands)

                # Budget checking removed

            except Exception as e:
                logger.error(f"Failed to process topic batch {i//self.batch_size + 1}: {e}")
                continue

        logger.info(f"Generated {len(all_commands)} optimization commands")
        return all_commands

    def _process_topic_batch(self, topics_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of topics for optimization."""
        # Create batch description
        batch_description = []
        for topic in topics_batch:
            topic_desc = {
                "id": topic.get("topic_id", topic.get("id")),
                "name": topic.get("name", f"Topic {topic.get('topic_id', 'Unknown')}"),
                "keywords": topic.get("keywords", [])[:8],  # Top 8 keywords
                "document_count": topic.get("document_count", 0),
                "quality_scores": topic.get("quality_scores", {})
            }
            batch_description.append(topic_desc)

        # Convert to string format for prompt
        topics_str = json.dumps(batch_description, indent=2)
        prompt = get_prompt_template("batch", topics_batch=topics_str)

        try:
            response = self._make_api_call(prompt, max_tokens=1500,
                                         json_schema=OPTIMIZATION_ANALYSIS_SCHEMA,
                                         schema_name="optimization_analysis")
            result = self._parse_json_response(response)

            # Extract recommendations
            if isinstance(result, dict) and "recommendations" in result:
                commands = result["recommendations"]
                if isinstance(commands, list):
                    return self._validate_commands(commands)

            logger.warning("No valid recommendations in batch response")
            return []

        except Exception as e:
            logger.error(f"Failed to process topic batch: {e}")
            return []

    def _validate_commands(self, commands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate optimization commands."""
        valid_commands = []

        for cmd in commands:
            if not isinstance(cmd, dict):
                continue

            action = cmd.get("action", "").upper()
            targets = cmd.get("targets", [])
            reason = cmd.get("reason", "")

            # Validate action type
            if action not in ["MERGE", "SPLIT"]:
                logger.warning(f"Invalid action type: {action}")
                continue

            # Validate targets
            if not isinstance(targets, list) or len(targets) == 0:
                logger.warning(f"Invalid targets for {action}: {targets}")
                continue

            # Additional validation for MERGE (need at least 2 targets)
            if action == "MERGE" and len(targets) < 2:
                logger.warning(f"MERGE action needs at least 2 targets: {targets}")
                continue

            # Additional validation for SPLIT (need exactly 1 target)
            if action == "SPLIT" and len(targets) != 1:
                logger.warning(f"SPLIT action needs exactly 1 target: {targets}")
                continue

            valid_commands.append({
                "action": action,
                "targets": targets,
                "reason": reason,
                "confidence": cmd.get("confidence", 0.5)
            })

        logger.debug(f"Validated {len(valid_commands)} out of {len(commands)} commands")
        return valid_commands

    def calculate_tokens_and_cost(self, text: str) -> Dict[str, Any]:
        """
        Estimate token count and cost for a text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with token count and cost estimates
        """
        try:
            tokens = len(self.tokenizer.encode(text))
            input_cost = (tokens / 1000) * self.model_config["input_cost_per_1k"]

            # Estimate output tokens (typically much smaller)
            estimated_output_tokens = min(tokens // 4, 500)  # Conservative estimate
            output_cost = (estimated_output_tokens / 1000) * self.model_config["output_cost_per_1k"]

            total_cost = input_cost + output_cost

            return {
                "input_tokens": tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_cost": total_cost,
                "input_cost": input_cost,
                "output_cost": output_cost
            }

        except Exception as e:
            logger.error(f"Failed to calculate tokens and cost: {e}")
            return {
                "input_tokens": 0,
                "estimated_output_tokens": 0,
                "estimated_cost": 0.0,
                "input_cost": 0.0,
                "output_cost": 0.0
            }

    def _normalize_response_content(self, content: Any) -> str:
        """Flatten structured response content into plain text."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                part_text = self._normalize_response_content(part)
                if part_text:
                    parts.append(part_text)
            return "\n".join(parts).strip()

        if isinstance(content, dict):
            # Common OpenAI structures: {"type": "text", "text": "..."}
            if "text" in content and isinstance(content["text"], (str, list, dict)):
                return self._normalize_response_content(content["text"])

            # Some reasoning models return {"type": "reasoning", "reasoning": [...]}
            if "reasoning" in content:
                return self._normalize_response_content(content["reasoning"])

            if "content" in content:
                return self._normalize_response_content(content["content"])

            return str(content).strip()

        # Fallback for objects with a "text" attribute (pydantic model objects)
        text_attr = getattr(content, "text", None)
        if text_attr is not None:
            return self._normalize_response_content(text_attr)

        # Last resort: convert to string
        return str(content).strip()

    def _make_api_call(self, prompt: str, max_tokens: int = 1000, json_schema: Optional[Dict] = None, schema_name: str = "response") -> str:
        """Make an API call with retries and error handling, supporting structured outputs."""
        system_prompt = get_system_prompt(self.model_name)

        # Cost estimation removed

        for attempt in range(self.max_retries):
            try:
                # Use correct parameter name based on model
                api_params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "timeout": 60
                }

                # Use Responses API for all models
                # Add JSON schema requirements to prompt for structured output
                if json_schema:
                    schema_instruction = f"\n\nIMPORTANT: Respond ONLY with valid JSON matching this exact schema:\n{json.dumps(json_schema, indent=2)}\n\nProvide no other text outside the JSON."
                    combined_prompt = f"{system_prompt}\n\n{prompt}{schema_instruction}"
                else:
                    combined_prompt = f"{system_prompt}\n\n{prompt}"

                response_params = {
                    "model": self.model_name,
                    "input": combined_prompt,
                }

                # Set reasoning effort based on GPT-5 model variant
                if "gpt-5-nano" in self.model_name:
                    response_params["reasoning"] = {"effort": "minimal"}  # Optimize for speed
                elif "gpt-5-mini" in self.model_name:
                    response_params["reasoning"] = {"effort": "low"}  # Balanced for mini
                else:  # gpt-5 full model
                    response_params["reasoning"] = {"effort": "medium"}  # Full reasoning

                # Set temperature for all models except GPT-5-nano (it doesn't support temperature)
                if "gpt-5-nano" not in self.model_name:
                    response_params["temperature"] = self.temperature

                response = self.client.responses.create(**response_params)

                # Handle Responses API format for all models
                self._track_usage_and_cost(response)

                # Extract content from Responses API format
                content = ""
                if hasattr(response, 'output'):
                    for item in response.output:
                        # Skip reasoning items (they have content=None)
                        if hasattr(item, 'content') and item.content is not None:
                            for content_item in item.content:
                                if hasattr(content_item, 'text'):
                                    content += content_item.text

                # Check for structured output in Responses API
                if hasattr(response, 'output_parsed') and response.output_parsed:
                    return json.dumps(response.output_parsed) if not isinstance(response.output_parsed, str) else response.output_parsed

                if not content:
                    logger.warning(
                        "Model returned empty content; "
                        "Raw response: %s", response
                    )

                return content

            except openai.RateLimitError:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)

            except openai.APITimeoutError:
                logger.warning(f"API timeout (attempt {attempt + 1})")
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise Exception(f"Failed to make API call after {self.max_retries} attempts")

    def _track_usage_and_cost(self, response) -> None:
        """Track API usage and costs from response."""
        # Handle Responses API format (used for all models now)
        if hasattr(response, 'usage'):
            usage = response.usage
            # Responses API usage format
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
            total_output_tokens = output_tokens + reasoning_tokens
        else:
            # Fallback: estimate if no usage data available
            input_tokens = 100  # Conservative estimate
            total_output_tokens = 50  # Conservative estimate

        actual_cost = (
            (input_tokens / 1000) * self.model_config["input_cost_per_1k"] +
            (total_output_tokens / 1000) * self.model_config["output_cost_per_1k"]
        )

        self.cost_tracker.add_usage(input_tokens, total_output_tokens, actual_cost)
        logger.debug(f"API call successful. Cost: ${actual_cost:.4f}, Total: ${self.cost_tracker.total_cost:.4f}")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with error handling."""
        try:
            # Clean up the response
            response = response.strip()

            # Find JSON content
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                # Try to find JSON array
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON content found")

            json_str = response[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response: {e}")

    def _check_budget_exceeded(self) -> bool:
        """Budget checking disabled."""
        return False

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost and usage summary."""
        return self.cost_tracker.get_summary()

    def reset_cost_tracker(self):
        """Reset the cost tracker."""
        self.cost_tracker = CostTracker(start_time=datetime.now())
        logger.info("Cost tracker reset")

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample data (only if API key is available)
    try:
        agent = LLMAgent(model_name="gpt-5-nano-2025-08-07")

        # Test topic naming
        keywords = ["cybersecurity", "threat", "protection", "security", "risk"]
        result = agent.get_topic_name_and_summary(keywords)
        print(f"✅ Topic naming test: {result}")

        # Test quality evaluation
        quality = agent.evaluate_topic_quality("Cybersecurity Measures", keywords)
        print(f"✅ Quality evaluation test: {quality}")

        # Test cost summary
        cost_summary = agent.get_cost_summary()
        print(f"💰 Cost summary: ${cost_summary['total_cost']:.4f}")

    except Exception as e:
        print(f"⚠️ Test skipped (API key required): {e}")
