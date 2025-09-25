"""
LLM Agent module for intelligent topic analysis and optimization.

This module implements the core intelligence of the framework using OpenAI's LLM models
to analyze topics, evaluate quality, and generate optimization commands with cost control.
"""

import json
import logging
import time
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
from openai import OpenAI
import pandas as pd

# Configuration imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import DEFAULT_LLM_MODELS, OPENAI_API_KEY, OPTIMIZER_CONFIG
from config.prompts import get_prompt_template, get_system_prompt

logger = logging.getLogger(__name__)

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
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_retries: int = 3,
        budget_limit: float = 50.0,
        batch_size: int = 10
    ):
        """
        Initialize the LLM Agent.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model_name: LLM model to use
            temperature: Temperature for generation (0.1 for stability)
            max_retries: Maximum retries for API calls
            budget_limit: Maximum budget for API calls
            batch_size: Topics per batch for analysis
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.budget_limit = budget_limit
        self.batch_size = batch_size

        # Model configuration
        if model_name not in DEFAULT_LLM_MODELS:
            logger.warning(f"Model {model_name} not in default configs, using gpt-4 pricing")
            self.model_config = DEFAULT_LLM_MODELS["gpt-4"]
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
        logger.info(f"Budget limit: ${budget_limit}")

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
        if self._check_budget_exceeded():
            raise Exception(f"Budget limit of ${self.budget_limit} exceeded")

        keywords_str = ", ".join(keywords[:10])  # Limit to top 10 keywords
        prompt = get_prompt_template("naming", keywords=keywords_str)

        try:
            response = self._make_api_call(prompt, max_tokens=200)
            result = self._parse_json_response(response)

            # Validate response format
            if not isinstance(result, dict) or 'name' not in result or 'summary' not in result:
                logger.warning("Invalid response format, using fallback")
                return {
                    "name": f"Topic: {keywords[0]} related",
                    "summary": f"This topic focuses on {', '.join(keywords[:3])} and related concepts."
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get topic name and summary: {e}")
            # Return fallback response
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
        if self._check_budget_exceeded():
            raise Exception(f"Budget limit of ${self.budget_limit} exceeded")

        keywords_str = ", ".join(keywords[:10])
        prompt = get_prompt_template("quality", topic_name=topic_name, keywords=keywords_str)

        try:
            response = self._make_api_call(prompt, max_tokens=300)
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
        if self._check_budget_exceeded():
            raise Exception(f"Budget limit of ${self.budget_limit} exceeded")

        logger.info(f"Generating optimization commands for {len(topics_data)} topics")

        all_commands = []

        # Process topics in batches to reduce API calls
        for i in range(0, len(topics_data), self.batch_size):
            batch = topics_data[i:i + self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}: {len(batch)} topics")

            try:
                batch_commands = self._process_topic_batch(batch)
                all_commands.extend(batch_commands)

                # Check budget after each batch
                if self._check_budget_exceeded():
                    logger.warning("Budget limit reached, stopping optimization command generation")
                    break

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
            response = self._make_api_call(prompt, max_tokens=1500)
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

    def _make_api_call(self, prompt: str, max_tokens: int = 1000) -> str:
        """Make an API call with retries and error handling."""
        system_prompt = get_system_prompt(self.model_name)

        # Estimate cost before making call
        cost_estimate = self.calculate_tokens_and_cost(system_prompt + prompt)
        if self.cost_tracker.total_cost + cost_estimate["estimated_cost"] > self.budget_limit:
            raise Exception(f"Estimated cost would exceed budget limit")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    timeout=60
                )

                # Track actual usage
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens

                actual_cost = (
                    (input_tokens / 1000) * self.model_config["input_cost_per_1k"] +
                    (output_tokens / 1000) * self.model_config["output_cost_per_1k"]
                )

                self.cost_tracker.add_usage(input_tokens, output_tokens, actual_cost)

                logger.debug(f"API call successful. Cost: ${actual_cost:.4f}, Total: ${self.cost_tracker.total_cost:.4f}")

                return response.choices[0].message.content.strip()

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
        """Check if budget limit has been exceeded."""
        if self.cost_tracker.total_cost >= self.budget_limit:
            logger.warning(f"Budget limit exceeded: ${self.cost_tracker.total_cost:.2f} >= ${self.budget_limit}")
            return True
        return False

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost and usage summary."""
        summary = self.cost_tracker.get_summary()
        summary.update({
            "budget_limit": self.budget_limit,
            "budget_remaining": max(0, self.budget_limit - self.cost_tracker.total_cost),
            "budget_used_percent": (self.cost_tracker.total_cost / self.budget_limit) * 100
        })
        return summary

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
        agent = LLMAgent(model_name="gpt-3.5-turbo", budget_limit=5.0)

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