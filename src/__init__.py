"""
Augmentation Agent Topic - LLM-assisted iterative topic modeling framework.

This package provides tools for:
- Loading and preprocessing document data
- Initial topic modeling with BERTopic
- LLM-assisted topic optimization
- Digital resilience scoring

Author: Claude Code
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from .data_loader import load_corpus
from .initial_topic_modeler import create_initial_topics
from .llm_agent import LLMAgent
from .iterative_optimizer import Optimizer
from .resilience_mapper import map_topics_to_resilience, calculate_resilience_scores

__all__ = [
    "load_corpus",
    "create_initial_topics",
    "LLMAgent",
    "Optimizer",
    "map_topics_to_resilience",
    "calculate_resilience_scores"
]