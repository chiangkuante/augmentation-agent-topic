"""
Configuration settings for the augmentation-agent-topic system.

This module integrates with the environment loader to provide
configuration management with dotenv support.
"""

import os
from pathlib import Path
from typing import Dict, Any
from .env_loader import get_env_loader

# Initialize environment loader
env_loader = get_env_loader()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "topics").mkdir(exist_ok=True)
(RESULTS_DIR / "scores").mkdir(exist_ok=True)
(RESULTS_DIR / "optimization").mkdir(exist_ok=True)
(RESULTS_DIR / "resilience").mkdir(exist_ok=True)
(RESULTS_DIR / "checkpoints").mkdir(exist_ok=True)

# Get configuration from environment
api_config = env_loader.get_api_config()
DEFAULT_API_BUDGET = api_config['max_budget']
OPENAI_API_KEY = api_config['openai_api_key']

# Get all configuration from environment
topic_config = env_loader.get_topic_modeling_config()
optimization_config = env_loader.get_optimization_config()
quality_config = env_loader.get_quality_thresholds()
resilience_config = env_loader.get_resilience_config()

# Model Configuration
DEFAULT_LLM_MODELS = {
    "gpt-4": {
        "name": "gpt-4",
        "input_cost_per_1k": 0.03,
        "output_cost_per_1k": 0.06,
        "context_limit": 8192
    },
    "gpt-3.5-turbo": {
        "name": "gpt-3.5-turbo",
        "input_cost_per_1k": 0.0015,
        "output_cost_per_1k": 0.002,
        "context_limit": 4096
    },
    "gpt-4-turbo": {
        "name": "gpt-4-turbo",
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "context_limit": 128000
    }
}

# Embedding Model Configuration (from environment)
EMBEDDING_MODELS = {
    "phase1": "sentence-transformers/all-mpnet-base-v2",  # MVP
    "phase2": "answerdotai/ModernBERT-base",  # Advanced
    "phase3": "answerdotai/ModernBERT-large"  # Experimental
}

# Topic Modeling Configuration (from environment)
TOPIC_MODEL_CONFIG = {
    "min_topic_size": topic_config['min_topic_size'],
    "nr_topics": None,  # Auto-detect
    "calculate_probabilities": True,
    "verbose": env_loader.get_logging_config()['verbose']
}

# UMAP Configuration (from environment)
UMAP_CONFIG = {
    **topic_config['umap_config'],
    "random_state": 42
}

# HDBSCAN Configuration (from environment)
HDBSCAN_CONFIG = {
    **topic_config['hdbscan_config'],
    "cluster_selection_method": "eom",
    "prediction_data": True
}

# Optimizer Configuration (from environment)
OPTIMIZER_CONFIG = {
    **optimization_config,
    "api_budget_limit": DEFAULT_API_BUDGET,
    "temperature": api_config['temperature'],
    "max_retries": api_config['max_retries']
}

# Quality Metrics Configuration (from environment)
QUALITY_THRESHOLDS = quality_config

# Resilience Framework Configuration (from environment)
RESILIENCE_DIMENSIONS = resilience_config['dimensions']

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "system.log"),
            "formatter": "detailed"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "results_dir": RESULTS_DIR,
            "logs_dir": LOGS_DIR
        },
        "api": {
            "budget": DEFAULT_API_BUDGET,
            "openai_key": OPENAI_API_KEY
        },
        "models": {
            "llm": DEFAULT_LLM_MODELS,
            "embeddings": EMBEDDING_MODELS
        },
        "topic_modeling": TOPIC_MODEL_CONFIG,
        "umap": UMAP_CONFIG,
        "hdbscan": HDBSCAN_CONFIG,
        "optimizer": OPTIMIZER_CONFIG,
        "quality": QUALITY_THRESHOLDS,
        "resilience": RESILIENCE_DIMENSIONS,
        "logging": LOGGING_CONFIG
    }

def validate_config() -> bool:
    """Validate configuration settings using environment loader."""
    is_valid, errors = env_loader.validate_config()

    if not is_valid:
        for error in errors:
            print(f"Configuration Error: {error}")

    return is_valid

def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags from environment."""
    return env_loader.get_feature_flags()

def get_environment_name() -> str:
    """Get current environment name."""
    return env_loader.env_name

def reload_config(env_name: str = None) -> None:
    """Reload configuration with optional environment change."""
    global env_loader, api_config, topic_config, optimization_config, quality_config, resilience_config
    global DEFAULT_API_BUDGET, OPENAI_API_KEY

    # Reload environment
    from .env_loader import load_environment
    env_loader = load_environment(env_name)

    # Reload all configs
    api_config = env_loader.get_api_config()
    topic_config = env_loader.get_topic_modeling_config()
    optimization_config = env_loader.get_optimization_config()
    quality_config = env_loader.get_quality_thresholds()
    resilience_config = env_loader.get_resilience_config()

    DEFAULT_API_BUDGET = api_config['max_budget']
    OPENAI_API_KEY = api_config['openai_api_key']