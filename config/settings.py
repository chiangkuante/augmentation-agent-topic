"""
Configuration settings for the augmentation-agent-topic system.

This module uses dotenv to directly load configuration from .env files.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Project paths
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

# Get configuration directly from environment variables
DEFAULT_API_BUDGET = float(os.getenv('OPENAI_API_BUDGET', '50.0'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Topic Modeling Configuration from environment variables
topic_config = {
    'min_topic_size': int(os.getenv('MIN_TOPIC_SIZE', '10')),
    'umap_config': {
        'n_neighbors': int(os.getenv('UMAP_N_NEIGHBORS', '15')),
        'n_components': int(os.getenv('UMAP_N_COMPONENTS', '5')),
        'metric': os.getenv('UMAP_METRIC', 'cosine'),
        'min_dist': float(os.getenv('UMAP_MIN_DIST', '0.0'))
    },
    'hdbscan_config': {
        'min_cluster_size': int(os.getenv('HDBSCAN_MIN_CLUSTER_SIZE', '10')),
        'metric': os.getenv('HDBSCAN_METRIC', 'euclidean'),
        'cluster_selection_epsilon': float(os.getenv('HDBSCAN_CLUSTER_SELECTION_EPSILON', '0.0'))
    }
}

# Optimization Configuration from environment variables
optimization_config = {
    'max_iterations': int(os.getenv('MAX_OPTIMIZATION_ITERATIONS', '5')),
    'convergence_threshold': float(os.getenv('CONVERGENCE_THRESHOLD', '0.02')),
    'batch_size': int(os.getenv('LLM_BATCH_SIZE', '10'))
}

# Quality Thresholds from environment variables
quality_config = {
    'min_coherence_score': float(os.getenv('MIN_COHERENCE_SCORE', '0.3')),
    'min_diversity_score': float(os.getenv('MIN_DIVERSITY_SCORE', '0.1')),
    'min_silhouette_score': float(os.getenv('MIN_SILHOUETTE_SCORE', '0.1'))
}

# Resilience Configuration from environment variables
resilience_config = {
    'dimensions': {
        'absorb': os.getenv('RESILIENCE_ABSORB_KEYWORDS', 'backup,alternative,stability,resistance,predict,defense').split(','),
        'adapt': os.getenv('RESILIENCE_ADAPT_KEYWORDS', 'adjust,flexible,reallocation,response,repositioning,adaptation').split(','),
        'transform': os.getenv('RESILIENCE_TRANSFORM_KEYWORDS', 'redesign,innovation,transformation,ecosystem,fundamental,revolutionary').split(',')
    }
}

# Model Configuration - GPT-5 Series Only
DEFAULT_LLM_MODELS = {
    "gpt-5": {
        "name": "gpt-5",
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03,
        "context_limit": 128000
    },
    "gpt-5-mini": {
        "name": "gpt-5-mini",
        "input_cost_per_1k": 0.005,
        "output_cost_per_1k": 0.015,
        "context_limit": 64000
    },
    "gpt-5-nano-2025-08-07": {
        "name": "gpt-5-nano-2025-08-07",
        "input_cost_per_1k": 0.001,
        "output_cost_per_1k": 0.002,
        "context_limit": 32000
    }
}

# Embedding Model Configuration (from environment)
EMBEDDING_MODELS = {
    "phase1": "sentence-transformers/all-mpnet-base-v2",  # MVP
    "phase2": "answerdotai/ModernBERT-base",  # Advanced
    "phase3": "answerdotai/ModernBERT-large",  # Experimental
    "modernbert": "answerdotai/ModernBERT-base"  # Direct ModernBERT access
}

# Topic Modeling Configuration (from environment)
TOPIC_MODEL_CONFIG = {
    "min_topic_size": topic_config['min_topic_size'],
    "nr_topics": None,  # Auto-detect
    "calculate_probabilities": True,
    "verbose": os.getenv('VERBOSE', 'false').lower() == 'true'
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
    "temperature": float(os.getenv('OPENAI_TEMPERATURE', '0.1')),
    "max_retries": int(os.getenv('OPENAI_MAX_RETRIES', '3'))
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
    """Validate configuration settings."""
    is_valid = True
    errors = []

    # Validate required environment variables
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")
        is_valid = False

    if DEFAULT_API_BUDGET <= 0:
        errors.append("OPENAI_API_BUDGET must be positive")
        is_valid = False

    if not is_valid:
        for error in errors:
            print(f"Configuration Error: {error}")

    return is_valid

def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags from environment."""
    return {
        'use_modernbert': os.getenv('USE_MODERNBERT', 'true').lower() == 'true',
        'enable_stopwords': os.getenv('ENABLE_STOPWORDS', 'true').lower() == 'true',
        'enable_optimization': os.getenv('ENABLE_OPTIMIZATION', 'true').lower() == 'true',
        'enable_resilience_mapping': os.getenv('ENABLE_RESILIENCE_MAPPING', 'true').lower() == 'true'
    }

def get_environment_name() -> str:
    """Get current environment name."""
    return os.getenv('ENVIRONMENT', 'development')

def reload_config(env_name: str = None) -> None:
    """Reload configuration by re-reading environment variables."""
    global topic_config, optimization_config, quality_config, resilience_config
    global DEFAULT_API_BUDGET, OPENAI_API_KEY

    # Always use unified .env file
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    # Reload all configuration variables
    DEFAULT_API_BUDGET = float(os.getenv('OPENAI_API_BUDGET', '50.0'))
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Reload configurations (recreate the dictionaries)
    topic_config = {
        'min_topic_size': int(os.getenv('MIN_TOPIC_SIZE', '10')),
        'umap_config': {
            'n_neighbors': int(os.getenv('UMAP_N_NEIGHBORS', '15')),
            'n_components': int(os.getenv('UMAP_N_COMPONENTS', '5')),
            'metric': os.getenv('UMAP_METRIC', 'cosine'),
            'min_dist': float(os.getenv('UMAP_MIN_DIST', '0.0'))
        },
        'hdbscan_config': {
            'min_cluster_size': int(os.getenv('HDBSCAN_MIN_CLUSTER_SIZE', '10')),
            'metric': os.getenv('HDBSCAN_METRIC', 'euclidean'),
            'cluster_selection_epsilon': float(os.getenv('HDBSCAN_CLUSTER_SELECTION_EPSILON', '0.0'))
        }
    }

    optimization_config = {
        'max_iterations': int(os.getenv('MAX_OPTIMIZATION_ITERATIONS', '5')),
        'convergence_threshold': float(os.getenv('CONVERGENCE_THRESHOLD', '0.02')),
        'batch_size': int(os.getenv('LLM_BATCH_SIZE', '10'))
    }

    quality_config = {
        'min_coherence_score': float(os.getenv('MIN_COHERENCE_SCORE', '0.3')),
        'min_diversity_score': float(os.getenv('MIN_DIVERSITY_SCORE', '0.1')),
        'min_silhouette_score': float(os.getenv('MIN_SILHOUETTE_SCORE', '0.1'))
    }

    resilience_config = {
        'dimensions': {
            'absorb': os.getenv('RESILIENCE_ABSORB_KEYWORDS', 'backup,alternative,stability,resistance,predict,defense').split(','),
            'adapt': os.getenv('RESILIENCE_ADAPT_KEYWORDS', 'adjust,flexible,reallocation,response,repositioning,adaptation').split(','),
            'transform': os.getenv('RESILIENCE_TRANSFORM_KEYWORDS', 'redesign,innovation,transformation,ecosystem,fundamental,revolutionary').split(',')
        }
    }