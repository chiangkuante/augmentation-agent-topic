"""
Environment variable loader with dotenv support.

This module handles loading environment variables from .env files
and provides utilities for environment configuration.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv, find_dotenv

logger = logging.getLogger(__name__)

class EnvironmentLoader:
    """Handles loading and managing environment variables."""

    def __init__(self, env_name: Optional[str] = None, auto_load: bool = True):
        """
        Initialize the environment loader.

        Args:
            env_name: Environment name ('development', 'production', 'testing')
            auto_load: Whether to automatically load environment files
        """
        self.project_root = Path(__file__).parent.parent
        self.env_name = env_name or os.getenv('ENVIRONMENT', 'development')
        self.loaded_files = []

        if auto_load:
            self.load_environment()

    def load_environment(self) -> None:
        """Load environment variables from appropriate .env files."""
        logger.info(f"Loading environment configuration for: {self.env_name}")

        # Priority order for loading .env files:
        # 1. .env.{environment} (specific environment)
        # 2. .env.local (local overrides, should be gitignored)
        # 3. .env (default/fallback)

        env_files = [
            self.project_root / f".env.{self.env_name}",
            self.project_root / ".env.local",
            self.project_root / ".env"
        ]

        # Also try to find .env files automatically
        auto_env = find_dotenv(usecwd=True)
        if auto_env and auto_env not in [str(f) for f in env_files]:
            env_files.append(Path(auto_env))

        for env_file in env_files:
            if env_file.exists():
                try:
                    load_dotenv(env_file, override=False)  # Don't override already set vars
                    self.loaded_files.append(str(env_file))
                    logger.debug(f"Loaded environment file: {env_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {env_file}: {e}")

        if self.loaded_files:
            logger.info(f"Loaded {len(self.loaded_files)} environment files")
        else:
            logger.warning("No environment files found, using system environment only")

    def get_env(
        self,
        key: str,
        default: Any = None,
        var_type: type = str,
        required: bool = False
    ) -> Any:
        """
        Get environment variable with type conversion and validation.

        Args:
            key: Environment variable name
            default: Default value if not found
            var_type: Type to convert to (str, int, float, bool, list)
            required: Whether the variable is required

        Returns:
            Converted environment variable value

        Raises:
            ValueError: If required variable is missing or conversion fails
        """
        value = os.getenv(key)

        if value is None:
            if required:
                raise ValueError(f"Required environment variable '{key}' is not set")
            return default

        # Convert string value to requested type
        try:
            if var_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif var_type == int:
                return int(value)
            elif var_type == float:
                return float(value)
            elif var_type == list:
                # Handle comma-separated lists
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                return str(value)

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert env var '{key}' = '{value}' to {var_type.__name__}: {e}")
            if required:
                raise ValueError(f"Invalid value for required env var '{key}': {value}")
            return default

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration from environment."""
        return {
            'openai_api_key': self.get_env('OPENAI_API_KEY', required=True),
            'max_budget': self.get_env('MAX_API_BUDGET', 50.0, float),
            'budget_warning_threshold': self.get_env('BUDGET_WARNING_THRESHOLD', 0.8, float),
            'budget_emergency_threshold': self.get_env('BUDGET_EMERGENCY_THRESHOLD', 0.95, float),
            'default_model': self.get_env('DEFAULT_LLM_MODEL', 'gpt-4'),
            'temperature': self.get_env('LLM_TEMPERATURE', 0.1, float),
            'max_retries': self.get_env('MAX_API_RETRIES', 3, int),
            'rate_limit': self.get_env('API_RATE_LIMIT', 60, int),
            'validate_keys': self.get_env('VALIDATE_API_KEYS', True, bool)
        }

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration from environment."""
        return {
            'max_iterations': self.get_env('MAX_OPTIMIZATION_ITERATIONS', 5, int),
            'convergence_threshold': self.get_env('CONVERGENCE_THRESHOLD', 0.02, float),
            'quality_history_window': self.get_env('QUALITY_HISTORY_WINDOW', 2, int),
            'batch_size': self.get_env('TOPIC_ANALYSIS_BATCH_SIZE', 10, int),
            'enable_split': self.get_env('ENABLE_TOPIC_SPLIT', False, bool),
            'save_checkpoints': self.get_env('SAVE_CHECKPOINTS', True, bool)
        }

    def get_topic_modeling_config(self) -> Dict[str, Any]:
        """Get topic modeling configuration from environment."""
        return {
            'min_topic_size': self.get_env('MIN_TOPIC_SIZE', 10, int),
            'embedding_phase': self.get_env('DEFAULT_EMBEDDING_PHASE', 'phase1'),
            'use_gpu': self.get_env('USE_GPU', False, bool),
            'batch_size': self.get_env('EMBEDDING_BATCH_SIZE', 32, int),
            'umap_config': {
                'n_neighbors': self.get_env('UMAP_N_NEIGHBORS', 15, int),
                'n_components': self.get_env('UMAP_N_COMPONENTS', 5, int),
                'min_dist': self.get_env('UMAP_MIN_DIST', 0.0, float),
                'metric': self.get_env('UMAP_METRIC', 'cosine')
            },
            'hdbscan_config': {
                'min_cluster_size': self.get_env('HDBSCAN_MIN_CLUSTER_SIZE', 10, int),
                'metric': self.get_env('HDBSCAN_METRIC', 'euclidean')
            }
        }

    def get_quality_thresholds(self) -> Dict[str, Any]:
        """Get quality threshold configuration from environment."""
        return {
            'min_coherence_improvement': self.get_env('MIN_COHERENCE_IMPROVEMENT', 0.1, float),
            'max_quality_degradation': self.get_env('MAX_QUALITY_DEGRADATION', -0.05, float),
            'min_silhouette_score': self.get_env('MIN_SILHOUETTE_SCORE', 0.1, float),
            'min_topic_diversity': self.get_env('MIN_TOPIC_DIVERSITY', 0.5, float)
        }

    def get_resilience_config(self) -> Dict[str, Any]:
        """Get resilience framework configuration from environment."""
        return {
            'confidence_threshold': self.get_env('RESILIENCE_CONFIDENCE_THRESHOLD', 0.3, float),
            'budget_ratio': self.get_env('RESILIENCE_ANALYSIS_BUDGET_RATIO', 0.4, float),
            'scoring_method': self.get_env('RESILIENCE_SCORING_METHOD', 'weighted_average'),
            'dimensions': {
                'absorb': 'Absorption Capacity',
                'adapt': 'Adaptive Capacity',
                'transform': 'Transformation Capacity'
            }
        }

    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration from environment."""
        return {
            'default_file': self.get_env('DEFAULT_DATA_FILE', 'data/corpus_semantic_chunks.csv'),
            'encoding_options': self.get_env('CSV_ENCODING_OPTIONS', ['utf-8', 'latin-1'], list),
            'min_text_length': self.get_env('MIN_TEXT_LENGTH', 10, int),
            'default_sample_size': self.get_env('DEFAULT_SAMPLE_SIZE', 0, int)
        }

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration from environment."""
        return {
            'level': self.get_env('LOG_LEVEL', 'INFO'),
            'verbose': self.get_env('VERBOSE_LOGGING', True, bool),
            'mask_sensitive': self.get_env('MASK_SENSITIVE_INFO', True, bool)
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration from environment."""
        return {
            'max_memory_gb': self.get_env('MAX_MEMORY_USAGE_GB', 8, int),
            'n_workers': self.get_env('N_WORKERS', 0, int),
            'use_gpu': self.get_env('USE_GPU', False, bool)
        }

    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags from environment."""
        return {
            'experimental_features': self.get_env('ENABLE_EXPERIMENTAL_FEATURES', False, bool),
            'modernbert': self.get_env('ENABLE_MODERNBERT', False, bool),
            'local_llm': self.get_env('ENABLE_LOCAL_LLM', False, bool),
            'advanced_visualizations': self.get_env('ENABLE_ADVANCED_VISUALIZATIONS', True, bool),
            'real_time_monitoring': self.get_env('ENABLE_REAL_TIME_MONITORING', False, bool),
            'development_mode': self.get_env('DEVELOPMENT_MODE', False, bool)
        }

    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return {
            'environment': self.env_name,
            'loaded_files': self.loaded_files,
            'api': self.get_api_config(),
            'optimization': self.get_optimization_config(),
            'topic_modeling': self.get_topic_modeling_config(),
            'quality_thresholds': self.get_quality_thresholds(),
            'resilience': self.get_resilience_config(),
            'data': self.get_data_config(),
            'logging': self.get_logging_config(),
            'performance': self.get_performance_config(),
            'feature_flags': self.get_feature_flags()
        }

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        Validate the current configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Validate API configuration
            api_config = self.get_api_config()

            if not api_config['openai_api_key'] or api_config['openai_api_key'] == 'your-openai-api-key-here':
                errors.append("OPENAI_API_KEY is not set or using placeholder value")

            if api_config['max_budget'] <= 0:
                errors.append("MAX_API_BUDGET must be positive")

            if not 0 <= api_config['temperature'] <= 1:
                errors.append("LLM_TEMPERATURE must be between 0.0 and 1.0")

            # Validate optimization configuration
            opt_config = self.get_optimization_config()

            if opt_config['max_iterations'] <= 0:
                errors.append("MAX_OPTIMIZATION_ITERATIONS must be positive")

            if opt_config['convergence_threshold'] <= 0:
                errors.append("CONVERGENCE_THRESHOLD must be positive")

            # Validate paths
            project_paths = [
                self.project_root / "data",
                self.project_root / "results",
                self.project_root / "logs"
            ]

            for path in project_paths:
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {path}")
                    except Exception as e:
                        errors.append(f"Cannot create directory {path}: {e}")

        except Exception as e:
            errors.append(f"Configuration validation error: {e}")

        return len(errors) == 0, errors

    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        config = self.get_all_config()

        print(f"\n{'='*60}")
        print(f"AUGMENTATION AGENT TOPIC - CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"Environment: {config['environment']}")
        print(f"Loaded files: {len(config['loaded_files'])}")
        for file in config['loaded_files']:
            print(f"  - {file}")

        print(f"\nAPI Configuration:")
        print(f"  - Model: {config['api']['default_model']}")
        print(f"  - Budget: ${config['api']['max_budget']}")
        print(f"  - Temperature: {config['api']['temperature']}")

        print(f"\nOptimization:")
        print(f"  - Max iterations: {config['optimization']['max_iterations']}")
        print(f"  - Convergence threshold: {config['optimization']['convergence_threshold']}")
        print(f"  - SPLIT enabled: {config['optimization']['enable_split']}")

        print(f"\nFeature Flags:")
        for flag, enabled in config['feature_flags'].items():
            status = "✅" if enabled else "❌"
            print(f"  - {flag}: {status}")

        print(f"{'='*60}\n")

# Global environment loader instance
_env_loader = None

def get_env_loader(env_name: Optional[str] = None, reload: bool = False) -> EnvironmentLoader:
    """
    Get the global environment loader instance.

    Args:
        env_name: Environment name to load
        reload: Whether to reload the environment

    Returns:
        EnvironmentLoader instance
    """
    global _env_loader

    if _env_loader is None or reload:
        _env_loader = EnvironmentLoader(env_name=env_name)

    return _env_loader

def load_environment(env_name: Optional[str] = None) -> EnvironmentLoader:
    """
    Load environment configuration.

    Args:
        env_name: Environment name ('development', 'production', etc.)

    Returns:
        EnvironmentLoader instance
    """
    return get_env_loader(env_name=env_name, reload=True)

if __name__ == "__main__":
    # Test environment loading
    import logging
    logging.basicConfig(level=logging.INFO)

    env_loader = load_environment()
    env_loader.print_config_summary()

    # Validate configuration
    is_valid, errors = env_loader.validate_config()

    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")