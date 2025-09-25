"""
Main pipeline orchestration for the augmentation-agent-topic system.

This module provides the command-line interface and orchestrates the complete
pipeline from data loading through resilience scoring.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables early
from config.env_loader import load_environment
env_loader = load_environment()

# Internal imports
from config.settings import get_config, validate_config, LOGS_DIR, RESULTS_DIR
from src.data_loader import load_corpus, get_corpus_statistics, sample_corpus
from src.initial_topic_modeler import create_initial_topics
from src.llm_agent import LLMAgent
from src.iterative_optimizer import Optimizer
from src.resilience_mapper import map_topics_to_resilience, calculate_resilience_scores, save_resilience_results

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    LOGS_DIR.mkdir(exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"pipeline_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with environment defaults."""
    # Get defaults from environment
    api_config = env_loader.get_api_config()
    optimization_config = env_loader.get_optimization_config()
    data_config = env_loader.get_data_config()
    logging_config = env_loader.get_logging_config()

    parser = argparse.ArgumentParser(
        description="Augmentation Agent Topic - LLM-assisted iterative topic modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Environment selection
    parser.add_argument(
        "--env",
        type=str,
        default=env_loader.env_name,
        help="Environment to use (development, production, testing)"
    )

    # Data arguments
    parser.add_argument(
        "--data-file",
        type=str,
        default=data_config['default_file'],
        help="Path to the corpus CSV file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=data_config['default_sample_size'] if data_config['default_sample_size'] > 0 else None,
        help="Use a sample of N documents (for testing)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=api_config['default_model'],
        choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        help="LLM model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=api_config['temperature'],
        help="Temperature for LLM generation (0.0-1.0)"
    )

    # Optimization arguments
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=optimization_config['max_iterations'],
        help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=api_config['max_budget'],
        help="API budget limit in USD"
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=optimization_config['convergence_threshold'],
        help="Quality improvement threshold for convergence"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip iterative optimization, use initial topics only"
    )
    parser.add_argument(
        "--skip-resilience",
        action="store_true",
        help="Skip resilience mapping and scoring"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results"
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default=logging_config['level'],
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    # Environment and configuration
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit"
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )

    return parser.parse_args()

def load_and_validate_data(data_file: str, sample_size: Optional[int] = None) -> tuple:
    """Load and validate corpus data."""
    logger = logging.getLogger(__name__)

    # Check if data file exists
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"Loading corpus data from: {data_file}")

    # Load corpus
    corpus_df = load_corpus(data_file)
    logger.info(f"Loaded {len(corpus_df)} documents")

    # Get statistics
    stats = get_corpus_statistics(corpus_df)
    logger.info(f"Corpus statistics:")
    logger.info(f"  - Companies: {stats['unique_companies']}")
    logger.info(f"  - Year range: {stats['year_range']}")
    logger.info(f"  - Avg text length: {stats['text_length_stats']['mean']:.0f} chars")

    # Sample if requested
    if sample_size and sample_size < len(corpus_df):
        logger.info(f"Creating sample of {sample_size} documents")
        corpus_df = sample_corpus(corpus_df, n_samples=sample_size, sample_by_company=True)
        stats = get_corpus_statistics(corpus_df)
        logger.info(f"Sample statistics: {stats['unique_companies']} companies, {len(corpus_df)} documents")

    # Extract documents list
    documents = corpus_df['text'].tolist()

    return corpus_df, documents, stats

def run_topic_modeling(documents: list, args: argparse.Namespace) -> tuple:
    """Run initial topic modeling."""
    logger = logging.getLogger(__name__)

    logger.info("Starting initial topic modeling...")

    # Create initial topics
    topic_model, topics_df, metadata = create_initial_topics(
        documents,
        model_phase="phase1"  # MVP uses proven sentence-transformers
    )

    logger.info(f"Initial topic modeling completed:")
    logger.info(f"  - Topics created: {metadata['n_topics']}")
    logger.info(f"  - Quality metrics: {metadata['quality_metrics']}")

    return topic_model, topics_df, metadata

def run_optimization(
    documents: list,
    topic_model,
    topics_df,
    args: argparse.Namespace
) -> tuple:
    """Run iterative optimization."""
    logger = logging.getLogger(__name__)

    if args.skip_optimization:
        logger.info("Skipping optimization (--skip-optimization specified)")
        return topic_model, topics_df, {"skipped": True}

    logger.info("Starting iterative optimization...")

    # Initialize optimizer
    optimizer = Optimizer(
        max_iterations=args.max_iterations,
        api_budget_limit=args.budget,
        temperature=args.temperature,
        model_name=args.model,
        convergence_threshold=args.convergence_threshold,
        enable_split=False,  # Phase 1: MERGE-only
        save_checkpoints=args.save_intermediate
    )

    # Run optimization
    result = optimizer.run(documents)

    if result.success:
        logger.info("Optimization completed successfully:")
        logger.info(f"  - Iterations: {result.metadata.get('total_iterations', 0)}")
        logger.info(f"  - Quality improvement: {result.metadata.get('improvement', 0):+.3f}")
        logger.info(f"  - Total cost: ${result.cost_summary.get('total_cost', 0):.2f}")
    else:
        logger.error(f"Optimization failed: {result.error_message}")

    return result.final_model, result.final_topics_df, result.metadata

def run_resilience_analysis(
    corpus_df,
    documents: list,
    topic_model,
    topics_df,
    args: argparse.Namespace
) -> tuple:
    """Run resilience mapping and scoring."""
    logger = logging.getLogger(__name__)

    if args.skip_resilience:
        logger.info("Skipping resilience analysis (--skip-resilience specified)")
        return None, None, {"skipped": True}

    logger.info("Starting resilience analysis...")

    try:
        # Initialize LLM agent for resilience mapping
        llm_agent = LLMAgent(
            model_name="gpt-3.5-turbo",  # More cost-effective for classification
            temperature=0.1,
            budget_limit=min(20.0, args.budget * 0.4)  # Reserve budget for mapping
        )

        # Map topics to resilience dimensions
        logger.info("Mapping topics to resilience framework...")
        topic_mappings, mapping_metadata = map_topics_to_resilience(
            topics_df,
            llm_agent=llm_agent,
            batch_size=5
        )

        logger.info(f"Topic mapping completed:")
        logger.info(f"  - Successful mappings: {mapping_metadata['successful_mappings']}")
        logger.info(f"  - Mapping cost: ${mapping_metadata['cost_summary']['total_cost']:.2f}")

        # Calculate resilience scores
        logger.info("Calculating company resilience scores...")
        topic_assignments = topic_model.topics_

        resilience_scores = calculate_resilience_scores(
            corpus_df,
            topic_assignments,
            topic_mappings,
            aggregation_method="weighted_average"
        )

        logger.info(f"Resilience scoring completed:")
        logger.info(f"  - Company-year combinations: {len(resilience_scores)}")
        logger.info(f"  - Average total resilience score: {resilience_scores['total_resilience_score'].mean():.3f}")

        # Combine metadata
        resilience_metadata = {
            **mapping_metadata,
            "scoring_method": "weighted_average",
            "total_company_years": len(resilience_scores)
        }

        return resilience_scores, topic_mappings, resilience_metadata

    except Exception as e:
        logger.error(f"Resilience analysis failed: {e}")
        return None, None, {"error": str(e)}

def save_results(
    corpus_df,
    topic_model,
    topics_df,
    topic_metadata,
    optimization_metadata,
    resilience_scores,
    topic_mappings,
    resilience_metadata,
    args: argparse.Namespace
) -> str:
    """Save all results to output directory."""
    logger = logging.getLogger(__name__)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / "pipeline" / f"run_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {output_dir}")

    try:
        # Save corpus statistics
        corpus_stats = get_corpus_statistics(corpus_df)
        with open(output_dir / "corpus_statistics.json", 'w') as f:
            json.dump(corpus_stats, f, indent=2, default=str)

        # Save topic model and topics
        import pickle
        with open(output_dir / "topic_model.pkl", 'wb') as f:
            pickle.dump(topic_model, f)

        topics_df.to_csv(output_dir / "topics.csv", index=False)
        topics_df.to_pickle(output_dir / "topics.pkl")

        # Save topic modeling metadata
        with open(output_dir / "topic_metadata.json", 'w') as f:
            json.dump(topic_metadata, f, indent=2, default=str)

        # Save optimization metadata
        if optimization_metadata and not optimization_metadata.get("skipped"):
            with open(output_dir / "optimization_metadata.json", 'w') as f:
                json.dump(optimization_metadata, f, indent=2, default=str)

        # Save resilience analysis results
        if resilience_scores is not None and not resilience_metadata.get("skipped"):
            save_resilience_results(
                resilience_scores,
                topic_mappings,
                resilience_metadata,
                output_dir / "resilience"
            )

        # Save pipeline configuration
        pipeline_config = {
            "arguments": vars(args),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "pipeline_stages": {
                "data_loading": True,
                "topic_modeling": True,
                "optimization": not args.skip_optimization,
                "resilience_analysis": not args.skip_resilience
            }
        }

        with open(output_dir / "pipeline_config.json", 'w') as f:
            json.dump(pipeline_config, f, indent=2, default=str)

        # Create summary report
        create_summary_report(
            output_dir,
            corpus_df,
            topics_df,
            topic_metadata,
            optimization_metadata,
            resilience_scores,
            resilience_metadata,
            args
        )

        logger.info("✅ All results saved successfully")
        return str(output_dir)

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def create_summary_report(
    output_dir: Path,
    corpus_df,
    topics_df,
    topic_metadata,
    optimization_metadata,
    resilience_scores,
    resilience_metadata,
    args: argparse.Namespace
) -> None:
    """Create a summary report of the pipeline run."""
    logger = logging.getLogger(__name__)

    try:
        report = {
            "pipeline_summary": {
                "timestamp": datetime.now().isoformat(),
                "data_file": args.data_file,
                "model_used": args.model,
                "total_budget": args.budget,
                "sample_size": args.sample_size
            },
            "data_summary": {
                "total_documents": len(corpus_df),
                "unique_companies": corpus_df['ticker'].nunique(),
                "year_range": f"{corpus_df['year'].min()}-{corpus_df['year'].max()}",
                "avg_text_length": corpus_df['text'].str.len().mean()
            },
            "topic_modeling_summary": {
                "initial_topics": topic_metadata.get('n_topics', 0),
                "quality_metrics": topic_metadata.get('quality_metrics', {}),
                "embedding_model": topic_metadata.get('embedding_model', 'unknown')
            }
        }

        # Add optimization summary
        if optimization_metadata and not optimization_metadata.get("skipped"):
            report["optimization_summary"] = {
                "iterations_completed": optimization_metadata.get('total_iterations', 0),
                "quality_improvement": optimization_metadata.get('improvement', 0),
                "api_cost": optimization_metadata.get('cost_summary', {}).get('total_cost', 0),
                "converged": optimization_metadata.get('converged', False),
                "budget_exceeded": optimization_metadata.get('budget_exceeded', False)
            }

        # Add resilience analysis summary
        if resilience_scores is not None and not resilience_metadata.get("skipped"):
            report["resilience_summary"] = {
                "company_year_combinations": len(resilience_scores),
                "successful_mappings": resilience_metadata.get('successful_mappings', 0),
                "mapping_cost": resilience_metadata.get('cost_summary', {}).get('total_cost', 0),
                "avg_resilience_scores": {
                    "absorb": resilience_scores['absorb_score'].mean(),
                    "adapt": resilience_scores['adapt_score'].mean(),
                    "transform": resilience_scores['transform_score'].mean(),
                    "total": resilience_scores['total_resilience_score'].mean()
                }
            }

        # Calculate total cost
        total_cost = 0
        if optimization_metadata and not optimization_metadata.get("skipped"):
            total_cost += optimization_metadata.get('cost_summary', {}).get('total_cost', 0)
        if resilience_metadata and not resilience_metadata.get("skipped"):
            total_cost += resilience_metadata.get('cost_summary', {}).get('total_cost', 0)

        report["cost_summary"] = {
            "total_api_cost": total_cost,
            "budget_used_percent": (total_cost / args.budget) * 100,
            "budget_remaining": max(0, args.budget - total_cost)
        }

        # Save report
        with open(output_dir / "pipeline_summary.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("📊 Summary report created")

    except Exception as e:
        logger.warning(f"Failed to create summary report: {e}")

def main() -> int:
    """Main pipeline function."""
    global env_loader
    try:
        # Parse arguments
        args = parse_arguments()

        # Handle environment switching
        if hasattr(args, 'env') and args.env != env_loader.env_name:
            from config.env_loader import load_environment
            env_loader = load_environment(args.env)
            from config.settings import reload_config
            reload_config(args.env)

        # Handle configuration commands
        if args.show_config:
            env_loader.print_config_summary()
            return 0

        if args.validate_config:
            is_valid, errors = env_loader.validate_config()
            if is_valid:
                print("✅ Configuration is valid")
                return 0
            else:
                print("❌ Configuration errors found:")
                for error in errors:
                    print(f"  - {error}")
                return 1

        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)

        logger.info("🚀 Starting Augmentation Agent Topic Pipeline")
        logger.info(f"Environment: {env_loader.env_name}")
        logger.info(f"Configuration: {vars(args)}")

        # Show configuration summary in debug mode
        feature_flags = env_loader.get_feature_flags()
        if feature_flags['development_mode'] or args.log_level == 'DEBUG':
            logger.info("Configuration Summary:")
            config_summary = env_loader.get_all_config()
            for section, values in config_summary.items():
                if section not in ['loaded_files']:
                    logger.debug(f"  {section}: {values}")

        # Validate configuration
        if not validate_config():
            logger.error("❌ Configuration validation failed")
            return 1

        # Step 1: Load and validate data
        logger.info("\n=== Step 1: Data Loading ===")
        corpus_df, documents, corpus_stats = load_and_validate_data(
            args.data_file, args.sample_size
        )

        # Step 2: Initial topic modeling
        logger.info("\n=== Step 2: Initial Topic Modeling ===")
        topic_model, topics_df, topic_metadata = run_topic_modeling(documents, args)

        # Step 3: Iterative optimization
        logger.info("\n=== Step 3: Iterative Optimization ===")
        optimized_model, optimized_topics_df, optimization_metadata = run_optimization(
            documents, topic_model, topics_df, args
        )

        # Step 4: Resilience analysis
        logger.info("\n=== Step 4: Resilience Analysis ===")
        resilience_scores, topic_mappings, resilience_metadata = run_resilience_analysis(
            corpus_df, documents, optimized_model, optimized_topics_df, args
        )

        # Step 5: Save results
        logger.info("\n=== Step 5: Saving Results ===")
        output_path = save_results(
            corpus_df,
            optimized_model,
            optimized_topics_df,
            topic_metadata,
            optimization_metadata,
            resilience_scores,
            topic_mappings,
            resilience_metadata,
            args
        )

        # Success summary
        logger.info("\n🎉 Pipeline completed successfully!")
        logger.info(f"📁 Results saved to: {output_path}")

        if not args.quiet:
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📁 Results location: {output_path}")
            print(f"📊 Final topics: {len(optimized_topics_df)}")
            if resilience_scores is not None:
                print(f"🏢 Companies analyzed: {resilience_scores['ticker'].nunique()}")
                print(f"📈 Avg resilience score: {resilience_scores['total_resilience_score'].mean():.3f}")
            print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if not args.quiet:
            print(f"\n❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())