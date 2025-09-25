"""
Iterative optimizer module for LLM-assisted topic model refinement.

This module implements the core optimization loop with comprehensive quality control,
cost management, and Phase 1 MVP focus on MERGE operations.
"""

import logging
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import copy

# BERTopic and ML libraries
from bertopic import BERTopic
from sklearn.metrics import silhouette_score

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import OPTIMIZER_CONFIG, QUALITY_THRESHOLDS, RESULTS_DIR
from .llm_agent import LLMAgent
from .initial_topic_modeler import create_initial_topics, _calculate_quality_metrics

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for topic model evaluation."""
    coherence: float = 0.0
    silhouette: float = 0.0
    topic_diversity: float = 0.0
    n_topics: int = 0
    coverage: float = 0.0
    iteration: int = 0
    timestamp: str = ""

    def get_combined_score(self) -> float:
        """Get weighted combined quality score."""
        weights = {"coherence": 0.4, "silhouette": 0.3, "topic_diversity": 0.2, "coverage": 0.1}
        return (
            weights["coherence"] * self.coherence +
            weights["silhouette"] * max(0, self.silhouette) +  # Silhouette can be negative
            weights["topic_diversity"] * self.topic_diversity +
            weights["coverage"] * self.coverage
        )

@dataclass
class OptimizationResult:
    """Results from the optimization process."""
    final_model: BERTopic
    final_topics_df: pd.DataFrame
    quality_history: List[QualityMetrics]
    cost_summary: Dict[str, Any]
    optimization_commands: List[List[Dict[str, Any]]]
    metadata: Dict[str, Any]
    success: bool = True
    error_message: str = ""

class Optimizer:
    """
    Iterative topic model optimizer with LLM assistance.

    Phase 1 MVP: MERGE-only operations with strict cost control.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.02,
        quality_history_window: int = 2,
        temperature: float = 0.1,
        model_name: str = "gpt-5",
        enable_split: bool = False,  # Phase 1: False, Phase 2: True
        save_checkpoints: bool = True
    ):
        """
        Initialize the optimizer.

        Args:
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Minimum improvement threshold for convergence
            quality_history_window: Window size for plateau detection
            temperature: LLM temperature for stable outputs
            model_name: LLM model to use
            enable_split: Whether to enable SPLIT operations (Phase 2)
            save_checkpoints: Whether to save intermediate results
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.quality_history_window = quality_history_window
        self.temperature = temperature
        self.model_name = model_name
        self.enable_split = enable_split
        self.save_checkpoints = save_checkpoints

        # Initialize LLM agent
        try:
            self.llm_agent = LLMAgent(
                model_name=model_name,
                temperature=temperature,
                batch_size=OPTIMIZER_CONFIG.get("batch_size", 10)
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM agent: {e}")
            raise

        # Tracking variables
        self.quality_history: List[QualityMetrics] = []
        self.optimization_commands: List[List[Dict[str, Any]]] = []
        self.best_model: Optional[BERTopic] = None
        self.best_quality: Optional[QualityMetrics] = None
        self.current_embeddings: Optional[np.ndarray] = None

        logger.info(f"Optimizer initialized:")
        logger.info(f"  - Max iterations: {max_iterations}")
        logger.info("  - Budget limit: Disabled")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - SPLIT enabled: {enable_split}")

    def run(self,
            documents: List[str],
            initial_model: Optional[BERTopic] = None,
            initial_topics_df: Optional[pd.DataFrame] = None,
            model_phase: str = "phase1"
           ) -> OptimizationResult:
        """
        Run the complete optimization process.

        Args:
            documents: List of documents to analyze
            initial_model: Pre-trained topic model to optimize (optional)
            initial_topics_df: Pre-computed topics dataframe (optional)
            model_phase: Embedding model phase to use if creating new model

        Returns:
            OptimizationResult with final model and analysis
        """
        logger.info(f"Starting optimization process with {len(documents)} documents")
        start_time = datetime.now()

        try:
            # Step 1: Use existing model or create initial topic model
            if initial_model is not None and initial_topics_df is not None:
                logger.info("Step 1: Using provided topic model...")
                # Calculate metadata for consistency
                initial_metadata = {
                    "n_topics": len(initial_model.get_topic_info()) - 1,  # Exclude -1 topic
                    "model_phase": model_phase
                }
            else:
                logger.info("Step 1: Creating initial topic model...")
                initial_model, initial_topics_df, initial_metadata = create_initial_topics(
                    documents, model_phase=model_phase
                )

            # Store embeddings for quality calculations
            # We'll need to regenerate them for consistency
            from sentence_transformers import SentenceTransformer
            from config.settings import EMBEDDING_MODELS
            embedding_model_name = EMBEDDING_MODELS.get(model_phase, EMBEDDING_MODELS["phase1"])
            embedding_model = SentenceTransformer(embedding_model_name)
            self.current_embeddings = embedding_model.encode(
                documents, show_progress_bar=True
            )

            # Initialize tracking
            current_model = initial_model
            current_topics_df = initial_topics_df
            self.best_model = copy.deepcopy(current_model)

            # Record baseline quality
            baseline_quality = self._calculate_quality_metrics(
                current_model, self.current_embeddings, documents, iteration=0
            )
            self.quality_history.append(baseline_quality)
            self.best_quality = baseline_quality

            logger.info(f"Baseline quality metrics:")
            logger.info(f"  - Combined score: {baseline_quality.get_combined_score():.3f}")
            logger.info(f"  - Coherence: {baseline_quality.coherence:.3f}")
            logger.info(f"  - Silhouette: {baseline_quality.silhouette:.3f}")

            # Step 2: Iterative optimization loop
            logger.info("Step 2: Starting iterative optimization...")

            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n=== Iteration {iteration}/{self.max_iterations} ===")

                # Check budget before iteration
                if self._check_budget_exceeded():
                    logger.warning("Budget limit reached, stopping optimization")
                    break

                # Analyze current topics and get optimization commands
                try:
                    commands = self._analyze_topics_and_get_commands(current_topics_df)
                    self.optimization_commands.append(commands)

                    if not commands:
                        logger.info("No optimization commands generated, convergence reached")
                        break

                    logger.info(f"Generated {len(commands)} optimization commands")

                    # Execute commands
                    modified_model = self._execute_commands(
                        current_model, current_topics_df, commands, documents
                    )

                    if modified_model is None:
                        logger.warning("Command execution failed, keeping current model")
                        continue

                    # Calculate new quality metrics
                    new_quality = self._calculate_quality_metrics(
                        modified_model, self.current_embeddings, documents, iteration
                    )

                    logger.info(f"Iteration {iteration} quality:")
                    logger.info(f"  - Combined score: {new_quality.get_combined_score():.3f}")
                    logger.info(f"  - Improvement: {new_quality.get_combined_score() - baseline_quality.get_combined_score():+.3f}")

                    # Check for improvement and update best model
                    if self._is_improvement(new_quality):
                        logger.info("✅ Quality improved, updating best model")
                        self.best_model = copy.deepcopy(modified_model)
                        self.best_quality = new_quality
                        current_model = modified_model
                        current_topics_df = self._extract_topics_dataframe(current_model, documents)
                        # Only add to history when we actually accept the change
                        self.quality_history.append(new_quality)
                    else:
                        logger.info("❌ No significant improvement, keeping previous model")
                        # Do not add rejected changes to quality history

                    # Check convergence
                    if self._check_convergence():
                        logger.info("Convergence criteria met, stopping optimization")
                        break

                    # Save checkpoint if enabled
                    if self.save_checkpoints:
                        self._save_checkpoint(iteration, current_model, current_topics_df)

                except Exception as e:
                    logger.error(f"Iteration {iteration} failed: {e}")
                    continue

            # Step 3: Prepare final results
            logger.info("Step 3: Preparing final results...")

            final_topics_df = self._extract_topics_dataframe(self.best_model, documents)

            # Enhance topics with LLM analysis (if budget allows)
            if not self._check_budget_exceeded():
                final_topics_df = self._enhance_topics_with_llm(final_topics_df)

            # Create result object
            result = OptimizationResult(
                final_model=self.best_model,
                final_topics_df=final_topics_df,
                quality_history=self.quality_history,
                cost_summary=self.llm_agent.get_cost_summary(),
                optimization_commands=self.optimization_commands,
                metadata={
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_iterations": len(self.quality_history) - 1,
                    "initial_quality": asdict(baseline_quality),
                    "final_quality": asdict(self.best_quality) if self.best_quality else {},
                    "improvement": (self.best_quality.get_combined_score() - baseline_quality.get_combined_score()) if self.best_quality else 0,
                    "converged": self._check_convergence(),
                    "budget_exceeded": self._check_budget_exceeded()
                },
                success=True
            )

            # Save final results
            self._save_final_results(result)

            logger.info("🎉 Optimization completed successfully!")
            logger.info(f"Final improvement: {result.metadata['improvement']:+.3f}")
            logger.info(f"Total cost: ${result.cost_summary['total_cost']:.2f}")

            return result

        except Exception as e:
            logger.error(f"Optimization process failed: {e}")
            return OptimizationResult(
                final_model=current_model if 'current_model' in locals() else None,
                final_topics_df=current_topics_df if 'current_topics_df' in locals() else pd.DataFrame(),
                quality_history=self.quality_history,
                cost_summary=self.llm_agent.get_cost_summary() if hasattr(self, 'llm_agent') else {},
                optimization_commands=self.optimization_commands,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )

    def _analyze_topics_and_get_commands(self, topics_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze topics and generate optimization commands."""
        try:
            # Prepare topics data for LLM analysis
            topics_data = []
            for _, row in topics_df.iterrows():
                topic_data = {
                    "topic_id": row["topic_id"],
                    "name": row["name"],
                    "keywords": row["keywords"][:8] if isinstance(row["keywords"], list) else [],
                    "document_count": row.get("document_count", 0),
                    "quality_scores": row.get("quality_scores", {})
                }
                topics_data.append(topic_data)

            # Get optimization commands from LLM
            commands = self.llm_agent.generate_optimization_commands(topics_data)

            # Filter commands based on current phase
            if not self.enable_split:
                commands = [cmd for cmd in commands if cmd["action"] == "MERGE"]
                if len(commands) < len([cmd for cmd in commands if cmd["action"] == "SPLIT"]):
                    logger.info("Filtered out SPLIT commands (Phase 1 MVP)")

            return commands

        except Exception as e:
            logger.error(f"Failed to analyze topics and get commands: {e}")
            return []

    def _execute_commands(
        self,
        model: BERTopic,
        topics_df: pd.DataFrame,
        commands: List[Dict[str, Any]],
        documents: List[str]
    ) -> Optional[BERTopic]:
        """Execute optimization commands on the topic model."""
        try:
            # Create a copy of the model to modify
            modified_model = copy.deepcopy(model)

            executed_commands = 0
            for i, command in enumerate(commands):
                action = command["action"]
                targets = command["targets"]
                reason = command.get("reason", "")

                logger.info(f"Executing command {i+1}: {action} on topics {targets}")
                logger.debug(f"Reason: {reason}")

                try:
                    if action == "MERGE":
                        # Validate targets exist
                        valid_targets = [t for t in targets if t in topics_df["topic_id"].values]
                        if len(valid_targets) < 2:
                            logger.warning(f"Insufficient valid targets for MERGE: {valid_targets}")
                            continue

                        # Execute merge operation
                        modified_model.merge_topics(documents, valid_targets)
                        executed_commands += 1
                        logger.info(f"✅ Successfully merged topics {valid_targets}")

                    elif action == "SPLIT" and self.enable_split:
                        # Phase 2: Implement topic splitting
                        logger.info("SPLIT operation not yet implemented (Phase 2)")
                        continue

                    else:
                        logger.warning(f"Unknown or disabled action: {action}")
                        continue

                except Exception as e:
                    logger.error(f"Failed to execute {action} on {targets}: {e}")
                    continue

            if executed_commands > 0:
                logger.info(f"Successfully executed {executed_commands} commands")
                return modified_model
            else:
                logger.warning("No commands were successfully executed")
                return None

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return None

    def _calculate_quality_metrics(
        self,
        model: BERTopic,
        embeddings: np.ndarray,
        documents: List[str],
        iteration: int
    ) -> QualityMetrics:
        """Calculate quality metrics for a topic model."""
        try:
            topics = model.topics_

            # Use the existing quality calculation function
            from .initial_topic_modeler import _calculate_quality_metrics
            base_metrics = _calculate_quality_metrics(model, embeddings, topics, documents)

            return QualityMetrics(
                coherence=base_metrics.get("coherence", 0.0),
                silhouette=base_metrics.get("silhouette", 0.0),
                topic_diversity=base_metrics.get("topic_diversity", 0.0),
                n_topics=base_metrics.get("n_topics", 0),
                coverage=base_metrics.get("coverage", 0.0),
                iteration=iteration,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {e}")
            return QualityMetrics(iteration=iteration, timestamp=datetime.now().isoformat())

    def _is_improvement(self, new_quality: QualityMetrics) -> bool:
        """Check if new quality metrics represent an improvement."""
        if not self.quality_history:
            return True

        previous_quality = self.quality_history[-2] if len(self.quality_history) >= 2 else self.quality_history[-1]

        improvement = new_quality.get_combined_score() - previous_quality.get_combined_score()

        # Check for significant improvement
        significant_improvement = improvement >= self.convergence_threshold

        # Check for significant degradation
        max_degradation = QUALITY_THRESHOLDS.get("max_quality_degradation", -0.05)
        significant_degradation = improvement <= max_degradation

        logger.debug(f"Quality change: {improvement:+.3f} (threshold: {self.convergence_threshold})")

        if significant_degradation:
            logger.warning(f"Significant quality degradation detected: {improvement:+.3f}")
            return False

        return significant_improvement

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.quality_history) < self.quality_history_window + 1:
            return False

        # Get recent quality scores
        recent_scores = [q.get_combined_score() for q in self.quality_history[-self.quality_history_window-1:]]

        # Check if improvement has plateaued
        improvements = [recent_scores[i+1] - recent_scores[i] for i in range(len(recent_scores)-1)]
        avg_improvement = np.mean(improvements)

        logger.debug(f"Recent improvements: {improvements}")
        logger.debug(f"Average improvement: {avg_improvement:.4f}")

        return abs(avg_improvement) < self.convergence_threshold

    def _check_budget_exceeded(self) -> bool:
        """Budget checking disabled."""
        return False

    def _extract_topics_dataframe(self, model: BERTopic, documents: List[str]) -> pd.DataFrame:
        """Extract topics information as DataFrame."""
        try:
            from .initial_topic_modeler import _extract_topic_info
            return _extract_topic_info(model, documents, model.topics_)
        except Exception as e:
            logger.error(f"Failed to extract topics dataframe: {e}")
            return pd.DataFrame()

    def _enhance_topics_with_llm(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance topic names and summaries using LLM."""
        logger.info("Enhancing topics with LLM analysis...")

        enhanced_df = topics_df.copy()

        for idx, row in enhanced_df.iterrows():
            try:
                if self._check_budget_exceeded():
                    logger.warning("Budget exceeded, stopping topic enhancement")
                    break

                keywords = row["keywords"][:8] if isinstance(row["keywords"], list) else []

                # Get enhanced name and summary
                enhancement = self.llm_agent.get_topic_name_and_summary(keywords)

                enhanced_df.at[idx, "name"] = enhancement.get("name", row["name"])
                enhanced_df.at[idx, "summary"] = enhancement.get("summary", "")

                # Get quality scores
                quality = self.llm_agent.evaluate_topic_quality(
                    enhanced_df.at[idx, "name"], keywords
                )
                enhanced_df.at[idx, "llm_quality_scores"] = quality

            except Exception as e:
                logger.error(f"Failed to enhance topic {row.get('topic_id', idx)}: {e}")
                continue

        logger.info("Topic enhancement completed")
        return enhanced_df

    def _save_checkpoint(self, iteration: int, model: BERTopic, topics_df: pd.DataFrame):
        """Save checkpoint during optimization."""
        try:
            checkpoint_dir = RESULTS_DIR / "checkpoints" / f"iteration_{iteration}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = checkpoint_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save topics
            topics_df.to_csv(checkpoint_dir / "topics.csv", index=False)
            topics_df.to_pickle(checkpoint_dir / "topics.pkl")

            # Save quality history
            quality_data = [asdict(q) for q in self.quality_history]
            with open(checkpoint_dir / "quality_history.json", 'w') as f:
                json.dump(quality_data, f, indent=2)

            logger.debug(f"Checkpoint saved: {checkpoint_dir}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _save_final_results(self, result: OptimizationResult):
        """Save final optimization results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = RESULTS_DIR / "optimization" / f"optimization_{timestamp}"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save final model
            model_path = results_dir / "final_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result.final_model, f)

            # Save final topics
            result.final_topics_df.to_csv(results_dir / "final_topics.csv", index=False)
            result.final_topics_df.to_pickle(results_dir / "final_topics.pkl")

            # Save complete results
            results_data = {
                "metadata": result.metadata,
                "cost_summary": result.cost_summary,
                "quality_history": [asdict(q) for q in result.quality_history],
                "optimization_commands": result.optimization_commands,
                "success": result.success,
                "error_message": result.error_message
            }

            with open(results_dir / "optimization_results.json", 'w') as f:
                json.dump(results_data, f, indent=2, default=str)

            logger.info(f"Final results saved to: {results_dir}")

        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample documents
    sample_docs = [
        "The company implemented new cybersecurity measures",
        "Digital transformation improved operational efficiency",
        "Supply chain resilience through diversification",
        "Remote work enabled business continuity",
        "Cloud infrastructure provided scalability",
        "Data analytics for predictive maintenance",
        "AI automation of business processes",
        "Customer experience digitalization",
        "Sustainability and environmental compliance",
        "Financial risk management strategies"
    ] * 10  # Repeat for enough documents

    try:
        # Test optimizer (requires OpenAI API key)
        optimizer = Optimizer(
            max_iterations=2,
            model_name="gpt-5-nano-2025-08-07"
        )

        print("🔄 Testing optimization process...")
        result = optimizer.run(sample_docs)

        if result.success:
            print("✅ Optimization completed successfully!")
            print(f"💰 Total cost: ${result.cost_summary['total_cost']:.4f}")
            print(f"📈 Quality improvement: {result.metadata.get('improvement', 0):+.3f}")
            print(f"🎯 Final topics: {len(result.final_topics_df)}")
        else:
            print(f"❌ Optimization failed: {result.error_message}")

    except Exception as e:
        print(f"⚠️ Test skipped (requires configuration): {e}")