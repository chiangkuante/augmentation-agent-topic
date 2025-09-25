"""
Resilience mapping module for digital resilience framework analysis.

This module maps optimized topics to the digital resilience framework
(Absorption, Adaptive, Transformation capacities) and calculates resilience scores.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

# Internal imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import RESILIENCE_DIMENSIONS, RESULTS_DIR
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

@dataclass
class ResilienceMapping:
    """Represents the mapping of a topic to resilience dimensions."""
    topic_id: int
    topic_name: str
    absorb_score: float
    adapt_score: float
    transform_score: float
    primary_dimension: str
    confidence: float
    reasoning: str

class ResilienceMappingError(Exception):
    """Custom exception for resilience mapping errors."""
    pass

def map_topics_to_resilience(
    topics_df: pd.DataFrame,
    llm_agent: Optional[LLMAgent] = None,
    batch_size: int = 5,
    confidence_threshold: float = 0.3
) -> Tuple[Dict[int, ResilienceMapping], Dict[str, Any]]:
    """
    Map topics to digital resilience framework dimensions.

    Args:
        topics_df: DataFrame with topic information
        llm_agent: LLM agent for analysis (creates new if None)
        batch_size: Number of topics to process in each batch
        confidence_threshold: Minimum confidence threshold for mapping

    Returns:
        Tuple of (mapping dictionary, metadata)

    Raises:
        ResilienceMappingError: If mapping process fails
    """
    logger.info(f"Starting resilience mapping for {len(topics_df)} topics")

    # Initialize LLM agent if not provided
    if llm_agent is None:
        try:
            llm_agent = LLMAgent(
                model_name="gpt-3.5-turbo",  # More cost-effective for classification
                temperature=0.1,
                budget_limit=20.0  # Conservative budget for mapping
            )
        except Exception as e:
            raise ResilienceMappingError(f"Failed to initialize LLM agent: {e}")

    try:
        mappings = {}
        failed_mappings = []
        metadata = {
            "total_topics": len(topics_df),
            "successful_mappings": 0,
            "failed_mappings": 0,
            "low_confidence_mappings": 0,
            "timestamp": datetime.now().isoformat(),
            "resilience_distribution": {"absorb": 0, "adapt": 0, "transform": 0}
        }

        # Process topics in batches to manage cost and efficiency
        for i in range(0, len(topics_df), batch_size):
            batch = topics_df.iloc[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: topics {i} to {min(i+batch_size-1, len(topics_df)-1)}")

            # Check budget before processing batch
            if llm_agent._check_budget_exceeded():
                logger.warning("Budget limit reached during resilience mapping")
                break

            # Process each topic in the batch
            for _, topic_row in batch.iterrows():
                try:
                    mapping = _map_single_topic_to_resilience(topic_row, llm_agent)

                    if mapping.confidence >= confidence_threshold:
                        mappings[mapping.topic_id] = mapping
                        metadata["successful_mappings"] += 1
                        metadata["resilience_distribution"][mapping.primary_dimension] += 1
                    else:
                        logger.warning(f"Low confidence mapping for topic {mapping.topic_id}: {mapping.confidence}")
                        mappings[mapping.topic_id] = mapping  # Keep but flag
                        metadata["low_confidence_mappings"] += 1

                except Exception as e:
                    logger.error(f"Failed to map topic {topic_row.get('topic_id', 'unknown')}: {e}")
                    failed_mappings.append({
                        "topic_id": topic_row.get("topic_id", "unknown"),
                        "error": str(e)
                    })
                    metadata["failed_mappings"] += 1

        # Add cost information to metadata
        metadata["cost_summary"] = llm_agent.get_cost_summary()
        metadata["failed_topic_details"] = failed_mappings

        logger.info(f"Resilience mapping completed:")
        logger.info(f"  - Successful: {metadata['successful_mappings']}")
        logger.info(f"  - Failed: {metadata['failed_mappings']}")
        logger.info(f"  - Low confidence: {metadata['low_confidence_mappings']}")
        logger.info(f"  - Total cost: ${metadata['cost_summary']['total_cost']:.2f}")

        return mappings, metadata

    except Exception as e:
        logger.error(f"Resilience mapping process failed: {e}")
        raise ResilienceMappingError(f"Mapping process failed: {e}")

def _map_single_topic_to_resilience(topic_row: pd.Series, llm_agent: LLMAgent) -> ResilienceMapping:
    """Map a single topic to resilience dimensions."""
    topic_id = topic_row.get("topic_id", 0)
    topic_name = topic_row.get("name", f"Topic_{topic_id}")
    keywords = topic_row.get("keywords", [])
    summary = topic_row.get("summary", "")

    # Prepare keywords string
    keywords_str = ", ".join(keywords[:10]) if isinstance(keywords, list) else str(keywords)

    try:
        # Get resilience mapping from LLM
        prompt = _create_resilience_mapping_prompt(topic_name, keywords_str, summary)
        response = llm_agent._make_api_call(prompt, max_tokens=400)
        result = llm_agent._parse_json_response(response)

        # Extract dimensions with validation
        dimensions = result.get("dimensions", {})
        absorb_score = max(0.0, min(1.0, float(dimensions.get("absorb", 0.0))))
        adapt_score = max(0.0, min(1.0, float(dimensions.get("adapt", 0.0))))
        transform_score = max(0.0, min(1.0, float(dimensions.get("transform", 0.0))))

        # Determine primary dimension
        primary_dimension = result.get("primary_dimension", "absorb")
        if primary_dimension not in ["absorb", "adapt", "transform"]:
            # Fallback to highest score
            scores = {"absorb": absorb_score, "adapt": adapt_score, "transform": transform_score}
            primary_dimension = max(scores, key=scores.get)

        # Calculate overall confidence
        total_score = absorb_score + adapt_score + transform_score
        confidence = min(1.0, total_score) if total_score > 0 else 0.0

        reasoning = result.get("reasoning", "No reasoning provided")

        return ResilienceMapping(
            topic_id=topic_id,
            topic_name=topic_name,
            absorb_score=absorb_score,
            adapt_score=adapt_score,
            transform_score=transform_score,
            primary_dimension=primary_dimension,
            confidence=confidence,
            reasoning=reasoning
        )

    except Exception as e:
        logger.error(f"Failed to map topic {topic_id} to resilience: {e}")
        # Return fallback mapping
        return ResilienceMapping(
            topic_id=topic_id,
            topic_name=topic_name,
            absorb_score=0.33,
            adapt_score=0.33,
            transform_score=0.34,
            primary_dimension="absorb",
            confidence=0.1,
            reasoning=f"Fallback mapping due to error: {e}"
        )

def _create_resilience_mapping_prompt(topic_name: str, keywords: str, summary: str) -> str:
    """Create resilience mapping prompt."""
    from config.prompts import get_prompt_template

    return get_prompt_template(
        "resilience",
        topic_name=topic_name,
        keywords=keywords,
        summary=summary if summary else "No summary available"
    )

def calculate_resilience_scores(
    documents_df: pd.DataFrame,
    topic_assignments: List[int],
    topic_resilience_map: Dict[int, ResilienceMapping],
    aggregation_method: str = "weighted_average"
) -> pd.DataFrame:
    """
    Calculate resilience scores for companies and years.

    Args:
        documents_df: DataFrame with ['ticker', 'year', 'text'] columns
        topic_assignments: Topic assignments for each document
        topic_resilience_map: Mapping of topics to resilience dimensions
        aggregation_method: Method for aggregating scores

    Returns:
        DataFrame with resilience scores by company-year

    Raises:
        ResilienceMappingError: If calculation fails
    """
    logger.info(f"Calculating resilience scores for {len(documents_df)} documents")

    if len(documents_df) != len(topic_assignments):
        raise ResilienceMappingError(
            f"Mismatch between documents ({len(documents_df)}) and topic assignments ({len(topic_assignments)})"
        )

    try:
        # Add topic assignments to documents
        docs_with_topics = documents_df.copy()
        docs_with_topics['topic_id'] = topic_assignments

        # Add resilience scores for each document
        docs_with_topics['absorb_score'] = docs_with_topics['topic_id'].apply(
            lambda t: topic_resilience_map.get(t, _get_default_mapping()).absorb_score
        )
        docs_with_topics['adapt_score'] = docs_with_topics['topic_id'].apply(
            lambda t: topic_resilience_map.get(t, _get_default_mapping()).adapt_score
        )
        docs_with_topics['transform_score'] = docs_with_topics['topic_id'].apply(
            lambda t: topic_resilience_map.get(t, _get_default_mapping()).transform_score
        )

        # Calculate company-year aggregated scores
        if aggregation_method == "weighted_average":
            resilience_scores = _calculate_weighted_average_scores(docs_with_topics)
        elif aggregation_method == "simple_average":
            resilience_scores = _calculate_simple_average_scores(docs_with_topics)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # Add total resilience score
        resilience_scores['total_resilience_score'] = (
            resilience_scores['absorb_score'] * 0.33 +
            resilience_scores['adapt_score'] * 0.33 +
            resilience_scores['transform_score'] * 0.34
        )

        # Add confidence metrics
        resilience_scores = _add_confidence_metrics(
            resilience_scores, docs_with_topics, topic_resilience_map
        )

        # Sort by ticker and year
        resilience_scores = resilience_scores.sort_values(['ticker', 'year']).reset_index(drop=True)

        logger.info(f"Calculated resilience scores for {len(resilience_scores)} company-year combinations")

        return resilience_scores

    except Exception as e:
        logger.error(f"Failed to calculate resilience scores: {e}")
        raise ResilienceMappingError(f"Score calculation failed: {e}")

def _calculate_weighted_average_scores(docs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted average scores by document length."""
    # Add document length weights
    docs_df['doc_length'] = docs_df['text'].str.len()

    # Group by ticker and year, calculate weighted averages
    grouped = docs_df.groupby(['ticker', 'year']).agg({
        'absorb_score': lambda x: np.average(x, weights=docs_df.loc[x.index, 'doc_length']),
        'adapt_score': lambda x: np.average(x, weights=docs_df.loc[x.index, 'doc_length']),
        'transform_score': lambda x: np.average(x, weights=docs_df.loc[x.index, 'doc_length']),
        'doc_length': 'sum',
        'topic_id': 'count'
    }).reset_index()

    # Rename count column
    grouped = grouped.rename(columns={'topic_id': 'document_count'})

    return grouped

def _calculate_simple_average_scores(docs_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple average scores."""
    grouped = docs_df.groupby(['ticker', 'year']).agg({
        'absorb_score': 'mean',
        'adapt_score': 'mean',
        'transform_score': 'mean',
        'text': 'count'
    }).reset_index()

    # Rename count column and add total length
    grouped = grouped.rename(columns={'text': 'document_count'})
    grouped['doc_length'] = docs_df.groupby(['ticker', 'year'])['text'].apply(lambda x: x.str.len().sum()).values

    return grouped

def _add_confidence_metrics(
    scores_df: pd.DataFrame,
    docs_with_topics: pd.DataFrame,
    topic_resilience_map: Dict[int, ResilienceMapping]
) -> pd.DataFrame:
    """Add confidence metrics to resilience scores."""
    confidence_data = []

    for _, row in scores_df.iterrows():
        ticker = row['ticker']
        year = row['year']

        # Get documents for this company-year
        company_docs = docs_with_topics[
            (docs_with_topics['ticker'] == ticker) &
            (docs_with_topics['year'] == year)
        ]

        # Calculate average confidence
        confidences = [
            topic_resilience_map.get(topic_id, _get_default_mapping()).confidence
            for topic_id in company_docs['topic_id']
        ]

        avg_confidence = np.mean(confidences) if confidences else 0.0
        min_confidence = np.min(confidences) if confidences else 0.0

        # Calculate topic diversity (unique topics / total documents)
        unique_topics = len(company_docs['topic_id'].unique())
        total_docs = len(company_docs)
        topic_diversity = unique_topics / total_docs if total_docs > 0 else 0.0

        confidence_data.append({
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'topic_diversity': topic_diversity,
            'unique_topics': unique_topics
        })

    # Add confidence metrics to scores DataFrame
    confidence_df = pd.DataFrame(confidence_data)
    result_df = pd.concat([scores_df, confidence_df], axis=1)

    return result_df

def _get_default_mapping() -> ResilienceMapping:
    """Get default resilience mapping for unknown topics."""
    return ResilienceMapping(
        topic_id=-1,
        topic_name="Unknown Topic",
        absorb_score=0.33,
        adapt_score=0.33,
        transform_score=0.34,
        primary_dimension="absorb",
        confidence=0.1,
        reasoning="Default mapping for unknown topic"
    )

def save_resilience_results(
    resilience_scores: pd.DataFrame,
    topic_mappings: Dict[int, ResilienceMapping],
    metadata: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Save resilience analysis results.

    Args:
        resilience_scores: Company-year resilience scores
        topic_mappings: Topic to resilience mappings
        metadata: Analysis metadata
        save_path: Path to save results (auto-generated if None)

    Returns:
        Path where results were saved
    """
    # Generate save path if not provided
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = RESULTS_DIR / "resilience" / f"resilience_analysis_{timestamp}"

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save resilience scores
        resilience_scores.to_csv(save_path / "resilience_scores.csv", index=False)
        resilience_scores.to_pickle(save_path / "resilience_scores.pkl")

        # Save topic mappings
        mappings_data = []
        for topic_id, mapping in topic_mappings.items():
            mappings_data.append({
                "topic_id": mapping.topic_id,
                "topic_name": mapping.topic_name,
                "absorb_score": mapping.absorb_score,
                "adapt_score": mapping.adapt_score,
                "transform_score": mapping.transform_score,
                "primary_dimension": mapping.primary_dimension,
                "confidence": mapping.confidence,
                "reasoning": mapping.reasoning
            })

        mappings_df = pd.DataFrame(mappings_data)
        mappings_df.to_csv(save_path / "topic_resilience_mappings.csv", index=False)

        # Save metadata
        with open(save_path / "analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Generate summary statistics
        summary_stats = _generate_summary_statistics(resilience_scores, mappings_df)
        with open(save_path / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)

        logger.info(f"Resilience results saved to: {save_path}")
        return str(save_path)

    except Exception as e:
        logger.error(f"Failed to save resilience results: {e}")
        raise ResilienceMappingError(f"Failed to save results: {e}")

def _generate_summary_statistics(
    resilience_scores: pd.DataFrame,
    mappings_df: pd.DataFrame
) -> Dict[str, Any]:
    """Generate summary statistics for resilience analysis."""
    stats = {
        "resilience_scores_summary": {
            "total_company_years": len(resilience_scores),
            "unique_companies": resilience_scores['ticker'].nunique(),
            "year_range": (
                int(resilience_scores['year'].min()),
                int(resilience_scores['year'].max())
            ),
            "score_statistics": {
                "absorb": {
                    "mean": float(resilience_scores['absorb_score'].mean()),
                    "std": float(resilience_scores['absorb_score'].std()),
                    "min": float(resilience_scores['absorb_score'].min()),
                    "max": float(resilience_scores['absorb_score'].max())
                },
                "adapt": {
                    "mean": float(resilience_scores['adapt_score'].mean()),
                    "std": float(resilience_scores['adapt_score'].std()),
                    "min": float(resilience_scores['adapt_score'].min()),
                    "max": float(resilience_scores['adapt_score'].max())
                },
                "transform": {
                    "mean": float(resilience_scores['transform_score'].mean()),
                    "std": float(resilience_scores['transform_score'].std()),
                    "min": float(resilience_scores['transform_score'].min()),
                    "max": float(resilience_scores['transform_score'].max())
                },
                "total": {
                    "mean": float(resilience_scores['total_resilience_score'].mean()),
                    "std": float(resilience_scores['total_resilience_score'].std()),
                    "min": float(resilience_scores['total_resilience_score'].min()),
                    "max": float(resilience_scores['total_resilience_score'].max())
                }
            }
        },
        "topic_mappings_summary": {
            "total_topics": len(mappings_df),
            "dimension_distribution": {
                "absorb": int((mappings_df['primary_dimension'] == 'absorb').sum()),
                "adapt": int((mappings_df['primary_dimension'] == 'adapt').sum()),
                "transform": int((mappings_df['primary_dimension'] == 'transform').sum())
            },
            "confidence_statistics": {
                "mean": float(mappings_df['confidence'].mean()),
                "std": float(mappings_df['confidence'].std()),
                "min": float(mappings_df['confidence'].min()),
                "max": float(mappings_df['confidence'].max())
            }
        }
    }

    return stats

def load_resilience_results(load_path: str) -> Tuple[pd.DataFrame, Dict[int, ResilienceMapping], Dict[str, Any]]:
    """
    Load previously saved resilience analysis results.

    Args:
        load_path: Path to the saved results directory

    Returns:
        Tuple of (resilience_scores, topic_mappings, metadata)

    Raises:
        FileNotFoundError: If results files not found
        ResilienceMappingError: If loading fails
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Results path does not exist: {load_path}")

    try:
        # Load resilience scores
        scores_path = load_path / "resilience_scores.pkl"
        if scores_path.exists():
            resilience_scores = pd.read_pickle(scores_path)
        else:
            resilience_scores = pd.read_csv(load_path / "resilience_scores.csv")

        # Load topic mappings
        mappings_df = pd.read_csv(load_path / "topic_resilience_mappings.csv")
        topic_mappings = {}
        for _, row in mappings_df.iterrows():
            topic_mappings[row['topic_id']] = ResilienceMapping(
                topic_id=row['topic_id'],
                topic_name=row['topic_name'],
                absorb_score=row['absorb_score'],
                adapt_score=row['adapt_score'],
                transform_score=row['transform_score'],
                primary_dimension=row['primary_dimension'],
                confidence=row['confidence'],
                reasoning=row['reasoning']
            )

        # Load metadata
        with open(load_path / "analysis_metadata.json", 'r') as f:
            metadata = json.load(f)

        logger.info(f"Successfully loaded resilience results from: {load_path}")
        return resilience_scores, topic_mappings, metadata

    except Exception as e:
        logger.error(f"Failed to load resilience results from {load_path}: {e}")
        raise ResilienceMappingError(f"Results loading failed: {e}")

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    sample_topics_df = pd.DataFrame([
        {
            "topic_id": 0,
            "name": "Cybersecurity Measures",
            "keywords": ["cybersecurity", "security", "protection", "threat"],
            "summary": "This topic covers cybersecurity measures and threat protection"
        },
        {
            "topic_id": 1,
            "name": "Digital Transformation",
            "keywords": ["digital", "transformation", "technology", "automation"],
            "summary": "This topic focuses on digital transformation initiatives"
        }
    ])

    sample_docs_df = pd.DataFrame([
        {"ticker": "AAPL", "year": 2023, "text": "We implemented new cybersecurity measures"},
        {"ticker": "AAPL", "year": 2023, "text": "Digital transformation improved our efficiency"},
        {"ticker": "MSFT", "year": 2023, "text": "Cybersecurity is our top priority"},
    ])

    sample_topic_assignments = [0, 1, 0]

    try:
        print("🔄 Testing resilience mapping...")

        # Test topic mapping (requires OpenAI API key)
        try:
            mappings, metadata = map_topics_to_resilience(sample_topics_df)
            print(f"✅ Successfully mapped {len(mappings)} topics")

            # Test score calculation
            scores = calculate_resilience_scores(
                sample_docs_df, sample_topic_assignments, mappings
            )
            print(f"✅ Successfully calculated scores for {len(scores)} company-years")
            print(f"📊 Sample scores: {scores.head()}")

        except Exception as e:
            print(f"⚠️ Mapping test skipped (requires OpenAI API key): {e}")

    except Exception as e:
        print(f"❌ Test failed: {e}")