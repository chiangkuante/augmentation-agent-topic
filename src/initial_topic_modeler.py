"""
Initial topic modeling module using BERTopic with configurable embedding models.

This module implements the Phase 1 MVP approach with sentence-transformers,
and provides hooks for Phase 2 ModernBERT integration.
"""

import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from datetime import datetime
import warnings

# Core ML libraries
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# BERTopic
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

# Configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    EMBEDDING_MODELS, TOPIC_MODEL_CONFIG, UMAP_CONFIG,
    HDBSCAN_CONFIG, RESULTS_DIR
)

logger = logging.getLogger(__name__)

class TopicModelingError(Exception):
    """Custom exception for topic modeling errors."""
    pass

def create_initial_topics(
    documents: List[str],
    model_phase: str = "phase1",
    embedding_model: Optional[str] = None,
    save_path: Optional[str] = None,
    batch_size: int = 32,
    use_gpu: bool = False,
    **kwargs
) -> Tuple[BERTopic, pd.DataFrame, Dict[str, Any]]:
    """
    Create initial topics using BERTopic with specified embedding model.

    Args:
        documents: List of text documents
        model_phase: Phase of implementation ("phase1", "phase2", "phase3")
        embedding_model: Override default embedding model
        save_path: Path to save the model (if None, auto-generates)
        batch_size: Batch size for embedding computation
        use_gpu: Whether to use GPU if available
        **kwargs: Additional parameters for BERTopic

    Returns:
        Tuple of (BERTopic model, topics dataframe, metadata dict)

    Raises:
        TopicModelingError: If topic modeling fails
        ValueError: If invalid parameters provided
    """
    logger.info(f"Starting initial topic modeling with {len(documents)} documents")
    logger.info(f"Model phase: {model_phase}")

    # Validate inputs
    if not documents:
        raise ValueError("No documents provided")

    if len(documents) < 10:
        raise ValueError(f"Need at least 10 documents, got {len(documents)}")

    # Get embedding model
    if embedding_model is None:
        embedding_model = EMBEDDING_MODELS.get(model_phase, EMBEDDING_MODELS["phase1"])

    logger.info(f"Using embedding model: {embedding_model}")

    try:
        # Phase-specific model loading
        embeddings = _load_embedding_model(embedding_model, use_gpu, batch_size)

        # Generate embeddings with progress tracking
        logger.info("Generating document embeddings...")
        document_embeddings = _generate_embeddings(documents, embeddings, batch_size)

        # Create BERTopic model
        logger.info("Creating BERTopic model...")
        topic_model = _create_bertopic_model(**kwargs)

        # Fit the model
        logger.info("Fitting topic model...")
        topics, probabilities = topic_model.fit_transform(documents, document_embeddings)

        # Extract topic information
        logger.info("Extracting topic information...")
        topics_df = _extract_topic_info(topic_model, documents, topics)

        # Calculate quality metrics
        logger.info("Calculating quality metrics...")
        quality_metrics = _calculate_quality_metrics(
            topic_model, document_embeddings, topics, documents
        )

        # Prepare metadata
        metadata = {
            "model_phase": model_phase,
            "embedding_model": embedding_model,
            "n_documents": len(documents),
            "n_topics": len(topic_model.get_topic_info()) - 1,  # Exclude -1 topic
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": quality_metrics,
            "model_config": {
                "umap_params": UMAP_CONFIG,
                "hdbscan_params": HDBSCAN_CONFIG,
                "topic_params": TOPIC_MODEL_CONFIG
            }
        }

        # Save model if requested
        if save_path:
            _save_model_and_results(topic_model, topics_df, metadata, save_path)
        else:
            # Auto-save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_path = RESULTS_DIR / "topics" / f"initial_model_{timestamp}"
            _save_model_and_results(topic_model, topics_df, metadata, auto_save_path)

        logger.info(f"Topic modeling completed successfully:")
        logger.info(f"  - Created {metadata['n_topics']} topics")
        logger.info(f"  - Coherence score: {quality_metrics.get('coherence', 'N/A'):.3f}")
        logger.info(f"  - Silhouette score: {quality_metrics.get('silhouette', 'N/A'):.3f}")

        return topic_model, topics_df, metadata

    except Exception as e:
        logger.error(f"Topic modeling failed: {str(e)}")
        raise TopicModelingError(f"Failed to create initial topics: {str(e)}") from e

def _load_embedding_model(model_name: str, use_gpu: bool, batch_size: int) -> SentenceTransformer:
    """Load the appropriate embedding model based on phase."""
    try:
        if "ModernBERT" in model_name:
            # Phase 2/3: ModernBERT integration
            logger.info("Loading ModernBERT model (Phase 2/3)")
            # For now, fall back to sentence-transformers until custom wrapper is ready
            logger.warning("ModernBERT not yet implemented, falling back to all-mpnet-base-v2")
            model_name = "sentence-transformers/all-mpnet-base-v2"

        # Phase 1: Standard sentence-transformers
        logger.info(f"Loading sentence-transformer model: {model_name}")
        model = SentenceTransformer(model_name)

        if use_gpu:
            try:
                model = model.to('cuda')
                logger.info("Using GPU for embeddings")
            except Exception as e:
                logger.warning(f"Could not use GPU, falling back to CPU: {e}")

        return model

    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        # Fallback to a basic model
        logger.info("Falling back to all-MiniLM-L6-v2")
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def _generate_embeddings(documents: List[str], model: SentenceTransformer, batch_size: int) -> np.ndarray:
    """Generate embeddings for documents with memory-efficient batching."""
    try:
        # Use sentence-transformers built-in batching
        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise TopicModelingError(f"Embedding generation failed: {e}")

def _create_bertopic_model(**kwargs) -> BERTopic:
    """Create a configured BERTopic model."""
    # Merge default config with any overrides
    config = {**TOPIC_MODEL_CONFIG, **kwargs}

    # UMAP for dimensionality reduction
    umap_model = umap.UMAP(**UMAP_CONFIG)

    # HDBSCAN for clustering
    hdbscan_model = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)

    # Create BERTopic model without representation model for initial stability
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=10,
        verbose=config.get("verbose", True),
        calculate_probabilities=config.get("calculate_probabilities", True),
        nr_topics=config.get("nr_topics", None),
        min_topic_size=config.get("min_topic_size", 10)
    )

    return topic_model

def _extract_topic_info(topic_model: BERTopic, documents: List[str], topics: List[int]) -> pd.DataFrame:
    """Extract comprehensive topic information."""
    # Get basic topic info
    topic_info = topic_model.get_topic_info()

    # Remove outlier topic (-1) for analysis
    topic_info_clean = topic_info[topic_info['Topic'] != -1].copy()

    # Add additional metrics per topic
    enriched_topics = []
    for _, row in topic_info_clean.iterrows():
        topic_id = row['Topic']
        topic_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        topic_data = {
            'topic_id': topic_id,
            'name': f"Topic_{topic_id}",  # Will be enhanced by LLM later
            'keywords': [word for word, _ in topic_model.get_topic(topic_id)],
            'keyword_scores': [score for _, score in topic_model.get_topic(topic_id)],
            'document_count': len(topic_docs),
            'document_percentage': len(topic_docs) / len(documents) * 100,
            'representative_docs': topic_docs[:3] if topic_docs else [],
            'avg_doc_length': np.mean([len(doc) for doc in topic_docs]) if topic_docs else 0
        }
        enriched_topics.append(topic_data)

    topics_df = pd.DataFrame(enriched_topics)

    if topics_df.empty:
        logger.warning("No valid topics found (all documents may be outliers)")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['topic_id', 'name', 'keywords', 'keyword_scores',
                                   'document_count', 'document_percentage', 'representative_docs', 'avg_doc_length'])

    # Sort by document count descending
    topics_df = topics_df.sort_values('document_count', ascending=False).reset_index(drop=True)

    logger.info(f"Extracted information for {len(topics_df)} topics")
    return topics_df

def _calculate_quality_metrics(
    topic_model: BERTopic,
    embeddings: np.ndarray,
    topics: List[int],
    documents: List[str]
) -> Dict[str, float]:
    """Calculate various quality metrics for the topic model."""
    metrics = {}

    try:
        # Topic coherence using BERTopic's built-in method
        try:
            coherence_score = topic_model.get_topic_info()['Count'].std() / topic_model.get_topic_info()['Count'].mean()
            metrics['coherence'] = float(coherence_score)
        except:
            metrics['coherence'] = 0.0

        # Silhouette score (only for non-outlier topics)
        valid_indices = np.array(topics) != -1
        if np.sum(valid_indices) > 1:
            silhouette_avg = silhouette_score(
                embeddings[valid_indices],
                np.array(topics)[valid_indices]
            )
            metrics['silhouette'] = float(silhouette_avg)
        else:
            metrics['silhouette'] = 0.0

        # Topic diversity (average pairwise distance between topic centers)
        topic_centers = []
        unique_topics = list(set(topics))
        if -1 in unique_topics:
            unique_topics.remove(-1)

        for topic_id in unique_topics:
            topic_embeddings = embeddings[np.array(topics) == topic_id]
            if len(topic_embeddings) > 0:
                center = np.mean(topic_embeddings, axis=0)
                topic_centers.append(center)

        if len(topic_centers) > 1:
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(topic_centers)
            # Get upper triangle (excluding diagonal)
            upper_triangle = distances[np.triu_indices_from(distances, k=1)]
            metrics['topic_diversity'] = float(np.mean(upper_triangle))
        else:
            metrics['topic_diversity'] = 0.0

        # Number of topics (excluding outliers)
        n_topics = len([t for t in unique_topics if t != -1])
        metrics['n_topics'] = n_topics

        # Coverage (percentage of documents not in outlier topic)
        outlier_count = sum(1 for t in topics if t == -1)
        coverage = (len(topics) - outlier_count) / len(topics)
        metrics['coverage'] = float(coverage)

        logger.debug(f"Quality metrics calculated: {metrics}")

    except Exception as e:
        logger.warning(f"Failed to calculate some quality metrics: {e}")
        # Return partial metrics
        metrics.update({
            'coherence': 0.0,
            'silhouette': 0.0,
            'topic_diversity': 0.0,
            'n_topics': len(set(topics)) - (1 if -1 in topics else 0),
            'coverage': 1.0 - (topics.count(-1) / len(topics))
        })

    return metrics

def _save_model_and_results(
    topic_model: BERTopic,
    topics_df: pd.DataFrame,
    metadata: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Save the topic model and related results."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save BERTopic model
        model_path = save_path / "bertopic_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(topic_model, f)

        # Save topics dataframe
        topics_path = save_path / "topics_info.csv"
        topics_df.to_csv(topics_path, index=False)

        # Save topics dataframe as pickle (preserves lists)
        topics_pkl_path = save_path / "topics_info.pkl"
        topics_df.to_pickle(topics_pkl_path)

        # Save metadata
        import json
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save topic visualization (if possible)
        try:
            fig = topic_model.visualize_topics()
            fig.write_html(str(save_path / "topic_visualization.html"))
        except Exception as e:
            logger.warning(f"Could not save topic visualization: {e}")

        logger.info(f"Model and results saved to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to save model and results: {e}")
        raise TopicModelingError(f"Save operation failed: {e}")

def load_topic_model(load_path: Union[str, Path]) -> Tuple[BERTopic, pd.DataFrame, Dict[str, Any]]:
    """
    Load a previously saved topic model.

    Args:
        load_path: Path to the saved model directory

    Returns:
        Tuple of (BERTopic model, topics dataframe, metadata dict)

    Raises:
        FileNotFoundError: If model files not found
        TopicModelingError: If loading fails
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {load_path}")

    try:
        # Load BERTopic model
        model_path = load_path / "bertopic_model.pkl"
        with open(model_path, 'rb') as f:
            topic_model = pickle.load(f)

        # Load topics dataframe (prefer pickle version)
        topics_pkl_path = load_path / "topics_info.pkl"
        if topics_pkl_path.exists():
            topics_df = pd.read_pickle(topics_pkl_path)
        else:
            topics_path = load_path / "topics_info.csv"
            topics_df = pd.read_csv(topics_path)

        # Load metadata
        import json
        metadata_path = load_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Successfully loaded model from: {load_path}")
        return topic_model, topics_df, metadata

    except Exception as e:
        logger.error(f"Failed to load model from {load_path}: {e}")
        raise TopicModelingError(f"Model loading failed: {e}")

def evaluate_topic_model(
    topic_model: BERTopic,
    documents: List[str],
    embeddings: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate a topic model with comprehensive metrics.

    Args:
        topic_model: Fitted BERTopic model
        documents: Original documents
        embeddings: Document embeddings (will generate if None)

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating topic model...")

    if embeddings is None:
        # Generate embeddings for evaluation
        logger.info("Generating embeddings for evaluation...")
        sentence_model = SentenceTransformer(EMBEDDING_MODELS["phase1"])
        embeddings = sentence_model.encode(documents, show_progress_bar=True)

    # Get topics
    topics = topic_model.topics_

    # Calculate metrics
    quality_metrics = _calculate_quality_metrics(topic_model, embeddings, topics, documents)

    # Add model-specific metrics
    topic_info = topic_model.get_topic_info()
    evaluation = {
        **quality_metrics,
        'model_info': {
            'total_topics': len(topic_info),
            'valid_topics': len(topic_info[topic_info['Topic'] != -1]),
            'outlier_documents': sum(1 for t in topics if t == -1),
            'largest_topic_size': topic_info[topic_info['Topic'] != -1]['Count'].max() if len(topic_info) > 1 else 0,
            'smallest_topic_size': topic_info[topic_info['Topic'] != -1]['Count'].min() if len(topic_info) > 1 else 0,
            'avg_topic_size': topic_info[topic_info['Topic'] != -1]['Count'].mean() if len(topic_info) > 1 else 0
        }
    }

    return evaluation

if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test with sample documents
    sample_docs = [
        "The company implemented new cybersecurity measures to protect against threats",
        "Digital transformation initiatives improved our operational efficiency",
        "Supply chain resilience was enhanced through diversification strategies",
        "Remote work technologies enabled business continuity during disruptions",
        "Cloud infrastructure provided scalability and flexibility",
        "Data analytics helped predict and prevent potential system failures",
        "Artificial intelligence solutions automated routine business processes",
        "Customer digital experience was prioritized in product development",
        "Sustainability initiatives aligned with environmental regulations",
        "Financial risk management strategies protected against market volatility"
    ] * 5  # Duplicate to have enough documents

    print("Testing initial topic modeling...")
    try:
        model, topics_df, metadata = create_initial_topics(
            sample_docs,
            model_phase="phase1"
        )
        print(f"✅ Successfully created {metadata['n_topics']} topics")
        print(f"📊 Quality metrics: {metadata['quality_metrics']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")