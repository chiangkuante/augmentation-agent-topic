"""
Data loading and preprocessing module for augmentation-agent-topic system.

This module handles loading and preprocessing of corporate report data
from CSV files with proper error handling and data validation.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Set, Union
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up logging
logger = logging.getLogger(__name__)

def _ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

def get_stopwords(languages: Union[str, List[str]] = 'english',
                  custom_stopwords: Optional[Set[str]] = None) -> Set[str]:
    """
    Get stopwords for specified language(s) with optional custom additions.

    Args:
        languages: Language code(s) for stopwords ('english', 'chinese', etc.)
                  Can be a string or list of strings
        custom_stopwords: Additional custom stopwords to include

    Returns:
        Set of stopwords
    """
    _ensure_nltk_data()

    if isinstance(languages, str):
        languages = [languages]

    stop_words = set()

    # Add NLTK stopwords for each language
    for lang in languages:
        try:
            lang_stopwords = set(stopwords.words(lang))
            stop_words.update(lang_stopwords)
            logger.debug(f"Added {len(lang_stopwords)} {lang} stopwords")
        except OSError:
            logger.warning(f"Stopwords not available for language: {lang}")

    # Add common financial/business stopwords
    financial_stopwords = {
        'company', 'companies', 'business', 'businesses', 'corporation', 'corp',
        'inc', 'ltd', 'llc', 'llp', 'organization', 'org', 'management', 'team',
        'year', 'years', 'quarter', 'quarters', 'month', 'months', 'period',
        'fiscal', 'financial', 'revenue', 'sales', 'million', 'billion',
        'report', 'reporting', 'section', 'item', 'page', 'table', 'figure',
        'see', 'also', 'including', 'include', 'includes', 'may', 'could',
        'would', 'should', 'might', 'must', 'shall', 'will', 'can', 'cannot'
    }
    stop_words.update(financial_stopwords)

    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)
        logger.debug(f"Added {len(custom_stopwords)} custom stopwords")

    logger.info(f"Total stopwords loaded: {len(stop_words)}")
    return stop_words

def remove_stopwords(text: str,
                    stopwords_set: Set[str],
                    min_word_length: int = 2,
                    preserve_numbers: bool = True) -> str:
    """
    Remove stopwords from text while preserving meaningful content.

    Args:
        text: Input text to process
        stopwords_set: Set of stopwords to remove
        min_word_length: Minimum length for words to keep
        preserve_numbers: Whether to preserve numeric values

    Returns:
        Processed text with stopwords removed
    """
    if not text or not isinstance(text, str):
        return ""

    # Tokenize the text
    try:
        tokens = word_tokenize(text.lower())
    except:
        # Fallback to simple split if tokenization fails
        tokens = re.findall(r'\b\w+\b', text.lower())

    filtered_tokens = []

    for token in tokens:
        # Skip if it's a stopword
        if token in stopwords_set:
            continue

        # Skip very short words unless they're numbers
        if len(token) < min_word_length:
            if preserve_numbers and token.isdigit():
                filtered_tokens.append(token)
            continue

        # Skip pure punctuation
        if not any(c.isalnum() for c in token):
            continue

        filtered_tokens.append(token)

    return ' '.join(filtered_tokens)

def load_corpus(file_path: str,
               encoding: Optional[str] = None,
               remove_stopwords_flag: bool = False,
               stopword_languages: Union[str, List[str]] = 'english',
               custom_stopwords: Optional[Set[str]] = None,
               min_word_length: int = 2) -> pd.DataFrame:
    """
    Load and preprocess the corpus data from CSV file with optional stopword removal.

    Args:
        file_path: Path to the corpus CSV file
        encoding: File encoding (if None, will try to detect)
        remove_stopwords_flag: Whether to remove stopwords from text
        stopword_languages: Language(s) for stopwords ('english', 'chinese', etc.)
        custom_stopwords: Additional custom stopwords to remove
        min_word_length: Minimum length for words to keep after stopword removal

    Returns:
        pandas.DataFrame with columns ['ticker', 'year', 'text']

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
        pd.errors.EmptyDataError: If the file is empty
    """
    logger.info(f"Loading corpus data from: {file_path}")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    # Try different encodings if not specified
    encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    df = None
    for enc in encodings:
        try:
            logger.debug(f"Trying encoding: {enc}")
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"Successfully loaded data with encoding: {enc}")
            break
        except (UnicodeDecodeError, UnicodeError):
            logger.debug(f"Failed to load with encoding: {enc}")
            continue
        except Exception as e:
            logger.error(f"Error loading file with encoding {enc}: {e}")
            continue

    if df is None:
        raise ValueError(f"Could not load file with any of the tried encodings: {encodings}")

    # Validate and process the data
    df = _validate_and_clean_data(df, remove_stopwords_flag, stopword_languages, custom_stopwords, min_word_length)

    logger.info(f"Loaded corpus: {len(df)} documents across {df['ticker'].nunique()} companies")
    return df

def _validate_and_clean_data(df: pd.DataFrame,
                             remove_stopwords_flag: bool = False,
                             stopword_languages: Union[str, List[str]] = 'english',
                             custom_stopwords: Optional[Set[str]] = None,
                             min_word_length: int = 2) -> pd.DataFrame:
    """
    Validate and clean the loaded data with optional stopword removal.

    Args:
        df: Raw dataframe from CSV
        remove_stopwords_flag: Whether to remove stopwords from text
        stopword_languages: Language(s) for stopwords
        custom_stopwords: Additional custom stopwords to remove
        min_word_length: Minimum length for words to keep after stopword removal

    Returns:
        Cleaned and validated dataframe

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.debug("Validating and cleaning data")

    # Check for required columns
    required_columns = ['ticker', 'year', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        # Try common alternative column names
        column_mapping = {
            'company': 'ticker',
            'symbol': 'ticker',
            'company_symbol': 'ticker',
            'content': 'text',
            'document': 'text',
            'document_text': 'text'
        }

        for alt_name, standard_name in column_mapping.items():
            if alt_name in df.columns and standard_name in missing_columns:
                df = df.rename(columns={alt_name: standard_name})
                missing_columns.remove(standard_name)
                logger.info(f"Mapped column '{alt_name}' to '{standard_name}'")

    if missing_columns:
        available_columns = list(df.columns)
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {available_columns}"
        )

    # Remove rows with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=['ticker', 'text'])

    # Clean text data
    df['text'] = df['text'].astype(str).str.strip()

    # Apply stopword removal if requested
    if remove_stopwords_flag:
        logger.info("Applying stopword removal to text data...")
        stopwords_set = get_stopwords(stopword_languages, custom_stopwords)

        # Store original text length for statistics
        original_lengths = df['text'].str.len()

        # Apply stopword removal
        df['text'] = df['text'].apply(
            lambda x: remove_stopwords(x, stopwords_set, min_word_length)
        )

        # Log statistics about stopword removal
        new_lengths = df['text'].str.len()
        avg_reduction = ((original_lengths - new_lengths) / original_lengths * 100).mean()
        logger.info(f"Stopword removal completed:")
        logger.info(f"  - Average text length reduction: {avg_reduction:.1f}%")
        logger.info(f"  - Processed {len(df)} documents")

    df = df[df['text'].str.len() > 10]  # Remove very short texts

    # Clean year data
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

        # Filter reasonable years (e.g., 1990-2030)
        df = df[(df['year'] >= 1990) & (df['year'] <= 2030)]

    # Clean ticker data
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    df = df[df['ticker'].str.len() > 0]

    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker', 'year', 'text'])

    final_count = len(df)
    removed_count = initial_count - final_count

    if removed_count > 0:
        logger.warning(f"Removed {removed_count} invalid/duplicate records during cleaning")

    if final_count == 0:
        raise ValueError("No valid data remaining after cleaning")

    # Sort data for consistency
    df = df.sort_values(['ticker', 'year']).reset_index(drop=True)

    # Log statistics
    logger.info(f"Data validation complete:")
    logger.info(f"  - Final documents: {final_count}")
    logger.info(f"  - Unique companies: {df['ticker'].nunique()}")
    logger.info(f"  - Year range: {df['year'].min()}-{df['year'].max()}")
    logger.info(f"  - Average text length: {df['text'].str.len().mean():.0f} characters")

    return df

def get_corpus_statistics(df: pd.DataFrame) -> dict:
    """
    Get detailed statistics about the corpus.

    Args:
        df: Corpus dataframe

    Returns:
        Dictionary with corpus statistics
    """
    stats = {
        'total_documents': len(df),
        'unique_companies': df['ticker'].nunique(),
        'year_range': (int(df['year'].min()), int(df['year'].max())),
        'companies': df['ticker'].value_counts().to_dict(),
        'years': df['year'].value_counts().sort_index().to_dict(),
        'text_length_stats': {
            'mean': df['text'].str.len().mean(),
            'median': df['text'].str.len().median(),
            'min': df['text'].str.len().min(),
            'max': df['text'].str.len().max(),
            'std': df['text'].str.len().std()
        }
    }

    return stats

def sample_corpus(df: pd.DataFrame,
                  n_samples: Optional[int] = None,
                  sample_by_company: bool = True,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Create a sample of the corpus for testing or development.

    Args:
        df: Full corpus dataframe
        n_samples: Number of samples to return (if None, returns 1000 or 10% whichever is smaller)
        sample_by_company: If True, sample proportionally by company
        random_state: Random seed for reproducibility

    Returns:
        Sampled dataframe
    """
    if n_samples is None:
        n_samples = min(1000, len(df) // 10)

    if n_samples >= len(df):
        logger.warning(f"Requested sample size ({n_samples}) >= total size ({len(df)}), returning full dataset")
        return df.copy()

    if sample_by_company:
        # Sample proportionally by company
        company_counts = df['ticker'].value_counts()
        sample_counts = (company_counts / company_counts.sum() * n_samples).round().astype(int)

        sampled_dfs = []
        for company, count in sample_counts.items():
            if count > 0:
                company_data = df[df['ticker'] == company]
                sample_size = min(count, len(company_data))
                sampled = company_data.sample(n=sample_size, random_state=random_state)
                sampled_dfs.append(sampled)

        result = pd.concat(sampled_dfs, ignore_index=True)
    else:
        # Simple random sample
        result = df.sample(n=n_samples, random_state=random_state)

    result = result.sort_values(['ticker', 'year']).reset_index(drop=True)

    logger.info(f"Created sample: {len(result)} documents from {result['ticker'].nunique()} companies")

    return result

def validate_corpus_format(df: pd.DataFrame) -> List[str]:
    """
    Validate that the corpus has the expected format.

    Args:
        df: Corpus dataframe to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required columns
    required_columns = ['ticker', 'year', 'text']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return errors

    # Check data types and content
    if not df['ticker'].dtype == 'object':
        errors.append("ticker column should be string type")

    if not pd.api.types.is_numeric_dtype(df['year']):
        errors.append("year column should be numeric type")

    if not df['text'].dtype == 'object':
        errors.append("text column should be string type")

    # Check for empty or null values
    if df['ticker'].isnull().any():
        errors.append("ticker column contains null values")

    if df['year'].isnull().any():
        errors.append("year column contains null values")

    if df['text'].isnull().any():
        errors.append("text column contains null values")

    # Check text length
    short_texts = (df['text'].str.len() < 10).sum()
    if short_texts > 0:
        errors.append(f"{short_texts} texts are too short (<10 characters)")

    # Check year range
    if df['year'].min() < 1990 or df['year'].max() > 2030:
        errors.append(f"Years outside expected range (1990-2030): {df['year'].min()}-{df['year'].max()}")

    return errors

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    # This would be the actual usage:
    # df = load_corpus("data/corpus_semantic_chunks.csv")
    # stats = get_corpus_statistics(df)
    # sample_df = sample_corpus(df, n_samples=100)

    print("Data loader module ready for use")
    print("Use load_corpus('path/to/your/file.csv') to load data")