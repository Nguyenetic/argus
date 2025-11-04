"""
Embedding generation using sentence-transformers
Local embeddings with caching for performance
Based on sage-mcp and SurfSense implementations
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
import hashlib
import logging
from functools import lru_cache

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================
# EMBEDDING MODEL SINGLETON
# ============================================

class EmbeddingModel:
    """Singleton wrapper for sentence-transformers model"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model"""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        """Get maximum sequence length"""
        return self._model.max_seq_length

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar for large batches
            normalize_embeddings: Normalize to unit length (recommended for cosine similarity)

        Returns:
            numpy array of embeddings
        """
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise


# Global model instance
_embedding_model = EmbeddingModel()


# ============================================
# CACHING LAYER
# ============================================

class EmbeddingCache:
    """In-memory cache for embeddings with SHA256 keys"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: dict = {}

    @staticmethod
    def _get_cache_key(text: str) -> str:
        """Generate SHA256 hash as cache key"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = self._get_cache_key(text)
        return self._cache.get(key)

    def set(self, text: str, embedding: np.ndarray):
        """Store embedding in cache"""
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            # Remove first (oldest) entry
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        key = self._get_cache_key(text)
        self._cache[key] = embedding

    def clear(self):
        """Clear cache"""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)


# Global cache instance
_embedding_cache = EmbeddingCache(max_size=10000)


# ============================================
# PUBLIC API
# ============================================

async def get_embedding(
    text: str,
    use_cache: bool = True
) -> List[float]:
    """
    Get embedding for a single text (async wrapper)

    Args:
        text: Input text
        use_cache: Use cached embedding if available

    Returns:
        List of floats (embedding vector)
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return [0.0] * settings.EMBEDDING_DIMENSION

    # Check cache
    if use_cache:
        cached = _embedding_cache.get(text)
        if cached is not None:
            return cached.tolist()

    # Generate embedding
    try:
        embedding = _embedding_model.encode(text)

        # Cache result
        if use_cache:
            _embedding_cache.set(text, embedding)

        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * settings.EMBEDDING_DIMENSION


async def get_embeddings_batch(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False,
    use_cache: bool = True
) -> List[List[float]]:
    """
    Get embeddings for multiple texts (async wrapper)

    Args:
        texts: List of input texts
        batch_size: Batch size for processing
        show_progress: Show progress bar
        use_cache: Use cached embeddings if available

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Check which texts need embedding
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
        elif use_cache:
            cached = _embedding_cache.get(text)
            if cached is not None:
                embeddings.append(cached.tolist())
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        else:
            embeddings.append(None)  # Placeholder
            texts_to_embed.append(text)
            indices_to_embed.append(i)

    # Generate embeddings for uncached texts
    if texts_to_embed:
        try:
            new_embeddings = _embedding_model.encode(
                texts_to_embed,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )

            # Fill in results and cache
            for i, idx in enumerate(indices_to_embed):
                embedding = new_embeddings[i]
                embeddings[idx] = embedding.tolist()

                if use_cache:
                    _embedding_cache.set(texts_to_embed[i], embedding)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Fill with zero vectors
            for idx in indices_to_embed:
                embeddings[idx] = [0.0] * settings.EMBEDDING_DIMENSION

    return embeddings


def get_embedding_sync(text: str, use_cache: bool = True) -> List[float]:
    """
    Synchronous version of get_embedding (for non-async contexts)
    """
    if not text or not text.strip():
        return [0.0] * settings.EMBEDDING_DIMENSION

    if use_cache:
        cached = _embedding_cache.get(text)
        if cached is not None:
            return cached.tolist()

    try:
        embedding = _embedding_model.encode(text)
        if use_cache:
            _embedding_cache.set(text, embedding)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * settings.EMBEDDING_DIMENSION


# ============================================
# CHUNKING UTILITIES
# ============================================

def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = 50
) -> List[str]:
    """
    Split text into chunks for better semantic search

    Args:
        text: Input text
        chunk_size: Max chunk size (defaults to model's max_seq_length)
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = _embedding_model.max_seq_length

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)

            if break_point > chunk_size // 2:  # Only break if past halfway
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


# ============================================
# CACHE MANAGEMENT
# ============================================

def get_cache_stats() -> dict:
    """Get embedding cache statistics"""
    return {
        "size": _embedding_cache.size(),
        "max_size": _embedding_cache.max_size,
        "model": settings.EMBEDDING_MODEL,
        "dimension": _embedding_model.dimension,
        "max_seq_length": _embedding_model.max_seq_length
    }


def clear_embedding_cache():
    """Clear the embedding cache"""
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


# ============================================
# MODEL INFO
# ============================================

def get_model_info() -> dict:
    """Get embedding model information"""
    return {
        "model_name": settings.EMBEDDING_MODEL,
        "dimension": _embedding_model.dimension,
        "max_seq_length": _embedding_model.max_seq_length,
        "cache_size": _embedding_cache.size(),
    }
