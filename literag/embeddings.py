"""
Embedding Service for LiteRAG

Handles embedding generation using Gemini API.
Embeddings are stored in LanceDB for efficient retrieval.
"""

import hashlib
from typing import Dict, List
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings.
    Uses Gemini text-embedding-004 by default.
    
    Embeddings are stored in LanceDB for persistence and efficient similarity search.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "models/text-embedding-004",
        batch_size: int = 100,
        max_retries: int = 3,
        use_litellm: bool = False,
        litellm_base_url: str = "http://localhost:4000"
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.use_litellm = use_litellm
        self.litellm_base_url = litellm_base_url
        
        # In-memory cache for query embeddings only (transient, for current session)
        self._query_cache: Dict[str, np.ndarray] = {}
        
        # Initialize client
        self._init_client()

    def __repr__(self):
        """Safely mask the API key when printing the service configuration."""
        key_masked = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
        return f"<EmbeddingService model={self.model} api_key={key_masked}>"
    
    def _init_client(self):
        """Initialize the embedding client."""
        if self.use_litellm:
            self._client_type = "litellm"
        else:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client_type = "genai"
            except ImportError:
                # Fallback to requests
                self._client_type = "requests"
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _embed_batch_genai(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using Google's genai library."""
        import google.generativeai as genai
        
        result = genai.embed_content(
            model=self.model,
            content=texts,
            task_type="retrieval_document"
        )
        return [np.array(emb) for emb in result['embedding']]
    
    def _embed_batch_requests(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using direct API requests."""
        import requests
        
        url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        embeddings = []
        for text in texts:
            data = {
                "model": self.model,
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT"
            }
            
            response = requests.post(url, json=data, params=params, headers=headers)
            response.raise_for_status()
            result = response.json()
            embeddings.append(np.array(result['embedding']['values']))
        
        return embeddings
    
    def _embed_batch_litellm(self, texts: List[str]) -> List[np.ndarray]:
        """Embed using LiteLLM safely mapped with base URLs."""
        import litellm
        
        response = litellm.embedding(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            api_base=self.litellm_base_url
        )
        
        return [np.array(item['embedding']) for item in response.data]
    
    def embed(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        """
        key = self._get_cache_key(text)
        if key in self._query_cache:
            return self._query_cache[key]
        
        embeddings = self.embed_batch([text])
        embedding = embeddings[0]
        
        self._query_cache[key] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []
        
        all_embeddings: List[np.ndarray] = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = texts[batch_start:batch_start + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    if self._client_type == "genai":
                        embeddings = self._embed_batch_genai(batch_texts)
                    elif self._client_type == "litellm":
                        embeddings = self._embed_batch_litellm(batch_texts)
                    else:
                        embeddings = self._embed_batch_requests(batch_texts)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to embed after {self.max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def clear_cache(self):
        """Clear the query embedding cache."""
        self._query_cache.clear()
        logger.info("Query embedding cache cleared")
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))