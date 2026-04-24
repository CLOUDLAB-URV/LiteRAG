"""
Google FileSearch Engine Adapter

Wraps Google's Gemini FileSearch (RAG) tool in the standard SearchEngineInterface
for benchmarking. FileSearch is an out-of-the-box RAG solution that handles
chunking, embedding, and retrieval internally.

Docs: https://ai.google.dev/gemini-api/docs/file-search
"""

import os
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .engine_interface import SearchEngineInterface, QueryResult, EngineType

LOGGER = logging.getLogger(__name__)


@dataclass
class FileSearchConfig:
    """Configuration for Google FileSearch engine"""
    # API key (falls back to GOOGLE_API_KEY env var)
    api_key: str = ""

    # Model to use for generate_content
    model: str = "gemini-2.5-flash-lite"

    # Option 1: Reuse an existing FileSearch store
    file_search_store_name: str = ""

    # Option 2: Create a new store and upload files
    corpus_files: List[str] = field(default_factory=list)
    display_name: str = "benchmark-store"

    @classmethod
    def from_yaml(cls, path: str) -> "FileSearchConfig":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GoogleFileSearchEngine(SearchEngineInterface):
    """
    Google FileSearch (Gemini API) adapter for the benchmark interface.

    Uses Google's built-in RAG solution which handles chunking, embedding,
    indexing and retrieval internally. The adapter manages store lifecycle
    and routes queries through generate_content with the FileSearch tool.

    Supports two modes:
    - Reuse an existing store (recommended for repeated benchmarks)
    - Create a new store and upload files during initialization
    """

    def __init__(
        self,
        config_path: Path,
        prompt_price_per_m: float = 0.15,
        completion_price_per_m: float = 0.60,
        **kwargs
    ):
        """
        Initialize Google FileSearch adapter.

        Args:
            config_path: Path to FileSearch YAML configuration file
            prompt_price_per_m: Price per million input tokens (EUR)
            completion_price_per_m: Price per million output tokens (EUR)
            **kwargs: Additional configuration overrides
        """
        self.config_path = Path(config_path)
        self.prompt_price_per_m = prompt_price_per_m
        self.completion_price_per_m = completion_price_per_m
        self.config_overrides = kwargs

        self._client = None
        self._config: Optional[FileSearchConfig] = None
        self._store_name: Optional[str] = None
        self._store_created_by_me = False
        self._initialized = False
        self._files_uploaded = 0

    @property
    def name(self) -> str:
        return "Google FileSearch"

    @property
    def engine_type(self) -> EngineType:
        return EngineType.GOOGLE_FILESEARCH

    async def initialize(self) -> None:
        """Initialize Google FileSearch engine"""
        if self._initialized:
            return

        # Load configuration
        if not self.config_path.exists():
            raise ValueError(f"FileSearch config file not found: {self.config_path}")

        self._config = FileSearchConfig.from_yaml(str(self.config_path))

        # Apply runtime overrides
        for key, value in self.config_overrides.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Import and create Google GenAI client
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai not installed. Install with: pip install google-genai"
            )

        api_key = self._resolve_env_ref(self._config.api_key) or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Google API key required. Set 'api_key' in config or GOOGLE_API_KEY env var."
            )

        self._client = genai.Client(api_key=api_key)

        # Determine store: reuse existing or create new
        if self._config.file_search_store_name:
            # Mode 1: Reuse existing store
            self._store_name = self._config.file_search_store_name
            print(f"    Using existing FileSearch store: {self._store_name}")
        elif self._config.corpus_files:
            # Mode 2: Create new store and upload files
            await self._create_and_populate_store()
        else:
            raise ValueError(
                "Either 'file_search_store_name' or 'corpus_files' must be set in config."
            )

        self._initialized = True

    async def _create_and_populate_store(self) -> None:
        """Create a new FileSearch store and upload corpus files"""
        import time as _time

        print(f"    Creating FileSearch store: {self._config.display_name}")
        store = self._client.file_search_stores.create(
            config={'display_name': self._config.display_name}
        )
        self._store_name = store.name
        self._store_created_by_me = True
        print(f"    Store created: {self._store_name}")

        # Upload each file
        for file_path in self._config.corpus_files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"    ⚠️  File not found, skipping: {file_path}")
                continue

            print(f"    Uploading: {file_path.name}...", end=" ", flush=True)
            operation = self._client.file_search_stores.upload_to_file_search_store(
                file=str(file_path),
                file_search_store_name=self._store_name,
                config={'display_name': file_path.name}
            )

            # Wait for indexing to complete
            while not operation.done:
                _time.sleep(5)
                operation = self._client.operations.get(operation)

            self._files_uploaded += 1
            print("✓")

        print(f"    Uploaded {self._files_uploaded} files to store")

    async def search(self, query: str) -> QueryResult:
        """Execute a search query using Google FileSearch"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        from google.genai import types

        start = time.perf_counter()

        try:
            # Call generate_content with FileSearch tool
            response = self._client.models.generate_content(
                model=self._config.model,
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[self._store_name]
                            )
                        )
                    ]
                )
            )

            latency = time.perf_counter() - start

            # Extract answer text
            answer = response.text or ""

            # Extract token usage from usage_metadata
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0

            total_tokens = prompt_tokens + completion_tokens

            # Calculate cost
            cost = self.calculate_cost(
                prompt_tokens,
                completion_tokens,
                self.prompt_price_per_m,
                self.completion_price_per_m
            )

            # Extract grounding metadata (citations) if available
            context_text = None
            grounding_chunks = []
            if (response.candidates and
                    response.candidates[0].grounding_metadata):
                gm = response.candidates[0].grounding_metadata
                if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        chunk_text = getattr(chunk, 'text', None)
                        if chunk_text:
                            grounding_chunks.append(str(chunk_text))
                    if grounding_chunks:
                        context_text = "\n\n".join(grounding_chunks)

            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer=answer,
                context_text=context_text,
                latency_seconds=round(latency, 4),
                llm_calls=1,  # FileSearch is a single API call
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_eur=cost,
                extra_info={
                    "model": self._config.model,
                    "store_name": self._store_name,
                    "grounding_chunks": len(grounding_chunks),
                }
            )

        except Exception as e:
            if LOGGER.isEnabledFor(logging.INFO):
                LOGGER.exception("Google FileSearch query failed")
            latency = time.perf_counter() - start
            return QueryResult(
                engine=self.engine_type.value,
                prompt_name="",
                prompt_text=query,
                answer="",
                latency_seconds=round(latency, 4),
                error=str(e),
            )

    @staticmethod
    def _resolve_env_ref(value: Optional[str]) -> str:
        """Resolve values in the form 'os.environ/VAR_NAME' or pass through."""
        if not value:
            return ""
        if isinstance(value, str) and value.startswith("os.environ/"):
            return os.getenv(value.split("/", 1)[1], "")
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get FileSearch engine statistics"""
        if not self._initialized:
            return {"engine": self.name, "initialized": False}

        return {
            "engine": self.name,
            "model": self._config.model,
            "store_name": self._store_name,
            "files_uploaded": self._files_uploaded,
            "initialized": True,
        }

    async def close(self) -> None:
        """Clean up by deleting the store if we created it"""
        if self._initialized and self._store_created_by_me and self._store_name:
            try:
                print(f"    Deleting FileSearch store: {self._store_name}...")
                self._client.file_search_stores.delete(name=self._store_name)
                print("    ✓ Store deleted")
            except Exception as e:
                print(f"    ⚠️ Failed to delete FileSearch store: {e}")
