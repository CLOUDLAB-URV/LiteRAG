"""
Vector Store for LiteRAG

LanceDB wrapper providing consistent interface for vector operations.
Supports reusing existing GraphRAG LanceDB indexes when available.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Table names used by GraphRAG
GRAPHRAG_ENTITY_TABLE = "default-entity-description"
GRAPHRAG_COMMUNITY_TABLE = "default-community-full_content"
GRAPHRAG_TEXTUNIT_TABLE = "default-text_unit-text"

# Table names for LiteRAG-specific data (when GraphRAG tables don't exist)
LiteRAG_ENTITY_TABLE = "LiteRAG-entity-description"
LiteRAG_COMMUNITY_TABLE = "LiteRAG-community-description"


class LiteRAGVectorStore:
    """
    LanceDB wrapper for LiteRAG vector operations.
    
    Provides a clean interface for:
    - Connecting to existing GraphRAG LanceDB databases
    - Creating new tables when needed
    - Performing similarity search
    - Adding/updating embeddings
    
    Example:
        store = LiteRAGVectorStore("/path/to/lancedb")
        
        # Check for existing GraphRAG embeddings
        if store.has_table(GRAPHRAG_ENTITY_TABLE):
            results = store.similarity_search(GRAPHRAG_ENTITY_TABLE, query_vec, k=10)
    """
    
    def __init__(self, db_uri: str):
        """
        Initialize connection to LanceDB.
        
        Args:
            db_uri: Path to LanceDB directory
        """
        self.db_uri = str(db_uri)
        self._db = None
        self._connect()
    
    def _connect(self):
        """Establish connection to LanceDB."""
        try:
            import lancedb
            
            # Ensure directory exists
            Path(self.db_uri).mkdir(parents=True, exist_ok=True)
            
            self._db = lancedb.connect(self.db_uri)
            logger.info(f"Connected to LanceDB at {self.db_uri}")
            
            # Log available tables
            tables = self._db.table_names()
            if tables:
                logger.info(f"Found existing tables: {tables}")
                
        except ImportError:
            raise ImportError(
                "LanceDB is required for LiteRAG vector operations. "
                "Install it with: pip install lancedb"
            )
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise
    
    def has_table(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            return table_name in self._db.table_names()
        except Exception as e:
            logger.warning(f"Error checking table existence: {e}")
            return False
    
    def get_table(self, table_name: str):
        """
        Get an existing table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            LanceDB table object
            
        Raises:
            ValueError: If table doesn't exist
        """
        if not self.has_table(table_name):
            raise ValueError(f"Table '{table_name}' does not exist")
        return self._db.open_table(table_name)
    
    def create_table(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        mode: str = "overwrite"
    ):
        """
        Create a new table with initial data.
        
        Args:
            table_name: Name for the new table
            data: List of dicts with 'id', 'text', 'vector' keys
            mode: 'overwrite' or 'create' (fail if exists)
            
        Returns:
            LanceDB table object
        """
        return self._db.create_table(table_name, data, mode=mode)
    
    def similarity_search(
        self,
        table_name: str,
        query_vector: np.ndarray,
        k: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in a table.
        
        Args:
            table_name: Table to search in
            query_vector: Query embedding vector
            k: Number of results to return
            filter_expr: Optional filter expression
            
        Returns:
            List of (id, distance) tuples, sorted by distance (ascending)
        """
        try:
            table = self.get_table(table_name)
            
            # Perform search
            query = table.search(query_vector.tolist()).limit(k)
            
            if filter_expr:
                query = query.where(filter_expr)
            
            results = query.to_list()
            
            # Extract id and distance
            # LanceDB returns _distance field for distance
            output = []
            for row in results:
                # Try different ID field names (GraphRAG uses 'id', we might use 'entity_id')
                entity_id = row.get('id') or row.get('entity_id') or row.get('title', '')
                distance = row.get('_distance', 0.0)
                
                # Convert distance to similarity score (1 - distance for L2, or use cosine directly)
                # LanceDB uses L2 distance by default, convert to similarity-like score
                similarity = 1.0 / (1.0 + distance)
                
                output.append((str(entity_id), similarity))
            
            return output
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def similarity_search_by_text(
        self,
        table_name: str,
        query_text: str,
        embed_fn,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search by text, embedding it first.
        
        Args:
            table_name: Table to search in
            query_text: Text to search for
            embed_fn: Function to convert text to embedding
            k: Number of results
            
        Returns:
            List of (id, similarity) tuples
        """
        query_vector = embed_fn(query_text)
        return self.similarity_search(table_name, query_vector, k)
    
    def add_embeddings(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        mode: str = "append"
    ) -> None:
        """
        Add embeddings to a table.
        
        Args:
            table_name: Target table
            records: List of dicts with at least 'id', 'text', 'vector' keys
            mode: 'append' to add, 'overwrite' to replace all
        """
        if not records:
            return
            
        try:
            if self.has_table(table_name):
                table = self.get_table(table_name)
                if mode == "overwrite":
                    # Drop and recreate
                    self._db.drop_table(table_name)
                    self.create_table(table_name, records)
                else:
                    table.add(records)
            else:
                self.create_table(table_name, records)
                
            logger.info(f"Added {len(records)} embeddings to {table_name}")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def get_all_ids(self, table_name: str) -> List[str]:
        """
        Get all IDs from a table.
        
        Args:
            table_name: Table to query
            
        Returns:
            List of all entity IDs in the table
        """
        try:
            table = self.get_table(table_name)
            df = table.to_pandas()
            
            # Try different ID column names
            for col in ['id', 'entity_id', 'title']:
                if col in df.columns:
                    return df[col].astype(str).tolist()
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get IDs from {table_name}: {e}")
            return []
    
    def get_all_embeddings(self, table_name: str) -> Dict[str, np.ndarray]:
        """
        Load all embeddings from a table into memory.
        
        This is efficient for caching entity embeddings for fast
        semantic similarity computation during graph exploration.
        
        Args:
            table_name: Table to load from
            
        Returns:
            Dict mapping entity id/title to embedding vector
        """
        try:
            table = self.get_table(table_name)
            df = table.to_pandas()
            
            # Find the ID column
            id_col = None
            for col in ['id', 'entity_id', 'title']:
                if col in df.columns:
                    id_col = col
                    break
            
            if id_col is None:
                logger.warning(f"No ID column found in {table_name}")
                return {}
            
            # Build dict of id -> embedding
            embeddings = {}
            for _, row in df.iterrows():
                entity_id = str(row[id_col])
                vector = np.array(row['vector'])
                embeddings[entity_id] = vector
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {table_name} into memory")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {table_name}: {e}")
            return {}
    
    def table_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        try:
            table = self.get_table(table_name)
            return table.count_rows()
        except Exception:
            return 0
    
    def close(self):
        """Close the database connection."""
        self._db = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def detect_graphrag_lancedb(data_dir: str) -> Optional[str]:
    """
    Detect if a GraphRAG LanceDB exists in the given data directory.
    
    GraphRAG typically stores LanceDB in: {data_dir}/lancedb or {data_dir}/output/lancedb
    
    Args:
        data_dir: Path to GraphRAG data directory
        
        
    Returns:
        Path to LanceDB if found, None otherwise
    """
    possible_paths = [
        Path(data_dir) / "lancedb",
        Path(data_dir) / "output" / "lancedb",
        Path(data_dir).parent / "output" / "lancedb",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            # Check for LanceDB signature files/directories
            if any((path / table).exists() for table in [
                f"{GRAPHRAG_ENTITY_TABLE}.lance",
                f"{GRAPHRAG_COMMUNITY_TABLE}.lance",
                f"{GRAPHRAG_TEXTUNIT_TABLE}.lance",
            ]):
                logger.info(f"Found GraphRAG LanceDB at {path}")
                return str(path)
    
    return None
