import os
import json

import numpy as np

# The semantic cache is a latency optimisation, not a dependency. If the Postgres
# driver or the server is unavailable the pipeline must still answer queries, so
# the import is optional and every operation degrades to a no-op.
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

# --- CONSTANTS ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Allow overriding the DATABASE_URL, fallback to a default dev connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/agentic_search")

class SemanticCache:
    """A Postgres vector-based semantic cache for storing and retrieving pipeline results."""

    def __init__(self, threshold=0.85):
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.threshold = threshold
        self.model = None     # don't load on startup
        self.enabled = PGVECTOR_AVAILABLE and self._init_db()
        if not self.enabled:
            print("[CACHE] Disabled — running without the semantic cache.")

    def _init_db(self) -> bool:
        """Creates the pgvector extension and the cache table if they don't exist."""
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cached_queries (
                        id SERIAL PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        embedding vector(384),
                        table_data JSONB NOT NULL
                    );
                """)
            conn.close()
            print("[CACHE] Postgres pgvector initialized.")
            return True
        except Exception as e:
            print(f"[CACHE] Error initializing Postgres: {e}")
            return False

    def _get_conn(self):
        """Returns a new psycopg2 connection with pgvector registered."""
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
        return conn

    def _get_model(self):
        """Lazy loads the sentence-transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def warm(self):
        """Loads the embedding model ahead of the first query.

        _get_model() is lazy, so without this the first visitor after a restart
        pays for loading all-MiniLM-L6-v2 inside their request. Measured on the
        t3.micro deployment: 35.3s for the first query against 8.9s for the second.
        """
        if not self.enabled:
            return
        try:
            self._get_model().encode(["warm"])
            print("[CACHE] Embedding model warmed.")
        except Exception as e:
            print(f"[CACHE] Warm-up failed: {e}")

    def search(self, query: str):
        """Queries the Postgres pgvector to find a cached result exceeding the similarity threshold."""
        if not self.enabled:
            return None
        try:
            vec = self._get_model().encode([query])[0]
        except Exception as e:
            print(f"[CACHE] Embedding failed, skipping cache: {e}")
            return None
        
        # Calculate cosine distance threshold. Cosine similarity = 1 - cosine_distance
        # If similarity threshold is 0.85, max distance is 0.15
        max_distance = 1.0 - self.threshold
        
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                # <=> is the cosine distance operator in pgvector
                cur.execute(
                    """
                    SELECT query_text, table_data, embedding <=> %s AS distance
                    FROM cached_queries
                    ORDER BY distance ASC
                    LIMIT 1;
                    """,
                    (vec,)
                )
                row = cur.fetchone()
                
            conn.close()
            
            if row:
                query_text, table_data, distance = row
                if distance <= max_distance:
                    score = 1.0 - distance
                    print(f"[CACHE HIT] score={score:.2f} for '{query}'")
                    return {"query": query_text, "table": table_data}
                
                print(f"[CACHE MISS] best={1.0 - distance:.2f} for '{query}'")
            else:
                print(f"[CACHE MISS] no entries in cache for '{query}'")
                
            return None
        except Exception as e:
            print(f"[CACHE] Search error: {e}")
            return None

    def search_all(self) -> list[dict]:
        """Returns metadata for all globally cached entries."""
        if not self.enabled:
            return []
        try:
            conn = self._get_conn()
            results = []
            with conn.cursor() as cur:
                cur.execute("SELECT query_text, table_data FROM cached_queries;")
                rows = cur.fetchall()
                for query_text, table_data in rows:
                    results.append({
                        "query": query_text,
                        "entity_count": len(table_data.get("entities", [])),
                    })
            conn.close()
            return results
        except Exception as e:
            print(f"[CACHE] Search_all error: {e}")
            return []

    def save(self, query: str, table_data: dict):
        """Embeds and saves a new query string and its final result table to Postgres."""
        if not self.enabled:
            return
        try:
            vec = self._get_model().encode([query])[0]
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cached_queries (query_text, embedding, table_data)
                    VALUES (%s, %s, %s);
                    """,
                    (query, vec, json.dumps(table_data))
                )
            conn.commit()
            conn.close()
            print(f"[CACHE SAVED] '{query}'")
        except Exception as e:
            print(f"[CACHE] Save error: {e}")

    def clear(self):
        """Truncates the cache table."""
        if not self.enabled:
            return
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute("TRUNCATE cached_queries;")
            conn.commit()
            conn.close()
            print("[CACHE] Cleared successfully.")
        except Exception as e:
            print(f"[CACHE] Clear error: {e}")