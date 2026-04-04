import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONSTANTS ---
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
STORE_PATH = os.path.join(CACHE_DIR, "store.pkl")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class SemanticCache:
    """A local vector-based semantic cache for storing and retrieving pipeline results."""

    def __init__(self, threshold=0.85):
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.threshold = threshold
        self.model = None     # don't load on startup
        self.index = faiss.IndexFlatIP(self.dimension)
        self.cache_store = []
        os.makedirs(CACHE_DIR, exist_ok=True)
        if os.path.exists(INDEX_PATH) and os.path.exists(STORE_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(STORE_PATH, "rb") as f:
                self.cache_store = pickle.load(f)
            print(f"[CACHE] Loaded {self.index.ntotal} cached queries from disk")
        else:
            print("[CACHE] Starting fresh cache")

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def _fresh_index(self):
        """Creates a completely new empty FAISS index."""
        index = faiss.IndexFlatIP(self.dimension)
        return index

    def search(self, query: str):
        """Queries the FAISS index to find a cached result exceeding the similarity threshold."""
        if self.index.ntotal == 0:
            return None
        vec = self._get_model().encode([query]).astype(np.float32)
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, 1)
        best_score = distances[0][0]
        best_idx = indices[0][0]
        if best_score >= self.threshold and best_idx != -1:
            print(f"[CACHE HIT] score={best_score:.2f} for '{query}'")
            entry = self.cache_store[best_idx]
            if isinstance(entry, dict) and "table" in entry:
                return entry
            else:
                return {"query": query, "table": entry}
        print(f"[CACHE MISS] best={best_score:.2f} for '{query}'")
        return None

    def search_all(self) -> list[dict]:
        """Returns metadata for all globally cached entries."""
        results = []
        for entry in self.cache_store:
            if isinstance(entry, dict) and "table" in entry:
                results.append({
                    "query": entry.get("query", ""),
                    "entity_count": len(entry["table"].get("entities", [])),
                })
            else:
                results.append({
                    "query": "",
                    "entity_count": len(entry.get("entities", [])) if isinstance(entry, dict) else 0,
                })
        return results

    def save(self, query: str, table_data: dict):
        """Embeds and saves a new query string and its final result table to the cache."""
        vec = self._get_model().encode([query]).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.cache_store.append({"query": query, "table": table_data})
        self._save_to_disk()
        print(f"[CACHE SAVED] '{query}' — {self.index.ntotal} total cached")
    
    def _save_to_disk(self):
        """Persists the FAISS index and local store to the file system."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(STORE_PATH, "wb") as f:
            pickle.dump(self.cache_store, f)