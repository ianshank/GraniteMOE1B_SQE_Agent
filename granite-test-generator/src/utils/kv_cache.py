import pickle
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time
import logging

logger = logging.getLogger(__name__)

class KVCache:
    """Key-Value cache for storing precomputed embeddings and responses"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 15000):  # Increased from 10k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata: Dict[str, Any] = {}  # Initialize metadata
        self.load_metadata()
        logger.info(f"KVCache initialized at {self.cache_dir} with max_size={max_size}")
    
    def _generate_key(self, content: str, context: Dict[str, Any]) -> str:
        """Generate consistent hash key for content and context"""
        combined = f"{content}_{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def load_metadata(self):
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.debug(f"Loaded {len(self.metadata)} cache entries from metadata.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode metadata.json: {e}. Starting with empty metadata.")
                self.metadata = {}
        else:
            self.metadata = {}
            logger.debug("No metadata file found. Starting with empty metadata.")
    
    def save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved {len(self.metadata)} cache entries to metadata.")
        except Exception as e:
            logger.error(f"Failed to save metadata.json: {e}.")
    
    def store(self, content: str, context: Dict[str, Any], 
              embedding: Any = None, response: Any = None, 
              tags: Optional[List[str]] = None) -> str:  # Changed tags to Optional
        """Store content with embeddings and generated responses"""
        key = self._generate_key(content, context)
        
        cache_entry = {
            'content': content,
            'context': context,
            'embedding': embedding,
            'response': response,
            'tags': tags or [],
            'timestamp': time.time()
        }
        
        # Store to disk
        try:
            with open(self.cache_dir / f"{key}.pkl", 'wb') as f:
                pickle.dump(cache_entry, f)
            logger.debug(f"Stored cache entry for key: {key}")
        except Exception as e:
            logger.error(f"Failed to store cache entry for key {key}: {e}.")
            return ""
        
        # Update metadata
        self.metadata[key] = {
            'tags': tags or [],
            'timestamp': cache_entry['timestamp'],
            'size': len(content)  # Store content size for potential eviction strategies
        }
        self.save_metadata()
        
        # Basic eviction strategy (LRU-like by timestamp, or size-based)
        if len(self.metadata) > self.max_size:
            self._evict_oldest_entry()
        
        return key
    
    def retrieve(self, content: str, context: Dict[str, Any]) -> Optional[Dict]:
        """Retrieve cached entry"""
        key = self._generate_key(content, context)
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                logger.debug(f"Retrieved cache entry for key: {key}")
                return entry
            except Exception as e:
                logger.error(f"Failed to retrieve/load cache entry for key {key}: {e}.")
                return None
        logger.debug(f"No cache entry found for key: {key}")
        return None
    
    def retrieve_by_tags(self, tags: List[str]) -> List[Dict]:
        """Retrieve entries matching specific tags"""
        matching_entries = []
        for key, meta in self.metadata.items():
            if any(tag in meta['tags'] for tag in tags):
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            matching_entries.append(pickle.load(f))
                    except Exception as e:
                        logger.error(f"Failed to load tagged cache entry {key}: {e}.")
        logger.debug(f"Retrieved {len(matching_entries)} entries for tags: {tags}")
        return matching_entries
    
    def _evict_oldest_entry(self):
        """Evict the oldest entry from the cache based on timestamp."""
        if not self.metadata:
            return
        
        oldest_key = min(self.metadata, key=lambda k: self.metadata[k]['timestamp'])
        oldest_file = self.cache_dir / f"{oldest_key}.pkl"
        if oldest_file.exists():
            try:
                oldest_file.unlink()
                del self.metadata[oldest_key]
                self.save_metadata()
                logger.info(f"Evicted oldest cache entry: {oldest_key}")
            except Exception as e:
                logger.error(f"Failed to evict cache entry {oldest_key}: {e}.")
        else:
            del self.metadata[oldest_key]  # Remove from metadata even if file is gone
            self.save_metadata()
            logger.warning(f"Metadata entry for {oldest_key} found, but file missing. Removed metadata.")