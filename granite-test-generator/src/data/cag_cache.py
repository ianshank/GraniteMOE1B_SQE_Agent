from typing import Dict, List, Any, Optional, TYPE_CHECKING
import time
from dataclasses import dataclass, asdict
import json

@dataclass
class CacheEntry:
    key: str
    content: str
    generated_response: str
    context: Dict[str, Any]
    embeddings: Optional[List[float]]
    tags: List[str]
    team_context: str
    timestamp: float
    access_count: int = 0
    
if TYPE_CHECKING:
    from src.utils.kv_cache import KVCache

class CAGCache:
    """Cache-Augmented Generation system optimized for test case patterns"""
    
    def __init__(self, kv_cache: 'KVCache'):
        self.kv_cache = kv_cache
        self.pattern_cache = {}  # In-memory cache for frequent patterns
        self.team_caches = {}   # Separate caches per team
        
    def preload_common_patterns(self, team_contexts: List[str]):
        """Preload frequently used test case patterns for each team"""
        common_patterns = [
            "login functionality test case",
            "api endpoint validation test case", 
            "database integration test case",
            "user input validation test case",
            "error handling test case",
            "performance test case"
        ]
        
        for team in team_contexts:
            team_cache = {}
            for pattern in common_patterns:
                # Retrieve existing test cases matching this pattern
                cached_entries = self.kv_cache.retrieve_by_tags([pattern, team])
                if cached_entries:
                    team_cache[pattern] = cached_entries
            
            self.team_caches[team] = team_cache
    
    def get_cached_response(self, query: str, team_context: str) -> Optional[Dict]:
        """Get cached response for similar queries"""
        # Check team-specific cache first
        if team_context in self.team_caches:
            for pattern, entries in self.team_caches[team_context].items():
                if self._query_matches_pattern(query, pattern):
                    # Return most recent matching entry
                    return max(entries, key=lambda x: x['timestamp'])
        
        # Fall back to general cache
        return self.kv_cache.retrieve(query, {'team': team_context})
    
    def cache_response(self, query: str, response: str, context: Dict, 
                      team_context: str, tags: List[str] = None):
        """Cache generated response with context"""
        tags = tags or []
        tags.extend([team_context, 'test_case'])
        
        return self.kv_cache.store(
            content=query,
            context=context,
            response=response,
            tags=tags
        )
    
    def _query_matches_pattern(self, query: str, pattern: str, threshold: float = 0.7) -> bool:
        """Simple similarity matching - could be enhanced with embeddings"""
        query_words = set(query.lower().split())
        pattern_words = set(pattern.lower().split())
        
        intersection = len(query_words.intersection(pattern_words))
        union = len(query_words.union(pattern_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity >= threshold
