# src/utils/chunking.py
from typing import List, Dict, Any
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_type: str  # requirements, user_story, documentation
    team_context: str

class IntelligentChunker:
    """Chunks documents based on semantic structure for test case generation"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_requirements(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk requirements documents preserving semantic boundaries"""
        # Split on requirement boundaries (e.g., "REQ-", "US-", numbered lists)
        req_pattern = r'(?=(?:REQ-|US-|\d+\.|[A-Z]+\d+))'
        sections = re.split(req_pattern, text)
        
        chunks = []
        for i, section in enumerate(sections):
            if len(section.strip()) > 0:
                chunks.append(DocumentChunk(
                    content=section.strip(),
                    metadata={**metadata, 'section_index': i},
                    chunk_id=f"{metadata.get('doc_id', 'unknown')}_{i}",
                    source_type='requirements',
                    team_context=metadata.get('team', 'unknown')
                ))
        
        return self._apply_sliding_window(chunks)
    
    def chunk_user_stories(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """Chunk user stories maintaining story integrity"""
        # User stories typically follow "As a... I want... So that..." pattern
        story_pattern = r'(?=As\s+a\s+)'
        stories = re.split(story_pattern, text, flags=re.IGNORECASE)
        
        chunks = []
        for i, story in enumerate(stories):
            if len(story.strip()) > 50:  # Filter out very short segments
                chunks.append(DocumentChunk(
                    content=story.strip(),
                    metadata={**metadata, 'story_index': i},
                    chunk_id=f"story_{metadata.get('doc_id', 'unknown')}_{i}",
                    source_type='user_story',
                    team_context=metadata.get('team', 'unknown')
                ))
                
        return chunks
