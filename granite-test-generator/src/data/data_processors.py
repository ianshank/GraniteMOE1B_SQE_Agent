# src/data/data_processors.py
from typing import List, Dict, Any, Tuple
from src.utils.chunking import IntelligentChunker
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.models.test_case_schemas import TestCase
import json
import pandas as pd
from pathlib import Path

class TestCaseDataProcessor:
    """Process various data formats into training-ready test cases"""
    
    def __init__(self, chunker: IntelligentChunker):
        self.chunker = chunker
        
    def process_requirements_files(self, requirements_dir: str) -> List[Tuple[str, Dict]]:
        """Process requirements documents from multiple teams"""
        processed_docs = []
        req_path = Path(requirements_dir)
        
        for file_path in req_path.glob("**/*.txt"):
            team_name = file_path.parent.name
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                'doc_id': file_path.stem,
                'team': team_name,
                'file_path': str(file_path)
            }
            
            chunks = self.chunker.chunk_requirements(content, metadata)
            processed_docs.extend([(chunk.content, chunk.metadata) for chunk in chunks])
        
        return processed_docs
    
    def process_user_stories(self, stories_file: str) -> List[Tuple[str, Dict]]:
        """Process user stories from JSON/CSV format"""
        stories_path = Path(stories_file)
        processed_stories = []
        
        if stories_path.suffix == '.json':
            with open(stories_path, 'r') as f:
                stories_data = json.load(f)
        elif stories_path.suffix == '.csv':
            stories_data = pd.read_csv(stories_path).to_dict('records')
        else:
            raise ValueError("Supported formats: JSON, CSV")
        
        for story in stories_data:
            story_text = story.get('story', '') or story.get('description', '')
            metadata = {
                'story_id': story.get('id', ''),
                'team': story.get('team', 'unknown'),
                'priority': story.get('priority', 'medium'),
                'epic': story.get('epic', '')
            }
            
            chunks = self.chunker.chunk_user_stories(story_text, metadata)
            processed_stories.extend([(chunk.content, chunk.metadata) for chunk in chunks])
        
        return processed_stories
    
    def create_synthetic_test_cases(self, requirements: List[str], 
                                   team_context: str) -> List['TestCase']:
        """Create synthetic test cases for initial training"""
        synthetic_cases = []
        
        templates = self._get_test_case_templates()
        
        for req in requirements:
            # Select appropriate template based on requirement content
            template = self._select_template(req, templates)
            
            # Generate test case
            test_case = self._generate_from_template(template, req, team_context)
            synthetic_cases.append(test_case)
        
        return synthetic_cases
    
    def _get_test_case_templates(self) -> List[Dict]:
        """Define test case templates for different scenarios"""
        return [
            {
                'type': 'login',
                'pattern': ['login', 'authenticate', 'sign in'],
                'template': {
                    'summary': 'Verify {functionality} functionality',
                    'steps': [
                        'Navigate to login page',
                        'Enter valid credentials',
                        'Click login button',
                        'Verify successful login'
                    ],
                    'expected': 'User should be successfully logged in and redirected to dashboard'
                }
            },
            {
                'type': 'api',
                'pattern': ['api', 'endpoint', 'service'],
                'template': {
                    'summary': 'Validate {functionality} API endpoint',
                    'steps': [
                        'Send request to API endpoint',
                        'Verify response status code',
                        'Validate response structure',
                        'Check error handling'
                    ],
                    'expected': 'API should return correct response with proper status code'
                }
            },
        ]
