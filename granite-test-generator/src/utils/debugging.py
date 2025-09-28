# src/utils/debugging.py
import logging
from typing import Any, Dict, List
from functools import wraps
import time
import traceback

class DebugLogger:
    """Comprehensive debugging system for the test generation pipeline"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.performance_metrics = {}
        self.path_validation_cache = {}
    
    def validate_path_resolution(self, input_directory: str, team_name: str) -> Dict[str, Any]:
        """Validate and debug path resolution for team connectors.
        
        Args:
            input_directory: Input directory path to validate
            team_name: Team name for context
            
        Returns:
            Validation result with detailed diagnostics
        """
        cache_key = f"{team_name}:{input_directory}"
        if cache_key in self.path_validation_cache:
            return self.path_validation_cache[cache_key]
        
        result = {
            'team_name': team_name,
            'input_directory': input_directory,
            'is_valid': False,
            'issues': [],
            'suggestions': [],
            'path_info': {}
        }
        
        try:
            input_path = Path(input_directory)
            
            # Record path information
            result['path_info'] = {
                'raw_path': input_directory,
                'resolved_path': str(input_path.resolve()),
                'absolute_path': str(input_path.absolute()),
                'is_absolute': input_path.is_absolute(),
                'current_working_directory': os.getcwd()
            }
            
            # Check if path exists
            if not input_path.exists():
                result['issues'].append(f"Input directory does not exist: {input_path}")
                
                # Suggest alternatives
                parent = input_path.parent
                if parent.exists():
                    subdirs = [d.name for d in parent.iterdir() if d.is_dir()]
                    if subdirs:
                        result['suggestions'].append(f"Available directories in {parent}: {subdirs}")
                
                # Check for common path mistakes
                if 'granite-test-generator' in str(input_path):
                    corrected = str(input_path).replace('granite-test-generator/', '')
                    result['suggestions'].append(f"Try removing 'granite-test-generator/' prefix: {corrected}")
                
                self.path_validation_cache[cache_key] = result
                return result
            
            # Check if it's a directory
            if not input_path.is_dir():
                result['issues'].append(f"Path exists but is not a directory: {input_path}")
                self.path_validation_cache[cache_key] = result
                return result
            
            # Check for readable files
            file_types = ['.md', '.txt', '.json']
            file_counts = {}
            
            for file_type in file_types:
                files = list(input_path.glob(f"*{file_type}"))
                file_counts[file_type] = len(files)
            
            total_files = sum(file_counts.values())
            result['path_info']['file_counts'] = file_counts
            result['path_info']['total_files'] = total_files
            
            if total_files == 0:
                result['issues'].append("No readable files found in directory")
                result['suggestions'].append("Ensure directory contains .md, .txt, or .json files")
            else:
                result['is_valid'] = True
                result['suggestions'].append(f"Found {total_files} files: {file_counts}")
        
        except Exception as e:
            result['issues'].append(f"Path validation error: {e}")
        
        self.path_validation_cache[cache_key] = result
        return result
    
    def _setup_logger(self, log_level):
        """Set up structured logging"""
        logger = logging.getLogger('granite_test_generator')
        logger.setLevel(log_level)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler('debug.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_performance(self, func_name: str, execution_time: float, **kwargs):
        """Log performance metrics"""
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = []
        
        self.performance_metrics[func_name].append({
            'execution_time': execution_time,
            'timestamp': time.time(),
            **kwargs
        })
        
        self.logger.info(f"Performance - {func_name}: {execution_time:.2f}s")
    
    def log_moe_routing(self, input_text: str, activated_experts: List[int], 
                       routing_weights: List[float]):
        """Log MoE expert routing decisions"""
        self.logger.info(f"MoE Routing - Input: {input_text[:100]}...")
        self.logger.info(f"Activated experts: {activated_experts}")
        self.logger.info(f"Routing weights: {routing_weights}")
    
    def log_test_case_quality(self, test_case: Any, quality_score: float):
        """Log test case quality metrics"""
        self.logger.info(f"Test Case Quality - ID: {test_case.id}, Score: {quality_score}")
        
    def performance_monitor(self, func):
        """Decorator to monitor function performance"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.log_performance(func.__name__, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.log_performance(func.__name__, execution_time, error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                self.log_performance(func.__name__, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.log_performance(func.__name__, execution_time, error=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# Test suite
# tests/test_granite_system.py
import unittest
import asyncio
from unittest.mock import Mock, patch
from src.models.test_case_schemas import TestCase, TestCasePriority, TestCaseType

class TestGraniteSystem(unittest.TestCase):
    """Comprehensive test suite for the Granite test generation system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_test_case = TestCase(
            id="test_001",
            summary="Test user login functionality",
            priority=TestCasePriority.HIGH,
            test_type=TestCaseType.FUNCTIONAL,
            steps=[],
            expected_results="User should be able to login successfully"
        )
    
    def test_test_case_schema_validation(self):
        """Test Pydantic schema validation"""
        # Valid test case should pass
        self.assertIsInstance(self.sample_test_case, TestCase)
        self.assertEqual(self.sample_test_case.priority, TestCasePriority.HIGH)
        
        # Invalid priority should raise validation error
        with self.assertRaises(ValueError):
            TestCase(
                id="invalid",
                summary="Test",
                priority="invalid_priority",  # This should fail
                test_type=TestCaseType.FUNCTIONAL,
                steps=[],
                expected_results="Test"
            )
    
    @patch('src.models.granite_moe.generate')
    def test_granite_generation(self, mock_generate):
        """Test Granite MoE model generation"""
        mock_generate.return_value = "[TEST_CASE][SUMMARY]Login Test[/SUMMARY][/TEST_CASE]"
        
        from src.models.granite_moe import GraniteMoETrainer
        trainer = GraniteMoETrainer()
        
        # Mock the model loading
        trainer.mlx_model = Mock()
        trainer.mlx_tokenizer = Mock()
        
        result = trainer._generate_test_case("Test login functionality")
        
        self.assertIn("Login Test", result)
        mock_generate.assert_called_once()
    
    def test_rag_chunking(self):
        """Test document chunking for RAG"""
        from src.utils.chunking import IntelligentChunker
        
        chunker = IntelligentChunker()
        test_text = """
        REQ-001: User login functionality
        The system shall allow users to login with username and password.
        
        REQ-002: Password validation
        The system shall validate password strength.
        """
        
        chunks = chunker.chunk_requirements(
            test_text, 
            {'doc_id': 'test_doc', 'team': 'test_team'}
        )
        
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].team_context, 'test_team')
    
    async def test_agent_workflow(self):
        """Test the agent-based test generation workflow"""
        from src.agents.generation_agent import TestGenerationAgent
        
        # Mock dependencies
        mock_granite = Mock()
        mock_rag = Mock()
        mock_cag = Mock()
        
        agent = TestGenerationAgent(mock_granite, mock_rag, mock_cag)
        
        # Mock the tool functions
        agent._retrieve_requirements = Mock(return_value="Sample requirement")
        agent._check_cache = Mock(return_value="No cached response found")
        agent._generate_test_case = Mock(return_value="[TEST_CASE][SUMMARY]Test[/SUMMARY][/TEST_CASE]")
        
        result = await agent.generate_test_cases_for_team("test_team", ["Sample requirement"])
        
        self.assertIsInstance(result, list)
    
    def test_kv_cache_functionality(self):
        """Test KV cache operations"""
        from src.utils.kv_cache import KVCache
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = KVCache(cache_dir=temp_dir)
            
            # Test store and retrieve
            key = cache.store(
                content="test content",
                context={'team': 'test'},
                response="test response",
                tags=['test']
            )
            
            retrieved = cache.retrieve("test content", {'team': 'test'})
            
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved['response'], "test response")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
