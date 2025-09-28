import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.models.granite_moe import GraniteMoETrainer
from src.data.rag_retriever import RAGRetriever
from src.data.cag_cache import CAGCache
from src.utils.kv_cache import KVCache
from src.utils.chunking import IntelligentChunker
from src.agents.generation_agent import TestGenerationAgent
from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration
from src.integration.team_connectors import JiraConnector, GitHubConnector, LocalFileSystemConnector
from src.data.data_processors import TestCaseDataProcessor
from src.utils.constants import (
    DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_DIR, DEFAULT_REQUIREMENTS_DIR,
    APP_NAME, APP_DESCRIPTION
)
from src.utils.config_utils import get_config_with_defaults

logger = logging.getLogger(__name__)

class GraniteTestCaseGenerator:
    """Main class orchestrating the entire test case generation system"""
    
    def __init__(self, config_path: str = "config/model_config.yaml", config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the GraniteTestCaseGenerator.
        
        Args:
            config_path: Path to configuration file (used if config_dict is None)
            config_dict: Configuration dictionary (overrides config_path if provided)
        """
        self.config = self._load_config(config_path, config_dict)
        self.components = {}
        
    def _load_config(self, config_path: str, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration from YAML file or use provided dictionary.
        
        Args:
            config_path: Path to configuration file (used if config_dict is None)
            config_dict: Configuration dictionary (overrides config_path if provided)
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config_path does not exist and config_dict is None
            yaml.YAMLError: If config_path exists but contains invalid YAML
        """
        # If config_dict is provided, use it directly
        if config_dict is not None:
            logger.debug("Using provided configuration dictionary")
            return config_dict
        
        # Otherwise, load from file with environment variable substitution
        try:
            config = get_config_with_defaults(config_path)
            logger.debug(f"Loaded configuration from {config_path}")
            return config or {}
        except Exception as e:
            logger.warning(f"Failed to load configuration from {config_path}: {e}")
            return {}
    
    async def initialize_system(self):
        """Initialize all system components"""
        print(f"Initializing {APP_NAME}...")
        
        # Initialize core components
        self.components['chunker'] = IntelligentChunker()
        self.components['kv_cache'] = KVCache(cache_dir="./cache")
        self.components['cag_cache'] = CAGCache(self.components['kv_cache'])
        self.components['rag_retriever'] = RAGRetriever()
        
        # Initialize Granite MoE trainer
        model_name = self.config.get('model_name', DEFAULT_MODEL_NAME)
        self.components['granite_trainer'] = GraniteMoETrainer(model_name)
        
        # Initialize agent
        self.components['test_agent'] = TestGenerationAgent(
            self.components['granite_trainer'],
            self.components['rag_retriever'],
            self.components['cag_cache']
        )
        
        # Initialize workflow orchestrator
        self.components['orchestrator'] = WorkflowOrchestrator(self.components['test_agent'])
        
        print("System initialization complete!")
    
    async def setup_data_pipeline(self):
        """Set up data processing and indexing pipeline.
        
        Processes requirements documents and user stories from configured paths.
        Indexes documents in the RAG system and preloads the CAG cache.
        """
        print("Setting up data pipeline...")
        logger.info("Setting up data pipeline...")
        
        processor = TestCaseDataProcessor(self.components['chunker'])
        
        # Get requirements directory from config or use default
        requirements_dir = self.config.get('paths', {}).get('requirements_dir', DEFAULT_REQUIREMENTS_DIR)
        requirements_path = Path(requirements_dir)
        
        # Get user stories path from config or use default
        user_stories_path = self.config.get('paths', {}).get('user_stories_path', "data/user_stories.json")
        user_stories_file = Path(user_stories_path)
        
        # Process requirements documents
        requirements_docs = []
        if requirements_path.exists():
            requirements_docs = processor.process_requirements_files(str(requirements_path))
            logger.info(f"Processed {len(requirements_docs)} requirement documents from {requirements_path}")
            print(f"Processed {len(requirements_docs)} requirement documents")
            # Index in RAG system if RAG is enabled
            if self.config.get('rag', {}).get('enabled', True) and requirements_docs:
                from src.utils.chunking import DocumentChunk
                chunks = [DocumentChunk(content=content, metadata=metadata, 
                                      chunk_id=f"req_{i}", source_type='requirements',
                                      team_context=metadata.get('team', 'default'))
                         for i, (content, metadata) in enumerate(requirements_docs)]
                self.components['rag_retriever'].index_documents(chunks)
                logger.info(f"Indexed {len(chunks)} document chunks in RAG system")
        else:
            logger.warning(f"Requirements directory not found: {requirements_path}")

        # Persist whether we have any local requirements data for later decisions
        self.components['has_requirements'] = bool(requirements_docs)
        
        # Process user stories if available
        user_stories = []
        if user_stories_file.exists():
            user_stories = processor.process_user_stories(str(user_stories_file))
            logger.info(f"Processed {len(user_stories)} user stories from {user_stories_file}")
            print(f"Processed {len(user_stories)} user stories")
            
            # Index user stories in RAG system if RAG is enabled
            if user_stories and self.config.get('rag', {}).get('enabled', True):
                from src.utils.chunking import DocumentChunk
                chunks = [DocumentChunk(content=content, metadata=metadata, 
                                      chunk_id=f"story_{i}", source_type='user_story',
                                      team_context=metadata.get('team', 'default'))
                         for i, (content, metadata) in enumerate(user_stories)]
                
                self.components['rag_retriever'].index_documents(chunks)
                logger.info(f"Indexed {len(chunks)} user story chunks in RAG system")
        else:
            logger.debug(f"User stories file not found: {user_stories_file}")
        
        # Preload CAG cache with common patterns if CAG is enabled
        if self.config.get('cag', {}).get('preload_patterns', True):
            # Collect team contexts from all documents
            team_contexts = set()
            for doc in requirements_docs:
                team_contexts.add(doc[1].get('team', 'default'))
            for doc in user_stories:
                team_contexts.add(doc[1].get('team', 'default'))
                
            # If no team contexts found, use default
            if not team_contexts:
                team_contexts = {'default'}
                
            # Preload CAG cache
            self.components['cag_cache'].preload_common_patterns(list(team_contexts))
            logger.info(f"Preloaded CAG cache with patterns for {len(team_contexts)} teams")
        
        logger.info("Data pipeline setup complete!")
        print("Data pipeline setup complete!")
    
    async def fine_tune_model(self):
        """Fine-tune Granite MoE model if training data is available"""
        if not Path("data/training").exists():
            print("No training data found. Skipping fine-tuning.")
            return
        
        print("Starting model fine-tuning...")
        
        # This function is kept as a stub for future implementation.
        # Fine-tuning should only be performed with real training data obtained from external
        # sources, which should be placed in the data/training directory.
        logger.warning("Fine-tuning with synthetic test cases has been disabled.")
        logger.info("To use fine-tuning, obtain real training data from external sources and place it in the data/training directory.")
        
        print("Fine-tuning skipped: no synthetic test generation allowed.")
        return None
    
    def register_teams(self):
        """Register team connectors based on configuration.
        
        Supports the following connector types:
        - jira: Jira connector for teams using Jira for requirements management
        - github: GitHub connector for teams using GitHub Issues
        - local: LocalFileSystem connector for teams using local files
        """
        teams_config = list(self.config.get('teams', []) or [])

        # Optionally merge integration config file specified via env var
        integ_path = os.getenv("INTEGRATION_CONFIG_PATH")
        if integ_path and Path(integ_path).exists():
            try:
                import yaml
                with open(integ_path, 'r', encoding='utf-8') as f:
                    integ_cfg = yaml.safe_load(f) or {}
                extra_teams = integ_cfg.get('teams', []) or []
                # Normalize connector fields for local connectors (accept 'path' alias)
                for t in extra_teams:
                    conn = t.get('connector', {}) or {}
                    if conn.get('type', '').lower() == 'local':
                        if 'path' in conn and 'input_directory' not in conn:
                            conn['input_directory'] = str(conn['path'])
                        # Normalize 'output' key if provided
                        if 'output' in conn and 'output_directory' not in conn:
                            conn['output_directory'] = str(conn['output'])
                        # Cleanup aliases
                        conn.pop('path', None)
                        conn.pop('output', None)
                teams_config.extend(extra_teams)
                logger.info(f"Merged {len(extra_teams)} teams from integration config: {integ_path}")
            except Exception as e:
                logger.error(f"Failed to load integration config from {integ_path}: {e}")
        
        if not teams_config:
            logger.warning("No teams configured. Using default local team.")
            # Create a default local team if none are configured
            teams_config = [{
                'name': 'default',
                'connector': {
                    'type': 'local',
                    'input_directory': DEFAULT_REQUIREMENTS_DIR,
                    'output_directory': DEFAULT_OUTPUT_DIR
                },
                'rag_enabled': True,
                'cag_enabled': True,
                'auto_push': False
            }]
        
        registered_count = 0
        for team_config in teams_config:
            try:
                # Create team configuration using factory method
                team_configuration = self.components['orchestrator'].create_team_configuration(team_config)
                
                # Register team
                self.components['orchestrator'].register_team(team_configuration)
                team_name = team_configuration.team_name
                connector_type = team_config.get('connector', {}).get('type', 'unknown')
                logger.info(f"Registered team: {team_name} with connector type: {connector_type}")
                print(f"Registered team: {team_name}")
                registered_count += 1
                
            except ValueError as e:
                logger.error(f"Error creating team configuration: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error registering team: {e}")
                continue
        
        # Check if any teams were registered
        if registered_count == 0:
            logger.error("No valid teams were registered. Cannot proceed.")
            raise ValueError("No valid teams were registered. Check your configuration.")
        
        logger.info(f"Registered {registered_count} teams")
        return registered_count
    
    async def generate_test_cases(self):
        """Main execution: generate test cases for all registered teams.
        
        Processes all registered teams, generates test cases, and saves results to output directory.
        Also generates a quality report for all teams.
        
        Returns:
            Dictionary mapping team names to lists of generated test cases
        """
        print("Starting test case generation for all teams...")
        logger.info("Starting test case generation for all registered teams")
        
        # If no teams registered yet, attempt to register from config (includes default local fallback)
        orchestrator = self.components['orchestrator']
        has_requirements = self.components.get('has_requirements', False)
        if not getattr(orchestrator, 'team_configs', {}) and has_requirements:
            try:
                self.register_teams()
            except Exception as e:
                logger.warning(f"Team registration failed or none configured: {e}")
        results = await orchestrator.process_all_teams()
        
        # Get output directory from config or use default
        output_dir_path = self.config.get('paths', {}).get('output_dir', DEFAULT_OUTPUT_DIR)
        output_dir = Path(output_dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_generated = 0
        for team_name, test_cases in results.items():
            total_generated += len(test_cases)
            
            # Save test cases as JSON
            output_file = output_dir / f"{team_name}_test_cases.json"
            with open(output_file, 'w') as f:
                test_cases_dict = [
                    (tc.model_dump() if hasattr(tc, 'model_dump') else tc.dict())
                    for tc in test_cases
                ]
                json.dump(test_cases_dict, f, indent=2, default=str)
            
            logger.info(f"Generated {len(test_cases)} test cases for team: {team_name}")
            print(f"Generated {len(test_cases)} test cases for team: {team_name}")
        
        # Generate quality report
        quality_report = self.components['orchestrator'].generate_quality_report()
        quality_report_file = output_dir / "quality_report.json"
        with open(quality_report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        logger.info(f"Total test cases generated: {total_generated}")
        logger.info(f"Results saved to {output_dir} directory")
        print(f"Total test cases generated: {total_generated}")
        print(f"Results saved to {output_dir} directory")
        
        return results

async def main(config_path: str = "config/model_config.yaml", config_dict: Optional[Dict[str, Any]] = None):
    """Main execution function.
    
    Args:
        config_path: Path to configuration file (used if config_dict is None)
        config_dict: Configuration dictionary (overrides config_path if provided)
    """
    generator = GraniteTestCaseGenerator(config_path=config_path, config_dict=config_dict)
    
    try:
        # Initialize system
        await generator.initialize_system()
        
        # Set up data pipeline
        await generator.setup_data_pipeline()
        
        # Fine-tune model (optional, if training data exists and enabled)
        if generator.config.get('fine_tuning', {}).get('enabled', False):
            await generator.fine_tune_model()
        
        # Register teams
        generator.register_teams()
        
        # Generate test cases
        results = await generator.generate_test_cases()
        
        logger.info("Test case generation completed successfully!")
        print("\nTest case generation completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
