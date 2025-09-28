import asyncio
import json
from pathlib import Path
from typing import Dict, Any
from src.models.granite_moe import GraniteMoETrainer
from src.data.rag_retriever import RAGRetriever
from src.data.cag_cache import CAGCache
from src.utils.kv_cache import KVCache
from src.utils.chunking import IntelligentChunker
from src.agents.test_generation_agent import TestGenerationAgent
from src.integration.workflow_orchestrator import WorkflowOrchestrator, TeamConfiguration
from src.integration.team_connectors import JiraConnector, GitHubConnector, LocalFileSystemConnector
from src.data.data_processors import TestCaseDataProcessor

class GraniteTestCaseGenerator:
    """Main class orchestrating the entire test case generation system"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.components = {}
        # Attempt to load integration config if teams are not provided in the
        # initial config. This keeps defaults flexible and avoids hard failures
        # when a separate integration config file is in use.
        self._maybe_merge_integration_config()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _maybe_merge_integration_config(self) -> None:
        """Merge integration config into loaded config when present.

        Behavior:
        - If `teams` already present in the current config, do nothing.
        - Else, try to load path from env var `INTEGRATION_CONFIG_PATH`; if not
          set, try default `config/integration_config.yaml` if it exists.
        - Merge `teams` into `self.config`.
        """
        if self.config.get("teams"):
            return
        import os, yaml
        candidate = os.environ.get("INTEGRATION_CONFIG_PATH")
        default_path = Path("config/integration_config.yaml")
        try_path = Path(candidate) if candidate else default_path
        if try_path.exists():
            try:
                with open(try_path, "r") as f:
                    integ = yaml.safe_load(f) or {}
                if integ.get("teams"):
                    self.config["teams"] = integ["teams"]
            except Exception:
                # Log soft error; do not crash initialization
                print(f"Warning: failed to load integration config from {try_path}")
    
    async def initialize_system(self):
        """Initialize all system components"""
        print("Initializing Granite Test Case Generation System...")
        
        # Initialize core components
        self.components['chunker'] = IntelligentChunker()
        self.components['kv_cache'] = KVCache(cache_dir="./cache")
        self.components['cag_cache'] = CAGCache(self.components['kv_cache'])
        self.components['rag_retriever'] = RAGRetriever()
        
        # Initialize Granite MoE trainer
        model_name = self.config.get('model_name', 'ibm-granite/granite-3.0-1b-a400m-instruct')
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
        """Set up data processing and indexing pipeline"""
        print("Setting up data pipeline...")
        
        processor = TestCaseDataProcessor(self.components['chunker'])
        
        # Process requirements documents
        requirements_docs = []
        if Path("data/requirements").exists():
            requirements_docs = processor.process_requirements_files("data/requirements")
            print(f"Processed {len(requirements_docs)} requirement documents")
            
            # Index in RAG system
            from src.utils.chunking import DocumentChunk
            chunks = [DocumentChunk(content=content, metadata=metadata, 
                                   chunk_id=f"req_{i}", source_type='requirements',
                                   team_context=metadata.get('team', 'default'))
                     for i, (content, metadata) in enumerate(requirements_docs)]
            
            self.components['rag_retriever'].index_documents(chunks)
            # Store raw texts for possible fallback generation
            self.components['requirements_texts'] = [content for content, _ in requirements_docs]
        
        # Process user stories if available
        if Path("data/user_stories.json").exists():
            try:
                user_stories = processor.process_user_stories("data/user_stories.json")
                print(f"Processed {len(user_stories)} user stories")
                # Index user stories in RAG as well
                from src.utils.chunking import DocumentChunk
                user_story_chunks = [DocumentChunk(content=content, metadata=metadata,
                                                   chunk_id=f"us_{i}", source_type='user_story',
                                                   team_context=metadata.get('team', 'default'))
                                      for i, (content, metadata) in enumerate(user_stories)]
                self.components['rag_retriever'].index_documents(user_story_chunks)
                # Store raw texts for possible fallback generation
                self.components['user_stories_texts'] = [content for content, _ in user_stories]
            except Exception as e:
                print(f"Error processing user stories: {e}")
        
        # Preload CAG cache with common patterns
        team_contexts = list(set([doc[1].get('team', 'default') 
                                for doc in requirements_docs])) if requirements_docs else ['default']
        self.components['cag_cache'].preload_common_patterns(team_contexts)
        
        print("Data pipeline setup complete!")
    
    async def fine_tune_model(self):
        """Fine-tune Granite MoE model if training data is available"""
        if not Path("data/training").exists():
            print("No training data found. Skipping fine-tuning.")
            return
        
        print("Starting model fine-tuning...")
        
        # Load training data
        processor = TestCaseDataProcessor(self.components['chunker'])
        
        # You would load your existing test cases here
        # For now, we'll create synthetic examples
        sample_requirements = [
            "User should be able to login with valid credentials",
            "System should validate API input parameters",
            "Application should handle database connection errors gracefully"
        ]
        
        synthetic_test_cases = processor.create_synthetic_test_cases(
            sample_requirements, "default"
        )
        
        # Prepare training dataset
        training_dataset = self.components['granite_trainer'].prepare_training_data(
            synthetic_test_cases, sample_requirements
        )
        
        # Fine-tune model
        trainer = self.components['granite_trainer'].fine_tune(
            training_dataset,
            output_dir="./models/fine_tuned_granite",
            num_epochs=2,  # Reduced for Mac Mini
            batch_size=2   # Smaller batch size for memory constraints
        )
        
        print("Model fine-tuning complete!")
        
        return trainer
    
    def register_teams(self):
        """Register team connectors based on configuration"""
        teams_config = self.config.get('teams', [])
        
        for team_config in teams_config:
            team_name = team_config['name']
            connector_type = team_config['connector']['type']
            
            if connector_type == 'jira':
                connector = JiraConnector(
                    base_url=team_config['connector']['base_url'],
                    username=team_config['connector']['username'],
                    api_token=team_config['connector']['api_token'],
                    project_key=team_config['connector']['project_key']
                )
            elif connector_type == 'github':
                connector = GitHubConnector(
                    repo_owner=team_config['connector']['repo_owner'],
                    repo_name=team_config['connector']['repo_name'],
                    token=team_config['connector']['token']
                )
            elif connector_type == 'local':
                connector = LocalFileSystemConnector(
                    directory=team_config['connector']['path'],
                    team_name=team_name
                )
            else:
                print(f"Unknown connector type: {connector_type}")
                continue
            
            team_configuration = TeamConfiguration(
                team_name=team_name,
                connector=connector,
                rag_enabled=team_config.get('rag_enabled', True),
                cag_enabled=team_config.get('cag_enabled', True),
                auto_push=team_config.get('auto_push', False)
            )
            
            self.components['orchestrator'].register_team(team_configuration)
            print(f"Registered team: {team_name}")
    
    async def generate_test_cases(self):
        """Main execution: generate test cases for all registered teams"""
        print("Starting test case generation for all teams...")
        
        results = await self.components['orchestrator'].process_all_teams()
        # Fallback: if no teams configured or no results, generate from local documents
        total_before_fallback = sum(len(v) for v in results.values()) if results else 0
        if (not self.components['orchestrator'].team_configs) or total_before_fallback == 0:
            fallback_inputs = self.components.get('requirements_texts') or self.components.get('user_stories_texts') or []
            if fallback_inputs:
                print(f"No remote teams or results; generating test cases from local documents ({len(fallback_inputs)})...")
                fallback_cases = await self.components['test_agent'].generate_test_cases_for_team(
                    'default', fallback_inputs
                )
                results = results or {}
                results['default'] = fallback_cases
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        total_generated = 0
        for team_name, test_cases in results.items():
            total_generated += len(test_cases)
            
            # Save test cases as JSON
            output_file = output_dir / f"{team_name}_test_cases.json"
            with open(output_file, 'w') as f:
                test_cases_dict = [tc.dict() for tc in test_cases]
                json.dump(test_cases_dict, f, indent=2, default=str)
            
            print(f"Generated {len(test_cases)} test cases for team: {team_name}")
        
        # Generate quality report
        quality_report = self.components['orchestrator'].generate_quality_report()
        with open(output_dir / "quality_report.json", 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        print(f"Total test cases generated: {total_generated}")
        print("Results saved to output/ directory")
        
        return results

async def main():
    """Main execution function"""
    generator = GraniteTestCaseGenerator()
    
    try:
        # Initialize system
        await generator.initialize_system()
        
        # Set up data pipeline
        await generator.setup_data_pipeline()
        
        # Fine-tune model (optional). On Apple MPS, skip if bfloat16 unsupported
        try:
            await generator.fine_tune_model()
        except Exception as e:
            print(f"Skipping fine-tuning due to environment constraint: {e}")
        
        # Register teams
        generator.register_teams()
        
        # Generate test cases
        results = await generator.generate_test_cases()
        
        print("\nTest case generation completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
