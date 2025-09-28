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
from src.integration.team_connectors import JiraConnector, GitHubConnector
from src.data.data_processors import TestCaseDataProcessor

class GraniteTestCaseGenerator:
    """Main class orchestrating the entire test case generation system"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.components = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
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
        
        # Process user stories if available
        if Path("data/user_stories.json").exists():
            user_stories = processor.process_user_stories("data/user_stories.json")
            print(f"Processed {len(user_stories)} user stories")
        
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
        
        # Fine-tune model (optional, if training data exists)
        await generator.fine_tune_model()
        
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
