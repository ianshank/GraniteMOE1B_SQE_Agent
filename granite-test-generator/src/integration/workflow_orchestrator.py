import asyncio
import time
from typing import Dict, List, Any, TYPE_CHECKING
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.integration.team_connectors import TeamConnector
    from src.agents.test_generation_agent import TestGenerationAgent
    from src.models.test_case_schemas import TestCase

@dataclass
class TeamConfiguration:
    team_name: str
    connector: 'TeamConnector'
    rag_enabled: bool = True
    cag_enabled: bool = True
    auto_push: bool = False

class WorkflowOrchestrator:
    """Orchestrates test case generation workflow across multiple teams"""
    
    def __init__(self, test_generation_agent: 'TestGenerationAgent'):
        self.agent = test_generation_agent
        self.team_configs: Dict[str, TeamConfiguration] = {}
        self.results_cache: Dict[str, List['TestCase']] = {}
        logger.info("WorkflowOrchestrator initialized.")
    
    def register_team(self, config: TeamConfiguration):
        """Register a team for automated test case generation"""
        self.team_configs[config.team_name] = config
        logger.info(f"Team '{config.team_name}' registered with connector type: {type(config.connector).__name__}, "
                   f"RAG: {config.rag_enabled}, CAG: {config.cag_enabled}, auto_push: {config.auto_push}")
    
    async def process_all_teams(self) -> Dict[str, List['TestCase']]:
        """Process test case generation for all registered teams"""
        results = {}
        tasks = []
        logger.info(f"Starting test case generation for {len(self.team_configs)} registered teams.")
        
        if not self.team_configs:
            logger.warning("No teams registered for processing.")
            return results
        
        for team_name, config in self.team_configs.items():
            logger.debug(f"Creating processing task for team: {team_name}")
            task = self._process_team(team_name, config)
            tasks.append(task)
        
        logger.info(f"Executing {len(tasks)} team processing tasks in parallel.")
        team_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        successful_teams = 0
        failed_teams = 0
        
        for i, (team_name, _) in enumerate(self.team_configs.items()):
            if isinstance(team_results[i], Exception):
                logger.error(f"Error processing team '{team_name}': {team_results[i]}", 
                           exc_info=team_results[i])
                results[team_name] = []
                failed_teams += 1
            else:
                results[team_name] = team_results[i]
                test_count = len(team_results[i])
                logger.info(f"Team '{team_name}' processing completed with {test_count} test cases.")
                successful_teams += 1
        
        logger.info(f"All teams processing completed. Successful: {successful_teams}, Failed: {failed_teams}")
        return results
    
    async def _process_team(self, team_name: str, config: TeamConfiguration) -> List['TestCase']:
        """Process test case generation for a single team"""
        logger.info(f"Starting test case generation for team: {team_name}")
        start_time = time.time()
        
        try:
            # Fetch requirements
            logger.debug(f"Fetching requirements for team: {team_name}")
            raw_requirements = config.connector.fetch_requirements()
            logger.info(f"Fetched {len(raw_requirements)} requirements for team: {team_name}")
            
            if not raw_requirements:
                logger.warning(f"No requirements found for team: {team_name}")
                return []
            
            # Convert to simple text list for processing
            requirements_text = []
            for req in raw_requirements:
                req_text = f"{req['summary']}\n{req.get('description', '')}"
                requirements_text.append(req_text)
                logger.debug(f"Processing requirement {req['id']}: {req['summary'][:50]}...")
            
            # Generate test cases using the agent
            logger.info(f"Generating test cases for team: {team_name} with {len(requirements_text)} requirements")
            test_cases = await self.agent.generate_test_cases_for_team(
                team_name, requirements_text
            )
            logger.info(f"Generated {len(test_cases)} test cases for team: {team_name}")
            
            # Add traceability
            traced_count = 0
            for i, test_case in enumerate(test_cases):
                if i < len(raw_requirements):
                    test_case.requirements_traced = [raw_requirements[i]['id']]
                    traced_count += 1
                    logger.debug(f"Added traceability for test case {test_case.id} to requirement {raw_requirements[i]['id']}")
            
            logger.info(f"Added traceability to {traced_count} test cases for team: {team_name}")
            
            # Auto-push if enabled
            if config.auto_push and test_cases:
                logger.info(f"Auto-pushing {len(test_cases)} test cases for team: {team_name}")
                try:
                    success = config.connector.push_test_cases(test_cases)
                    if success:
                        logger.info(f"Successfully pushed all test cases for team: {team_name}")
                    else:
                        logger.warning(f"Failed to push some test cases for team: {team_name}")
                except Exception as push_error:
                    logger.error(f"Error pushing test cases for team {team_name}: {push_error}", 
                               exc_info=True)
            elif config.auto_push and not test_cases:
                logger.debug(f"No test cases to push for team: {team_name}")
            elif not config.auto_push:
                logger.debug(f"Auto-push disabled for team: {team_name}")
            
            # Cache results
            self.results_cache[team_name] = test_cases
            logger.debug(f"Cached {len(test_cases)} results for team: {team_name}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Team '{team_name}' processing completed in {elapsed_time:.2f} seconds")
            
            return test_cases
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error processing team '{team_name}' after {elapsed_time:.2f} seconds: {str(e)}", 
                       exc_info=True)
            return []
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate quality metrics report across all teams"""
        logger.info("Generating quality report for all processed teams.")
        
        if not self.results_cache:
            logger.warning("No results cached for quality report generation.")
            return {
                'total_test_cases': 0,
                'teams_processed': 0,
                'team_metrics': {},
                'generation_timestamp': time.time(),
                'report_status': 'no_data'
            }
        
        total_test_cases = 0
        team_metrics = {}
        
        for team_name, test_cases in self.results_cache.items():
            team_test_count = len(test_cases)
            total_test_cases += team_test_count
            logger.debug(f"Processing metrics for team '{team_name}' with {team_test_count} test cases")
            
            # Calculate metrics
            priority_distribution = {}
            type_distribution = {}
            total_steps = 0
            
            for tc in test_cases:
                # Priority distribution
                priority_key = tc.priority.value
                priority_distribution[priority_key] = priority_distribution.get(priority_key, 0) + 1
                
                # Type distribution
                type_key = tc.test_type.value
                type_distribution[type_key] = type_distribution.get(type_key, 0) + 1
                
                # Step count
                total_steps += len(tc.steps)
            
            avg_steps = total_steps / team_test_count if team_test_count > 0 else 0
            
            team_metrics[team_name] = {
                'test_case_count': team_test_count,
                'priority_distribution': priority_distribution,
                'type_distribution': type_distribution,
                'average_steps_per_test': round(avg_steps, 2),
                'total_steps': total_steps
            }
            
            logger.debug(f"Team '{team_name}' metrics: {team_test_count} tests, "
                       f"avg {avg_steps:.2f} steps/test, "
                       f"priorities: {priority_distribution}, "
                       f"types: {type_distribution}")
        
        report = {
            'total_test_cases': total_test_cases,
            'teams_processed': len(self.team_configs),
            'teams_with_results': len(self.results_cache),
            'team_metrics': team_metrics,
            'generation_timestamp': time.time(),
            'report_status': 'success'
        }
        
        logger.info(f"Quality report generated: {total_test_cases} total test cases across "
                   f"{len(self.results_cache)} teams with results (out of {len(self.team_configs)} registered)")
        
        return report