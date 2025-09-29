import asyncio
import time
from typing import Dict, List, Any, TYPE_CHECKING, Optional, Union, Type
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Default feature flags for auto-generated teams. These constants avoid scattering
# literal booleans across the codebase and allow central tuning or config overrides.
DEFAULT_RAG_ENABLED: bool = True
DEFAULT_CAG_ENABLED: bool = True
DEFAULT_AUTO_PUSH: bool = False

if TYPE_CHECKING:
    from src.integration.team_connectors import TeamConnector, JiraConnector, GitHubConnector, LocalFileSystemConnector
    from src.agents.generation_agent import TestGenerationAgent
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
    
    def __init__(self, test_generation_agent: 'TestGenerationAgent', *, local_only: bool = False):
        self.agent = test_generation_agent
        self.team_configs: Dict[str, TeamConfiguration] = {}
        self.results_cache: Dict[str, List['TestCase']] = {}
        self.local_only = local_only
        logger.info("WorkflowOrchestrator initialized.")
        if self.local_only:
            logger.info("WorkflowOrchestrator running in local-only connector mode.")
        
    @staticmethod
    def create_connector(connector_config: Dict[str, Any]) -> 'TeamConnector':
        """Create a connector instance based on configuration.
        
        Factory method to create the appropriate connector based on the connector type.
        Validates that all required fields are present for the specified connector type.
        
        Args:
            connector_config: Dictionary containing connector configuration
            
        Returns:
            An instance of the appropriate TeamConnector subclass
            
        Raises:
            ValueError: If connector_config is invalid or missing required fields
            ImportError: If the required connector class cannot be imported
            Exception: For other errors during connector creation
        """
        # Import connector classes here to avoid circular imports
        from src.integration.team_connectors import JiraConnector, GitHubConnector, LocalFileSystemConnector
        
        # Validate basic configuration
        if not connector_config:
            raise ValueError("Connector configuration is empty")
            
        if 'type' not in connector_config:
            raise ValueError("Connector configuration missing required field: type")
            
        connector_type = connector_config['type'].lower()
        
        # Create connector based on type
        if connector_type == 'jira':
            # Validate Jira connector config
            required_fields = ['base_url', 'username', 'api_token', 'project_key']
            for field in required_fields:
                if field not in connector_config:
                    raise ValueError(f"Jira connector missing required field: {field}")
                    
            return JiraConnector(
                base_url=connector_config['base_url'],
                username=connector_config['username'],
                api_token=connector_config['api_token'],
                project_key=connector_config['project_key']
            )
            
        elif connector_type == 'github':
            # Validate GitHub connector config
            required_fields = ['repo_owner', 'repo_name', 'token']
            for field in required_fields:
                if field not in connector_config:
                    raise ValueError(f"GitHub connector missing required field: {field}")
                    
            return GitHubConnector(
                repo_owner=connector_config['repo_owner'],
                repo_name=connector_config['repo_name'],
                token=connector_config['token']
            )
            
        elif connector_type == 'local':
            # Validate LocalFileSystem connector config
            if 'input_directory' not in connector_config:
                raise ValueError("LocalFileSystem connector missing required field: input_directory")
                
            # Get parameters with defaults
            input_directory = connector_config['input_directory']
            team_name = connector_config.get('team_name', 'default')
            output_directory = connector_config.get('output_directory', None)
            file_types = connector_config.get('file_types', None)
            
            return LocalFileSystemConnector(
                input_directory=input_directory,
                team_name=team_name,
                output_directory=output_directory,
                file_types=file_types
            )
            
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
    
    def create_team_configuration(self, team_config: Dict[str, Any]) -> TeamConfiguration:
        """Create a TeamConfiguration instance from a configuration dictionary.
        
        Args:
            team_config: Dictionary containing team configuration
            
        Returns:
            TeamConfiguration instance
            
        Raises:
            ValueError: If team_config is invalid or missing required fields
            Exception: For other errors during team configuration creation
        """
        # Validate team config
        if 'name' not in team_config:
            raise ValueError("Team configuration missing required field: name")
            
        if 'connector' not in team_config:
            raise ValueError(f"Team {team_config['name']} missing connector configuration")
            
        # Create connector
        connector_config = team_config['connector']
        # Add team name to connector config for LocalFileSystemConnector
        if connector_config.get('type', '').lower() == 'local' and 'team_name' not in connector_config:
            connector_config['team_name'] = team_config['name']
            
        if self.local_only:
            from src.integration.team_connectors import LocalFileSystemConnector  # Lazy import to avoid cycles

            input_directory = (
                connector_config.get('input_directory')
                or connector_config.get('path')
                or f"data/requirements/{team_config['name']}"
            )
            output_directory = connector_config.get('output_directory') or connector_config.get('output') or 'output'
            file_types = connector_config.get('file_types')

            logger.debug(
                "Local-only mode active; overriding connector for team '%s' with LocalFileSystemConnector (input=%s, output=%s)",
                team_config['name'],
                input_directory,
                output_directory,
            )

            connector = LocalFileSystemConnector(
                input_directory=input_directory,
                team_name=team_config['name'],
                output_directory=output_directory,
                file_types=file_types,
            )
        else:
            connector = self.create_connector(connector_config)
        
        # Create team configuration
        return TeamConfiguration(
            team_name=team_config['name'],
            connector=connector,
            rag_enabled=team_config.get('rag_enabled', True),
            cag_enabled=team_config.get('cag_enabled', True),
            auto_push=team_config.get('auto_push', False)
        )
    
    def register_team(self, config: TeamConfiguration):
        """Register a team for automated test case generation"""
        self.team_configs[config.team_name] = config
        logger.info(f"Team '{config.team_name}' registered with connector type: {type(config.connector).__name__}, "
                   f"RAG: {config.rag_enabled}, CAG: {config.cag_enabled}, auto_push: {config.auto_push}")
    
    async def process_all_teams(self, generate_multiple_suites: bool = False) -> Dict[str, List['TestCase']]:
        """Process test case generation for all registered teams
        
        Args:
            generate_multiple_suites: If True, generate functional, regression, and E2E test suites
            
        Returns:
            Dictionary mapping team names to lists of generated test cases
        """
        results = {}
        tasks = []
        logger.info(f"Starting test case generation for {len(self.team_configs)} registered teams.")
        
        if not self.team_configs:
            logger.warning("No teams registered for processing.")
            
            # Check if we have any local requirements
            from pathlib import Path
            default_requirements_dir = Path("data/requirements")
            if default_requirements_dir.exists() and any(default_requirements_dir.glob("**/*")):
                logger.info("Found local requirements but no teams registered. Creating default team.")
                from src.integration.team_connectors import LocalFileSystemConnector
                
                # Create a default team with local connector
                default_team = TeamConfiguration(
                    team_name="default",
                    connector=LocalFileSystemConnector(
                        input_directory=str(default_requirements_dir),
                        output_directory="output"
                    ),
                    rag_enabled=DEFAULT_RAG_ENABLED,
                    cag_enabled=DEFAULT_CAG_ENABLED,
                    auto_push=DEFAULT_AUTO_PUSH
                )
                
                # Register the default team
                self.team_configs["default"] = default_team
                logger.info("Registered default team with local requirements.")
            else:
                # No teams and no local requirements
                return results
        
        for team_name, config in self.team_configs.items():
            logger.debug(f"Creating processing task for team: {team_name}")
            task = self._process_team(team_name, config, generate_multiple_suites)
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
    
    async def _process_team(self, team_name: str, config: TeamConfiguration, generate_multiple_suites: bool = False) -> List['TestCase']:
        """Process test case generation for a single team
        
        Args:
            team_name: Name of the team to process
            config: Team configuration object
            generate_multiple_suites: If True, generate functional, regression, and E2E test suites
            
        Returns:
            List of generated test cases
        """
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

            # Validate and convert to simple text list for processing
            requirements_text: List[str] = []
            valid_requirements: List[Dict[str, Any]] = []
            skipped = 0
            for req in raw_requirements:
                if not req.get('id') or not req.get('summary'):
                    logger.error(
                        f"Skipping requirement missing mandatory fields (id/summary): {str(req)[:120]}"
                    )
                    skipped += 1
                    continue
                req_text = f"{req['summary']}\n{req.get('description', '')}"
                requirements_text.append(req_text)
                valid_requirements.append(req)
                logger.debug(
                    f"Processing requirement {req['id']}: {req['summary'][:50]}..."
                )
            if skipped:
                logger.warning(f"Skipped {skipped} invalid requirements for team: {team_name}")

            # Generate test cases using the agent - either multiple suites or just functional
            logger.info(f"Generating test cases for team: {team_name} with {len(requirements_text)} requirements")
            
            test_cases = []
            if generate_multiple_suites:
                from src.models.test_case_schemas import TestCaseType
                
                # Generate functional test cases
                logger.info(f"Generating functional test cases for team: {team_name}")
                functional_cases = await self.agent.generate_test_cases_for_team(
                    team_name, 
                    requirements_text,
                    test_types=[TestCaseType.FUNCTIONAL]
                )
                test_cases.extend(functional_cases)
                logger.info(f"Generated {len(functional_cases)} functional test cases for team: {team_name}")
                
                # Generate regression test cases
                logger.info(f"Generating regression test cases for team: {team_name}")
                regression_cases = await self.agent.generate_test_cases_for_team(
                    team_name, 
                    requirements_text,
                    test_types=[TestCaseType.REGRESSION]
                )
                test_cases.extend(regression_cases)
                logger.info(f"Generated {len(regression_cases)} regression test cases for team: {team_name}")
                
                # Generate integration/E2E test cases
                logger.info(f"Generating integration/E2E test cases for team: {team_name}")
                integration_cases = await self.agent.generate_test_cases_for_team(
                    team_name, 
                    requirements_text,
                    test_types=[TestCaseType.INTEGRATION]
                )
                test_cases.extend(integration_cases)
                logger.info(f"Generated {len(integration_cases)} integration/E2E test cases for team: {team_name}")
            else:
                # Generate only functional test cases (default behavior)
                test_cases = await self.agent.generate_test_cases_for_team(
                    team_name, 
                    requirements_text
                )
            
            logger.info(f"Generated {len(test_cases)} test cases for team: {team_name}")

            # Add traceability and provenance
            traced_count = 0
            from src.models.test_case_schemas import TestCaseProvenance
            for i, test_case in enumerate(test_cases):
                if i < len(valid_requirements):
                    source_req = valid_requirements[i]
                    test_case.requirements_traced = [source_req['id']]
                    traced_count += 1
                    # Build provenance, prefer connector-provided source details
                    src_meta = source_req.get('source') or {}
                    system = src_meta.get('system')
                    source_id = src_meta.get('source_id') or source_req['id']
                    url = src_meta.get('url')
                    if not system:
                        # Infer from connector type when absent
                        cname = type(config.connector).__name__.lower()
                        if 'jira' in cname:
                            system = 'jira'
                        elif 'github' in cname:
                            system = 'github'
                        else:
                            system = 'unknown'
                        # Best-effort URL for GitHub from team string if present
                        if not url and system == 'github':
                            team_str = source_req.get('team', '')
                            if '/' in team_str:
                                owner, repo = team_str.split('/', 1)
                                url = f"https://github.com/{owner}/{repo}/issues/{source_id}"
                    test_case.provenance = TestCaseProvenance(
                        system=system,
                        source_id=str(source_id),
                        url=url,
                        summary=source_req.get('summary'),
                        extra={'team': source_req.get('team')}
                    )
                    logger.debug(
                        f"Provenance set for test case {test_case.id}: system={system}, source_id={source_id}, url={url}"
                    )

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
