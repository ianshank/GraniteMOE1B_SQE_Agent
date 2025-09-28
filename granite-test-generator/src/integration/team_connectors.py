from typing import Dict, List, Any, TYPE_CHECKING, Optional
import requests
import json
from abc import ABC, abstractmethod
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.models.test_case_schemas import TestCase

class TeamConnector(ABC):
    """Abstract base class for team system integrations"""
    
    @abstractmethod
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Fetches requirements from the connected system."""
        pass


class LocalFileSystemConnector(TeamConnector):
    """Connector that reads requirements from local text/markdown files.

    Files are read from a specified directory. Each file becomes a requirement with
    the first line used as the summary and the full content as the description.

    Design choices:
    - Safe-by-default: ignores non-text/markdown files.
    - Structured logging at error/decision points.
    - No-op `push_test_cases` (returns True) since local file systems typically
      don’t accept push operations for test cases.
    """

    def __init__(self, directory: str, team_name: Optional[str] = None) -> None:
        self.directory = str(directory)
        self._team_name = team_name
        logger.debug(f"LocalFileSystemConnector initialized for directory: {self.directory}")

    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Load .txt/.md files in the directory as requirements.

        Returns:
            A list of dicts with keys: id, summary, description, type, priority, team
        """
        base = Path(self.directory)
        requirements: List[Dict[str, Any]] = []

        if not base.exists():
            logger.error(f"Local requirements directory does not exist: {base}")
            return requirements
        if not base.is_dir():
            logger.error(f"Local requirements path is not a directory: {base}")
            return requirements

        team_value = self._team_name or base.name

        for path in sorted(base.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read requirement file {path}: {e}")
                continue

            first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), path.name)
            requirements.append({
                "id": path.name,
                "summary": first_line,
                "description": content,
                "type": "File",
                "priority": "medium",
                "team": team_value,
            })

        logger.info(f"Fetched {len(requirements)} local requirements from {base}")
        return requirements

    def push_test_cases(self, test_cases: List['TestCase']) -> bool:
        """No-op push for local connector. Always returns True.

        Rationale: For a local-only workflow, pushing is typically handled by a
        different component (e.g., writing JSON files). This method exists to
        satisfy the `TeamConnector` contract.
        """
        logger.info("LocalFileSystemConnector.push_test_cases called (no-op).")
        return True

class JiraConnector(TeamConnector):
    """Connector for teams using Jira for requirements management"""
    
    def __init__(self, base_url: str, username: str, api_token: str, project_key: str):
        self.base_url = base_url
        self.auth = (username, api_token)
        self.project_key = project_key
        logger.debug(f"JiraConnector initialized for project: {project_key}")
    
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Fetch user stories and requirements from Jira"""
        jql = f'project = "{self.project_key}" AND type in ("Story", "Requirement")'
        search_url = f"{self.base_url}/rest/api/3/search"
        params = {'jql': jql, 'maxResults': 1000}
        logger.info(f"Fetching requirements from Jira: {search_url} with JQL: {jql}")
        
        try:
            response = requests.get(search_url, params=params, auth=self.auth, timeout=30)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            issues = response.json()['issues']
            requirements = []
            
            for issue in issues:
                req_data = {
                    'id': issue['key'],
                    'summary': issue['fields']['summary'],
                    'description': issue['fields'].get('description', ''),
                    'type': issue['fields']['issuetype']['name'],
                    'priority': issue['fields']['priority']['name'],
                    'team': self.project_key
                }
                requirements.append(req_data)
            logger.info(f"Successfully fetched {len(requirements)} requirements from Jira.")
            return requirements
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 401:
                msg = (
                    "Unauthorized (401) when fetching from Jira. Verify username/API token and permissions."
                )
                logger.error(msg)
                raise Exception(msg) from e
            logger.error(f"Failed to fetch from Jira: {e}")
            raise Exception(f"Failed to fetch from Jira: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Jira: {e}")
            raise Exception(f"Invalid JSON response from Jira: {e}")
    
    def push_test_cases(self, test_cases: List['TestCase']) -> bool:
        """Create test cases as Jira Test issues"""
        success_count = 0
        issue_create_url = f"{self.base_url}/rest/api/3/issue"
        logger.info(f"Attempting to push {len(test_cases)} test cases to Jira.")
        
        for test_case in test_cases:
            issue_data = {
                "fields": {
                    "project": {"key": self.project_key},
                    "summary": test_case.summary,
                    "description": self._format_test_case_description(test_case),
                    "issuetype": {"name": "Test"}  # Assuming 'Test' issue type exists
                }
            }
            try:
                response = requests.post(issue_create_url, json=issue_data, auth=self.auth, timeout=30)
                response.raise_for_status()
                success_count += 1
                logger.debug(f"Successfully created Jira issue for test case: {test_case.summary}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to create test case '{test_case.summary}' in Jira: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while pushing test case '{test_case.summary}': {e}")
        
        if success_count == len(test_cases):
            logger.info(f"Successfully pushed all {success_count} test cases to Jira.")
        else:
            logger.warning(f"Pushed {success_count} out of {len(test_cases)} test cases to Jira.")
        return success_count == len(test_cases)
    
    def _format_test_case_description(self, test_case: 'TestCase') -> str:
        """Format test case for Jira description using Jira's markdown-like syntax"""
        description = f"*Priority:* {test_case.priority.value}\n"
        description += f"*Type:* {test_case.test_type.value}\n\n"
        
        if test_case.preconditions:
            description += "*Preconditions:*\n"
            for precond in test_case.preconditions:
                description += f"* {precond}\n"
            description += "\n"
        
        description += "*Test Steps:*\n"
        for step in test_case.steps:
            description += f"{step.step_number}. {step.action}\n"
            description += f"   _Expected:_ {step.expected_result}\n"
        
        description += f"\n*Overall Expected Result:*\n{test_case.expected_results}"
        
        if test_case.requirements_traced:
            description += "\n*Traced Requirements:*\n"
            for req_id in test_case.requirements_traced:
                description += f"* {req_id}\n"
        
        return description

class GitHubConnector(TeamConnector):
    """Connector for teams using GitHub Issues"""
    
    def __init__(self, repo_owner: str, repo_name: str, token: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        # Use GitHub REST API base for the target repository
        self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        logger.debug(f"GitHubConnector initialized for repo: {repo_owner}/{repo_name}")
    
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Fetch requirements from GitHub Issues"""
        issues_url = f"{self.base_url}/issues"
        params = {'labels': 'requirement,user-story', 'state': 'open'}
        logger.info(f"Fetching requirements from GitHub Issues: {issues_url}")
        
        try:
            response = requests.get(issues_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            issues = response.json()
            requirements = []
            
            for issue in issues:
                req_data = {
                    'id': str(issue['number']),
                    'summary': issue['title'],
                    'description': issue.get('body', ''),
                    'type': 'Issue',
                    'priority': 'medium',  # Default, could parse from labels
                    'team': f"{self.repo_owner}/{self.repo_name}"
                }
                requirements.append(req_data)
            logger.info(f"Successfully fetched {len(requirements)} requirements from GitHub.")
            return requirements
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 401:
                msg = (
                    "Unauthorized (401) when fetching from GitHub. Ensure the token is valid and has appropriate scopes (e.g., 'repo')."
                )
                logger.error(msg)
                raise Exception(msg) from e
            logger.error(f"Failed to fetch from GitHub Issues: {e}")
            raise Exception(f"Failed to fetch from GitHub: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from GitHub: {e}")
            raise Exception(f"Invalid JSON response from GitHub: {e}")
    
    def push_test_cases(self, test_cases: List['TestCase']) -> bool:
        """Create test cases as GitHub Issues"""
        success_count = 0
        issues_create_url = f"{self.base_url}/issues"
        logger.info(f"Attempting to push {len(test_cases)} test cases to GitHub Issues.")
        
        for test_case in test_cases:
            issue_data = {
                "title": f"Test Case: {test_case.summary}",
                "body": self._format_test_case_markdown(test_case),
                "labels": ["test-case", test_case.test_type.value, test_case.priority.value]
            }
            try:
                response = requests.post(issues_create_url, json=issue_data, headers=self.headers, timeout=30)
                response.raise_for_status()
                success_count += 1
                logger.debug(f"Successfully created GitHub issue for test case: {test_case.summary}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to create test case '{test_case.summary}' in GitHub: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while pushing test case '{test_case.summary}': {e}")
        
        if success_count == len(test_cases):
            logger.info(f"Successfully pushed all {success_count} test cases to GitHub Issues.")
        else:
            logger.warning(f"Pushed {success_count} out of {len(test_cases)} test cases to GitHub Issues.")
        return success_count == len(test_cases)
    
    def _format_test_case_markdown(self, test_case: 'TestCase') -> str:
        """Format test case as Markdown for GitHub"""
        markdown = f"## Test Case: {test_case.summary}\n\n"
        markdown += f"**Priority:** {test_case.priority.value}\n"
        markdown += f"**Type:** {test_case.test_type.value}\n\n"
        
        if test_case.preconditions:
            markdown += "### Preconditions\n"
            for precond in test_case.preconditions:
                markdown += f"- {precond}\n"
            markdown += "\n"
        
        if test_case.input_data:
            markdown += "### Input Data\n"
            markdown += f"```json\n{json.dumps(test_case.input_data, indent=2)}\n```\n\n"
        else:
            markdown += "### Input Data\n_None_\n\n"
        
        markdown += "### Test Steps\n"
        for step in test_case.steps:
            markdown += f"{step.step_number}. {step.action}\n"
            markdown += f"   **Expected:** {step.expected_result}\n\n"
        
        markdown += f"### Expected Results\n{test_case.expected_results}\n"
        
        if test_case.requirements_traced:
            markdown += "\n### Traced Requirements\n"
            for req in test_case.requirements_traced:
                markdown += f"- {req}\n"
        
        return markdown
