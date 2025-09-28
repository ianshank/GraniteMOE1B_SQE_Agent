from typing import Dict, List, Any, TYPE_CHECKING, Optional, Union
import requests
import json
from abc import ABC, abstractmethod
import logging
import re
import os
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
    
    @abstractmethod
    def push_test_cases(self, test_cases: List['TestCase']) -> bool:
        """Pushes generated test cases to the connected system."""
        pass

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
                    'team': self.project_key,
                    'source': {
                        'system': 'jira',
                        'source_id': issue['key'],
                        'url': f"{self.base_url}/browse/{issue['key']}",
                    },
                }
                requirements.append(req_data)
            logger.info(f"Successfully fetched {len(requirements)} requirements from Jira.")
            return requirements
        except requests.exceptions.RequestException as e:
            # Provide actionable guidance for common auth failures
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 401:
                detail = "Unauthorized (401): API token invalid or missing"
            elif status == 403:
                detail = "Forbidden (403)"
            else:
                detail = str(e)
            logger.error(f"Failed to fetch from Jira: {detail}")
            raise Exception(f"Failed to fetch from Jira: {detail}")
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

    def _safe_headers(self) -> Dict[str, str]:
        """Return headers safe for logging without exposing credentials."""
        redacted = {}
        for k, v in self.headers.items():
            if k.lower() == 'authorization':
                redacted[k] = 'token ******'
            else:
                redacted[k] = v
        return redacted
    
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
                    'team': f"{self.repo_owner}/{self.repo_name}",
                    'source': {
                        'system': 'github',
                        'source_id': str(issue['number']),
                        'url': f"https://github.com/{self.repo_owner}/{self.repo_name}/issues/{issue['number']}",
                    },
                }
                requirements.append(req_data)
            logger.info(f"Successfully fetched {len(requirements)} requirements from GitHub.")
            return requirements
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status == 401:
                detail = "Unauthorized (401): token invalid or missing"
            elif status == 403:
                detail = "Forbidden (403)"
            else:
                detail = str(e)
            logger.error(
                f"Failed to fetch from GitHub Issues for repo {self.repo_owner}/{self.repo_name}: {detail}"
            )
            # Avoid leaking tokens
            logger.debug(f"Request URL: {issues_url}, Headers: {self._safe_headers()}, Params: {params}")
            if 'response' in dir(e) and getattr(e.response, 'text', None):
                logger.debug(f"Response body: {e.response.text[:500]}")
            raise Exception(f"Failed to fetch from GitHub repo {self.repo_owner}/{self.repo_name}: {detail}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from GitHub repo {self.repo_owner}/{self.repo_name}: {e}")
            raise Exception(f"Invalid JSON response from GitHub repo {self.repo_owner}/{self.repo_name}: {e}")
    
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


class LocalFileSystemConnector(TeamConnector):
    """Connector for teams using local files for requirements and test cases.
    
    This connector reads requirements from local files in various formats (markdown, text, JSON)
    and writes generated test cases to local files. It supports both individual files and
    collections in JSON format.
    
    Attributes:
        input_directory: Path to directory containing requirement files
        output_directory: Path to directory where test cases will be written
        team_name: Name of the team associated with this connector
        file_types: List of file extensions to process (default: ['.md', '.txt', '.json'])
    """
    
    def __init__(self, 
                 input_directory: Optional[str] = None,
                 team_name: str = "default",
                 output_directory: Optional[str] = None,
                 file_types: Optional[List[str]] = None,
                 directory: Optional[str] = None):
        """Initialize the LocalFileSystemConnector.
        
        Args:
            input_directory: Path to directory containing requirement files
            team_name: Name of the team associated with this connector
            output_directory: Path to directory where test cases will be written (defaults to 'output/{team_name}')
            file_types: List of file extensions to process (default: ['.md', '.txt', '.json'])
        """
        # Accept either input_directory or directory (alias used by some callers/tests)
        base_dir = input_directory or directory
        if not base_dir:
            raise ValueError("input_directory or directory must be provided")
        self.input_directory = Path(base_dir)
        self.team_name = team_name
        self.output_directory = Path(output_directory) if output_directory else Path(f"output/{team_name}")
        self.file_types = file_types or [".md", ".txt", ".json"]
        logger.info(f"LocalFileSystemConnector initialized for team {team_name} at {input_directory}")

    def _normalize_description(self, description: Any) -> str:
        """Normalize a description field to a string without double-encoding.

        - If description is a dict, serialize compactly using json.dumps (consistent with
          existing behavior and tests).
        - If it's already a string, return as-is.
        - For other primitive types, coerce to string for robustness.
        """
        if isinstance(description, dict):
            try:
                return json.dumps(description)
            except Exception:
                return str(description)
        return description if isinstance(description, str) else str(description)
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extract a title from the content of a file.
        
        Looks for:
        1. Markdown headers (# Title)
        2. YAML frontmatter (title: Title)
        3. First non-empty line
        
        Args:
            content: The content of the file
            
        Returns:
            The extracted title or None if no title could be found
        """
        # Check for markdown headers
        header_match = re.search(r'^\s*#\s+(.+)$', content, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # Check for YAML frontmatter title
        yaml_match = re.search(r'^---[\s\S]*?title:\s*([^\n]+)[\s\S]*?---', content)
        if yaml_match:
            return yaml_match.group(1).strip()
        
        # Use first non-empty line as title
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                # Limit title length
                return line[:100] if len(line) > 100 else line
        
        return None
    
    def _extract_priority(self, content: str) -> Optional[str]:
        """Extract priority from the content of a file.
        
        Looks for:
        1. Priority: X
        2. [PX] or (PX) tags
        3. Keywords like 'high priority', 'critical', etc.
        
        Args:
            content: The content of the file
            
        Returns:
            The extracted priority ('high', 'medium', 'low') or None if no priority could be found
        """
        # Check for explicit priority markers
        priority_match = re.search(r'priority:\s*(high|medium|low)', content, re.IGNORECASE)
        if priority_match:
            return priority_match.group(1).lower()
        
        # Check for priority tags
        tag_match = re.search(r'\[(P[123])\]|\((P[123])\)', content)
        if tag_match:
            tag = tag_match.group(1) or tag_match.group(2)
            if tag == 'P1':
                return 'high'
            elif tag == 'P2':
                return 'medium'
            elif tag == 'P3':
                return 'low'
        
        # Check for priority keywords
        if re.search(r'\b(critical|urgent|high\s+priority)\b', content, re.IGNORECASE):
            return 'high'
        elif re.search(r'\b(medium\s+priority|normal)\b', content, re.IGNORECASE):
            return 'medium'
        elif re.search(r'\b(low\s+priority|minor)\b', content, re.IGNORECASE):
            return 'low'
        
        # Default priority
        return 'medium'
    
    def fetch_requirements(self) -> List[Dict[str, Any]]:
        """Fetch requirements from local files.
        
        Scans the input directory for files with the specified extensions and parses them
        into requirement dictionaries compatible with the system.
        
        Returns:
            List of requirement dictionaries
        
        Raises:
            FileNotFoundError: If the input directory does not exist
        """
        if not self.input_directory.exists():
            logger.warning(f"Input directory does not exist: {self.input_directory}")
            return []
            
        requirements = []
        file_count = 0
        skipped_count = 0
        
        # Process individual files with specified extensions
        for file_type in self.file_types:
            for file_path in self.input_directory.glob(f"**/*{file_type}"):
                file_count += 1
                try:
                    # Skip directories and hidden files
                    if file_path.is_dir() or file_path.name.startswith('.'):
                        continue
                        
                    # Read file content
                    with open(str(file_path), 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Special handling for JSON files that might contain collections
                    if file_path.suffix.lower() == '.json':
                        # Only auto-process as a collection when the top-level is an array.
                        # Nested collections are intentionally ignored during directory scans
                        # to avoid over-ingesting; callers may use _process_json_file directly
                        # when they explicitly want nested extraction.
                        stripped = content.lstrip()
                        if stripped.startswith('['):
                            json_reqs = self._process_json_file(file_path, content)
                            if json_reqs:
                                requirements.extend(json_reqs)
                                continue
                        elif stripped.startswith('{'):
                            try:
                                data = json.loads(content)
                                # Skip nested collections by default during directory scans
                                if isinstance(data, dict) and any(
                                    isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
                                    for v in data.values()
                                ):
                                    logger.debug(
                                        f"Skipping nested JSON collection by default: {file_path}"
                                    )
                                    continue
                            except Exception:
                                # If JSON is invalid or unexpected, fall back to single-file handling
                                pass
                    
                    # Process as individual file
                    file_id = file_path.name
                    title = self._extract_title(content) or file_id
                    priority = self._extract_priority(content) or 'medium'
                    
                    req_data = {
                        'id': file_id,
                        'summary': title,
                        'description': content,
                        'type': 'File',
                        'priority': priority,
                        'team': self.team_name,
                        'source': {
                            'system': 'file',
                            'source_id': str(file_path),
                            'url': None,
                            'path': str(file_path)
                        }
                    }
                    requirements.append(req_data)
                    logger.debug(f"Loaded requirement from file: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    skipped_count += 1
        
        logger.info(f"Fetched {len(requirements)} requirements from {file_count} files in {self.input_directory} "
                   f"(skipped {skipped_count} files with errors)")
        return requirements
    
    def _process_json_file(self, file_path: Path, content: str) -> Optional[List[Dict[str, Any]]]:
        """Process a JSON file that might contain multiple requirements.
        
        Args:
            file_path: Path to the JSON file
            content: Content of the JSON file
            
        Returns:
            List of requirement dictionaries or None if the file doesn't contain a collection
        """
        try:
            data = json.loads(content)
            requirements = []
            
            # Handle array of requirements
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                        
                    # Check if this looks like a requirement
                    if 'description' in item or 'summary' in item or 'title' in item:
                        req_id = item.get('id', f"{file_path.stem}_{i+1}")
                        summary = item.get('summary', item.get('title', f"Requirement {i+1}"))
                        description = self._normalize_description(item.get('description', ''))
                            
                        req_data = {
                            'id': str(req_id),
                            'summary': summary,
                            'description': description,
                            'type': item.get('type', 'Requirement'),
                            'priority': item.get('priority', 'medium').lower(),
                            'team': self.team_name,
                            'source': {
                                'system': 'file',
                                'source_id': f"{file_path.stem}_{req_id}",
                                'url': None,
                                'path': str(file_path)
                            }
                        }
                        requirements.append(req_data)
                        
            # Handle object with nested arrays or a single requirement object
            elif isinstance(data, dict):
                # Look for arrays in the object that might contain requirements
                found_nested = False
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        for i, item in enumerate(value):
                            if 'description' in item or 'summary' in item or 'title' in item:
                                req_id = item.get('id', f"{key}_{i+1}")
                                summary = item.get('summary', item.get('title', f"{key} {i+1}"))
                                
                                req_data = {
                                    'id': str(req_id),
                                    'summary': summary,
                                    'description': item.get('description', ''),
                                    'type': item.get('type', 'Requirement'),
                                    'priority': item.get('priority', 'medium').lower(),
                                    'team': self.team_name,
                                    'source': {
                                        'system': 'file',
                                        'source_id': f"{file_path.stem}_{req_id}",
                                        'url': None,
                                        'path': str(file_path)
                                    }
                                }
                                requirements.append(req_data)
                                found_nested = True
                # If not nested arrays, treat the dict itself as a single requirement when fields present
                if not found_nested and any(k in data for k in ('summary', 'title', 'description')):
                    req_id = data.get('id', file_path.stem)
                    summary = data.get('summary', data.get('title', file_path.stem))
                    description = self._normalize_description(data.get('description', ''))
                    req_data = {
                        'id': str(req_id),
                        'summary': summary,
                        'description': description,
                        'type': data.get('type', 'Requirement'),
                        'priority': data.get('priority', 'medium').lower(),
                        'team': self.team_name,
                        'source': {
                            'system': 'file',
                            'source_id': f"{file_path.stem}_{req_id}",
                            'url': None,
                            'path': str(file_path)
                        }
                    }
                    requirements.append(req_data)
            
            if requirements:
                logger.debug(f"Extracted {len(requirements)} requirements from JSON file: {file_path}")
                return requirements
            return None
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON file: {file_path}")
            return None
    
    def push_test_cases(self, test_cases: List['TestCase']) -> bool:
        """Write generated test cases to local files.
        
        Args:
            test_cases: List of TestCase objects to write
            
        Returns:
            True if all test cases were written successfully, False otherwise
        """
        if not test_cases:
            logger.info(f"No test cases to write for team: {self.team_name}")
            return True
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Write test cases to file
        output_file = self.output_directory / f"{self.team_name}_test_cases.json"
        success = False
        
        try:
            # Convert test cases to dictionaries (Pydantic v2+ compatible)
            test_cases_dict = [
                (tc.model_dump() if hasattr(tc, 'model_dump') else tc.dict())
                for tc in test_cases
            ]
            
            # Write to file
            with open(str(output_file), 'w', encoding='utf-8') as f:
                json.dump(test_cases_dict, f, indent=2, default=str)
                
            logger.info(f"Successfully wrote {len(test_cases)} test cases to {output_file}")
            success = True
            
        except Exception as e:
            logger.error(f"Failed to write test cases to {output_file}: {e}")
            success = False
            
        return success
