"""
Unit tests for the LocalFileSystemConnector class.

These tests verify that the LocalFileSystemConnector can correctly:
1. Read requirements from local files in various formats
2. Extract titles and priorities from file content
3. Process JSON files containing collections of requirements
4. Write test cases to output files
5. Handle error conditions gracefully
"""

import json
import os
import pytest
from pathlib import Path
from typing import List, Dict, Any

from src.integration.team_connectors import LocalFileSystemConnector
from src.models.test_case_schemas import TestCase, TestStep, TestCasePriority, TestCaseType

# Test data
MARKDOWN_CONTENT = """# Test Requirement

This is a test requirement with priority: high

## Details

Some details about the requirement.
"""

TEXT_CONTENT = """Test Requirement

This is a test requirement with [P2] priority.

Details:
Some details about the requirement.
"""

JSON_CONTENT_SINGLE = """
{
    "id": "REQ-001",
    "summary": "JSON Requirement",
    "description": "This is a requirement in JSON format",
    "priority": "low"
}
"""

JSON_CONTENT_COLLECTION = """
[
    {
        "id": "REQ-001",
        "summary": "First Requirement",
        "description": "This is the first requirement"
    },
    {
        "id": "REQ-002",
        "summary": "Second Requirement",
        "description": "This is the second requirement",
        "priority": "high"
    }
]
"""

JSON_CONTENT_NESTED = """
{
    "requirements": [
        {
            "id": "REQ-001",
            "title": "Nested Requirement 1",
            "description": "This is a nested requirement"
        },
        {
            "id": "REQ-002",
            "title": "Nested Requirement 2",
            "description": "This is another nested requirement"
        }
    ]
}
"""

# Fixtures
@pytest.fixture
def temp_input_dir(tmp_path):
    """Create a temporary input directory with test files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    # Create markdown file
    md_file = input_dir / "requirement.md"
    md_file.write_text(MARKDOWN_CONTENT)
    
    # Create text file
    txt_file = input_dir / "requirement.txt"
    txt_file.write_text(TEXT_CONTENT)
    
    # Create JSON files
    json_file_single = input_dir / "requirement_single.json"
    json_file_single.write_text(JSON_CONTENT_SINGLE)
    
    json_file_collection = input_dir / "requirement_collection.json"
    json_file_collection.write_text(JSON_CONTENT_COLLECTION)
    
    json_file_nested = input_dir / "requirement_nested.json"
    json_file_nested.write_text(JSON_CONTENT_NESTED)
    
    return input_dir

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir

@pytest.fixture
def connector(temp_input_dir, temp_output_dir):
    """Create a LocalFileSystemConnector instance."""
    return LocalFileSystemConnector(
        input_directory=str(temp_input_dir),
        team_name="test_team",
        output_directory=str(temp_output_dir)
    )

@pytest.fixture
def test_cases():
    """Create a list of test cases for testing."""
    return [
        TestCase(
            id="TC-001",
            summary="Test Case 1",
            priority=TestCasePriority.HIGH,
            test_type=TestCaseType.FUNCTIONAL,
            steps=[
                TestStep(
                    step_number=1,
                    action="Do something",
                    expected_result="Something happens"
                )
            ],
            expected_results="Test passes",
            team_context="test_team"
        ),
        TestCase(
            id="TC-002",
            summary="Test Case 2",
            priority=TestCasePriority.MEDIUM,
            test_type=TestCaseType.INTEGRATION,
            steps=[
                TestStep(
                    step_number=1,
                    action="Do something else",
                    expected_result="Something else happens"
                )
            ],
            expected_results="Test passes",
            team_context="test_team"
        )
    ]

# Tests
def test_connector_initialization(temp_input_dir, temp_output_dir):
    """Test that the connector initializes correctly."""
    # Test with all parameters
    connector = LocalFileSystemConnector(
        input_directory=str(temp_input_dir),
        team_name="test_team",
        output_directory=str(temp_output_dir),
        file_types=[".md", ".txt"]
    )
    
    assert connector.input_directory == temp_input_dir
    assert connector.team_name == "test_team"
    assert connector.output_directory == temp_output_dir
    assert connector.file_types == [".md", ".txt"]
    
    # Test with default output_directory
    connector = LocalFileSystemConnector(
        input_directory=str(temp_input_dir),
        team_name="test_team"
    )
    
    assert connector.input_directory == temp_input_dir
    assert connector.team_name == "test_team"
    assert connector.output_directory == Path("output/test_team")
    assert connector.file_types == [".md", ".txt", ".json"]

def test_extract_title():
    """Test the _extract_title method."""
    connector = LocalFileSystemConnector(
        input_directory="dummy",
        team_name="test_team"
    )
    
    # Test markdown header
    title = connector._extract_title(MARKDOWN_CONTENT)
    assert title == "Test Requirement"
    
    # Test first line
    title = connector._extract_title(TEXT_CONTENT)
    assert title == "Test Requirement"
    
    # Test empty content
    title = connector._extract_title("")
    assert title is None
    
    # Test content with no title
    title = connector._extract_title("   \n  \n  ")
    assert title is None
    
    # Test YAML frontmatter
    yaml_content = """---
title: YAML Title
date: 2023-01-01
---

Content here
"""
    title = connector._extract_title(yaml_content)
    assert title == "YAML Title"

def test_extract_priority():
    """Test the _extract_priority method."""
    connector = LocalFileSystemConnector(
        input_directory="dummy",
        team_name="test_team"
    )
    
    # Test explicit priority
    priority = connector._extract_priority(MARKDOWN_CONTENT)
    assert priority == "high"
    
    # Test priority tag
    priority = connector._extract_priority(TEXT_CONTENT)
    assert priority == "medium"  # P2 = medium
    
    # Test priority keywords
    priority = connector._extract_priority("This is critical")
    assert priority == "high"
    
    priority = connector._extract_priority("This is medium priority")
    assert priority == "medium"
    
    priority = connector._extract_priority("This is low priority")
    assert priority == "low"
    
    # Test default priority
    priority = connector._extract_priority("No priority specified")
    assert priority == "medium"

@pytest.mark.regression
def test_fetch_requirements(connector):
    """Test fetching requirements from files."""
    requirements = connector.fetch_requirements()
    
    # We should have 5 requirements: 1 from markdown, 1 from text, 1 from single JSON,
    # 2 from collection JSON, and 0 from nested JSON (it's not processed by default)
    assert len(requirements) == 5
    
    # Check that requirements have the expected fields
    for req in requirements:
        assert "id" in req
        assert "summary" in req
        assert "description" in req
        assert "priority" in req
        assert "team" in req
        assert "source" in req
        assert req["team"] == "test_team"
    
    # Check that the source field has the expected structure
    for req in requirements:
        assert "system" in req["source"]
        assert "source_id" in req["source"]
        assert "path" in req["source"]
        assert req["source"]["system"] == "file"

def test_process_json_file(connector, temp_input_dir):
    """Test processing JSON files."""
    # Test single JSON
    json_file_single = temp_input_dir / "requirement_single.json"
    with open(json_file_single, 'r') as f:
        content = f.read()
    
    requirements = connector._process_json_file(json_file_single, content)
    # Single JSON object should be processed when it contains a requirement
    assert requirements is not None
    assert isinstance(requirements, list)
    assert len(requirements) == 1
    assert requirements[0]["id"] == "REQ-001"
    
    # Test collection JSON
    json_file_collection = temp_input_dir / "requirement_collection.json"
    with open(json_file_collection, 'r') as f:
        content = f.read()
    
    requirements = connector._process_json_file(json_file_collection, content)
    assert requirements is not None
    assert len(requirements) == 2
    assert requirements[0]["id"] == "REQ-001"
    assert requirements[1]["id"] == "REQ-002"
    
    # Test nested JSON
    json_file_nested = temp_input_dir / "requirement_nested.json"
    with open(json_file_nested, 'r') as f:
        content = f.read()
    
    requirements = connector._process_json_file(json_file_nested, content)
    assert requirements is not None
    assert len(requirements) == 2
    assert requirements[0]["summary"] == "Nested Requirement 1"
    assert requirements[1]["summary"] == "Nested Requirement 2"
    
    # Test invalid JSON
    invalid_json = "{ invalid json }"
    requirements = connector._process_json_file(Path("invalid.json"), invalid_json)
    assert requirements is None

def test_push_test_cases(connector, test_cases, temp_output_dir):
    """Test pushing test cases to output files."""
    # Push test cases
    result = connector.push_test_cases(test_cases)
    assert result is True
    
    # Check that the output file exists
    output_file = temp_output_dir / "test_team_test_cases.json"
    assert output_file.exists()
    
    # Check that the output file contains the expected content
    with open(output_file, 'r') as f:
        content = json.load(f)
    
    assert len(content) == 2
    assert content[0]["id"] == "TC-001"
    assert content[1]["id"] == "TC-002"
    
    # Test with empty test cases list
    result = connector.push_test_cases([])
    assert result is True

def test_fetch_requirements_nonexistent_dir():
    """Test fetching requirements from a nonexistent directory."""
    connector = LocalFileSystemConnector(
        input_directory="nonexistent_dir",
        team_name="test_team"
    )
    
    requirements = connector.fetch_requirements()
    assert requirements == []

def test_push_test_cases_error(connector, test_cases, monkeypatch):
    """Test error handling in push_test_cases."""
    # Mock open to raise an exception
    def mock_open(*args, **kwargs):
        raise IOError("Mock IO error")
    
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Push test cases should return False on error
    result = connector.push_test_cases(test_cases)
    assert result is False

@pytest.mark.regression
def test_fetch_requirements_error(connector, monkeypatch):
    """Test error handling in fetch_requirements."""
    # Mock open to raise an exception for specific files
    original_open = open
    
    def mock_open(*args, **kwargs):
        if "requirement.md" in args[0]:
            raise IOError("Mock IO error")
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr("builtins.open", mock_open)
    
    # Fetch requirements should skip files with errors
    requirements = connector.fetch_requirements()
    assert len(requirements) == 4  # 5 - 1 skipped
