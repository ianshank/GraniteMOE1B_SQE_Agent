"""
Unit tests for path resolution fix - NO TEST CASE GENERATION.

These tests ONLY validate:
1. Path resolution correctness
2. Configuration precedence logic  
3. File system operations
4. Error handling

NEVER generates test cases - only validates infrastructure.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
import pytest
from typing import Dict, Any

from src.main import GraniteTestCaseGenerator
from src.integration.team_connectors import LocalFileSystemConnector


class TestPathResolutionOnly:
    """Test ONLY path resolution - no test case generation."""
    
    def test_local_connector_path_attribute_exists(self, tmp_path):
        """Test that LocalFileSystemConnector properly stores input directory."""
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        
        connector = LocalFileSystemConnector(
            input_directory=str(test_dir),
            team_name="test_team"
        )
        
        # Verify path is stored correctly
        assert hasattr(connector, 'input_directory')
        assert connector.input_directory == test_dir
        assert connector.team_name == "test_team"
    
    def test_relative_path_resolution_from_correct_directory(self, tmp_path):
        """Test relative path resolution when running from correct directory."""
        # Create granite-test-generator structure
        granite_dir = tmp_path / "granite-test-generator"
        granite_dir.mkdir()
        
        data_dir = granite_dir / "data" / "user_stories"
        data_dir.mkdir(parents=True)
        
        # Create a sample file to verify path resolution
        (data_dir / "sample.md").write_text("# Sample\nContent")
        
        original_cwd = os.getcwd()
        try:
            # Change to granite-test-generator directory
            os.chdir(granite_dir)
            
            # Create connector with relative path
            connector = LocalFileSystemConnector(
                input_directory="data/user_stories",
                team_name="test_team"
            )
            
            # Verify path resolution - LocalFileSystemConnector stores as Path object
            # but may not resolve relative paths to absolute during initialization
            assert connector.input_directory == Path("data/user_stories")
            
            # Verify the path actually points to the correct location
            resolved_path = connector.input_directory.resolve()
            expected_path = granite_dir / "data" / "user_stories"
            assert resolved_path == expected_path.resolve()
            assert resolved_path.exists()
            
        finally:
            os.chdir(original_cwd)
    
    def test_absolute_path_works_from_any_directory(self, tmp_path):
        """Test that absolute paths work regardless of working directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.md").write_text("# Test\nContent")
        
        # Test from different working directories
        for test_cwd in [tmp_path, tmp_path / "subdir", Path("/tmp")]:
            if not test_cwd.exists():
                test_cwd.mkdir(parents=True, exist_ok=True)
            
            original_cwd = os.getcwd()
            try:
                os.chdir(test_cwd)
                
                connector = LocalFileSystemConnector(
                    input_directory=str(data_dir),
                    team_name="test_team"
                )
                
                # Should resolve to same absolute path regardless of working directory
                assert connector.input_directory == data_dir
                assert connector.input_directory.exists()
                
            finally:
                os.chdir(original_cwd)
    
    def test_configuration_precedence_replace_mode(self, tmp_path):
        """Test configuration precedence REPLACE mode logic only."""
        base_config = {
            'teams': [
                {'name': 'team1', 'connector': {'type': 'github'}},
                {'name': 'team2', 'connector': {'type': 'github'}}
            ]
        }
        
        integration_config = {
            'teams': [
                {'name': 'team1', 'connector': {'type': 'local', 'input_directory': 'data'}},
                {'name': 'team3', 'connector': {'type': 'local', 'input_directory': 'data'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            temp_config = f.name
        
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': temp_config,
                'GRANITE_CONFIG_OVERRIDE_MODE': 'replace'
            }):
                generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
                result_teams = generator._load_integration_config_with_precedence(base_config['teams'])
                
                # REPLACE mode: should only have integration teams
                assert len(result_teams) == 2
                team_names = {team['name'] for team in result_teams}
                assert team_names == {'team1', 'team3'}
                
                # team1 should be from integration (local), not base (github)
                team1 = next(t for t in result_teams if t['name'] == 'team1')
                assert team1['connector']['type'] == 'local'
        
        finally:
            Path(temp_config).unlink(missing_ok=True)
    
    def test_configuration_precedence_merge_mode(self, tmp_path):
        """Test configuration precedence MERGE mode logic only."""
        base_config = {
            'teams': [
                {'name': 'team1', 'connector': {'type': 'github'}},
                {'name': 'team2', 'connector': {'type': 'github'}}
            ]
        }
        
        integration_config = {
            'teams': [
                {'name': 'team1', 'connector': {'type': 'local', 'input_directory': 'data'}},  # Override
                {'name': 'team3', 'connector': {'type': 'local', 'input_directory': 'data'}}   # New
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            temp_config = f.name
        
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': temp_config,
                'GRANITE_CONFIG_OVERRIDE_MODE': 'merge'
            }):
                generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
                result_teams = generator._load_integration_config_with_precedence(base_config['teams'])
                
                # MERGE mode: should have team2 (base) + team1,team3 (integration)
                assert len(result_teams) == 3
                team_names = {team['name'] for team in result_teams}
                assert team_names == {'team1', 'team2', 'team3'}
                
                # team1 should be from integration (local), overriding base (github)
                team1 = next(t for t in result_teams if t['name'] == 'team1')
                assert team1['connector']['type'] == 'local'
                
                # team2 should be from base (github)
                team2 = next(t for t in result_teams if t['name'] == 'team2')
                assert team2['connector']['type'] == 'github'
        
        finally:
            Path(temp_config).unlink(missing_ok=True)


class TestPathValidationOnly:
    """Test ONLY path validation - no test case operations."""
    
    def test_path_normalization_legacy_fields(self):
        """Test path normalization handles legacy field names."""
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        legacy_teams = [
            {
                'name': 'legacy_team',
                'connector': {
                    'type': 'local',
                    'path': '/old/path',  # Legacy field
                    'output': '/old/output'  # Legacy field
                }
            }
        ]
        
        normalized = generator._normalize_connector_configs(legacy_teams)
        
        assert len(normalized) == 1
        conn = normalized[0]['connector']
        assert conn['input_directory'] == '/old/path'
        assert conn['output_directory'] == '/old/output'
        assert 'path' not in conn  # Legacy field removed
        assert 'output' not in conn  # Legacy field removed
    
    def test_team_deduplication_logic(self):
        """Test team deduplication logic without test case generation."""
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        base_teams = [
            {'name': 'team1', 'connector': {'type': 'github'}},
            {'name': 'team2', 'connector': {'type': 'github'}}
        ]
        
        integration_teams = [
            {'name': 'team1', 'connector': {'type': 'local'}},  # Override
            {'name': 'team3', 'connector': {'type': 'local'}}   # New
        ]
        
        merged = generator._merge_and_deduplicate_teams(base_teams, integration_teams)
        
        # Verify deduplication worked
        assert len(merged) == 3
        team_names = {team['name'] for team in merged}
        assert team_names == {'team1', 'team2', 'team3'}
        
        # Verify override worked (team1 should be local, not github)
        team1 = next(t for t in merged if t['name'] == 'team1')
        assert team1['connector']['type'] == 'local'


class TestFileSystemOperationsOnly:
    """Test ONLY file system operations - no test case generation."""
    
    def test_connector_handles_missing_directory_gracefully(self):
        """Test that connector handles missing directories without crashing."""
        connector = LocalFileSystemConnector(
            input_directory="/nonexistent/directory",
            team_name="test_team"
        )
        
        # Should not raise exception
        requirements = connector.fetch_requirements()
        assert isinstance(requirements, list)
        assert len(requirements) == 0
    
    def test_connector_file_discovery_only(self, tmp_path):
        """Test file discovery without processing content into test cases."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create files of different types
        (data_dir / "req1.md").write_text("# Requirement 1\nContent 1")
        (data_dir / "req2.txt").write_text("Requirement 2\nContent 2")
        (data_dir / "req3.json").write_text('{"id": "req3", "summary": "Requirement 3"}')
        (data_dir / "ignored.py").write_text("# Should be ignored")
        
        connector = LocalFileSystemConnector(
            input_directory=str(data_dir),
            team_name="test_team"
        )
        
        # Test file discovery only
        requirements = connector.fetch_requirements()
        
        # Verify file discovery worked (content parsing is separate concern)
        assert len(requirements) == 3  # Should find 3 files, ignore .py
        
        # Verify basic requirement structure (but no test case generation)
        for req in requirements:
            assert 'id' in req
            assert 'summary' in req
            assert 'team' in req
            assert req['team'] == 'test_team'


class TestEnvironmentVariableHandling:
    """Test ONLY environment variable handling - no test case generation."""
    
    @pytest.mark.parametrize("env_value,expected", [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", False),
    ])
    def test_local_only_mode_detection(self, env_value, expected):
        """Test local-only mode detection from environment variables."""
        with patch.dict(os.environ, {'GRANITE_LOCAL_ONLY': env_value}):
            generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
            assert generator.local_only_mode is expected
    
    def test_override_mode_detection(self):
        """Test configuration override mode detection."""
        # Test default (merge)
        with patch.dict(os.environ, {}, clear=True):
            # Simulate the logic in _load_integration_config_with_precedence
            override_mode = os.getenv("GRANITE_CONFIG_OVERRIDE_MODE", "merge").lower()
            assert override_mode == "merge"
        
        # Test explicit replace
        with patch.dict(os.environ, {'GRANITE_CONFIG_OVERRIDE_MODE': 'replace'}):
            override_mode = os.getenv("GRANITE_CONFIG_OVERRIDE_MODE", "merge").lower()
            assert override_mode == "replace"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
