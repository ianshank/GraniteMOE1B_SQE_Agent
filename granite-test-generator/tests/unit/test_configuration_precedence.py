"""
Unit tests for configuration precedence and override behavior.

Tests the enhanced configuration loading logic that properly handles
base config vs integration config precedence with deduplication.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, Any, List

from src.main import GraniteTestCaseGenerator


class TestConfigurationPrecedence:
    """Test configuration precedence and override behavior."""
    
    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Base configuration with GitHub teams."""
        return {
            'model_name': 'test-model',
            'teams': [
                {
                    'name': 'team1',
                    'connector': {'type': 'github', 'repo_owner': 'org', 'repo_name': 'repo1', 'token': 'token1'},
                    'rag_enabled': True
                },
                {
                    'name': 'team2', 
                    'connector': {'type': 'github', 'repo_owner': 'org', 'repo_name': 'repo2', 'token': 'token2'},
                    'rag_enabled': True
                }
            ]
        }
    
    @pytest.fixture
    def integration_config(self) -> Dict[str, Any]:
        """Integration config with local teams."""
        return {
            'teams': [
                {
                    'name': 'team1',  # Override team1
                    'connector': {'type': 'local', 'input_directory': 'data/team1'},
                    'rag_enabled': False
                },
                {
                    'name': 'team3',  # New team
                    'connector': {'type': 'local', 'input_directory': 'data/team3'},
                    'rag_enabled': True
                }
            ]
        }
    
    @pytest.fixture
    def temp_integration_file(self, integration_config: Dict[str, Any]) -> str:
        """Create temporary integration config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(integration_config, f)
            return f.name
    
    def test_load_integration_config_replace_mode(self, base_config: Dict[str, Any], temp_integration_file: str):
        """Test REPLACE mode: integration config completely replaces base config."""
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': temp_integration_file,
                'GRANITE_CONFIG_OVERRIDE_MODE': 'replace'
            }):
                generator = GraniteTestCaseGenerator(config_dict=base_config)
                base_teams = base_config['teams']
                
                result_teams = generator._load_integration_config_with_precedence(base_teams)
                
                # Should have only integration teams
                assert len(result_teams) == 2
                team_names = {team['name'] for team in result_teams}
                assert team_names == {'team1', 'team3'}
                
                # team1 should have local connector (from integration)
                team1 = next(t for t in result_teams if t['name'] == 'team1')
                assert team1['connector']['type'] == 'local'
                assert team1['rag_enabled'] is False
                
        finally:
            Path(temp_integration_file).unlink(missing_ok=True)
    
    def test_load_integration_config_merge_mode(self, base_config: Dict[str, Any], temp_integration_file: str):
        """Test MERGE mode: integration config merges with base config, overriding duplicates."""
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': temp_integration_file,
                'GRANITE_CONFIG_OVERRIDE_MODE': 'merge'
            }):
                generator = GraniteTestCaseGenerator(config_dict=base_config)
                base_teams = base_config['teams']
                
                result_teams = generator._load_integration_config_with_precedence(base_teams)
                
                # Should have team2 (base only), team1 (overridden), team3 (integration only)
                assert len(result_teams) == 3
                team_names = {team['name'] for team in result_teams}
                assert team_names == {'team1', 'team2', 'team3'}
                
                # team1 should be from integration (local connector)
                team1 = next(t for t in result_teams if t['name'] == 'team1')
                assert team1['connector']['type'] == 'local'
                assert team1['rag_enabled'] is False
                
                # team2 should be from base (github connector)
                team2 = next(t for t in result_teams if t['name'] == 'team2')
                assert team2['connector']['type'] == 'github'
                assert team2['rag_enabled'] is True
                
        finally:
            Path(temp_integration_file).unlink(missing_ok=True)
    
    def test_load_integration_config_default_merge_mode(self, base_config: Dict[str, Any], temp_integration_file: str):
        """Test default behavior (merge mode when GRANITE_CONFIG_OVERRIDE_MODE not set)."""
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': temp_integration_file
            }, clear=False):
                # Ensure override mode is not set
                os.environ.pop('GRANITE_CONFIG_OVERRIDE_MODE', None)
                
                generator = GraniteTestCaseGenerator(config_dict=base_config)
                base_teams = base_config['teams']
                
                result_teams = generator._load_integration_config_with_precedence(base_teams)
                
                # Should merge (default behavior)
                assert len(result_teams) == 3
                team_names = {team['name'] for team in result_teams}
                assert team_names == {'team1', 'team2', 'team3'}
                
        finally:
            Path(temp_integration_file).unlink(missing_ok=True)
    
    def test_normalize_connector_configs(self, base_config: Dict[str, Any]):
        """Test connector config normalization handles legacy field names."""
        generator = GraniteTestCaseGenerator(config_dict=base_config)
        
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
        assert 'path' not in conn
        assert 'output' not in conn
    
    def test_merge_and_deduplicate_teams(self, base_config: Dict[str, Any]):
        """Test team merging and deduplication logic."""
        generator = GraniteTestCaseGenerator(config_dict=base_config)
        
        base_teams = [
            {'name': 'team1', 'connector': {'type': 'github'}},
            {'name': 'team2', 'connector': {'type': 'github'}}
        ]
        
        integration_teams = [
            {'name': 'team1', 'connector': {'type': 'local'}},  # Override
            {'name': 'team3', 'connector': {'type': 'local'}}   # New
        ]
        
        merged = generator._merge_and_deduplicate_teams(base_teams, integration_teams)
        
        assert len(merged) == 3
        team_names = {team['name'] for team in merged}
        assert team_names == {'team1', 'team2', 'team3'}
        
        # team1 should be from integration (local)
        team1 = next(t for t in merged if t['name'] == 'team1')
        assert team1['connector']['type'] == 'local'
        
        # team2 should be from base (github)
        team2 = next(t for t in merged if t['name'] == 'team2')
        assert team2['connector']['type'] == 'github'
    
    def test_no_integration_config_file(self, base_config: Dict[str, Any]):
        """Test behavior when no integration config file exists."""
        with patch.dict(os.environ, {}, clear=True):
            generator = GraniteTestCaseGenerator(config_dict=base_config)
            base_teams = base_config['teams']
            
            result_teams = generator._load_integration_config_with_precedence(base_teams)
            
            # Should return base teams unchanged
            assert result_teams == base_teams
    
    def test_invalid_integration_config_file(self, base_config: Dict[str, Any]):
        """Test behavior when integration config file is invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_file = f.name
        
        try:
            with patch.dict(os.environ, {
                'INTEGRATION_CONFIG_PATH': invalid_file
            }):
                generator = GraniteTestCaseGenerator(config_dict=base_config)
                base_teams = base_config['teams']
                
                result_teams = generator._load_integration_config_with_precedence(base_teams)
                
                # Should fallback to base teams
                assert result_teams == base_teams
                
        finally:
            Path(invalid_file).unlink(missing_ok=True)


class TestConfigurationPrecedenceIntegration:
    """Integration tests for configuration precedence in full workflow."""
    
    def test_local_only_mode_forces_local_connectors(self):
        """Test that GRANITE_LOCAL_ONLY=true forces all connectors to local."""
        config = {
            'model_name': 'test-model',
            'teams': [
                {'name': 'github_team', 'connector': {'type': 'github', 'repo_owner': 'org', 'repo_name': 'repo', 'token': 'token'}},
                {'name': 'jira_team', 'connector': {'type': 'jira', 'base_url': 'url', 'username': 'user', 'api_token': 'token', 'project_key': 'key'}}
            ]
        }
        
        with patch.dict(os.environ, {'GRANITE_LOCAL_ONLY': 'true'}):
            generator = GraniteTestCaseGenerator(config_dict=config)
            
            # Local only mode should be detected
            assert generator.local_only_mode is True
    
    @pytest.mark.parametrize("env_value,expected", [
        ("true", True),
        ("1", True), 
        ("yes", True),
        ("on", True),
        ("True", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
    ])
    def test_local_only_mode_environment_values(self, env_value: str, expected: bool):
        """Test various environment variable values for local-only mode."""
        config = {'model_name': 'test-model'}
        
        with patch.dict(os.environ, {'GRANITE_LOCAL_ONLY': env_value}):
            generator = GraniteTestCaseGenerator(config_dict=config)
            assert generator.local_only_mode is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
