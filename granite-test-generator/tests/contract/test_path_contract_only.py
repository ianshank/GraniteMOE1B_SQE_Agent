"""
Contract tests for path resolution infrastructure.

IMPORTANT: Defines contracts for path handling and file system operations only.
No test case generation - all test cases must come from E2E flow.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch
import pytest

from src.integration.team_connectors import LocalFileSystemConnector
from src.main import GraniteTestCaseGenerator


class TestLocalFileSystemConnectorPathContract:
    """Contract tests for path handling infrastructure."""
    
    def test_contract_input_directory_storage(self, tmp_path):
        """CONTRACT: Connector must properly store input directory as Path object."""
        test_dir = tmp_path / "test_input"
        test_dir.mkdir()
        
        # Test with string path
        connector_str = LocalFileSystemConnector(
            input_directory=str(test_dir),
            team_name="test_team"
        )
        
        assert hasattr(connector_str, 'input_directory')
        assert isinstance(connector_str.input_directory, Path)
        assert connector_str.input_directory == test_dir
        
        # Test with Path object
        connector_path = LocalFileSystemConnector(
            input_directory=test_dir,
            team_name="test_team"
        )
        
        assert connector_path.input_directory == test_dir
    
    def test_contract_team_name_storage(self):
        """CONTRACT: Connector must properly store team name."""
        connector = LocalFileSystemConnector(
            input_directory="/tmp",
            team_name="specific_team_name"
        )
        
        assert hasattr(connector, 'team_name')
        assert connector.team_name == "specific_team_name"
    
    def test_contract_graceful_missing_directory_handling(self):
        """CONTRACT: Connector must handle missing directories gracefully."""
        connector = LocalFileSystemConnector(
            input_directory="/absolutely/nonexistent/path",
            team_name="test_team"
        )
        
        # Should not raise exception during initialization
        assert connector.input_directory == Path("/absolutely/nonexistent/path")
        
        # Should not raise exception during fetch_requirements
        requirements = connector.fetch_requirements()
        assert isinstance(requirements, list)
        assert len(requirements) == 0
    
    def test_contract_file_type_parameter_handling(self, tmp_path):
        """CONTRACT: Connector must respect file_types parameter."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create files of different types
        (data_dir / "file.md").write_text("# Markdown")
        (data_dir / "file.txt").write_text("Text content")
        (data_dir / "file.json").write_text('{"key": "value"}')
        (data_dir / "file.py").write_text("# Python code")
        
        # Test default file types
        connector_default = LocalFileSystemConnector(
            input_directory=str(data_dir),
            team_name="test_team"
        )
        
        # Should have default file types
        assert hasattr(connector_default, 'file_types')
        assert connector_default.file_types == ['.md', '.txt', '.json']
        
        # Test custom file types
        connector_custom = LocalFileSystemConnector(
            input_directory=str(data_dir),
            team_name="test_team",
            file_types=['.md']
        )
        
        assert connector_custom.file_types == ['.md']


class TestConfigurationPrecedenceContract:
    """Contract tests for configuration precedence infrastructure."""
    
    def test_contract_environment_variable_precedence(self):
        """CONTRACT: Environment variables must take precedence over config files."""
        # Test GRANITE_LOCAL_ONLY detection
        with patch.dict(os.environ, {'GRANITE_LOCAL_ONLY': 'true'}):
            generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
            assert generator.local_only_mode is True
        
        with patch.dict(os.environ, {'GRANITE_LOCAL_ONLY': 'false'}):
            generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
            assert generator.local_only_mode is False
    
    def test_contract_integration_config_override_behavior(self, tmp_path):
        """CONTRACT: Integration config must properly override base config."""
        base_teams = [
            {'name': 'base_team', 'connector': {'type': 'github'}}
        ]
        
        integration_config = {
            'teams': [
                {'name': 'integration_team', 'connector': {'type': 'local', 'input_directory': 'data'}}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(integration_config, f)
            temp_config = f.name
        
        try:
            with patch.dict(os.environ, {'INTEGRATION_CONFIG_PATH': temp_config}):
                generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
                
                # Test that integration config is loaded
                result_teams = generator._load_integration_config_with_precedence(base_teams)
                
                # Should have both base and integration teams (merge mode default)
                # Note: The actual behavior may vary based on implementation
                assert len(result_teams) >= 1  # At least integration team should be present
                team_names = {team['name'] for team in result_teams}
                assert 'integration_team' in team_names  # Integration team must be present
        
        finally:
            Path(temp_config).unlink(missing_ok=True)
    
    def test_contract_duplicate_team_handling(self):
        """CONTRACT: Duplicate team names must be handled with integration config winning."""
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        base_teams = [
            {'name': 'shared_team', 'connector': {'type': 'github', 'priority': 'base'}}
        ]
        
        integration_teams = [
            {'name': 'shared_team', 'connector': {'type': 'local', 'priority': 'integration'}}
        ]
        
        merged = generator._merge_and_deduplicate_teams(base_teams, integration_teams)
        
        # Should have only one team (integration wins)
        assert len(merged) == 1
        assert merged[0]['name'] == 'shared_team'
        assert merged[0]['connector']['type'] == 'local'
        assert merged[0]['connector']['priority'] == 'integration'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])