"""
Unit tests for team configuration validation.

Tests the comprehensive team validation logic that ensures
configuration integrity before team registration.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import os

from src.main import GraniteTestCaseGenerator


class TestTeamConfigurationValidation:
    """Test team configuration validation logic."""
    
    def test_validate_team_configurations_valid_teams(self, tmp_path):
        """Test validation passes for valid team configurations."""
        # Create test directory for local connector
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        
        valid_teams = [
            {
                'name': 'local_team',
                'connector': {
                    'type': 'local',
                    'input_directory': str(test_dir)
                },
                'rag_enabled': True
            },
            {
                'name': 'github_team',
                'connector': {
                    'type': 'github',
                    'repo_owner': 'owner',
                    'repo_name': 'repo',
                    'token': 'token'
                },
                'rag_enabled': False
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        validated = generator._validate_team_configurations(valid_teams)
        
        assert len(validated) == 2
        assert validated == valid_teams
    
    def test_validate_team_configurations_missing_name(self):
        """Test validation fails for teams missing name field."""
        invalid_teams = [
            {
                # Missing 'name' field
                'connector': {'type': 'local', 'input_directory': '/tmp'},
                'rag_enabled': True
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        with pytest.raises(ValueError, match="No valid team configurations found"):
            generator._validate_team_configurations(invalid_teams)
    
    def test_validate_team_configurations_invalid_connector(self):
        """Test validation fails for invalid connector configurations."""
        invalid_teams = [
            {
                'name': 'invalid_team',
                'connector': 'not_a_dict',  # Should be dict
                'rag_enabled': True
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        with pytest.raises(ValueError, match="No valid team configurations found"):
            generator._validate_team_configurations(invalid_teams)
    
    def test_validate_team_configurations_missing_connector_type(self):
        """Test validation fails for connectors missing type field."""
        invalid_teams = [
            {
                'name': 'no_type_team',
                'connector': {
                    # Missing 'type' field
                    'input_directory': '/tmp'
                },
                'rag_enabled': True
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        with pytest.raises(ValueError, match="No valid team configurations found"):
            generator._validate_team_configurations(invalid_teams)
    
    def test_validate_team_configurations_mixed_valid_invalid(self, tmp_path):
        """Test validation handles mixed valid and invalid teams."""
        test_dir = tmp_path / "valid_data"
        test_dir.mkdir()
        
        mixed_teams = [
            {
                'name': 'valid_team',
                'connector': {
                    'type': 'local',
                    'input_directory': str(test_dir)
                },
                'rag_enabled': True
            },
            {
                'name': 'invalid_team',
                'connector': 'not_a_dict',  # Invalid
                'rag_enabled': True
            },
            {
                # Missing name
                'connector': {'type': 'local'},
                'rag_enabled': True
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        validated = generator._validate_team_configurations(mixed_teams)
        
        # Should keep only the valid team
        assert len(validated) == 1
        assert validated[0]['name'] == 'valid_team'
    
    def test_validate_team_configurations_empty_list(self):
        """Test validation handles empty team list gracefully."""
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        validated = generator._validate_team_configurations([])
        
        assert validated == []
    
    def test_validate_team_configurations_local_path_warning(self, tmp_path):
        """Test validation logs warning for non-existent local paths."""
        nonexistent_teams = [
            {
                'name': 'missing_path_team',
                'connector': {
                    'type': 'local',
                    'input_directory': '/absolutely/nonexistent/path'
                },
                'rag_enabled': True
            }
        ]
        
        generator = GraniteTestCaseGenerator(config_dict={'model_name': 'test'})
        
        # Should not raise exception, but should log warning
        validated = generator._validate_team_configurations(nonexistent_teams)
        
        # Team should still be included (connector handles missing paths gracefully)
        assert len(validated) == 1
        assert validated[0]['name'] == 'missing_path_team'


class TestTeamValidationIntegration:
    """Integration tests for team validation in full workflow."""
    
    def test_validation_integrates_with_register_teams(self, tmp_path):
        """Test that validation is properly integrated with team registration."""
        test_dir = tmp_path / "integration_test"
        test_dir.mkdir()
        
        config = {
            'model_name': 'test-model',
            'teams': [
                {
                    'name': 'valid_team',
                    'connector': {
                        'type': 'local',
                        'input_directory': str(test_dir)
                    },
                    'rag_enabled': True,
                    'cag_enabled': True,
                    'auto_push': False
                },
                {
                    'name': 'invalid_team',
                    'connector': 'invalid_connector',  # Invalid
                    'rag_enabled': True
                }
            ]
        }
        
        # Mock heavy components to focus on validation
        with patch('src.main.GraniteMoETrainer'), \
             patch('src.main.RAGRetriever'), \
             patch('src.main.CAGCache'), \
             patch('src.main.TestGenerationAgent'), \
             patch('src.main.IntelligentChunker'), \
             patch('src.main.KVCache'), \
             patch('src.main.WorkflowOrchestrator') as mock_orchestrator:
            
            mock_orchestrator_instance = MagicMock()
            mock_orchestrator_instance.create_team_configuration.return_value = MagicMock()
            mock_orchestrator_instance.register_team.return_value = None
            mock_orchestrator.return_value = mock_orchestrator_instance
    
            # Isolate the test from any global integration config
            with patch.dict(os.environ, {"INTEGRATION_CONFIG_PATH": ""}):
                generator = GraniteTestCaseGenerator(config_dict=config)
                generator.components['orchestrator'] = mock_orchestrator_instance
        
                # Should register only the valid team
                registered_count = generator.register_teams()
            assert registered_count == 1
            
            # Verify only valid team was processed
            mock_orchestrator_instance.create_team_configuration.assert_called_once()
            call_args = mock_orchestrator_instance.create_team_configuration.call_args[0][0]
            assert call_args['name'] == 'valid_team'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
