"""
Unit tests for main entry points.

Tests the main.py entry points for proper error handling, logging,
and workflow execution without generating test cases.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typing import Any


class TestRootMainEntryPoint:
    """Test root main.py entry point functionality."""
    
    def test_main_function_exists_and_callable(self):
        """Test that main function exists and is callable."""
        # Add root to path for import
        root_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root_path))
        
        try:
            import main
            assert hasattr(main, 'main')
            assert callable(main.main)
        finally:
            # Clean up path
            if str(root_path) in sys.path:
                sys.path.remove(str(root_path))
    
    def test_main_handles_missing_granite_directory(self):
        """Test main handles missing granite-test-generator directory gracefully."""
        root_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root_path))
        
        try:
            import main
            
            # Mock Path to simulate missing directory
            with patch.dict(
                os.environ,
                {
                    "GRANITE_LOCAL_ONLY": "true",
                    "GRANITE_PROJECT_ROOT": "/nonexistent/granite-test-generator",
                },
                clear=False,
            ):

                with pytest.raises(RuntimeError, match="Granite test generator directory not found"):
                    main.main()
        finally:
            if str(root_path) in sys.path:
                sys.path.remove(str(root_path))
    
    def test_main_handles_import_errors_gracefully(self):
        """Test main handles import errors with proper logging."""
        root_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root_path))
        
        try:
            import main
            
            # Mock os.chdir and import to simulate import failure
            with patch.dict(os.environ, {"GRANITE_LOCAL_ONLY": "true"}, clear=False), \
                 patch('main.os.chdir'), \
                 patch('asyncio.run') as mock_run:

                def _raise_module_not_found(coro):
                    if hasattr(coro, 'close'):
                        coro.close()
                    raise ImportError("Module not found")

                mock_run.side_effect = _raise_module_not_found
                
                with pytest.raises(ImportError, match="Module not found"):
                    main.main()
        finally:
            if str(root_path) in sys.path:
                sys.path.remove(str(root_path))


class TestGraniteMainEntryPoint:
    """Test granite-test-generator/main.py entry point functionality."""
    
    def test_granite_main_function_exists(self):
        """Test that granite main function exists and is callable."""
        granite_path = Path(__file__).parent.parent.parent / "granite-test-generator"
        sys.path.insert(0, str(granite_path))
        
        try:
            import main as granite_main
            assert hasattr(granite_main, 'main')
            assert callable(granite_main.main)
        finally:
            if str(granite_path) in sys.path:
                sys.path.remove(str(granite_path))
    
    def test_granite_main_handles_import_errors(self):
        """Test granite main handles import errors gracefully."""
        granite_path = Path(__file__).parent.parent.parent / "granite-test-generator"
        sys.path.insert(0, str(granite_path))
        
        try:
            import main as granite_main
            
            # Mock the asyncio.run to simulate import failure
            with patch.dict(os.environ, {"GRANITE_LOCAL_ONLY": "true"}, clear=False), \
                 patch('asyncio.run') as mock_run:

                def _raise_src_main_not_found(coro):
                    if hasattr(coro, 'close'):
                        coro.close()
                    raise ImportError("src.main not found")

                mock_run.side_effect = _raise_src_main_not_found
                
                with pytest.raises(ImportError, match="src.main not found"):
                    granite_main.main()
        finally:
            if str(granite_path) in sys.path:
                sys.path.remove(str(granite_path))


class TestMainEntryPointIntegration:
    """Integration tests for main entry point behavior."""
    
    def test_working_directory_changes_correctly(self):
        """Test that main entry point changes to correct working directory."""
        root_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root_path))
        
        try:
            import main
            
            original_cwd = os.getcwd()
            expected_granite_dir = root_path / "granite-test-generator"
            
            # Mock the granite_main to avoid full execution
            with patch.dict(
                os.environ,
                {
                    "GRANITE_LOCAL_ONLY": "true",
                    "GRANITE_PROJECT_ROOT": str(expected_granite_dir),
                },
                clear=False,
            ), \
                 patch('asyncio.run', new=lambda coro: coro.close() if hasattr(coro, 'close') else None), \
                 patch('main.os.chdir') as mock_chdir:
                
                # Should not raise exception
                main.main()
                
                # Verify chdir was called with correct directory
                mock_chdir.assert_called_once_with(expected_granite_dir)
        
        finally:
            if str(root_path) in sys.path:
                sys.path.remove(str(root_path))
    
    def test_logging_configuration(self):
        """Test that logging is properly configured in main entry points."""
        root_path = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(root_path))
        
        try:
            import main
            
            # Mock logging configuration
            with patch.dict(
                os.environ,
                {
                    "GRANITE_LOCAL_ONLY": "true",
                    "GRANITE_PROJECT_ROOT": str((root_path / "granite-test-generator")),
                },
                clear=False,
            ), \
                 patch('logging.basicConfig') as mock_logging, \
                 patch('asyncio.run', new=lambda coro: coro.close() if hasattr(coro, 'close') else None):

                main.main()

                # Verify logging was configured
                mock_logging.assert_called_once()
                call_kwargs = mock_logging.call_args[1]
                import logging as py_logging
                assert call_kwargs['level'] == py_logging.INFO
                assert 'format' in call_kwargs
        
        finally:
            if str(root_path) in sys.path:
                sys.path.remove(str(root_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
