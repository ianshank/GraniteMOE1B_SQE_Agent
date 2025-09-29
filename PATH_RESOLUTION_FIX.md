# Path Resolution Fix

## Summary

This document describes the changes made to fix the path resolution issue in the Granite MOE Test Generator. The issue was causing test case outputs to be written to the `granite-test-generator/output` directory instead of the root project's `output` directory.

## Changes Made

### 1. Updated Constants Module

Modified `granite-test-generator/src/utils/constants.py` to use absolute paths:

- Added imports for `os` and `Path` from `pathlib`
- Created a path hierarchy to determine the project structure:
  ```python
  _CURRENT_FILE = Path(__file__).resolve()
  _MODULE_DIR = _CURRENT_FILE.parent  # utils directory
  _SRC_DIR = _MODULE_DIR.parent       # src directory
  _GRANITE_DIR = _SRC_DIR.parent      # granite-test-generator directory
  _PROJECT_ROOT = _GRANITE_DIR.parent # project root directory
  ```
- Updated all file system paths to use absolute paths:
  ```python
  DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "output")
  DEFAULT_MODELS_DIR = str(_GRANITE_DIR / "models/fine_tuned_granite")
  DEFAULT_CACHE_DIR = str(_GRANITE_DIR / "cache")
  DEFAULT_LOGS_DIR = str(_GRANITE_DIR / "logs")
  DEFAULT_DATA_DIR = str(_GRANITE_DIR / "data")
  DEFAULT_REQUIREMENTS_DIR = str(_GRANITE_DIR / "data/requirements")
  DEFAULT_TRAINING_DIR = str(_GRANITE_DIR / "data/training")
  DEFAULT_USER_STORIES_DIR = str(_GRANITE_DIR / "data/user_stories")
  ```

### 2. Added Unit Tests

Created a new test file `granite-test-generator/tests/unit/test_path_resolution.py` to verify the path resolution:

- Test that the path hierarchy is correctly established
- Test that all paths are absolute
- Test that paths are correctly resolved even when the working directory changes
- Test that the output directory points to the project root

## Verification

The fix has been verified by:

1. Running the unit tests, which all pass
2. Running the main script and confirming that it now writes to the correct output directory
3. Checking the timestamps on the output files to confirm they were updated

## Benefits

This change provides several benefits:

1. **Consistency**: Output files are always written to the same location, regardless of the current working directory
2. **Clarity**: The path resolution is now more explicit and easier to understand
3. **Robustness**: The system is more resilient to changes in the working directory

## Future Considerations

1. **Configuration**: Consider adding an explicit configuration option for the output directory path
2. **Documentation**: Update documentation to clearly specify where output files are written
3. **Environment Variables**: Consider adding environment variables to override default paths

## Conclusion

The path resolution issue has been successfully fixed by using absolute paths in the constants module. This ensures that output files are always written to the correct location, regardless of the current working directory.
