from pathlib import Path
from typing import Dict
from src.utils.constants import DEFAULT_DATA_DIR

class StorageManager:
    """Intelligent storage management for Mac Mini deployment"""
    
    def __init__(self, base_path: str = f"./{DEFAULT_DATA_DIR}_workspace"):
        self.base_path = Path(base_path)
        self.storage_limits = {
            'model_cache': 3.0,      # GB
            'vector_db': 2.0,        # GB  
            'training_data': 5.0,    # GB
            'output_files': 2.0,     # GB
            'temp_files': 1.0        # GB
        }
    
    def optimize_storage(self):
        """Clean up unnecessary files and optimize storage"""
        # Remove old checkpoints, compress logs, clean temp files
        total_freed = self._clean_temp_files()
        total_freed += self._compress_old_logs()
        total_freed += self._remove_old_checkpoints()
        
        return total_freed
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get current storage usage by component"""
        usage = {}
        for component, limit in self.storage_limits.items():
            path = self.base_path / component
            if path.exists():
                usage[component] = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)
            else:
                usage[component] = 0.0
        return usage