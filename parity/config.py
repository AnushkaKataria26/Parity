import sys
import yaml
import copy

DEFAULT_CONFIG = {
    "db_path": "data/parity.db",
    "chroma_persist_dir": "data/chroma",
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3:8b",
}

def load_config(path: str = "config.yaml") -> dict:
    import os
    if not os.path.exists(path):
        print(f"Warning: config file '{path}' not found, using defaults", file=sys.stderr)
        return copy.deepcopy(DEFAULT_CONFIG)
    
    try:
        with open(path, 'r') as f:
            file_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error: failed to parse config file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
        
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if file_config and isinstance(file_config, dict):
        merged.update(file_config)
        
    return merged
