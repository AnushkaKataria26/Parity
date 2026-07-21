import requests

def check_ollama_reachable(host: str = "http://localhost:11434", timeout: float = 3.0) -> bool:
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        return response.status_code >= 200 and response.status_code < 300
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, Exception):
        return False

def check_model_available(model_name: str, host: str = "http://localhost:11434", timeout: float = 3.0) -> bool:
    if not check_ollama_reachable(host, timeout):
        return False
        
    try:
        response = requests.get(f"{host}/api/tags", timeout=timeout)
        data = response.json()
        
        models = data.get("models", [])
        
        # 1. Exact match
        for model in models:
            if model.get("name") == model_name:
                return True
                
        # 2. Family prefix match
        requested_family = model_name.split(":")[0] if ":" in model_name else model_name
        for model in models:
            actual_name = model.get("name", "")
            actual_family = actual_name.split(":")[0] if ":" in actual_name else actual_name
            if requested_family == actual_family:
                return True
                
        return False
    except Exception:
        return False
