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

class LLMCallError(Exception):
    """Custom exception for LLM call failures."""
    pass

def extract_claims_raw(doc_chunk_text: str, heading_path: str, model_name: str, host: str, timeout: float = 60.0, retry_message: str = None) -> str:
    import ollama
    
    # Generate the prompt
    from parity.extraction.prompts import build_extraction_prompt
    prompt = build_extraction_prompt(doc_chunk_text, heading_path)
    if retry_message:
        prompt += f"\n\n{retry_message}"
    
    try:
        client = ollama.Client(host=host)
        # We use temperature 0.0 for deterministic extraction since structured output is desired.
        response = client.generate(
            model=model_name,
            prompt=prompt,
            stream=False,
            options={"temperature": 0.0}
        )
        return response.get("response", "")
    except Exception as e:
        raise LLMCallError(f"Failed to generate claims via Ollama: {str(e)}") from e
