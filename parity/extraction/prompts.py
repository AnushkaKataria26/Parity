import json
from dataclasses import dataclass

CLAIM_TYPES = ["signature", "default_value", "env_var", "return_type", "behavior"]

@dataclass
class ExtractedClaim:
    claim_text: str
    claim_type: str
    referenced_symbol_guess: str | None

SYSTEM_PROMPT = """You are an expert technical documentation analyzer.
Your task is to extract atomic, structured claims from documentation prose.
A "claim" is a specific statement of fact about how the code behaves or is configured.

The output MUST be a JSON array of objects.
Each object MUST have exactly these three keys:
- "claim_text": The extracted claim as a string.
- "claim_type": Must be one of ["signature", "default_value", "env_var", "return_type", "behavior"].
- "referenced_symbol_guess": The name of the function, class, or parameter the claim is about, or null if it cannot be determined.

Rules for claim_type:
- signature: claim about parameter names, order, count, or types.
- default_value: claim about a specific default.
- env_var: claim about an environment variable's name, purpose, or default.
- return_type: claim about what a function returns.
- behavior: any other claim about runtime behavior.

If the documentation chunk contains no checkable claims at all (pure narrative/prose), output an empty JSON array [].
DO NOT include any chain-of-thought, preamble, or markdown code fences (like ```json). Just the raw JSON array.
"""

# We deliberately instruct the model to suppress chain-of-thought to control costs and hallucinations 
# per the project's GenAI pipeline design.

def build_extraction_prompt(doc_chunk_text: str, heading_path: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n"
    
    prompt += "--- Examples ---\n"
    prompt += "Input Context (Heading: config/timeout):\n"
    prompt += "The timeout parameter defaults to 30 seconds. Set `DEBUG=1` to enable verbose logging.\n"
    prompt += "Output:\n"
    prompt += json.dumps([
        {"claim_text": "timeout defaults to 30 seconds", "claim_type": "default_value", "referenced_symbol_guess": "timeout"},
        {"claim_text": "set DEBUG=1 to enable verbose logging", "claim_type": "env_var", "referenced_symbol_guess": "DEBUG"}
    ], indent=2) + "\n\n"
    
    prompt += "Input Context (Heading: api.retry()):\n"
    prompt += "The retry function takes a max_attempts argument and requests are retried with exponential backoff.\n"
    prompt += "Output:\n"
    prompt += json.dumps([
        {"claim_text": "the retry function takes a max_attempts argument", "claim_type": "signature", "referenced_symbol_guess": "retry"},
        {"claim_text": "requests are retried with exponential backoff", "claim_type": "behavior", "referenced_symbol_guess": "retry"}
    ], indent=2) + "\n\n"
    
    prompt += "Input Context (Heading: utils.process()):\n"
    prompt += "Returns a list of Result objects. This function is idempotent.\n"
    prompt += "Output:\n"
    prompt += json.dumps([
        {"claim_text": "returns a list of Result objects", "claim_type": "return_type", "referenced_symbol_guess": "process"},
        {"claim_text": "this function is idempotent", "claim_type": "behavior", "referenced_symbol_guess": "process"}
    ], indent=2) + "\n\n"
    
    prompt += "--- Real Task ---\n"
    prompt += f"Input Context (Heading: {heading_path}):\n"
    prompt += f"{doc_chunk_text}\n"
    prompt += "Output:\n"
    
    return prompt
