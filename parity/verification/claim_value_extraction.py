"""
Claimed-value extraction module for the verification engine.

Phase 4's `claims` table stores `claim_text` as free prose (e.g. "the retry function accepts a `max_attempts` argument defaulting to 3") 
— there is no structured field yet identifying *which* parameter, *which* default, *which* env var name. 
Before comparison is possible, this phase adds a **claimed-value extraction** step: 
a second, narrower LLM call per claim, using a `claim_type`-specific schema, 
to pull the literal thing being claimed out of the prose into a comparable structure. 

This is a deliberate two-stage design — Phase 4 stayed generic (any claim type, uniform schema) 
so extraction wasn't over-fit to the verification step that didn't exist yet; 
this phase narrows per-type on purpose, now that comparison logic needs typed structure.
"""

import json
import logging
from typing import Dict, Any

from parity.llm.ollama_client import call_ollama_json

CLAIMED_VALUE_SCHEMAS = {
    "signature":     {"param_names": list, "param_count": (int, type(None))},
    "default_value": {"param_name": (str, type(None)), "default": (str, type(None))},
    "env_var":       {"var_name": (str, type(None))},
    "return_type":   {"return_type": (str, type(None))},
}

def build_claimed_value_prompt(claim_text: str, claim_type: str) -> str:
    base_instructions = (
        f"You are a specialized parsing system. Extract the structured value from the following claim text.\n"
        f"The claim type is: {claim_type}\n"
        f"Claim text: \"{claim_text}\"\n\n"
        f"CRITICAL RULES:\n"
        f"- Output MUST be exactly ONE valid JSON object.\n"
        f"- Do NOT wrap the JSON in markdown formatting or fences (e.g., no ```json).\n"
        f"- Do NOT provide any conversational preamble, explanation, or notes.\n"
    )

    if claim_type == "signature":
        prompt = (
            base_instructions +
            "Instruct extraction of every parameter name explicitly mentioned. "
            "Extract an explicit count ONLY if the claim states a count (e.g., 'takes three arguments'). "
            "If no explicit count is stated, set param_count to null.\n\n"
            "Format:\n"
            "{\"param_names\": [\"name1\", \"name2\"], \"param_count\": 2}\n\n"
            "Examples:\n"
            "Claim: \"The initialize function takes a host and port.\"\n"
            "Output: {\"param_names\": [\"host\", \"port\"], \"param_count\": null}\n"
            "Claim: \"This accepts three arguments including debug_mode.\"\n"
            "Output: {\"param_names\": [\"debug_mode\"], \"param_count\": 3}\n"
        )
    elif claim_type == "default_value":
        prompt = (
            base_instructions +
            "Instruct extraction of the single parameter name and its claimed default, as a string. "
            "Preserve exactly how it was phrased (e.g., \"30\", \"None\", \"'utf-8'\").\n\n"
            "Format:\n"
            "{\"param_name\": \"name\", \"default\": \"value\"}\n\n"
            "Examples:\n"
            "Claim: \"The retry max_attempts defaults to 3.\"\n"
            "Output: {\"param_name\": \"max_attempts\", \"default\": \"3\"}\n"
            "Claim: \"timeout is None by default.\"\n"
            "Output: {\"param_name\": \"timeout\", \"default\": \"None\"}\n"
        )
    elif claim_type == "env_var":
        prompt = (
            base_instructions +
            "Instruct extraction of the literal environment variable name, uppercase as conventionally written. "
            "Set to null if the claim doesn't name a specific variable.\n\n"
            "Format:\n"
            "{\"var_name\": \"VAR_NAME\"}\n\n"
            "Examples:\n"
            "Claim: \"Reads the DEBUG environment variable to toggle verbosity.\"\n"
            "Output: {\"var_name\": \"DEBUG\"}\n"
            "Claim: \"Checks for an environment variable for configuration.\"\n"
            "Output: {\"var_name\": null}\n"
        )
    elif claim_type == "return_type":
        prompt = (
            base_instructions +
            "Instruct extraction of the claimed return type as a short type-name string.\n\n"
            "Format:\n"
            "{\"return_type\": \"type_name\"}\n\n"
            "Examples:\n"
            "Claim: \"Returns a list of parsed elements.\"\n"
            "Output: {\"return_type\": \"list\"}\n"
            "Claim: \"This function will return None on failure.\"\n"
            "Output: {\"return_type\": \"None\"}\n"
        )
    else:
        prompt = base_instructions
        
    return prompt

def extract_claimed_value(claim_text: str, claim_type: str, model_name: str, host: str) -> dict:
    if claim_type not in CLAIMED_VALUE_SCHEMAS:
        return {}

    prompt = build_claimed_value_prompt(claim_text, claim_type)
    retry_message = "Respond with ONLY a JSON object. No prose. No markdown fences. Ensure valid JSON."
    
    response_text = ""
    parsed_json = None
    
    for attempt in range(2):
        current_prompt = prompt if attempt == 0 else prompt + f"\n\n{retry_message}"
        try:
            response_text = call_ollama_json(current_prompt, model_name, host)
            # Remove any markdown wrapping just in case, despite instructions
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed_json = json.loads(cleaned)
            if isinstance(parsed_json, dict):
                break
            else:
                parsed_json = None # Must be a dict, not array
        except json.JSONDecodeError:
            parsed_json = None
        except Exception as e:
            logging.error(f"Error calling LLM for value extraction: {e}")
            parsed_json = None
            
    if parsed_json is None:
        return {}
        
    # Validate and loose coerce
    schema = CLAIMED_VALUE_SCHEMAS[claim_type]
    validated = {}
    for key, expected_type in schema.items():
        if key not in parsed_json or parsed_json[key] is None:
            if expected_type == list:
                validated[key] = []
            else:
                validated[key] = None
        else:
            val = parsed_json[key]
            # Coerce based on expected type
            if expected_type == list:
                if isinstance(val, list):
                    validated[key] = val
                else:
                    logging.warning(f"Coercing {key} from {type(val)} to list")
                    validated[key] = [str(val)]
            elif isinstance(expected_type, tuple):
                # e.g., (int, type(None)) or (str, type(None))
                primary_type = expected_type[0]
                try:
                    if primary_type == int:
                        validated[key] = int(val)
                    else:
                        validated[key] = str(val)
                except ValueError:
                    logging.warning(f"Failed to coerce {key} value '{val}' to {primary_type}")
                    validated[key] = None
            else:
                try:
                    validated[key] = expected_type(val)
                except ValueError:
                    validated[key] = None
                    
    return validated
