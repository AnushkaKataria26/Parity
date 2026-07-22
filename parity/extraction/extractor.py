import json
import sqlite3
import logging
from typing import List, Tuple

from parity.extraction.prompts import ExtractedClaim, CLAIM_TYPES, build_extraction_prompt
from parity.llm.ollama_client import extract_claims_raw, LLMCallError

logger = logging.getLogger(__name__)

def parse_and_validate_claims(raw_response: str) -> Tuple[List[ExtractedClaim], List[str]]:
    valid_claims = []
    validation_errors = []
    
    # 1. Strip whitespace and markdown code fences
    response = raw_response.strip()
    if response.startswith("```"):
        # Strip leading fence
        lines = response.splitlines()
        if len(lines) >= 2:
            lines = lines[1:]  # remove first line
            # Strip trailing fence if present
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines).strip()
            
    # 2. Attempt JSON parsing with recovery heuristic
    parsed_data = None
    try:
        parsed_data = json.loads(response)
    except json.JSONDecodeError as e:
        # Recovery heuristic
        first_bracket = response.find('[')
        last_bracket = response.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            substring = response[first_bracket:last_bracket+1]
            try:
                parsed_data = json.loads(substring)
            except json.JSONDecodeError:
                pass
        
        if parsed_data is None:
            return [], [f"failed to parse JSON: {str(e)}"]

    # 3. Check if list
    if not isinstance(parsed_data, list):
        return [], [f"expected a JSON array, got {type(parsed_data).__name__}"]

    # 4. Validate each item
    for item in parsed_data:
        if not isinstance(item, dict):
            validation_errors.append("expected a JSON object in array")
            continue
            
        if "claim_text" not in item or "claim_type" not in item or "referenced_symbol_guess" not in item:
            validation_errors.append("missing required keys in claim object")
            continue
            
        # claim_text validation
        claim_text = item.get("claim_text", "")
        if not isinstance(claim_text, str):
            claim_text = str(claim_text)
        claim_text = claim_text.strip()
        if not claim_text:
            validation_errors.append("claim_text is missing or empty")
            continue
            
        # claim_type validation
        claim_type = item.get("claim_type", "")
        if not isinstance(claim_type, str):
            claim_type = str(claim_type)
        claim_type = claim_type.strip().lower()
        if claim_type not in CLAIM_TYPES:
            logger.warning(f"Unrecognized claim_type '{claim_type}', defaulting to 'behavior'")
            claim_type = "behavior"
            
        # referenced_symbol_guess validation
        guess = item.get("referenced_symbol_guess")
        if guess is not None and not isinstance(guess, str):
            logger.warning(f"referenced_symbol_guess is not a string, coercing: {guess}")
            guess = str(guess)
        if guess == "":
            guess = None
            
        valid_claims.append(ExtractedClaim(
            claim_text=claim_text,
            claim_type=claim_type,
            referenced_symbol_guess=guess
        ))
        
    return valid_claims, validation_errors

def extract_claims_for_chunk(doc_chunk_row, model_name: str, host: str) -> Tuple[List[ExtractedClaim], bool, bool]:
    # Support both sqlite3.Row / tuple and dict for easier testing
    try:
        chunk_id = doc_chunk_row['id']
        heading_path = doc_chunk_row['heading'] or ""
        doc_chunk_text = doc_chunk_row['text'] or ""
    except (TypeError, KeyError, IndexError):
        chunk_id = doc_chunk_row[0]
        heading_path = doc_chunk_row[3] if len(doc_chunk_row) > 3 else (doc_chunk_row[1] if len(doc_chunk_row) == 3 else "")
        doc_chunk_text = doc_chunk_row[4] if len(doc_chunk_row) > 4 else (doc_chunk_row[2] if len(doc_chunk_row) == 3 else "")

    def attempt_extraction(prompt_modifier: str = "") -> Tuple[List[ExtractedClaim], List[str], bool]:
        try:
            raw_response = extract_claims_raw(doc_chunk_text, heading_path, model_name, host, retry_message=prompt_modifier)
        except LLMCallError as e:
            logger.error(f"Error: LLM call failed for chunk id {chunk_id}: {str(e)}")
            return [], [], True # hard failure
            
        valid_claims, validation_errors = parse_and_validate_claims(raw_response)
        return valid_claims, validation_errors, False
        
    valid_claims, validation_errors, is_llm_error = attempt_extraction()
    if is_llm_error:
        return [], False, True
        
    if len(valid_claims) == 0 and len(validation_errors) > 0:
        # Retry exactly once
        valid_claims, validation_errors, is_llm_error = attempt_extraction(
            "Your previous response was not valid JSON. Respond with ONLY a JSON array, nothing else."
        )
        if is_llm_error:
            return [], False, True
            
        if len(valid_claims) == 0 and len(validation_errors) > 0:
            logger.warning(f"Warning: chunk id {chunk_id} failed claim extraction after retry, skipping")
            return [], True, False
            
    return valid_claims, False, False

def store_claims(conn: sqlite3.Connection, doc_chunk_id: int, claims: List[ExtractedClaim]) -> int:
    with conn:
        conn.execute("DELETE FROM claims WHERE doc_chunk_id = ?", (doc_chunk_id,))
        for claim in claims:
            conn.execute(
                '''
                INSERT INTO claims (doc_chunk_id, claim_text, claim_type, referenced_symbol_guess)
                VALUES (?, ?, ?, ?)
                ''',
                (doc_chunk_id, claim.claim_text, claim.claim_type, claim.referenced_symbol_guess)
            )
    return len(claims)
