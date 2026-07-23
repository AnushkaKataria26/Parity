import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from parity.verification.claim_value_extraction import extract_claimed_value
from parity.verification.resolve import ResolvedSymbol, module_path_from_file, resolve_symbol_dynamic, resolve_symbol_static
from parity.verification.compare import compare_default_values, find_param, normalize_value


@dataclass
class VerificationOutcome:
    status: str            # "Verified" | "Contradicted" | "Unverifiable"
    actual_value: Optional[str]
    claimed_value: Optional[str]


def verify_signature_claim(claimed: dict, resolved: ResolvedSymbol) -> VerificationOutcome:
    param_names = claimed.get("param_names")
    if not param_names:
        return VerificationOutcome("Unverifiable", None, None)
        
    actual_names = {p["name"].lower() for p in (resolved.parameters or [])}
    has_var_keyword = any(p["kind"] == "VAR_KEYWORD" for p in (resolved.parameters or []))
    
    contradicted = False
    unverifiable_due_to_kwargs = False
    
    for c_name in param_names:
        if c_name.lower() not in actual_names:
            if has_var_keyword:
                unverifiable_due_to_kwargs = True
            else:
                contradicted = True
                
    actual_val_str = ", ".join(sorted([p["name"] for p in (resolved.parameters or [])]))
    claimed_val_str = ", ".join(param_names)
    
    if contradicted:
        return VerificationOutcome("Contradicted", actual_val_str, claimed_val_str)
    elif unverifiable_due_to_kwargs:
        # Conservative choice to avoid false positives on flexible-signature functions
        return VerificationOutcome("Unverifiable", actual_val_str, claimed_val_str)
    else:
        return VerificationOutcome("Verified", actual_val_str, claimed_val_str)


def verify_default_value_claim(claimed: dict, resolved: ResolvedSymbol) -> VerificationOutcome:
    param_name = claimed.get("param_name")
    default_val = claimed.get("default")
    
    if not param_name or default_val is None:
        return VerificationOutcome("Unverifiable", None, None)
        
    param = find_param(resolved.parameters or [], param_name)
    has_var_keyword = any(p["kind"] == "VAR_KEYWORD" for p in (resolved.parameters or []))
    
    if param is None:
        if has_var_keyword:
            return VerificationOutcome("Unverifiable", "parameter not found", default_val)
        else:
            return VerificationOutcome("Contradicted", "parameter not found", default_val)
            
    if not param.get("has_default"):
        return VerificationOutcome("Contradicted", "no default", default_val)
        
    if not param.get("is_literal", False):
        return VerificationOutcome("Unverifiable", param.get("default_repr"), default_val)
        
    if compare_default_values(default_val, param.get("default_repr")):
        return VerificationOutcome("Verified", param.get("default_repr"), default_val)
    else:
        return VerificationOutcome("Contradicted", param.get("default_repr"), default_val)


def _scan_file_for_env_var(file_path: str, var_name: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False
        
    # case-sensitive on the var name itself
    patterns = [
        rf'os\.environ\.get\(\s*[\'"]{re.escape(var_name)}[\'"]',
        rf'os\.getenv\(\s*[\'"]{re.escape(var_name)}[\'"]',
        rf'os\.environ\[\s*[\'"]{re.escape(var_name)}[\'"]\s*\]'
    ]
    
    for pat in patterns:
        if re.search(pat, content):
            return True
            
    return False


def verify_env_var_claim(claimed: dict, repo_root: str, matched_file_path: str) -> VerificationOutcome:
    var_name = claimed.get("var_name")
    if not var_name:
        return VerificationOutcome("Unverifiable", None, None)
        
    # 1. Check matched file
    full_path = os.path.join(repo_root, matched_file_path)
    if _scan_file_for_env_var(full_path, var_name):
        return VerificationOutcome("Verified", f"{var_name} referenced in {matched_file_path}", var_name)
        
    # 2. Widen search to every .py file
    # Instead of importing Phase 1, we can just os.walk to keep it simple and independent
    for root, dirs, files in os.walk(repo_root):
        # Exclude hidden dirs and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', 'env', '__pycache__')]
        for file in files:
            if file.endswith('.py'):
                fp = os.path.join(root, file)
                if fp != full_path: # already checked
                    if _scan_file_for_env_var(fp, var_name):
                        rel_path = os.path.relpath(fp, repo_root)
                        return VerificationOutcome("Verified", f"{var_name} referenced in {rel_path}", var_name)
                        
    # Not found anywhere. Explicitly Unverifiable, not Contradicted.
    # Regex-based absence is not strong enough evidence to assert a contradiction.
    return VerificationOutcome("Unverifiable", "not found via static scan", var_name)


def verify_return_type_claim(claimed: dict, resolved: ResolvedSymbol) -> VerificationOutcome:
    return_type = claimed.get("return_type")
    if not return_type:
        return VerificationOutcome("Unverifiable", None, None)
        
    if not resolved.return_annotation:
        return VerificationOutcome("Unverifiable", "no return annotation", return_type)
        
    norm_claimed = normalize_value(return_type)
    norm_actual = normalize_value(resolved.return_annotation)
    
    if norm_claimed in norm_actual:
        return VerificationOutcome("Verified", resolved.return_annotation, return_type)
    else:
        return VerificationOutcome("Contradicted", resolved.return_annotation, return_type)


def verify_repo(conn, repo_id: int, repo_root: str, model_name: str, host: str) -> dict:
    # 1. Modify sys.path
    path_inserted = False
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        path_inserted = True
        
    try:
        module_cache = {}
        
        # 3. Query all claims + retrieval_results + code_chunks
        # Wait! It's a left join to code_chunks, since behavior claims or no_match claims might have matched_code_chunk_id=NULL
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                c.id as claim_id, c.claim_text, c.claim_type,
                r.match_status, r.matched_code_chunk_id,
                cc.file_path, cc.symbol_name, cc.symbol_type
            FROM claims c
            JOIN doc_chunks d ON c.doc_chunk_id = d.id
            LEFT JOIN retrieval_results r ON r.claim_id = c.id
            LEFT JOIN code_chunks cc ON r.matched_code_chunk_id = cc.id
            WHERE d.repo_id = ?
        """, (repo_id,))
        
        rows = cursor.fetchall()
        
        stats = {
            "total": len(rows),
            "verified": 0,
            "contradicted": 0,
            "unverifiable": 0,
            "dynamic_resolutions": 0,
            "static_resolutions": 0,
            "resolution_failures": 0
        }
        
        # Prepare for bulk insert
        cursor.execute("""
            DELETE FROM verification_results 
            WHERE claim_id IN (
                SELECT c.id FROM claims c 
                JOIN doc_chunks d ON c.doc_chunk_id = d.id 
                WHERE d.repo_id = ?
            )
        """, (repo_id,))
        
        insert_rows = []
        now_str = datetime.now(timezone.utc).isoformat()
        
        for row in rows:
            claim_id = row[0]
            claim_text = row[1]
            claim_type = row[2]
            match_status = row[3]
            matched_chunk_id = row[4]
            file_path = row[5]
            symbol_name = row[6]
            symbol_type = row[7]
            
            outcome = None
            
            if claim_type == "behavior":
                # Architecturally unverifiable by design
                outcome = VerificationOutcome("Unverifiable", None, None)
            elif match_status != "matched" or matched_chunk_id is None:
                # No code target to check against
                outcome = VerificationOutcome("Unverifiable", None, None)
            else:
                # 4. Extract claimed value
                claimed_dict = extract_claimed_value(claim_text, claim_type, model_name, host)
                if not claimed_dict:
                    outcome = VerificationOutcome("Unverifiable", None, None)
                else:
                    resolved = None
                    mod_path = module_path_from_file(file_path, repo_root)
                    
                    if mod_path is not None:
                        resolved = resolve_symbol_dynamic(mod_path, symbol_name, repo_root, module_cache)
                        if resolved:
                            stats["dynamic_resolutions"] += 1
                            
                    if resolved is None:
                        resolved = resolve_symbol_static(matched_chunk_id, repo_id, symbol_name, symbol_type)
                        if resolved and resolved.resolution_method == "static":
                            stats["static_resolutions"] += 1
                            
                    if resolved is None or resolved.resolution_method == "failed":
                        stats["resolution_failures"] += 1
                        outcome = VerificationOutcome("Unverifiable", "could not resolve symbol", None)
                    else:
                        # 5. Dispatch
                        if claim_type == "signature":
                            outcome = verify_signature_claim(claimed_dict, resolved)
                        elif claim_type == "default_value":
                            outcome = verify_default_value_claim(claimed_dict, resolved)
                        elif claim_type == "env_var":
                            outcome = verify_env_var_claim(claimed_dict, repo_root, file_path)
                        elif claim_type == "return_type":
                            outcome = verify_return_type_claim(claimed_dict, resolved)
                        else:
                            outcome = VerificationOutcome("Unverifiable", None, None)
                            
            if outcome.status == "Verified":
                stats["verified"] += 1
            elif outcome.status == "Contradicted":
                stats["contradicted"] += 1
            else:
                stats["unverifiable"] += 1
                
            insert_rows.append((
                claim_id, 
                matched_chunk_id, 
                outcome.status, 
                outcome.actual_value, 
                outcome.claimed_value, 
                now_str
            ))
            
        cursor.executemany("""
            INSERT INTO verification_results (claim_id, matched_code_chunk_id, status, actual_value, claimed_value, verified_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, insert_rows)
        conn.commit()
        
        return stats
        
    finally:
        if path_inserted:
            sys.path.remove(repo_root)
