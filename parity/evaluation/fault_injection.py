import os
import random
import re
import uuid
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from parity.verification.resolve import resolve_symbol_static

@dataclass
class InjectedFault:
    fault_id: str
    file_path: str
    symbol_name: str
    fault_type: str            # "rename_param" | "change_default" | "remove_param"
    original_snippet: str
    mutated_snippet: str
    target_claim_hint: str     # human-readable description for the eval report
    # Extra fields not in prompt but helpful internally:
    start_line: int = 0
    end_line: int = 0


def select_fault_targets(conn, repo_id: int, n: int) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    query = """
        SELECT c.id, c.claim_type, c.claim_text, cc.id, cc.file_path, cc.symbol_name, cc.symbol_type, cc.start_line, cc.end_line
        FROM claims c
        JOIN retrieval_results rr ON c.id = rr.claim_id
        JOIN code_chunks cc ON rr.matched_code_chunk_id = cc.id
        WHERE cc.repo_id = ?
          AND rr.match_status = 'matched'
          AND c.claim_type != 'behavior'
    """
    cursor.execute(query, (repo_id,))
    rows = cursor.fetchall()
    
    # Prefer signature and default_value
    preferred_targets = []
    other_targets = []
    
    for row in rows:
        target = {
            "claim_id": row[0],
            "claim_type": row[1],
            "claim_text": row[2],
            "chunk_id": row[3],
            "file_path": row[4],
            "symbol_name": row[5],
            "symbol_type": row[6],
            "start_line": row[7],
            "end_line": row[8]
        }
        if target["claim_type"] in ("signature", "default_value"):
            preferred_targets.append(target)
        else:
            other_targets.append(target)
            
    # Group by chunk_id to avoid hitting the same function twice
    def group_by_chunk(targets_list):
        chunk_map = {}
        for t in targets_list:
            if t["chunk_id"] not in chunk_map:
                chunk_map[t["chunk_id"]] = t
        return list(chunk_map.values())
        
    eligible_preferred = group_by_chunk(preferred_targets)
    eligible_other = group_by_chunk(other_targets)
    
    random.seed(42)
    selected = []
    
    if len(eligible_preferred) >= n:
        selected = random.sample(eligible_preferred, n)
    else:
        selected = eligible_preferred
        needed = n - len(selected)
        if len(eligible_other) >= needed:
            selected += random.sample(eligible_other, needed)
        else:
            selected += eligible_other
            
    if len(selected) < n:
        logging.warning(f"Requested {n} fault targets, but only {len(selected)} eligible targets exist.")
        
    return selected


def inject_faults(conn, repo_id: int, repo_copy_path: str, targets: List[Dict[str, Any]], rng_seed: int = 42) -> List[InjectedFault]:
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM repos WHERE id = ?", (repo_id,))
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Repo {repo_id} not found")
    orig_path = row[0]
    
    if os.path.abspath(repo_copy_path) == os.path.abspath(orig_path):
        raise ValueError("repo_copy_path must never be the original target repo path")

    random.seed(rng_seed)
    
    faults = []
    
    for t in targets:
        chunk_id = t["chunk_id"]
        symbol_name = t["symbol_name"]
        symbol_type = t["symbol_type"]
        file_path = t["file_path"]
        full_copy_path = os.path.join(repo_copy_path, file_path)
        
        # Resolve symbol static on the original repo to get params
        resolved = resolve_symbol_static(chunk_id, repo_id, symbol_name, symbol_type)
        if not resolved or not resolved.parameters:
            continue
            
        params = resolved.parameters
        
        # Try to find a param with literal default for change_default
        literal_params = [p for p in params if p["has_default"] and p["is_literal"]]
        
        # Randomly choose fault type
        fault_type = random.choice(["rename_param", "change_default"])
        if fault_type == "change_default" and not literal_params:
            fault_type = "rename_param"
            
        fault = None
        if fault_type == "rename_param":
            p = random.choice(params)
            old_name = p["name"]
            new_name = f"{old_name}_mutated"
            
            # Read chunk lines
            with open(full_copy_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            start = t["start_line"] - 1
            end = t["end_line"]
            
            original_snippet = "".join(lines[start:end])
            
            # Whole word regex replace for renaming param within the start-end range
            # Note: naive replace might hit keywords or other valid substrings, \b ensures word boundary
            mutated_snippet = re.sub(rf"\b{re.escape(old_name)}\b", new_name, original_snippet)
            
            if original_snippet != mutated_snippet:
                fault = InjectedFault(
                    fault_id=str(uuid.uuid4()),
                    file_path=file_path,
                    symbol_name=symbol_name,
                    fault_type="rename_param",
                    original_snippet=original_snippet,
                    mutated_snippet=mutated_snippet,
                    target_claim_hint=f"Renamed parameter '{old_name}' to '{new_name}'",
                    start_line=t["start_line"],
                    end_line=t["end_line"]
                )
                
        elif fault_type == "change_default":
            p = random.choice(literal_params)
            old_repr = p["default_repr"]
            
            # Change literal: int -> int, bool -> bool, str -> str
            new_repr = old_repr
            if old_repr in ("True", "False"):
                new_repr = "False" if old_repr == "True" else "True"
            elif old_repr.isdigit():
                new_repr = str(int(old_repr) + 42)
            elif old_repr.startswith("'") or old_repr.startswith('"'):
                new_repr = old_repr[:-1] + "_mutated" + old_repr[-1]
            else:
                new_repr = "None"
                
            # String replacement in the function definition lines
            with open(full_copy_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            start = t["start_line"] - 1
            end = t["end_line"]
            
            original_snippet = "".join(lines[start:end])
            # Replace exactly once to avoid matching wrong things. Wait, what if old_repr is just "0"?
            # Safer to replace 'old_name=old_repr' or 'old_name = old_repr'
            
            # Using regex to match param default assignment
            pattern = rf"({re.escape(p['name'])}\s*:\s*[^=]+=\s*|{re.escape(p['name'])}\s*=\s*){re.escape(old_repr)}"
            
            match = re.search(pattern, original_snippet)
            if match:
                mutated_snippet = original_snippet[:match.start()] + match.group(1) + new_repr + original_snippet[match.end():]
                fault = InjectedFault(
                    fault_id=str(uuid.uuid4()),
                    file_path=file_path,
                    symbol_name=symbol_name,
                    fault_type="change_default",
                    original_snippet=original_snippet,
                    mutated_snippet=mutated_snippet,
                    target_claim_hint=f"Changed default of '{p['name']}' from {old_repr} to {new_repr}",
                    start_line=t["start_line"],
                    end_line=t["end_line"]
                )
        
        if fault:
            apply_fault_to_file(full_copy_path, fault)
            faults.append(fault)
            
    return faults


def apply_fault_to_file(file_path: str, fault: InjectedFault) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start = fault.start_line - 1
    end = fault.end_line
    
    # We replace the entire chunk with the mutated snippet (which is the joined chunk)
    mutated_lines = fault.mutated_snippet.splitlines(True)
    
    new_lines = lines[:start] + mutated_lines + lines[end:]
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
