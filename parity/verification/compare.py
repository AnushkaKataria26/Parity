from typing import Any, Dict, List, Optional

def normalize_value(raw: str) -> str:
    """
    Normalize a string value for comparison.
    - lowercase
    - strip whitespace
    - strip matching quote characters
    - map boolean variants
    - float normalization for numerics
    """
    if not isinstance(raw, str):
        raw = str(raw)
        
    val = raw.strip()
    
    # Strip matching quotes
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1].strip()
        
    val = val.lower()
    
    # Booleans
    if val in ("true", "1"):
        return "true"
    if val in ("false", "0"):
        return "false"
        
    # None canonicalization
    if val in ("none", "null", "nil"):
        return "none"
        
    # Numerics
    try:
        fval = float(val)
        return str(fval)
    except ValueError:
        pass
        
    return val

def compare_default_values(claimed: str, actual_repr: str) -> bool:
    """
    Normalize both sides and compare for equality.
    """
    if claimed is None or actual_repr is None:
        return claimed == actual_repr
        
    norm_claimed = normalize_value(claimed)
    norm_actual = normalize_value(actual_repr)
    
    return norm_claimed == norm_actual

def find_param(parameters: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """
    Case-insensitive lookup by name field within the parameters list.
    """
    if not parameters:
        return None
        
    target = name.lower()
    for p in parameters:
        if p["name"].lower() == target:
            return p
            
    return None
