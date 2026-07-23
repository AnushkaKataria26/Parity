import ast
import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ResolvedSymbol:
    resolution_method: str        # "dynamic" | "static" | "failed"
    parameters: Optional[List[Dict[str, Any]]] # [{"name": ..., "kind": ..., "has_default": bool, "default_repr": str|None, "is_literal": bool}, ...]
    return_annotation: Optional[str]
    source_available: bool        # whether source_text was available for env-var scanning fallback


def module_path_from_file(file_path: str, repo_root: str) -> Optional[str]:
    """
    Convert a relative file path to a dotted module path.
    Example: "pkg/sub/mod.py" -> "pkg.sub.mod"
    "pkg/sub/__init__.py" -> "pkg.sub"
    Returns None if not ending in .py
    """
    if not file_path.endswith(".py"):
        return None
    
    # Remove .py
    clean_path = file_path[:-3]
    
    parts = clean_path.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
        
    if not parts:
        return "" # Root __init__.py? Edge case
        
    return ".".join(parts)


def resolve_symbol_dynamic(module_dotted_path: str, symbol_name: str, repo_root: str, module_cache: dict) -> Optional[ResolvedSymbol]:
    """
    Dynamically resolve a symbol via importlib and inspect.
    """
    # 1. Use cache or import
    if module_dotted_path in module_cache:
        mod = module_cache[module_dotted_path]
    else:
        try:
            # Broad except because executing arbitrary third-party code can raise anything
            mod = importlib.import_module(module_dotted_path)
            module_cache[module_dotted_path] = mod
        except Exception as e:
            logging.warning(f"Warning: failed to import module '{module_dotted_path}': {e}")
            module_cache[module_dotted_path] = None
            mod = None
            
    if mod is None:
        return None
        
    # 2. Traverse dotted symbol name
    obj = mod
    for part in symbol_name.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            # Note: Nested functions fail here intentionally as they aren't module attributes.
            return None
            
    # 3. Inspect signature
    try:
        sig = inspect.signature(obj, follow_wrapped=True)
    except (ValueError, TypeError):
        # Fallback for classes to __init__
        if inspect.isclass(obj) and hasattr(obj, "__init__"):
            try:
                sig = inspect.signature(obj.__init__, follow_wrapped=True)
            except (ValueError, TypeError):
                return None
        else:
            return None
            
    # 4. Build parameters list
    parameters = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        # Drop first parameter if named 'self' or 'cls'
        if i == 0 and name in ("self", "cls"):
            continue
            
        has_default = p.default is not inspect.Parameter.empty
        default_repr = repr(p.default) if has_default else None
        
        parameters.append({
            "name": name,
            "kind": str(p.kind),
            "has_default": has_default,
            "default_repr": default_repr,
            "is_literal": True # Dynamic resolution evaluates the default, so it's literal/exact for our comparison
        })
        
    # 5. Return annotation
    return_annotation = None
    if sig.return_annotation is not inspect.Signature.empty:
        return_annotation = str(sig.return_annotation)
        
    return ResolvedSymbol(
        resolution_method="dynamic",
        parameters=parameters,
        return_annotation=return_annotation,
        source_available=True
    )


def resolve_symbol_static(chunk_id: int, repo_id: int, symbol_name: str, symbol_type: str) -> Optional[ResolvedSymbol]:
    """
    Statically resolve a symbol via AST parsing of the saved chunk body.
    """
    body_path = f"data/code_chunk_bodies/{repo_id}/{chunk_id}.json"
    if not os.path.exists(body_path):
        return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        
    try:
        with open(body_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            source_text = data.get("text", "")
    except Exception:
        return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        
    if not tree.body:
        return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        
    node = tree.body[0]
    
    if symbol_type == "class":
        if not isinstance(node, ast.ClassDef):
            return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
            
        # Find __init__
        init_node = None
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                init_node = child
                break
        
        if not init_node:
            # Bare class, no explicit constructor claims can match against
            return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        node = init_node
        
    if getattr(node, "args", None) is None:
        return ResolvedSymbol(resolution_method="failed", parameters=None, return_annotation=None, source_available=False)
        
    args_node = node.args
    parameters = []
    
    # To map defaults correctly:
    # args_node.defaults applies to the LAST N elements of (posonlyargs + args)
    pos_args = getattr(args_node, 'posonlyargs', []) + getattr(args_node, 'args', [])
    num_pos_defaults = len(args_node.defaults) if getattr(args_node, 'defaults', None) else 0
    pos_default_offset = len(pos_args) - num_pos_defaults
    
    def add_param(name, kind, default_node):
        has_default = default_node is not None
        default_repr = None
        is_literal = False
        
        if has_default:
            try:
                # Literal evaluation
                val = ast.literal_eval(default_node)
                default_repr = repr(val)
                is_literal = True
            except Exception:
                # Non-literal expression (e.g. logging.INFO, Timeout.DEFAULT)
                default_repr = ast.unparse(default_node)
                is_literal = False
                
        parameters.append({
            "name": name,
            "kind": kind,
            "has_default": has_default,
            "default_repr": default_repr,
            "is_literal": is_literal
        })

    # 1. posonlyargs
    for i, a in enumerate(getattr(args_node, 'posonlyargs', [])):
        default = args_node.defaults[i - pos_default_offset] if i >= pos_default_offset else None
        add_param(a.arg, "POSITIONAL_ONLY", default)
        
    # 2. args
    posonly_len = len(getattr(args_node, 'posonlyargs', []))
    for i, a in enumerate(getattr(args_node, 'args', [])):
        total_idx = posonly_len + i
        default = args_node.defaults[total_idx - pos_default_offset] if total_idx >= pos_default_offset else None
        add_param(a.arg, "POSITIONAL_OR_KEYWORD", default)
        
    # 3. vararg
    if getattr(args_node, 'vararg', None):
        add_param(args_node.vararg.arg, "VAR_POSITIONAL", None)
        
    # 4. kwonlyargs
    kw_defaults = getattr(args_node, 'kw_defaults', [])
    for i, a in enumerate(getattr(args_node, 'kwonlyargs', [])):
        default = kw_defaults[i] if i < len(kw_defaults) else None
        add_param(a.arg, "KEYWORD_ONLY", default)
        
    # 5. kwarg
    if getattr(args_node, 'kwarg', None):
        add_param(args_node.kwarg.arg, "VAR_KEYWORD", None)
        
    # Strip self/cls
    if parameters and parameters[0]["name"] in ("self", "cls"):
        parameters = parameters[1:]
        
    # Return annotation
    return_annotation = None
    if getattr(node, "returns", None):
        try:
            return_annotation = ast.unparse(node.returns)
        except Exception:
            pass
            
    return ResolvedSymbol(
        resolution_method="static",
        parameters=parameters,
        return_annotation=return_annotation,
        source_available=True
    )
