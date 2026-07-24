import os
import ast
import hashlib
import fnmatch
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class CodeChunk:
    file_path: str          # relative to repo root
    symbol_name: str        # dotted path, e.g. "MyClass.my_method"
    symbol_type: str        # "function" | "async_function" | "class" | "method" | "async_method"
    start_line: int
    end_line: int
    source_text: str        # exact source slice, including decorators
    docstring: Optional[str]
    ast_hash: str

from parity.chunking.common import EXCLUDED_DIRS

def discover_python_files(repo_path: str) -> List[str]:
    """
    Walk repo_path recursively and discover Python files.
    Excludes test files and specific directories.
    """
    
    # We deliberately exclude tests because they document testing behavior,
    # not the public API that docs claim things about.
    
    discovered_files = []
    
    # os.walk by default has followlinks=False, which is what we want
    for root, dirs, files in os.walk(repo_path):
        # Filter directories in place so os.walk doesn't traverse them
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not fnmatch.fnmatch(d, '*.egg-info')]
        
        path_parts = set(root.split(os.sep))
        if any(p in {'test', 'tests', 'testing'} for p in path_parts):
            continue
            
        for file in files:
            if file.endswith('.py'):
                if file.startswith('test_') or file.endswith('_test.py'):
                    continue
                full_path = os.path.abspath(os.path.join(root, file))
                discovered_files.append(full_path)
                
    return sorted(discovered_files)

def _get_end_lineno(node: ast.AST) -> int:
    if hasattr(node, 'end_lineno') and node.end_lineno is not None:
        return node.end_lineno
    
    max_line = node.lineno
    for child in ast.walk(node):
        if hasattr(child, 'lineno') and child.lineno is not None:
            max_line = max(max_line, child.lineno)
        if hasattr(child, 'end_lineno') and child.end_lineno is not None:
            max_line = max(max_line, child.end_lineno)
    return max_line

def extract_chunks_from_file(file_path: str, repo_root: str) -> Tuple[List[CodeChunk], bool]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Warning: {file_path} has encoding issues, some characters replaced")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        print(f"Warning: skipping {file_path}, failed to parse: {e}")
        return [], True


    lines = content.splitlines(keepends=True)
    chunks = []
    symbol_counts = {}
    
    rel_path = os.path.relpath(file_path, repo_root).replace(os.sep, '/')

    def get_disambiguated_name(name: str) -> str:
        count = symbol_counts.get(name, 0)
        symbol_counts[name] = count + 1
        if count == 0:
            return name
        return f"{name}#{count + 1}"

    def process_node(node: ast.AST, parent_prefix: str = "", is_class_context: bool = False):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Determine base symbol name
            base_name = node.name
            if parent_prefix:
                raw_symbol_name = f"{parent_prefix}.{base_name}"
            else:
                raw_symbol_name = base_name
                
            symbol_name = get_disambiguated_name(raw_symbol_name)
            
            # Determine symbol type
            if isinstance(node, ast.ClassDef):
                symbol_type = "class"
            elif isinstance(node, ast.AsyncFunctionDef):
                symbol_type = "async_method" if is_class_context else "async_function"
            else:
                symbol_type = "method" if is_class_context else "function"

            start_line = node.lineno
            if getattr(node, 'decorator_list', None):
                start_line = min(dec.lineno for dec in node.decorator_list if hasattr(dec, 'lineno')) if node.decorator_list else node.lineno
                
            end_line = _get_end_lineno(node)
            
            source_text = "".join(lines[start_line - 1:end_line])
            docstring = ast.get_docstring(node, clean=True)
            
            # ast.dump without line/col attributes means hash reflects structural content only.
            # This makes the hash invariant to simple formatting changes that do not alter the AST structure,
            # which is needed for cache-invalidation semantics in Phase 8.
            ast_hash = hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()
            
            chunks.append(CodeChunk(
                file_path=rel_path,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                start_line=start_line,
                end_line=end_line,
                source_text=source_text,
                docstring=docstring,
                ast_hash=ast_hash
            ))
            
            for child in getattr(node, 'body', []):
                process_node(child, raw_symbol_name, is_class_context=isinstance(node, ast.ClassDef))
        else:
            # Traverse other nodes (like If, Try, Module, etc.) for nested defs
            for child in ast.iter_child_nodes(node):
                process_node(child, parent_prefix, is_class_context)

    process_node(tree)

    return chunks, False
