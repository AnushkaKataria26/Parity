import argparse
import sys
import os
import subprocess

from parity.config import load_config
from parity.db.connection import get_connection
from parity.db.migrate import apply_schema, upsert_repo
from parity.vectorstore.chroma_client import get_chroma_client, get_or_create_collections
from parity.llm.ollama_client import check_ollama_reachable, check_model_available
from parity.chunking.ast_chunker import discover_python_files, extract_chunks_from_file
from parity.chunking.doc_chunker import discover_doc_files, extract_chunks_from_markdown, extract_chunks_from_rst
from parity.db.chunk_ops import store_chunks, store_doc_chunks
from parity.embedding.model import embed_repo

def cmd_embed(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    conn = get_connection(config["db_path"])
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: repo '{repo_path}' not initialized — run 'init' first", file=sys.stderr)
        sys.exit(1)
        
    repo_id = row[0]
    
    # Sanity-check chunks exist
    cursor = conn.execute("SELECT COUNT(*) FROM code_chunks WHERE repo_id = ?", (repo_id,))
    code_count = cursor.fetchone()[0]
    if code_count == 0:
        print(f"Warning: code_chunks is empty for this repo — run chunk-code first", file=sys.stderr)
        
    cursor = conn.execute("SELECT COUNT(*) FROM doc_chunks WHERE repo_id = ?", (repo_id,))
    doc_count = cursor.fetchone()[0]
    if doc_count == 0:
        print(f"Warning: doc_chunks is empty for this repo — run chunk-docs first", file=sys.stderr)
        
    # Get chroma client and collections
    client = get_chroma_client(config["chroma_persist_dir"])
    code_col, doc_col = get_or_create_collections(client)
    
    # Embed
    summary = embed_repo(conn, repo_id, client, code_col, doc_col)
    
    print(f"Parity embed summary for {repo_path}")
    print(f"  Code chunks embedded:  {summary['code_chunks_embedded']}  ({summary['code_fallback_count']} used fallback text)")
    print(f"  Doc chunks embedded:   {summary['doc_chunks_embedded']}  ({summary['doc_fallback_count']} used fallback text)")
    print(f"  Model: BAAI/bge-small-en-v1.5")
    sys.exit(0)

def cmd_init(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"Error: '{repo_path}' is not a git repository (no .git directory found)", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    
    conn = get_connection(config["db_path"])
    apply_schema(conn)
    
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True)
    if result.returncode == 0:
        commit_sha = result.stdout.strip()
    else:
        commit_sha = None
        print("Warning: repo has no commits yet; last_ingested_commit_sha will be null", file=sys.stderr)
        
    repo_id = upsert_repo(conn, name=os.path.basename(repo_path), path=repo_path, commit_sha=commit_sha)
    
    client = get_chroma_client(config["chroma_persist_dir"])
    code_col, doc_col = get_or_create_collections(client)
    
    ollama_ok = check_ollama_reachable(config["ollama_host"])
    if ollama_ok:
        model_ok = check_model_available(config["ollama_model"], config["ollama_host"])
    else:
        model_ok = False
        
    print(f"Parity init summary for {repo_path}")
    print(f"  Database:      OK ({config['db_path']})")
    print(f"  Vector store:  OK ({config['chroma_persist_dir']}), collections: code_chunks, doc_chunks")
    
    sha_display = commit_sha[:8] if commit_sha else "no commits yet"
    print(f"  Git HEAD:      {sha_display}")
    
    ollama_display = f"OK ({config['ollama_host']})" if ollama_ok else f"NOT REACHABLE at {config['ollama_host']}"
    print(f"  Ollama:        {ollama_display}")
    
    model_display = f"OK ({config['ollama_model']})" if model_ok else f"NOT PULLED — run `ollama pull {config['ollama_model']}`"
    print(f"  Model:         {model_display}")

def cmd_chunk_code(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    conn = get_connection(config["db_path"])
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: repo '{repo_path}' not initialized — run 'init' first", file=sys.stderr)
        sys.exit(1)
        
    repo_id = row[0]
    
    python_files = discover_python_files(repo_path)
    
    all_chunks = []
    files_scanned = 0
    files_skipped = 0
    
    for file_path in python_files:
        files_scanned += 1
        chunks, is_skipped = extract_chunks_from_file(file_path, repo_path)
        if is_skipped:
            files_skipped += 1
        all_chunks.extend(chunks)
        
    store_chunks(conn, repo_id, all_chunks)
    
    total_chunks = len(all_chunks)
    functions = sum(1 for c in all_chunks if c.symbol_type in ("function", "async_function"))
    methods = sum(1 for c in all_chunks if c.symbol_type in ("method", "async_method"))
    classes = sum(1 for c in all_chunks if c.symbol_type == "class")
    
    print(f"Parity chunk-code summary for {repo_path}")
    print(f"  Files scanned:     {files_scanned}")
    print(f"  Files skipped:     {files_skipped} (syntax errors)")
    print(f"  Chunks extracted:  {total_chunks}")
    print(f"    functions:  {functions}")
    print(f"    methods:    {methods}")
    print(f"    classes:    {classes}")
    print(f"  Chunk bodies written to: data/code_chunk_bodies/{repo_id}/")

def cmd_chunk_docs(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    conn = get_connection(config["db_path"])
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: repo '{repo_path}' not initialized — run 'init' first", file=sys.stderr)
        sys.exit(1)
        
    repo_id = row[0]
    
    doc_files = discover_doc_files(repo_path)
    if not doc_files:
        print(f"Warning: no documentation files found in '{repo_path}'", file=sys.stderr)
    
    all_chunks = []
    
    for file_path in doc_files:
        if file_path.lower().endswith('.rst'):
            chunks = extract_chunks_from_rst(file_path, repo_path)
        else:
            chunks = extract_chunks_from_markdown(file_path, repo_path)
        all_chunks.extend(chunks)
        
    store_doc_chunks(conn, repo_id, all_chunks)
    
    total_chunks = len(all_chunks)
    with_headings = sum(1 for c in all_chunks if c.heading_level > 0)
    preamble_only = sum(1 for c in all_chunks if c.heading_level == 0)
    empty_sections = sum(1 for c in all_chunks if not c.text)
    total_code_blocks = sum(len(c.code_blocks) for c in all_chunks)
    
    print(f"Parity chunk-docs summary for {repo_path}")
    print(f"  Doc files scanned:   {len(doc_files)}")
    print(f"  Chunks extracted:    {total_chunks}")
    print(f"    with headings:  {with_headings}")
    print(f"    preamble-only:  {preamble_only}")
    print(f"    empty sections: {empty_sections}")
    print(f"  Code blocks extracted: {total_code_blocks}")
    print(f"  Chunk bodies written to: data/doc_chunk_bodies/{repo_id}/")

def main():
    parser = argparse.ArgumentParser(prog="parity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("repo_path")
    init_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    
    chunk_parser = subparsers.add_parser("chunk-code")
    chunk_parser.add_argument("repo_path")
    chunk_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    
    doc_chunk_parser = subparsers.add_parser("chunk-docs")
    doc_chunk_parser.add_argument("repo_path")
    doc_chunk_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    
    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("repo_path")
    embed_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    
    args = parser.parse_args()
    
    if args.command == "init":
        cmd_init(args)
    elif args.command == "chunk-code":
        cmd_chunk_code(args)
    elif args.command == "chunk-docs":
        cmd_chunk_docs(args)
    elif args.command == "embed":
        cmd_embed(args)

if __name__ == "__main__":
    main()
