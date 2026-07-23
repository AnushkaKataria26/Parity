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

def cmd_extract_claims(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    
    ollama_ok = check_ollama_reachable(config["ollama_host"])
    if not ollama_ok:
        print(f"Error: Ollama not reachable or model '{config['ollama_model']}' not available — run 'ollama pull {config['ollama_model']}' and ensure 'ollama serve' is running", file=sys.stderr)
        sys.exit(1)
        
    model_ok = check_model_available(config["ollama_model"], config["ollama_host"])
    if not model_ok:
        print(f"Error: Ollama not reachable or model '{config['ollama_model']}' not available — run 'ollama pull {config['ollama_model']}' and ensure 'ollama serve' is running", file=sys.stderr)
        sys.exit(1)

    conn = get_connection(config["db_path"])
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: repo '{repo_path}' not initialized — run 'init' first", file=sys.stderr)
        sys.exit(1)
        
    repo_id = row[0]
    
    query = "SELECT id, file_path, heading, text FROM doc_chunks WHERE repo_id = ? ORDER BY id"
    if args.limit:
        query += f" LIMIT {args.limit}"
        
    cursor = conn.execute(query, (repo_id,))
    doc_chunks = cursor.fetchall()
    
    if not doc_chunks:
        print(f"Warning: no doc chunks found — run chunk-docs first", file=sys.stderr)
        sys.exit(0)
        
    from parity.extraction.extractor import extract_claims_for_chunk, store_claims
    from parity.extraction.prompts import CLAIM_TYPES
    
    total_processed = 0
    total_claims = 0
    type_counts = {t: 0 for t in CLAIM_TYPES}
    type_counts["behavior"] = 0
    
    parse_failures = 0
    llm_errors = 0
    
    for chunk in doc_chunks:
        # We need to adapt the tuple to dictionary-like or pass as is because of extract_claims_for_chunk handling
        # Since it's a sqlite3.Row, it supports index mapping. We should map it properly:
        # id=0, file_path=1, heading=2, text=3
        # extract_claims_for_chunk fallback mapping handles this.
        chunk_dict = {
            "id": chunk[0],
            "file_path": chunk[1],
            "heading": chunk[2],
            "text": chunk[3]
        }
        
        claims, p_fail, l_error = extract_claims_for_chunk(chunk_dict, config["ollama_model"], config["ollama_host"])
        total_processed += 1
        
        if p_fail:
            parse_failures += 1
        if l_error:
            llm_errors += 1
            
        if claims:
            total_claims += len(claims)
            for c in claims:
                type_counts[c.claim_type] = type_counts.get(c.claim_type, 0) + 1
            store_claims(conn, chunk[0], claims)
            
    if llm_errors > 0 and llm_errors == total_processed:
        print("Warning: all extraction calls failed — check that Ollama is still running", file=sys.stderr)
        
    print(f"Parity extract-claims summary for {repo_path}")
    print(f"  Doc chunks processed:   {total_processed}")
    print(f"  Claims extracted:       {total_claims}")
    print(f"    signature:      {type_counts.get('signature', 0)}")
    print(f"    default_value:  {type_counts.get('default_value', 0)}")
    print(f"    env_var:        {type_counts.get('env_var', 0)}")
    print(f"    return_type:    {type_counts.get('return_type', 0)}")
    print(f"    behavior:       {type_counts.get('behavior', 0)}")
    print(f"  Chunks with parse failures (after retry): {parse_failures}")
    print(f"  Chunks with LLM call errors:              {llm_errors}")
    
    sys.exit(0)

def cmd_retrieve(args):
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
    
    # Check if there are claims for this repo
    cursor = conn.execute("SELECT COUNT(*) FROM claims c JOIN doc_chunks d ON c.doc_chunk_id = d.id WHERE d.repo_id = ?", (repo_id,))
    claims_count = cursor.fetchone()[0]
    if claims_count == 0:
        print(f"Warning: no claims found — run extract-claims first")
        sys.exit(0)
        
    # Check if there are embedded code chunks
    cursor = conn.execute("SELECT COUNT(*) FROM code_chunks WHERE repo_id = ? AND embedding_id IS NOT NULL", (repo_id,))
    embedded_chunks_count = cursor.fetchone()[0]
    if embedded_chunks_count == 0:
        print(f"Error: no embedded code chunks found — run 'embed' first", file=sys.stderr)
        sys.exit(1)
        
    client = get_chroma_client(config["chroma_persist_dir"])
    code_col, _ = get_or_create_collections(client)
    
    from parity.retrieval.retriever import retrieve_for_repo
    top_k = args.top_k if args.top_k else 5
    summary = retrieve_for_repo(conn, repo_id, code_col, "BAAI/bge-small-en-v1.5", top_k)
    
    print(f"Parity retrieve summary for {repo_path}")
    print(f"  Claims processed:  {summary['total_claims']}")
    if summary['total_claims'] > 0:
        matched_pct = (summary['matched'] / summary['total_claims']) * 100
        ambiguous_pct = (summary['ambiguous'] / summary['total_claims']) * 100
        no_match_pct = (summary['no_match'] / summary['total_claims']) * 100
        print(f"  Matched:      {summary['matched']} ({matched_pct:.1f}%)")
        print(f"  Ambiguous:    {summary['ambiguous']} ({ambiguous_pct:.1f}%)")
        print(f"  No match:     {summary['no_match']} ({no_match_pct:.1f}%)")
    
    sys.exit(0)

def cmd_verify(args):
    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"Error: repo path '{repo_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isdir(repo_path):
        print(f"Error: repo path '{repo_path}' is not a directory", file=sys.stderr)
        sys.exit(1)
        
    config = load_config(args.config if args.config else "config.yaml")
    
    ollama_ok = check_ollama_reachable(config["ollama_host"])
    if not ollama_ok:
        print(f"Error: Ollama not reachable or model '{config['ollama_model']}' not available — run 'ollama pull {config['ollama_model']}' and ensure 'ollama serve' is running", file=sys.stderr)
        sys.exit(1)
        
    model_ok = check_model_available(config["ollama_model"], config["ollama_host"])
    if not model_ok:
        print(f"Error: Ollama not reachable or model '{config['ollama_model']}' not available — run 'ollama pull {config['ollama_model']}' and ensure 'ollama serve' is running", file=sys.stderr)
        sys.exit(1)
        
    conn = get_connection(config["db_path"])
    
    cursor = conn.execute("SELECT id FROM repos WHERE path = ?", (repo_path,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: repo '{repo_path}' not initialized — run 'init' first", file=sys.stderr)
        sys.exit(1)
        
    repo_id = row[0]
    
    cursor = conn.execute("""
        SELECT COUNT(*) FROM retrieval_results r 
        JOIN claims c ON r.claim_id = c.id 
        JOIN doc_chunks d ON c.doc_chunk_id = d.id 
        WHERE d.repo_id = ?
    """, (repo_id,))
    count = cursor.fetchone()[0]
    
    if count == 0:
        print(f"Error: no retrieval results found — run 'retrieve' first", file=sys.stderr)
        sys.exit(1)
        
    from parity.verification.verify import verify_repo
    
    summary = verify_repo(conn, repo_id, repo_path, config["ollama_model"], config["ollama_host"])
    
    total = summary["total"]
    verified_pct = (summary["verified"] / total * 100) if total > 0 else 0.0
    contradicted_pct = (summary["contradicted"] / total * 100) if total > 0 else 0.0
    unverifiable_pct = (summary["unverifiable"] / total * 100) if total > 0 else 0.0
    
    print(f"Parity verify summary for {repo_path}")
    print(f"  Claims processed:    {total}")
    print(f"  Verified:            {summary['verified']} ({verified_pct:.1f}%)")
    print(f"  Contradicted:        {summary['contradicted']} ({contradicted_pct:.1f}%)")
    print(f"  Unverifiable:        {summary['unverifiable']} ({unverifiable_pct:.1f}%)")
    print(f"  Resolution methods:  dynamic={summary['dynamic_resolutions']}  static={summary['static_resolutions']}  failed={summary['resolution_failures']}")
    
    sys.exit(0)

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
    
    extract_parser = subparsers.add_parser("extract-claims")
    extract_parser.add_argument("repo_path")
    extract_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    extract_parser.add_argument("--limit", type=int, help="Optional limit on doc chunks processed")
    
    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("repo_path")
    retrieve_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    retrieve_parser.add_argument("--top-k", type=int, dest="top_k", help="Optional top-k for retrieval")
    
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("repo_path")
    verify_parser.add_argument("--config", dest="config", help="CONFIG_PATH")
    
    args = parser.parse_args()
    
    if args.command == "init":
        cmd_init(args)
    elif args.command == "chunk-code":
        cmd_chunk_code(args)
    elif args.command == "chunk-docs":
        cmd_chunk_docs(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "extract-claims":
        cmd_extract_claims(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)
    elif args.command == "verify":
        cmd_verify(args)

if __name__ == "__main__":
    main()
