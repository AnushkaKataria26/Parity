SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    last_ingested_commit_sha TEXT
);""",
    """CREATE TABLE IF NOT EXISTS code_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL REFERENCES repos(id),
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    embedding_id TEXT,
    ast_hash TEXT
);""",
    """CREATE TABLE IF NOT EXISTS doc_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL REFERENCES repos(id),
    file_path TEXT NOT NULL,
    heading TEXT,
    text TEXT NOT NULL,
    embedding_id TEXT
);""",
    """CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_chunk_id INTEGER NOT NULL REFERENCES doc_chunks(id),
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    referenced_symbol_guess TEXT
);""",
    """CREATE TABLE IF NOT EXISTS verification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    -- NOTE: matched_code_chunk_id conceptually belongs in retrieval_results, but is also stored
    -- here at verification time by copying the resolved match forward, rather than written by
    -- Phase 5 directly, to avoid partial/placeholder rows in verification_results.
    matched_code_chunk_id INTEGER REFERENCES code_chunks(id),
    status TEXT NOT NULL,
    actual_value TEXT,
    claimed_value TEXT,
    verified_at TEXT NOT NULL
);""",
    """CREATE TABLE IF NOT EXISTS retrieval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    matched_code_chunk_id INTEGER REFERENCES code_chunks(id),
    match_status TEXT NOT NULL,
    top_k_json TEXT NOT NULL,
    retrieved_at TEXT NOT NULL
);"""
]
