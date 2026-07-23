import os
import sqlite3
import subprocess
import pytest
from parity.cli.main import cmd_init, cmd_chunk_code, cmd_chunk_docs
import argparse

def get_args(repo_path, command="chunk-code", config="config.yaml", full=False):
    return argparse.Namespace(repo_path=str(repo_path), config=config, command=command, full=full)

def setup_repo(tmp_path):
    # Create config
    db_path = tmp_path / "parity.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
db_path: "{db_path}"
chroma_persist_dir: "{tmp_path}/chroma"
ollama_host: "http://localhost:11434"
ollama_model: "llama3"
embedding_model: "BAAI/bge-small-en-v1.5"
embedding_batch_size: 32
""")
    
    # Initialize Git
    repo_path = tmp_path / "testrepo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    
    (repo_path / "main.py").write_text("def hello(): pass")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True)
    
    return repo_path, config_path, db_path

def test_incremental_chunking(tmp_path):
    repo_path, config_path, db_path = setup_repo(tmp_path)
    
    # Init
    cmd_init(get_args(repo_path, "init", config=str(config_path)))
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # First chunk-code (should be full scan)
    changed = cmd_chunk_code(get_args(repo_path, "chunk-code", config=str(config_path)))
    assert isinstance(changed, set)
    assert len(changed) == 1
    
    count = conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
    assert count == 1
    
    # Second chunk-code (should be incremental but no changes)
    changed2 = cmd_chunk_code(get_args(repo_path, "chunk-code", config=str(config_path)))
    assert len(changed2) == 0
    
    count = conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
    assert count == 1
    
    # Modify file and commit
    (repo_path / "main.py").write_text("def hello(): pass\ndef world(): pass")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "mod"], cwd=repo_path, check=True)
    
    # Third chunk-code (should reprocess main.py)
    changed3 = cmd_chunk_code(get_args(repo_path, "chunk-code", config=str(config_path)))
    assert len(changed3) == 1
    assert list(changed3)[0] == "main.py"
    
    count = conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
    assert count == 2
    
    # Delete file, add new file, commit
    subprocess.run(["git", "rm", "main.py"], cwd=repo_path, check=True)
    (repo_path / "other.py").write_text("class Foo: pass")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "replace"], cwd=repo_path, check=True)
    
    changed4 = cmd_chunk_code(get_args(repo_path, "chunk-code", config=str(config_path)))
    assert len(changed4) == 2
    assert "main.py" in changed4
    assert "other.py" in changed4
    
    count = conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
    assert count == 1  # only Foo is present
    row = conn.execute("SELECT symbol_name FROM code_chunks").fetchone()
    assert row[0] == "Foo"
