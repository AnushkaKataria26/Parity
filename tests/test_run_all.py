import os
import sqlite3
import subprocess
import pytest
from parity.cli.main import main, get_connection
import argparse
from unittest.mock import patch

def setup_repo(tmp_path):
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
    
    repo_path = tmp_path / "testrepo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    
    (repo_path / "main.py").write_text("def hello(): pass")
    (repo_path / "docs.md").write_text("# Hello\nThis is a doc.")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True)
    
    return repo_path, config_path, db_path

@patch("parity.cli.main.cmd_extract_claims")
@patch("parity.cli.main.cmd_retrieve")
@patch("parity.cli.main.cmd_verify")
@patch("parity.cli.main.cmd_report")
@patch("parity.embedding.model.embed_texts")
def test_run_all_command(mock_embed, mock_report, mock_verify, mock_retrieve, mock_extract, tmp_path):
    mock_embed.return_value = [[0.1]*384, [0.1]*384]
    
    repo_path, config_path, db_path = setup_repo(tmp_path)
    
    # Run full pipeline
    test_args = ["parity", "run-all", str(repo_path), "--config", str(config_path)]
    with patch("sys.argv", test_args):
        main()
        
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0] == 1
    
    # Run pipeline again incrementally
    (repo_path / "main.py").write_text("def hello(): pass\ndef world(): pass")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "mod"], cwd=repo_path, check=True)
    
    mock_embed.return_value = [[0.1]*384, [0.1]*384] # for the two new code chunks
    
    with patch("sys.argv", test_args):
        main()
        
    assert conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0] == 2
