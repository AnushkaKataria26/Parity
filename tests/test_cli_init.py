import os
import pytest
import subprocess
import argparse
import sys
from unittest.mock import patch
from parity.cli.main import cmd_init

@pytest.fixture
def tmp_config_path(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
db_path: {tmp_path}/parity.db
chroma_persist_dir: {tmp_path}/chroma
ollama_host: http://localhost:11434
ollama_model: llama3:8b
""")
    return str(config_yaml)

def test_cmd_init_success(tmp_path, tmp_config_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    
    subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=str(repo_path), check=True)
    
    args = argparse.Namespace(repo_path=str(repo_path), config=tmp_config_path)
    
    try:
        cmd_init(args)
    except SystemExit:
        pytest.fail("cmd_init exited unexpectedly")
        
    assert os.path.exists(f"{tmp_path}/parity.db")
    assert os.path.isdir(f"{tmp_path}/chroma")

def test_cmd_init_nonexistent(tmp_path, tmp_config_path, capsys):
    args = argparse.Namespace(repo_path=str(tmp_path / "nonexistent"), config=tmp_config_path)
    
    with pytest.raises(SystemExit) as e:
        cmd_init(args)
        
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err

def test_cmd_init_no_git(tmp_path, tmp_config_path, capsys):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    
    args = argparse.Namespace(repo_path=str(repo_path), config=tmp_config_path)
    
    with pytest.raises(SystemExit) as e:
        cmd_init(args)
        
    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "not a git repository" in captured.err

def test_cmd_init_zero_commits(tmp_path, tmp_config_path, capsys):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
    
    args = argparse.Namespace(repo_path=str(repo_path), config=tmp_config_path)
    
    try:
        cmd_init(args)
    except SystemExit:
        pytest.fail("cmd_init exited unexpectedly")
        
    captured = capsys.readouterr()
    assert "Warning: repo has no commits yet" in captured.err

def test_cmd_init_twice(tmp_path, tmp_config_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_path), check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=str(repo_path), check=True)
    
    args = argparse.Namespace(repo_path=str(repo_path), config=tmp_config_path)
    
    cmd_init(args)
    cmd_init(args)
    
    import sqlite3
    conn = sqlite3.connect(f"{tmp_path}/parity.db")
    cursor = conn.execute("SELECT COUNT(*) FROM repos")
    assert cursor.fetchone()[0] == 1
