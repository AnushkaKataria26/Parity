import os
import subprocess
import pytest
from parity.versioning.git_diff import get_current_commit_sha, get_changed_files_since

def test_git_diff_no_commits(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    
    sha = get_current_commit_sha(str(tmp_path))
    assert sha is None
    
    changed, is_full = get_changed_files_since(str(tmp_path), None)
    assert is_full is True
    assert changed == set()

def test_git_diff_one_commit(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    (tmp_path / "file1.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    
    sha = get_current_commit_sha(str(tmp_path))
    assert sha is not None
    
    changed, is_full = get_changed_files_since(str(tmp_path), None)
    assert is_full is True

def test_git_diff_modifications(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.txt").write_text("world")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    
    sha1 = get_current_commit_sha(str(tmp_path))
    
    # modify file1, add file3
    (tmp_path / "file1.txt").write_text("hello modified")
    (tmp_path / "file3.txt").write_text("new")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "mod"], cwd=tmp_path, check=True)
    
    changed, is_full = get_changed_files_since(str(tmp_path), sha1)
    assert is_full is False
    assert "file1.txt" in changed
    assert "file3.txt" in changed
    assert "file2.txt" not in changed

def test_git_diff_uncommitted(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    (tmp_path / "file1.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    
    sha1 = get_current_commit_sha(str(tmp_path))
    
    # modify without commit
    (tmp_path / "file1.txt").write_text("hello uncommitted")
    (tmp_path / "untracked.txt").write_text("untracked")
    
    changed, is_full = get_changed_files_since(str(tmp_path), sha1)
    assert is_full is False
    assert "file1.txt" in changed
    assert "untracked.txt" in changed

def test_git_diff_deleted_and_renamed(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.txt").write_text("world")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    
    sha1 = get_current_commit_sha(str(tmp_path))
    
    subprocess.run(["git", "rm", "file1.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "mv", "file2.txt", "file2_renamed.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "mod"], cwd=tmp_path, check=True)
    
    changed, is_full = get_changed_files_since(str(tmp_path), sha1)
    assert is_full is False
    assert "file1.txt" in changed
    assert "file2.txt" in changed
    assert "file2_renamed.txt" in changed

def test_git_diff_bogus_sha(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True)
    (tmp_path / "file1.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    
    changed, is_full = get_changed_files_since(str(tmp_path), "bogus_sha")
    assert is_full is True
