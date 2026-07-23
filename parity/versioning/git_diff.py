import subprocess
import logging

def get_current_commit_sha(repo_path: str) -> str | None:
    """
    Returns the current HEAD commit SHA, or None if the repo has no commits.
    """
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    return None

def get_changed_files_since(repo_path: str, since_sha: str | None) -> tuple[set[str], bool]:
    """
    Returns a tuple of (changed_file_paths, is_full_rescan).
    Paths are relative to repo_path, forward-slash normalized.
    """
    if not since_sha:
        return set(), True
        
    changed_files = set()
    
    # 1. Diff against previous commit
    diff_result = subprocess.run(
        ["git", "diff", "--name-status", since_sha, "HEAD"], 
        cwd=repo_path, capture_output=True, text=True
    )
    
    if diff_result.returncode != 0:
        logging.warning(f"Warning: could not diff against previous commit {since_sha[:8]} (history may have changed) — falling back to full rescan")
        return set(), True
        
    for line in diff_result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split('\t')
        status = parts[0]
        if status.startswith('R') and len(parts) >= 3:
            # Rename, parts: status, old_path, new_path
            old_path = parts[1].replace('\\', '/')
            new_path = parts[2].replace('\\', '/')
            changed_files.add(old_path)
            changed_files.add(new_path)
        else:
            path = parts[-1].replace('\\', '/')
            changed_files.add(path)
            
    # 2. Check uncommitted changes (working tree)
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        cwd=repo_path, capture_output=True, text=True
    )
    
    if status_result.returncode != 0:
        logging.warning(f"Warning: git status failed — falling back to full rescan")
        return set(), True
        
    # parse null-terminated git status output
    status_output = status_result.stdout
    if status_output:
        parts = status_output.split('\0')
        i = 0
        while i < len(parts):
            if not parts[i]:
                i += 1
                continue
            
            line = parts[i]
            if len(line) < 3:
                i += 1
                continue
                
            status = line[:2]
            path = line[3:]
            changed_files.add(path.replace('\\', '/'))
            
            if status.startswith('R') or status.startswith('C'):
                i += 1 # The new path is the next item in -z output
                if i < len(parts) and parts[i]:
                    changed_files.add(parts[i].replace('\\', '/'))
                    
            i += 1

    return changed_files, False
