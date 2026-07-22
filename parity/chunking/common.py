# Common configuration and utilities for chunking

# Standard directories to exclude when walking the repository for files.
# This prevents parsing dependencies, build artifacts, test caches, and version control data.
EXCLUDED_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', 'env', '.tox',
    'build', 'dist', '.eggs', 'node_modules', '.mypy_cache', '.pytest_cache'
}
