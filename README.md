# Parity

Parity is an autonomous doc-to-code verification engine designed to detect and report drift between living codebase implementations and their prose documentation. Rather than generating documentation, Parity extracts checkable assertions from your existing docs (such as signatures, default values, and environment variables) and strictly validates them against the underlying code, ensuring honest alignment.

## Architecture

Parity operates as a 5-component pipeline:
1. **Incremental Chunking**: Parses Python code and markdown/RST documentation into semantic chunks, tracking changes via a content cache.
2. **Embedding & Vector Search**: Semantically embeds chunks using `BAAI/bge-small-en-v1.5` and indexes them in ChromaDB for fast retrieval.
3. **Claim Extraction**: Extracts verifiable claims from documentation using local LLMs (via Ollama).
4. **Retrieval**: Links extracted claims to the specific code chunks they describe based on semantic similarity and lexical matching.
5. **Verification**: Matches claimed values against statically and dynamically resolved code properties, detecting mismatches and generating a drift report.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Install Ollama (https://ollama.com/) and ensure the service is running.
3. Pull the required language model (e.g., `llama3.1`):
   ```bash
   ollama pull llama3.1
   ```
4. Warm up the embedding model (optional, downloads the BAAI embedding model):
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
   ```

## Quick Start

You can run the entire pipeline end-to-end on your repository using the `run-all` command:

```bash
python -m parity.cli.main run-all /path/to/your/repo
```

## CLI Reference

- `init <repo_path> [--config PATH]`: Initialize Parity for a new repository.
- `chunk-code <repo_path> [--config PATH] [--full]`: Extract code chunks incrementally or perform a full rescan.
- `chunk-docs <repo_path> [--config PATH] [--full]`: Extract documentation chunks incrementally or perform a full rescan.
- `embed <repo_path> [--config PATH]`: Generate embeddings and index them in ChromaDB.
- `extract-claims <repo_path> [--config PATH] [--limit N]`: Extract verifiable claims from documentation.
- `retrieve <repo_path> [--config PATH] [--top-k N]`: Link claims to code chunks.
- `verify <repo_path> [--config PATH]`: Verify claims against the code.
- `report <repo_path> [--config PATH] [--verbose] [--json-out PATH] [--text-out PATH]`: Generate a human-readable or structured JSON drift report.
- `run-all <repo_path> [--config PATH] [--full]`: Execute all pipeline steps sequentially.
- `eval-faults <repo_path> [--config PATH] [--n-faults N]`: Inject faults into a temporary copy of the repo and evaluate verification accuracy.
- `export-labels <repo_path> [--config PATH] [--out PATH]`: Export documentation chunks for human precision/recall labeling.
- `score-extraction <repo_path> [--config PATH] --labeled PATH`: Compute a coarse precision/recall extraction score based on human labels.
- `eval-summary <repo_path> [--config PATH]`: Render the metrics summary combining fault injection and extraction evaluations.

## Limitations

- **Nested Functions**: Dynamic resolution of nested functions is unsupported as they aren't accessible as module attributes.
- **Decorators**: Signature verification may fail for decorators that do not use `functools.wraps` to preserve the original signature.
- **RST Code Blocks**: Complex restructuredText code blocks with custom directives might be imperfectly stripped.
- **Environment Variables**: Environment variable detection relies on regex-based heuristics (e.g., `os.environ`, `os.getenv`).
- **Rename Detection**: Detecting file or function renames over time across commits is not yet implemented natively.
- **Extraction Evaluation**: The precision/recall scoring is a coarse, chunk-level proxy metric based on human labeling, rather than strict IR-level claim-for-claim matching.
