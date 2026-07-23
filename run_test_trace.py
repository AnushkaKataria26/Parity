import sqlite3
import tempfile
import os
from parity.db.migrate import apply_schema
from parity.evaluation.run_fault_eval import run_fault_injection_eval
from unittest.mock import patch, MagicMock

conn = sqlite3.connect(":memory:")
apply_schema(conn)

tmp_path = tempfile.mkdtemp()

conn.execute("INSERT INTO repos (id, name, path, last_ingested_commit_sha) VALUES (1, 'test', ?, 'sha')", (tmp_path,))

mock_select = MagicMock(return_value=[{"chunk_id": 1, "symbol_name": "foo", "file_path": "foo.py"}])
mock_inject = MagicMock(return_value=[])
mock_discover = MagicMock(return_value=[])
mock_extract = MagicMock(return_value=([], False))
mock_store = MagicMock()
mock_get_client = MagicMock()
mock_get_coll = MagicMock(return_value=(MagicMock(), MagicMock()))
mock_embed = MagicMock()
mock_retrieve = MagicMock()
mock_verify = MagicMock(side_effect=Exception("Forced crash"))

config = {"chroma_persist_dir": "", "ollama_model": "", "ollama_host": ""}

with patch("parity.evaluation.run_fault_eval.select_fault_targets", mock_select), \
     patch("parity.evaluation.run_fault_eval.inject_faults", mock_inject), \
     patch("parity.evaluation.run_fault_eval.discover_python_files", mock_discover), \
     patch("parity.evaluation.run_fault_eval.extract_chunks_from_file", mock_extract), \
     patch("parity.evaluation.run_fault_eval.store_chunks", mock_store), \
     patch("parity.evaluation.run_fault_eval.get_chroma_client", mock_get_client), \
     patch("parity.evaluation.run_fault_eval.get_or_create_collections", mock_get_coll), \
     patch("parity.evaluation.run_fault_eval.embed_repo", mock_embed), \
     patch("parity.evaluation.run_fault_eval.retrieve_for_repo", mock_retrieve), \
     patch("parity.evaluation.run_fault_eval.verify_repo", mock_verify):
     
    try:
        run_fault_injection_eval(conn, 1, tmp_path, config, n_faults=1)
    except Exception as e:
        import traceback
        traceback.print_exc()

