import os
import json
import pytest
import sys
from unittest.mock import patch, MagicMock
from parity.cli.main import main

def test_eval_summary_no_data(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # create data/eval empty dir
    os.makedirs("data/eval", exist_ok=True)
    
    with patch("sys.argv", ["parity", "eval-summary", "."]):
        with pytest.raises(SystemExit) as excinfo:
            main()
    
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "No evaluation data found" in captured.out

@patch("parity.evaluation.run_fault_eval.run_fault_injection_eval")
@patch("parity.cli.main.load_config")
@patch("parity.cli.main.get_connection")
def test_eval_faults_json_written(mock_get_conn, mock_load, mock_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    
    # Setup mock db
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)
    mock_get_conn.return_value = mock_conn
    mock_load.return_value = {"db_path": "test.db"}
    
    mock_run.return_value = {
        "n_faults": 1, "detected": 1, "missed": 0, "detection_rate": 100.0, "missed_details": []
    }
    
    with patch("sys.argv", ["parity", "eval-faults", "."]):
        main()
        
    # Check that json file was written
    eval_dir = os.path.join("data", "eval")
    assert os.path.exists(eval_dir)
    files = os.listdir(eval_dir)
    assert len(files) == 1
    assert files[0].startswith("fault_injection_")
    
    with open(os.path.join(eval_dir, files[0])) as f:
        data = json.load(f)
        assert data["n_faults"] == 1
