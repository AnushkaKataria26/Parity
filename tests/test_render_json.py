import pytest
import json
from parity.reporting.build_report import DriftReport, DriftEntry
from parity.reporting.render_json import render_json_report

@pytest.fixture
def sample_report():
    entries_doc1 = [
        DriftEntry("doc1.md", "h1", 10, 20, "claim 1", "signature", "Contradicted", "x", "y", "func1", "code1.py", 5, 10),
        DriftEntry("doc1.md", "h1", 25, 30, "claim 2", "behavior", "Verified", "a", "a", "func1", "code1.py", 5, 10),
    ]
    return DriftReport(
        repo_name="test",
        repo_path="/test",
        generated_at="2024-01-01T00:00:00Z",
        commit_sha="abc12345",
        entries_by_file={
            "doc1.md": entries_doc1
        },
        totals={"verified": 1, "contradicted": 1, "unverifiable": 0}
    )

def test_render_json_round_trip(sample_report):
    out = render_json_report(sample_report)
    parsed = json.loads(out)
    
    assert parsed["repo_name"] == "test"
    assert parsed["totals"]["verified"] == 1
    assert "doc1.md" in parsed["files"]
    
    # JSON includes Verified entries
    assert len(parsed["files"]["doc1.md"]) == 2
    assert parsed["files"]["doc1.md"][1]["status"] == "Verified"

def test_render_json_special_chars(sample_report):
    sample_report.entries_by_file["doc1.md"][0].claim_text = "claim \n \"quotes\" \u1234"
    out = render_json_report(sample_report)
    parsed = json.loads(out)
    assert parsed["files"]["doc1.md"][0]["claim_text"] == "claim \n \"quotes\" \u1234"
