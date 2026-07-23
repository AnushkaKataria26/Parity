import pytest
from parity.reporting.build_report import DriftReport, DriftEntry
from parity.reporting.render_text import render_text_report

@pytest.fixture
def sample_report():
    entries_doc1 = [
        DriftEntry("doc1.md", "h1", 10, 20, "claim 1", "signature", "Contradicted", "x", "y", "func1", "code1.py", 5, 10),
        DriftEntry("doc1.md", "h1", 25, 30, "claim 2", "behavior", "Verified", "a", "a", "func1", "code1.py", 5, 10),
    ]
    entries_doc2 = [
        DriftEntry("doc2.md", "h2", 15, 20, "claim 3", "return_type", "Verified", "b", "b", "func2", "code2.py", 15, 20),
    ]
    return DriftReport(
        repo_name="test",
        repo_path="/test",
        generated_at="2024-01-01T00:00:00Z",
        commit_sha="abc123456789",
        entries_by_file={
            "doc1.md": entries_doc1,
            "doc2.md": entries_doc2
        },
        totals={"verified": 2, "contradicted": 1, "unverifiable": 0}
    )

def test_render_text_non_verbose(sample_report):
    out = render_text_report(sample_report, verbose=False)
    assert "doc1.md  (1 issues)" in out
    assert "[CONTRADICTED] doc1.md:10" in out
    assert "[VERIFIED]" not in out
    # doc2.md should be completely absent because it only has Verified entries
    assert "doc2.md" not in out

def test_render_text_verbose(sample_report):
    out = render_text_report(sample_report, verbose=True)
    assert "doc1.md  (1 issues)" in out
    assert "[CONTRADICTED] doc1.md:10" in out
    assert "[VERIFIED] doc1.md:25" in out
    # doc2.md should be present
    assert "doc2.md  (0 issues)" in out
    assert "[VERIFIED] doc2.md:15" in out

def test_render_text_empty_report():
    report = DriftReport(
        repo_name="test",
        repo_path="/test",
        generated_at="2024-01-01T00:00:00Z",
        commit_sha="abc12345",
        entries_by_file={},
        totals={"verified": 0, "contradicted": 0, "unverifiable": 0}
    )
    out = render_text_report(report)
    assert "No drift detected — all claims verified, or no claims were checked." in out

def test_render_text_no_commit_sha(sample_report):
    sample_report.commit_sha = None
    out = render_text_report(sample_report)
    assert "Commit: unknown" in out
    
def test_unverifiable_reasons():
    report = DriftReport(
        repo_name="test", repo_path="/test", generated_at="now", commit_sha=None, totals={},
        entries_by_file={
            "doc.md": [
                DriftEntry("doc.md", "h1", 10, 20, "claim", "behavior", "Unverifiable", None, None, "func", "code.py", 5, 10),
                DriftEntry("doc.md", "h1", 25, 30, "claim", "signature", "Unverifiable", None, None, None, None, None, None),
            ]
        }
    )
    out = render_text_report(report)
    assert "Reason:   non-mechanically-checkable claim" in out
    assert "Reason:   no confident code match" in out

def test_claim_text_special_chars(sample_report):
    sample_report.entries_by_file["doc1.md"][0].claim_text = "claim with\nnewline and \"quotes\""
    out = render_text_report(sample_report)
    assert 'Claim:    "claim with\nnewline and "quotes""' in out
    assert "Claimed:  x" in out
