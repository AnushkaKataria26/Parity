from parity.evaluation.metrics_report import render_metrics_summary

def test_render_metrics_summary_both():
    f_res = {
        "n_faults": 10,
        "detected": 8,
        "missed": 2,
        "detection_rate": 80.0,
        "missed_details": [
            {"fault_type": "rename_param", "symbol_name": "foo", "actual_outcome": "Unverifiable"},
            {"fault_type": "change_default", "symbol_name": "bar", "actual_outcome": "Verified"}
        ]
    }
    
    e_res = {
        "labeled_count": 10,
        "zero_empty_correct": 2,
        "zero_empty_total": 3,
        "nonzero_correct": 6,
        "nonzero_total": 7,
        "exact_match": 8,
        "within_one": 9
    }
    
    out = render_metrics_summary(f_res, e_res)
    assert "Faults injected:     10" in out
    assert "Detected:            8 / 10  (80.0%)" in out
    assert "Missed:              2" in out
    assert "- rename_param on foo: actual outcome was Unverifiable" in out
    assert "- change_default on bar: actual outcome was Verified" in out
    
    assert "Chunks labeled:                10" in out
    assert "Zero-claim chunks correctly empty:  2 / 3" in out
    assert "Nonzero-claim chunks correctly nonzero: 6 / 7" in out
    assert "Exact count match:              80.0%" in out
    assert "Within ±1 count:                90.0%" in out

def test_render_metrics_summary_no_extraction():
    f_res = {
        "n_faults": 10,
        "detected": 10,
        "missed": 0,
        "detection_rate": 100.0,
        "missed_details": []
    }
    
    out = render_metrics_summary(f_res, None)
    assert "Detected:            10 / 10  (100.0%)" in out
    assert "Missed:              0" in out
    assert "Claim extraction eval skipped" in out
    assert "Chunks labeled" not in out
