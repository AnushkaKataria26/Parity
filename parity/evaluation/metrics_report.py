from typing import Dict, Optional

def render_metrics_summary(fault_eval_result: Dict, extraction_eval_result: Optional[Dict]) -> str:
    lines = []
    lines.append("================================================================================")
    lines.append("Parity Evaluation Summary")
    lines.append("================================================================================")
    lines.append("")
    lines.append("Fault Injection (Verification Accuracy)")
    lines.append(f"  Faults injected:     {fault_eval_result['n_faults']}")
    
    n = fault_eval_result['n_faults']
    d = fault_eval_result['detected']
    m = fault_eval_result['missed']
    rate = fault_eval_result['detection_rate']
    
    lines.append(f"  Detected:            {d} / {n}  ({rate:.1f}%)")
    lines.append(f"  Missed:              {m}")
    
    if m > 0:
        for md in fault_eval_result.get('missed_details', []):
            lines.append(f"    - {md['fault_type']} on {md['symbol_name']}: actual outcome was {md['actual_outcome']}")
            
    lines.append("")
    lines.append("Claim Extraction (Precision/Recall Proxy)")
    
    if extraction_eval_result is None:
        lines.append("  Claim extraction eval skipped — run export-labels and score-extraction after hand-labeling.")
    else:
        l_count = extraction_eval_result['labeled_count']
        lines.append(f"  Chunks labeled:                {l_count}")
        
        ze_c = extraction_eval_result['zero_empty_correct']
        ze_t = extraction_eval_result['zero_empty_total']
        nz_c = extraction_eval_result['nonzero_correct']
        nz_t = extraction_eval_result['nonzero_total']
        em = extraction_eval_result['exact_match']
        wo = extraction_eval_result['within_one']
        
        em_pct = (em / l_count * 100) if l_count > 0 else 0.0
        wo_pct = (wo / l_count * 100) if l_count > 0 else 0.0
        
        lines.append(f"  Zero-claim chunks correctly empty:  {ze_c} / {ze_t}  (over-extraction check)")
        lines.append(f"  Nonzero-claim chunks correctly nonzero: {nz_c} / {nz_t}  (under-extraction check)")
        lines.append(f"  Exact count match:              {em_pct:.1f}%")
        lines.append(f"  Within \u00b11 count:                {wo_pct:.1f}%")
        
    lines.append("================================================================================")
    
    return "\n".join(lines)
