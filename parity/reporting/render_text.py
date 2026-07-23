from parity.reporting.build_report import DriftReport, DriftEntry

def get_unverifiable_reason(entry: DriftEntry) -> str:
    if entry.claim_type == "behavior":
        return "non-mechanically-checkable claim"
    if entry.matched_symbol is None:
        return "no confident code match"
    if entry.claimed_value is None and entry.actual_value is None and entry.matched_symbol is not None:
        return "extraction failed"
    return "symbol could not be resolved"

def render_text_report(report: DriftReport, verbose: bool = False) -> str:
    lines = []
    
    commit_str = report.commit_sha[:8] if report.commit_sha else "unknown"
    
    lines.append("=" * 80)
    lines.append(f"Parity Drift Report — {report.repo_name}")
    lines.append(f"Generated: {report.generated_at}   Commit: {commit_str}")
    lines.append("=" * 80)
    lines.append("")
    
    total = sum(report.totals.values())
    verified = report.totals.get("verified", 0)
    contradicted = report.totals.get("contradicted", 0)
    unverifiable = report.totals.get("unverifiable", 0)
    
    lines.append(f"Summary: {total} claims checked — {contradicted} contradicted, {unverifiable} unverifiable, {verified} verified")
    lines.append("")
    
    if not report.entries_by_file:
        lines.append("No drift detected — all claims verified, or no claims were checked.")
        return "\n".join(lines)
        
    # Check if there are any files to display
    has_displayable_files = False
    for file_path, entries in report.entries_by_file.items():
        display_entries = entries if verbose else [e for e in entries if e.status != "Verified"]
        if display_entries:
            has_displayable_files = True
            break
            
    if not has_displayable_files:
        lines.append("No drift detected — all claims verified, or no claims were checked.")
        return "\n".join(lines)
        
    for file_path, entries in report.entries_by_file.items():
        if verbose:
            display_entries = entries
        else:
            display_entries = [e for e in entries if e.status != "Verified"]
            
        if not display_entries:
            continue
            
        issues_count = sum(1 for e in entries if e.status in ("Contradicted", "Unverifiable"))
        
        lines.append("-" * 80)
        lines.append(f"{file_path}  ({issues_count} issues)")
        lines.append("-" * 80)
        lines.append("")
        
        for entry in display_entries:
            doc_loc = f"{file_path}:{entry.doc_start_line if entry.doc_start_line is not None else ''}"
            if doc_loc.endswith(":"):
                doc_loc = doc_loc[:-1]
                
            status_tag = f"[{entry.status.upper()}]"
            
            lines.append(f"{status_tag} {doc_loc}  ({entry.claim_type})")
            lines.append(f"  Claim:    \"{entry.claim_text}\"")
            
            if entry.status == "Contradicted":
                lines.append(f"  Claimed:  {entry.claimed_value}")
                lines.append(f"  Actual:   {entry.actual_value}")
                match_str = f"{entry.matched_symbol} — {entry.matched_file_path}:{entry.matched_start_line}"
                lines.append(f"  Code:     {match_str}")
                
            elif entry.status == "Unverifiable":
                reason = get_unverifiable_reason(entry)
                lines.append(f"  Reason:   {reason}")
                
                if entry.matched_symbol:
                    loc = f"{entry.matched_file_path}:{entry.matched_start_line}"
                    code_str = f"{entry.matched_symbol} — {loc}"
                else:
                    code_str = "no match — n/a"
                lines.append(f"  Code:     {code_str}")
                
            elif entry.status == "Verified":
                lines.append(f"  Claimed:  {entry.claimed_value}")
                lines.append(f"  Actual:   {entry.actual_value}")
                match_str = f"{entry.matched_symbol} — {entry.matched_file_path}:{entry.matched_start_line}"
                lines.append(f"  Code:     {match_str}")
                
            lines.append("")
            
    return "\n".join(lines).strip() + "\n"
