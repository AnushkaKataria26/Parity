import json
import dataclasses
from parity.reporting.build_report import DriftReport

def render_json_report(report: DriftReport) -> str:
    # We include Verified entries always, unlike text output defaults,
    # because JSON is meant for programmatic consumption.
    data = dataclasses.asdict(report)
    
    result = {
        "repo_name": data["repo_name"],
        "repo_path": data["repo_path"],
        "generated_at": data["generated_at"],
        "commit_sha": data["commit_sha"],
        "totals": data["totals"],
        "files": data["entries_by_file"]
    }
    
    return json.dumps(result, indent=2)
