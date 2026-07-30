import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "security_summary.py"


def _run(tmp_path, *arguments):
    summary = tmp_path / "summary.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)],
        env={"GITHUB_STEP_SUMMARY": str(summary)},
        check=True,
        capture_output=True,
        text=True,
    )
    return summary.read_text(), result.stdout


def test_renders_sast_findings_and_annotations(tmp_path):
    bandit = tmp_path / "bandit.json"
    semgrep = tmp_path / "semgrep.json"
    bandit.write_text(json.dumps({"results": [{
        "filename": "kiokufux/example.py", "line_number": 7,
        "issue_severity": "HIGH", "test_id": "B999", "issue_text": "unsafe | example",
    }]}))
    semgrep.write_text(json.dumps({"results": [{
        "path": "kiokufux/other.py", "start": {"line": 11}, "check_id": "example.rule",
        "extra": {"severity": "WARNING", "message": "review this"},
    }]}))

    summary, annotations = _run(tmp_path, "--bandit", bandit, "--semgrep", semgrep)

    assert "**1 finding(s)** — high: 1" in summary
    assert "unsafe \\| example" in summary
    assert "Bandit B999" in annotations
    assert "file=kiokufux/other.py,line=11" in annotations


def test_sorts_findings_from_highest_to_lowest_severity(tmp_path):
    bandit = tmp_path / "bandit.json"
    semgrep = tmp_path / "semgrep.json"
    bandit.write_text(json.dumps({"results": [
        {"filename": "low.py", "line_number": 1, "issue_severity": "LOW", "test_id": "LOW"},
        {"filename": "high.py", "line_number": 2, "issue_severity": "HIGH", "test_id": "HIGH"},
        {"filename": "medium.py", "line_number": 3, "issue_severity": "MEDIUM", "test_id": "MEDIUM"},
    ]}))
    semgrep.write_text(json.dumps({"results": [
        {"path": "info.py", "start": {"line": 1}, "check_id": "INFO", "extra": {"severity": "INFO"}},
        {"path": "error.py", "start": {"line": 2}, "check_id": "ERROR", "extra": {"severity": "ERROR"}},
        {"path": "warning.py", "start": {"line": 3}, "check_id": "WARNING", "extra": {"severity": "WARNING"}},
    ]}))

    summary, annotations = _run(tmp_path, "--bandit", bandit, "--semgrep", semgrep)

    assert summary.index("`HIGH`") < summary.index("`MEDIUM`") < summary.index("`LOW`")
    assert summary.index("`ERROR`") < summary.index("`WARNING`") < summary.index("`INFO`")
    assert annotations.index("title=Bandit HIGH") < annotations.index("title=Bandit LOW")


def test_renders_dependency_fixes(tmp_path):
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"dependencies": [{
        "name": "example", "version": "1.0", "vulns": [{"id": "PYSEC-1", "fix_versions": ["1.1"]}],
    }]}))

    summary, annotations = _run(tmp_path, "--pip-audit", report)

    assert "**1 vulnerability finding(s)**" in summary
    assert "| `example` | `1.0` | `PYSEC-1` | 1.1 |" in summary
    assert not annotations
