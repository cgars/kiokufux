#!/usr/bin/env python3
"""Render concise GitHub summaries and annotations from security scanner JSON."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

MAX_ROWS = 20


def _severity_key(value: object, order: tuple[str, ...]) -> tuple[int, str]:
    severity = str(value).lower()
    try:
        return order.index(severity), severity
    except ValueError:
        return len(order), severity


def _cell(value: object) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _annotation(value: object) -> str:
    return str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as report:
        return json.load(report)


def _bandit(path: Path) -> tuple[str, list[str]]:
    findings = _load(path).get("results", [])
    findings.sort(key=lambda item: (
        _severity_key(item.get("issue_severity", "unknown"), ("high", "medium", "low")),
        str(item.get("filename", "")),
        int(item.get("line_number", 0)),
    ))
    counts = Counter(str(item.get("issue_severity", "unknown")).lower() for item in findings)
    lines = ["## Bandit", "", f"**{len(findings)} finding(s)** — " + ", ".join(
        f"{name}: {counts.get(name, 0)}" for name in ("high", "medium", "low")
    ), "", "_Ordered by severity: high → medium → low._"]
    annotations = []
    if findings:
        lines += ["", "| Severity | Rule | Location | Finding |", "|---|---|---|---|"]
    for item in findings[:MAX_ROWS]:
        filename = item.get("filename", "")
        line = item.get("line_number", 1)
        severity = str(item.get("issue_severity", "warning")).lower()
        rule = item.get("test_id", "Bandit")
        message = item.get("issue_text", "Security finding")
        lines.append(f"| {_cell(severity)} | `{_cell(rule)}` | `{_cell(filename)}:{line}` | {_cell(message)} |")
        level = "error" if severity == "high" else "warning"
        annotations.append(
            f"::{level} file={_annotation(filename)},line={line},title=Bandit {_annotation(rule)}::{_annotation(message)}"
        )
    return _finish(lines, len(findings)), annotations


def _semgrep(path: Path) -> tuple[str, list[str]]:
    findings = _load(path).get("results", [])
    findings.sort(key=lambda item: (
        _severity_key(item.get("extra", {}).get("severity", "unknown"), ("error", "warning", "info")),
        str(item.get("path", "")),
        int(item.get("start", {}).get("line", 0)),
    ))
    counts = Counter(str(item.get("extra", {}).get("severity", "unknown")).lower() for item in findings)
    lines = ["## Semgrep", "", f"**{len(findings)} finding(s)** — " + ", ".join(
        f"{name}: {counts.get(name, 0)}" for name in ("error", "warning", "info")
    ), "", "_Ordered by severity: error → warning → info._"]
    annotations = []
    if findings:
        lines += ["", "| Severity | Rule | Location | Finding |", "|---|---|---|---|"]
    for item in findings[:MAX_ROWS]:
        extra = item.get("extra", {})
        filename = item.get("path", "")
        line = item.get("start", {}).get("line", 1)
        severity = str(extra.get("severity", "warning")).lower()
        rule = item.get("check_id", "Semgrep")
        message = extra.get("message", "Security finding")
        lines.append(f"| {_cell(severity)} | `{_cell(rule)}` | `{_cell(filename)}:{line}` | {_cell(message)} |")
        level = "error" if severity == "error" else "warning"
        annotations.append(
            f"::{level} file={_annotation(filename)},line={line},title=Semgrep::{_annotation(rule)}: {_annotation(message)}"
        )
    return _finish(lines, len(findings)), annotations


def _pip_audit(path: Path) -> tuple[str, list[str]]:
    report = _load(path)
    dependencies = report.get("dependencies", []) if isinstance(report, dict) else report
    findings = [
        (dependency, vulnerability)
        for dependency in dependencies
        for vulnerability in dependency.get("vulns", [])
    ]
    findings.sort(key=lambda finding: (
        str(finding[0].get("name", "")).lower(),
        str(finding[1].get("id", "")),
    ))
    lines = [
        "## Dependency audit", "", f"**{len(findings)} vulnerability finding(s)**", "",
        "_Ordered by package and advisory ID; pip-audit does not provide a normalized severity._",
    ]
    if findings:
        lines += ["", "| Package | Installed | Advisory | Fixed in |", "|---|---|---|---|"]
    for dependency, vulnerability in findings[:MAX_ROWS]:
        fixed = ", ".join(vulnerability.get("fix_versions", [])) or "No fix listed"
        lines.append(
            f"| `{_cell(dependency.get('name', ''))}` | `{_cell(dependency.get('version', ''))}` "
            f"| `{_cell(vulnerability.get('id', ''))}` | {_cell(fixed)} |"
        )
    return _finish(lines, len(findings)), []


def _finish(lines: list[str], count: int) -> str:
    if count > MAX_ROWS:
        lines += ["", f"_Showing the first {MAX_ROWS} of {count}; download the JSON artifact for all findings._"]
    elif count:
        lines += ["", "_Download the JSON artifact for complete scanner metadata and remediation details._"]
    else:
        lines += ["", "✅ No findings reported."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit", type=Path)
    parser.add_argument("--semgrep", type=Path)
    parser.add_argument("--pip-audit", type=Path)
    args = parser.parse_args()
    renderers = ((args.bandit, _bandit), (args.semgrep, _semgrep), (args.pip_audit, _pip_audit))
    sections, annotations = [], []
    for path, renderer in renderers:
        if path and path.exists():
            section, messages = renderer(path)
            sections.append(section)
            annotations.extend(messages)
    summary = "\n".join(sections) or "## Security reports\n\n⚠️ No machine-readable report was produced.\n"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as output:
            output.write(summary)
    else:
        print(summary, end="")
    for message in annotations:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
