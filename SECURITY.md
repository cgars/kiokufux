# Security policy

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use the
repository's **Security** tab to submit a private vulnerability report. If
private vulnerability reporting is not enabled, contact the repository owner
privately and include the affected version, reproduction steps, and potential
impact. Do not include real photographs, face data, credentials, or other
personal information in a report.

## Automated security checks

The `Security` GitHub Actions workflow uses only free, open-source scanners:

- [Bandit](https://bandit.readthedocs.io/) checks Python code for common
  security defects.
- [Semgrep Community Edition](https://semgrep.dev/products/community-edition/)
  applies community Python, security-audit, and secret-detection rules. Metrics
  are disabled, and no Semgrep account or token is required.
- [pip-audit](https://pypi.org/project/pip-audit/) checks the installed Python
  dependency graph against the Python Packaging Advisory Database.
- GitHub Dependabot proposes updates for Python dependencies and GitHub Actions.

Checks run on pull requests, pushes to the default branch, every Monday, and on
manual request. The scheduled run catches newly published advisories even when
the source tree has not changed. All scanners run even if another scanner finds
an issue. Each job writes a severity breakdown and a compact, file-linked list
of up to 20 findings to the workflow summary. Bandit and Semgrep findings are
ordered from highest to lowest severity, with file and line as deterministic
tie-breakers. Code findings also appear in the same order as file annotations
on the workflow run. Machine-readable JSON reports are retained as
workflow artifacts for 14 days for full investigation, and the job fails when
it has findings or cannot complete successfully.

The final **Security gate** job passes only when both SAST scanners and the
dependency audit pass. Configure `Security gate` as a required status check in
the default branch's GitHub ruleset to prevent a pull request with findings from
being merged. Repository rulesets are an administrative GitHub setting and
cannot be enabled by a workflow without granting it repository-administration
credentials.

To run the same scanners locally in an isolated environment:

```bash
python -m venv .venv-security
. .venv-security/bin/activate
python -m pip install ".[dev]" "bandit[toml]" semgrep pip-audit
bandit --configfile pyproject.toml --recursive kiokufux
SEMGREP_SEND_METRICS=off semgrep scan --config p/python --config p/security-audit --config p/secrets --error --metrics off kiokufux tests
pip-audit --local --strict
```

The scanners and advisory feeds require network access when they are installed
or refreshed. GitHub-hosted runner usage remains subject to the free allowance
and public-repository terms of the repository's GitHub plan; none of the tools
requires a paid security product.
