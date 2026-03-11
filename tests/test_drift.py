import os
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFT_SCRIPT = os.path.join(BASE_DIR, "monitoring", "generate_evidently_drift.py")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DRIFT_SUMMARY_PATH = os.path.join(REPORTS_DIR, "drift_summary.json")


def test_drift_script_runs_successfully():
    result = subprocess.run(
        ["python", DRIFT_SCRIPT],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"


def test_drift_summary_file_created():
    assert os.path.exists(DRIFT_SUMMARY_PATH), "drift_summary.json was not created"


def test_drift_summary_has_expected_keys():
    with open(DRIFT_SUMMARY_PATH, "r") as f:
        data = json.load(f)

    assert "status" in data
    assert "baseline_rows" in data
    assert "current_rows" in data
