import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_jobs() -> dict[str, dict[str, str]]:
    with (REPO / "kaggle" / "jobs.csv").open(newline="", encoding="utf-8") as handle:
        return {row["job"]: row for row in csv.DictReader(handle)}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python kaggle/launch.py <job>")
    jobs = load_jobs()
    name = sys.argv[1]
    if name not in jobs:
        raise SystemExit(f"unknown job {name}, choose from {sorted(jobs)}")
    job = jobs[name]
    install = [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[kaggle]"]
    if subprocess.run(install, check=False).returncode != 0:
        raise SystemExit("package install failed")
    if job["device"] == "gpu" and shutil.which("nvidia-smi") is None:
        print("warning: job expects a GPU but none is visible", flush=True)
    print(f"running {name}: {job['command']}", flush=True)
    code = subprocess.run(job["command"], shell=True, cwd=REPO, check=False).returncode
    if code != 0:
        raise SystemExit(f"job {name} exited with code {code}")
    print(f"job {name} finished", flush=True)


if __name__ == "__main__":
    main()
