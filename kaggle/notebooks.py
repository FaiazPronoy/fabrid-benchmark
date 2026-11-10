import json
from pathlib import Path

from launch import load_jobs

HERE = Path(__file__).resolve().parent
CELL = """import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

JOB = "{job}"

os.environ.setdefault("FABRID_JOB_START", str(time.time()))
os.environ.setdefault("FABRID_DATA", "/kaggle/input")
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for key in ("GITHUB_TOKEN", "GIT_USER_NAME", "GIT_USER_EMAIL", "REPO_URL"):
        if not os.environ.get(key):
            try:
                os.environ[key] = secrets.get_secret(key)
            except Exception:
                print(f"secret {{key}} not set")
except ImportError:
    print("kaggle_secrets not available, using environment only")

REPO_URL = os.environ.get("REPO_URL") or "https://github.com/FaiazPronoy/fabrid-benchmark.git"
os.environ["REPO_URL"] = REPO_URL
REPO = Path("/kaggle/working/fabrid-benchmark")
auth = []
if os.environ.get("GITHUB_TOKEN"):
    encoded = base64.b64encode(f"x-access-token:{{os.environ['GITHUB_TOKEN']}}".encode()).decode()
    auth = ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {{encoded}}"]
if REPO.exists():
    shutil.rmtree(REPO)
if subprocess.run(["git", *auth, "clone", "--quiet", REPO_URL, str(REPO)]).returncode != 0:
    raise RuntimeError("git clone failed")
launch = [sys.executable, "-u", str(REPO / "kaggle" / "launch.py"), JOB]
process = subprocess.Popen(launch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait() != 0:
    raise RuntimeError(f"job {{JOB}} failed")
"""


def notebook(job: str) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL.format(job=job).rstrip("\n").splitlines(keepends=True),
    }
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return {"cells": [cell], "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    out_dir = HERE / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for job in load_jobs():
        path = out_dir / f"{job}.ipynb"
        path.write_text(json.dumps(notebook(job), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
