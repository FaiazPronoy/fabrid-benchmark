import base64
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml

PACKAGES = (
    "fabrid",
    "numpy",
    "pillow",
    "pyyaml",
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "pycocotools",
    "scipy",
    "scikit-image",
    "timm",
    "transformers",
    "albumentations",
)
DEFAULT_RUNS = "/kaggle/working/runs"
HEAVY_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".onnx", ".zip"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_LIMIT = 200 * 1024


def git(*args: str, cwd: Path | None = None, extra: list[str] | None = None):
    command = ["git", *(extra or []), *args]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def repo_root() -> Path:
    if os.environ.get("FABRID_REPO"):
        return Path(os.environ["FABRID_REPO"]).resolve()
    result = git("rev-parse", "--show-toplevel")
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).resolve()
    return Path.cwd().resolve()


def runs_dir(step: str, name: str) -> Path:
    path = Path(os.environ.get("FABRID_RUNS", DEFAULT_RUNS)) / step / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def versions() -> dict[str, str]:
    found = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return found


def commit_hash(root: Path) -> tuple[str | None, bool | None]:
    head = git("rev-parse", "HEAD", cwd=root)
    if head.returncode != 0:
        return None, None
    status = git("status", "--porcelain", "--untracked-files=no", cwd=root)
    return head.stdout.strip(), bool(status.stdout.strip())


def start_run(step: str, name: str, config: dict) -> Path:
    root = repo_root()
    run_dir = root / "results" / step / name
    run_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = commit_hash(root)
    record = {
        "config": config,
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": versions(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False)
    return run_dir


def redact(text: str, token: str | None) -> str:
    if not token:
        return text
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return text.replace(token, "***").replace(encoded, "***")


def auth_args(token: str) -> list[str]:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {encoded}"]


def publishable(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in HEAVY_SUFFIXES:
        return False
    return not (suffix in IMAGE_SUFFIXES and path.stat().st_size > IMAGE_LIMIT)


def publish(run_dir: str | Path, retries: int = 5) -> bool:
    run_dir = Path(run_dir).resolve()
    token = os.environ.get("GITHUB_TOKEN")
    repo_url = os.environ.get("REPO_URL")
    if not token or not repo_url:
        print(f"publish skipped for {run_dir}: GITHUB_TOKEN or REPO_URL not set")
        return False
    root = repo_root()
    files = [p for p in sorted(run_dir.rglob("*")) if p.is_file()]
    kept = [str(p.relative_to(root)) for p in files if publishable(p)]
    skipped = [str(p.relative_to(root)) for p in files if not publishable(p)]
    for path in skipped:
        print(f"not committed (large file): {path}")
    if not kept:
        print(f"publish skipped for {run_dir}: nothing to commit")
        return False
    identity = [
        "-c",
        f"user.name={os.environ.get('GIT_USER_NAME', 'fabrid')}",
        "-c",
        f"user.email={os.environ.get('GIT_USER_EMAIL', 'fabrid@users.noreply.github.com')}",
    ]
    git("add", "--", *kept, cwd=root)
    if git("diff", "--cached", "--quiet", cwd=root).returncode == 0:
        print(f"publish skipped for {run_dir}: no changes")
        return False
    message = f"Add {run_dir.relative_to(root / 'results').as_posix()} results"
    committed = git("commit", "-m", message, "--", *kept, cwd=root, extra=identity)
    if committed.returncode != 0:
        print(f"commit failed: {redact(committed.stderr.strip(), token)}")
        return False
    auth = [*auth_args(token), *identity]
    for attempt in range(1, retries + 1):
        steps = (
            ("pull", "--rebase", "--autostash", repo_url, "main"),
            ("push", repo_url, "HEAD:main"),
        )
        failure = None
        for step in steps:
            result = git(*step, cwd=root, extra=auth)
            if result.returncode != 0:
                failure = f"git {step[0]}: {redact(result.stderr.strip(), token)}"
                if step[0] == "pull":
                    git("rebase", "--abort", cwd=root)
                break
        if failure is None:
            print(f"published {run_dir.relative_to(root).as_posix()}")
            return True
        print(f"publish attempt {attempt} failed, {failure}")
        if attempt < retries:
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    print(f"publish gave up after {retries} attempts; results remain in {run_dir}")
    return False
