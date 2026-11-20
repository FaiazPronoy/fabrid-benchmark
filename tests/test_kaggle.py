import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))


def jobs() -> list[dict[str, str]]:
    with (ROOT / "kaggle" / "jobs.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_every_job_names_an_existing_module_and_config():
    for job in jobs():
        parts = job["command"].split()
        assert parts[:2] == ["python", "-m"]
        module = ROOT / "src" / Path(*parts[2].split(".")).with_suffix(".py")
        assert module.is_file(), module
        if "--config" in parts:
            config = ROOT / parts[parts.index("--config") + 1]
            assert config.is_file(), config
        assert job["device"] in ("cpu", "gpu")


def test_every_job_has_a_matching_notebook():
    import notebooks

    listed = {job["job"]: job for job in jobs()}
    for job in listed.values():
        path = ROOT / "kaggle" / "notebooks" / f"{job['job']}.ipynb"
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored == notebooks.notebook(job["job"], job["inputs"], listed)
        if job["inputs"]:
            assert stored["cells"][0]["cell_type"] == "markdown"
            text = "".join(stored["cells"][0]["source"])
            assert all(f"`{name}`" in text for name in job["inputs"].split())
        source = "".join(stored["cells"][-1]["source"])
        compile(source, str(path), "exec")
        assert f'\nJOB = "{job["job"]}"\n' in source
        assert 'os.environ["PYTHONUNBUFFERED"] = "1"' in source
        assert "print(os.environ" not in source
