import yaml

from fabrid import runio


def test_start_run_writes_exact_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FABRID_REPO", str(tmp_path))
    config = {"model": "m", "arms": ["session", "random"], "epochs": 3}
    run_dir = runio.start_run("baselines", "m_session_seed0", config)
    assert run_dir == tmp_path / "results" / "baselines" / "m_session_seed0"
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["config"] == config
    assert "python" in saved["versions"]
    assert "git_commit" in saved


def test_publish_without_token_does_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert runio.publish(tmp_path) is False
    assert "skipped" in capsys.readouterr().out


def test_redact_hides_token_and_header():
    token = "ghp_secretvalue"
    header = runio.auth_args(token)[1]
    text = f"fatal: https://{token}@github.com {header}"
    cleaned = runio.redact(text, token)
    assert token not in cleaned
    assert header.split()[-1] not in cleaned


def test_large_files_not_publishable(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"0")
    big = tmp_path / "plot.png"
    big.write_bytes(b"0" * (runio.IMAGE_LIMIT + 1))
    table = tmp_path / "metrics.json"
    table.write_text("{}", encoding="utf-8")
    assert not runio.publishable(weights)
    assert not runio.publishable(big)
    assert runio.publishable(table)
