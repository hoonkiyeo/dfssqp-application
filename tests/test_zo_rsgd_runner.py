"""ZO-RSGD runner checks use inexpensive losses; no neural models are trained."""
import csv
import fcntl
import json

import numpy as np
import pytest

import config
from experiments import run as storage
from experiments import zo_rsgd as runner


@pytest.fixture
def small(monkeypatch, tmp_path):
    settings = {
        "zo_rsgd_step_coefficient": 1.0, "zo_rsgd_mu": 0.01, "zo_rsgd_m": 1,
        "zo_rsgd_steps": 3, "zo_rsgd_run_seeds": [71, 72, 73],
        "zo_rsgd_evaluation_seeds": [101, 102],
        "zo_rsgd_evaluation_steps": [0, 1, 3],
        "zo_rsgd_experiment_name": "runner_test",
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value, raising=False)
    metadata = (runner.ROOT / "data/processed/metadata.json").read_text()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "fingerprints", lambda: {"inputs": "fixed"})
    data = tmp_path / "data/processed"
    data.mkdir(parents=True)
    (data / "metadata.json").write_text(metadata)
    for name in ("training", "validation"):
        np.savez(data / f"{name}.npz", z=np.zeros((64, 48)), u=np.zeros(64), y=np.zeros(64))
    calls = []
    target = np.linspace(-1, 1, 48)

    def loss(w, seed, training, validation):
        calls.append((np.asarray(w).copy(), int(seed)))
        return float(np.mean((np.asarray(w) - target)**2) + (seed % 17)*0.001)

    monkeypatch.setattr(storage, "evaluate", loss)
    return tmp_path, calls, loss


def events(folder):
    return [event for path in folder.rglob("fits.jsonl")
            for event in storage.read_journal(path)]


def scientific_results(folder):
    result = {}
    for directory in sorted(folder.glob("run_*")):
        if not directory.is_dir():
            continue
        result[directory.name] = {
            "history": storage.read_journal(directory / "history.jsonl"),
            "fits": {e["key"]: (e["seed"], e["w_sha256"], e["mse"])
                     for e in storage.read_journal(directory / "fits.jsonl")
                     if e["event"] == "done"},
        }
    result["initial"] = {
        e["key"]: (e["seed"], e["w_sha256"], e["mse"])
        for e in storage.read_journal(folder / "initial/fits.jsonl") if e["event"] == "done"
    }
    return result


def test_prepare_freezes_budget_without_fits(small):
    root, calls, _ = small
    folder = runner.prepare(root / "prepared")
    assert calls == [] and not list(folder.rglob("fits.jsonl"))
    assert json.loads((folder / "status.json").read_text())["state"] == "prepared"
    budget = json.loads((folder / "budget.json").read_text())
    assert budget["optimization_fits"] == 18
    assert budget["evaluation_fits"] == 14
    assert budget["total_fits"] == 32
    settings = json.loads((folder / "settings.json").read_text())["config"]
    assert settings["zo_rsgd_step_coefficient"] == 1.0
    assert settings["zo_rsgd_steps"] == 3 and settings["zo_rsgd_m"] == 1
    assert settings["zo_rsgd_mu"] == 0.01


def test_full_resume_frozen_horizon_and_idempotent_completion(small, monkeypatch):
    root, calls, _ = small
    full = runner.prepare(root / "full")
    runner.execute(full)
    expected = scientific_results(full)
    assert len(calls) == 32
    resumed = runner.prepare(root / "resumed")
    runner.execute(resumed, stop_after=1)
    assert json.loads((resumed / "status.json").read_text())["state"] == "paused"
    for name, value in {"zo_rsgd_steps": 99, "zo_rsgd_step_coefficient": 99.,
                        "zo_rsgd_mu": 0.5, "zo_rsgd_run_seeds": [999],
                        "zo_rsgd_evaluation_seeds": [998]}.items():
        monkeypatch.setattr(config, name, value)
    runner.execute(resumed)
    assert scientific_results(resumed) == expected
    assert (resumed / "summary.csv").read_text() == (full / "summary.csv").read_text()
    assert config.zo_rsgd_steps == 3 and config.zo_rsgd_step_coefficient == 1.
    for name, result in expected.items():
        if name == "initial":
            continue
        for row in result["history"][1:]:
            assert row["metrics"]["eta"] == pytest.approx(1 / np.sqrt(3))
    assert len(calls) == 64
    runner.execute(resumed)
    assert len(calls) == 64
    costs = json.loads((resumed / "costs.json").read_text())
    assert costs["opt"]["done"] == 18 and costs["eval"]["done"] == 14
    assert costs["total_done"] == 32
    assert json.loads((resumed / "status.json").read_text())["state"] == "completed"


@pytest.mark.parametrize("interrupt_call", [2, 3, 4, 5])
def test_interrupted_initial_current_probe_or_report_reuses_done_fits(small, monkeypatch, interrupt_call):
    root, calls, loss = small
    full = runner.prepare(root / "reference")
    runner.execute(full)
    resumed = runner.prepare(root / "interrupted")
    count = 0

    def interrupt(*args):
        nonlocal count
        count += 1
        if count == interrupt_call:
            raise KeyboardInterrupt("simulated interruption")
        return loss(*args)

    monkeypatch.setattr(storage, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(resumed)
    monkeypatch.setattr(storage, "evaluate", loss)
    runner.execute(resumed)
    assert scientific_results(resumed) == scientific_results(full)
    assert len(calls) == 64
    journal = events(resumed)
    assert sum(e["event"] == "done" for e in journal) == 32
    assert sum(e["event"] == "interrupted" for e in journal) == 1
    assert sum(e["event"] == "start" for e in journal) == 33


def test_reporting_independence_seed_pairing_and_summary_se(small, monkeypatch):
    root, _, _ = small
    dense = runner.prepare(root / "dense")
    runner.execute(dense)
    baseline = scientific_results(dense)
    initial_w = None
    for directory in dense.glob("run_*"):
        history = storage.read_journal(directory / "history.jsonl")
        assert [r["state"]["k"] for r in history] == [0, 1, 2, 3]
        w0 = history[0]["state"]["w"]
        assert initial_w is None or initial_w == w0
        initial_w = w0
        done = {e["key"]: e for e in storage.read_journal(directory / "fits.jsonl")
                if e["event"] == "done"}
        for k in range(3):
            pair = [e for key, e in done.items() if key.startswith(f"opt:{k}:")]
            assert len(pair) == 2 and pair[0]["seed"] == pair[1]["seed"] >= 2**32
        for key, event in done.items():
            if key.startswith("eval:"):
                assert event["seed"] in config.zo_rsgd_evaluation_seeds
                assert event["seed"] < 2**32
        assert all(row["metrics"]["feasibility"] < 1e-14 for row in history)
    with (dense / "summary.csv").open() as file:
        summary = list(csv.DictReader(file))
    assert len(summary) == 3
    for row in summary:
        k = int(row["k"])
        if k == 0:
            assert float(row["se_mse"]) == 0.
        means = []
        for directory in dense.glob("run_*"):
            panel = (baseline["initial"] if k == 0 else baseline[directory.name]["fits"])
            means.append(np.mean([panel[f"eval:{k}:{s}"][2]
                                  for s in config.zo_rsgd_evaluation_seeds]))
        assert int(row["runs"]) == 3
        assert float(row["mean_mse"]) == pytest.approx(np.mean(means))
        assert float(row["se_mse"]) == pytest.approx(np.std(means, ddof=1) / np.sqrt(3))
    monkeypatch.setattr(config, "zo_rsgd_evaluation_steps", [0, 3])
    monkeypatch.setattr(config, "zo_rsgd_evaluation_seeds", [103])
    sparse = runner.prepare(root / "sparse")
    runner.execute(sparse)
    reduced = scientific_results(sparse)
    for name, result in baseline.items():
        if name == "initial":
            continue
        assert result["history"] == reduced[name]["history"]
        assert {k: v for k, v in result["fits"].items() if k.startswith("opt:")} == {
            k: v for k, v in reduced[name]["fits"].items() if k.startswith("opt:")}


def test_zero_gradient_still_advances_and_reports(small, monkeypatch):
    root, _, _ = small
    monkeypatch.setattr(storage, "evaluate", lambda *args: 0.5)
    folder = runner.prepare(root / "zero")
    runner.execute(folder)
    for directory in folder.glob("run_*"):
        history = storage.read_journal(directory / "history.jsonl")
        assert [row["state"]["k"] for row in history] == [0, 1, 2, 3]
        for row in history:
            np.testing.assert_allclose(row["state"]["w"], history[0]["state"]["w"], atol=1e-15)
    assert sum(e["event"] == "done" for e in events(folder)) == 32
    assert json.loads((folder / "status.json").read_text())["state"] == "completed"


def test_lock_fingerprints_nonfinite_loss_and_recovery(small, monkeypatch):
    root, calls, loss = small
    folder = runner.prepare(root / "protected")
    with (folder / "runner.lock").open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already"):
            runner.execute(folder)
    monkeypatch.setattr(runner, "fingerprints", lambda: {"inputs": "changed"})
    with pytest.raises(ValueError, match="changed"):
        runner.execute(folder)
    assert calls == []
    monkeypatch.setattr(runner, "fingerprints", lambda: {"inputs": "fixed"})
    monkeypatch.setattr(storage, "evaluate", lambda *args: np.nan)
    with pytest.raises(FloatingPointError):
        runner.execute(folder)
    assert json.loads((folder / "status.json").read_text())["state"] == "failed"
    monkeypatch.setattr(storage, "evaluate", loss)
    runner.execute(folder)
    assert len(calls) == 32
    assert json.loads((folder / "costs.json").read_text())["eval"]["failed"] == 1


def test_pause_file_and_partial_iteration_replay(small):
    root, calls, _ = small
    full = runner.prepare(root / "reference")
    runner.execute(full)
    folder = runner.prepare(root / "partial")
    (folder / "PAUSE").touch()
    runner.execute(folder)
    assert len(calls) == 32
    assert json.loads((folder / "status.json").read_text())["state"] == "paused"
    runner.execute(folder, stop_after=1)
    path = next(folder.glob("run_*/history.jsonl"))
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[1][:40])
    runner.execute(folder)
    assert scientific_results(folder) == scientific_results(full)
    assert len(calls) == 64
    assert len(list(path.parent.glob("history.jsonl.partial-*"))) == 1
