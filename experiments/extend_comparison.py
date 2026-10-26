"""Prepare 10,000 iterations: continue DF/MiSTP, restart horizon-dependent ZO.

python -m experiments.extend_comparison --source runs/main_comparison_20261025 \
    --output runs/main_comparison_10000_20261026

Preparation only. Existing native runners handle any later authorized execution.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from experiments import run as df_runner, zo_rsgd as zo_runner, mistp as mistp_runner
from experiments.pilot import settings


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    ("df_ai", df_runner, "experiments.run", "sqp_steps", "evaluation_steps", 2),
    ("df_zi", df_runner, "experiments.run", "sqp_steps", "evaluation_steps", 2),
    ("zo_rsgd", zo_runner, "experiments.zo_rsgd", "zo_rsgd_steps", "zo_rsgd_evaluation_steps", 2),
    ("mistp", mistp_runner, "experiments.mistp", "mistp_steps", "mistp_evaluation_steps", 3),
)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(cases, steps, points, panel, calls, shared_initial):
    opt = cases * steps * calls
    evaluation = panel * (1 + cases * (len(points) - 1) if shared_initial else cases * len(points))
    return {"optimization_fits": opt, "evaluation_fits": evaluation, "total_fits": opt + evaluation}


def prepare(source, folder, steps=10000, report_every=500):
    """Freeze new settings, copy reusable prefixes, and record launch commands only."""
    source, folder = Path(source).resolve(), Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    if source == folder or source in folder.parents or folder in source.parents:
        raise ValueError("Source and output must be separate campaign folders.")
    if type(steps) is not int or type(report_every) is not int or report_every < 1:
        raise ValueError("Use an integer horizon and positive reporting interval.")
    read_json(source / "protocol.json")
    plans, reference, source_hashes = [], None, {}

    # Validate all four completed source campaigns before creating any output.
    for name, runner, module, horizon_key, points_key, calls in METHODS:
        child = source / name
        saved = read_json(child / "settings.json")
        cfg, current = saved["config"], runner.fingerprints()
        old_steps = cfg[horizon_key]
        if steps <= old_steps:
            raise ValueError("The new horizon must exceed every source horizon.")
        if read_json(child / "status.json")["state"] != "completed":
            raise ValueError(f"Source campaign is incomplete: {name}")
        if saved["runtime"] != df_runner.runtime():
            raise ValueError(f"Source runtime changed: {name}")
        changed = {}
        for path, old_hash in saved["sha256"].items():
            if current.get(path) != old_hash:
                if not path.endswith(".py"):
                    raise ValueError(f"Source data changed: {path}")
                changed[path] = {"source_sha256": old_hash, "prepared_sha256": current.get(path)}
        common = {key: cfg[key] for key in (
            "initialization_seed", "dimension", "outer_seeds", "evaluation_seeds",
            "training_steps", "learning_rate", "hidden_width", "activation", "batch_size",
            "iota1", "iota2", "iota3", "p1", "p2", "p3", "sigma", "epsilon",
            "L_f", "L_c", "jacobian_lower", "jacobian_upper", "psi", "p",
            "standardize", "standardization_ddof", "dtype", "device",
        )}
        identity = (old_steps, common)
        if reference is None:
            reference = identity
        if identity != reference:
            raise ValueError("Source campaigns do not share horizons, seeds and fixed settings.")
        prefix = "mistp_" if name == "mistp" else "zo_rsgd_" if name == "zo_rsgd" else ""
        seeds = cfg[prefix + "run_seeds"] if prefix else cfg["outer_seeds"]
        panel = cfg[prefix + "evaluation_seeds"]
        if seeds != cfg["outer_seeds"] or panel != cfg["evaluation_seeds"]:
            raise ValueError("Method-specific run or reporting seeds differ.")
        if name == "mistp" and len(cfg["mistp_alphas"]) != 1:
            raise ValueError("Continue a single fixed MiSTP step size.")
        directories = []
        for seed in seeds:
            relative = (f"{name[-2:].upper()}_{seed}" if name.startswith("df_") else
                        f"alpha_{float(cfg['mistp_alphas'][0])}/run_{seed}" if name == "mistp" else
                        f"run_{seed}")
            run = child / relative
            with (run / "history.jsonl").open() as file:
                history = [json.loads(line) for line in file]
            if [r["state"]["k"] for r in history] != list(range(old_steps + 1)):
                raise ValueError(f"Incomplete or non-contiguous source history: {run}")
            rows = list(csv.DictReader((run / "evaluations.csv").open()))
            if ([int(r["k"]) for r in rows] != cfg[points_key]
                    or any(r["panel_complete"] != "True" or int(r["evaluation_fits"]) != len(panel)
                           for r in rows)):
                raise ValueError(f"Incomplete source reporting panels: {run}")
            directories.append(Path(relative))
        if name == "mistp":
            directories.append(Path("initial"))
        points = sorted(set(cfg[points_key] + list(range((old_steps // report_every + 1) * report_every, steps, report_every)) + [steps]))
        if name == "mistp" and old_steps < (2 * steps) // 3 < steps:
            points = sorted(set(points + [(2 * steps) // 3]))
        total = budget(len(seeds), steps, points, len(panel), calls, bool(prefix))
        inherited = (budget(len(seeds), old_steps, cfg[points_key], len(panel), calls, bool(prefix))
                     if name != "zo_rsgd" else {key: 0 for key in total})
        additional = {key: total[key] - inherited[key] for key in total}
        # Hash only native scientific prefixes and settings; original files are never modified.
        copied = []
        if name != "zo_rsgd":
            for directory in directories:
                copied.extend(p for p in (child / directory).rglob("*") if p.is_file())
        tracked = copied + [child / "settings.json", child / "status.json"]
        hashes = {str(p.relative_to(source)): digest(p) for p in tracked}
        source_hashes.update(hashes)
        plans.append((name, runner, module, horizon_key, points_key, cfg, points,
                      total, inherited, additional, directories, changed))

    folder.mkdir(parents=True)
    campaigns = []
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    for name, runner, module, horizon_key, points_key, cfg, points, total, inherited, additional, directories, changed in plans:
        with settings(**dict(cfg, **{horizon_key: steps, points_key: points})):
            child = runner.prepare(folder / name)
        if name != "zo_rsgd":
            for relative in directories:
                shutil.copytree(source / name / relative, child / relative)
            for relative, expected in source_hashes.items():
                rel = Path(relative)
                if rel.parts[0] != name or rel.name in ("settings.json", "status.json"):
                    continue
                copied = folder / rel
                if digest(copied) != expected or copied.stat().st_ino == (source / rel).stat().st_ino:
                    raise ValueError(f"Prefix copy differs or is a hardlink: {rel}")
        shutil.copyfile(source / name / "settings.json", child / "source_settings.json")
        shutil.copyfile(source / name / "status.json", child / "source_status.json")
        df_runner.save_json(child / "budget.json", total)
        df_runner.save_json(child / "status.json", {
            "state": "prepared" if name == "zo_rsgd" else "prepared_extension", "pid": None,
            "k": 0 if name == "zo_rsgd" else reference[0],
        })
        command = [sys.executable, "-m", module, "--resume", str(child), "--background"]
        campaigns.append({"name": name, "folder": name, "module": module,
            "start_step": 0 if name == "zo_rsgd" else reference[0], "horizon": steps,
            "mode": "fresh_horizon_dependent_step" if name == "zo_rsgd" else "continue_saved_prefix",
            "source_changes": changed, "budget": total, "inherited_budget": inherited,
            "additional_budget": additional, "command": command, "shell_command": shlex.join(command),
            "settings_sha256": digest(child / "settings.json")})

    if any(digest(source / path) != value for path, value in source_hashes.items()):
        raise ValueError("Original source files changed during preparation.")
    totals = {scope: {key: sum(c[scope][key] for c in campaigns) for key in ("optimization_fits", "evaluation_fits", "total_fits")}
              for scope in ("budget", "inherited_budget", "additional_budget")}
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "git_commit": revision, "horizon": steps, "source_horizon": reference[0], "campaigns": campaigns,
        "source_files_sha256": source_hashes, "source_files_unchanged": True, **totals,
        "preparation_sha256": digest(Path(__file__)), "test_data_accessed": False,
        "note": "Preparation only. Current code changes are versioned for independent audit; ZO is fresh with c/sqrt(new N)."}
    df_runner.save_json(folder / "protocol.json", protocol)
    df_runner.save_json(folder / "budget.json", totals)
    df_runner.save_json(folder / "status.json", {"state": "prepared", "pid": None})
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.steps))
