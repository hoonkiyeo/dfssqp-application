"""Daily returns -> monthly log realized variances -> next-month prediction pairs.

Run from the repository root: python -m data.prepare
The formulas are in prepare_monthly; read_daily only reads and checks the ZIPs.
"""

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config

ROOT = Path(__file__).resolve().parents[1]


def prepare_monthly(returns, training_months, validation_months, test_months, ddof):
    """Build unstandardized z, u, y; both months of each pair stay in one split."""
    # 1. Select training and validation days. The test period remains reserved.
    intervals = [pd.PeriodIndex(months, freq="M") for months in
                 (training_months, validation_months, test_months)]
    if any(len(p) != 2 or p[1] < p[0] + 1 for p in intervals):
        raise ValueError("Each split needs at least two months.")
    if any(a[1] + 1 != b[0] for a, b in zip(intervals[:-1], intervals[1:])):
        raise ValueError("Splits must be adjacent and disjoint.")
    if ddof not in (0, 1):
        raise ValueError("Benchmark SD ddof must be 0 or 1.")
    segments = []
    for first, last in intervals[:2]:
        daily_months = returns.index.to_period("M")
        selected = returns.loc[(daily_months >= first) & (daily_months <= last)]
        months = pd.period_range(first, last, freq="M")
        if not selected.index.to_period("M").unique().equals(months):
            raise ValueError("Missing month in the selected daily data.")
        values = selected.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Missing or nonfinite training/validation return.")
        segments.append((selected.index, months, values))

    # 2. Fit fixed benchmark weights b_j proportional to 1 / SD_j on TRAINING days.
    sd = np.std(segments[0][2], axis=0, ddof=ddof)
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError("Training industry SDs must be finite and positive.")
    inverse_sd = np.min(sd) / sd  # Scaled reciprocals; the scale cancels below.
    b = inverse_sd / np.sum(inverse_sd)

    data = {}
    for name, (dates, months, r) in zip(("training", "validation"), segments):
        # 3. Aggregate DAILY returns first, then sum squares within each month.
        # r_aggregate,t = sum_j b_j r_tj; RV_m = sum_{t in month m} r_t**2.
        r_aggregate = r @ b
        month_index = months.get_indexer(dates.to_period("M"))
        industry_rv = np.zeros((len(months), r.shape[1]), dtype=np.float64)
        aggregate_rv = np.zeros(len(months), dtype=np.float64)
        np.add.at(industry_rv, month_index, np.square(r))
        np.add.at(aggregate_rv, month_index, np.square(r_aggregate))
        if (not np.isfinite(industry_rv).all() or not np.isfinite(aggregate_rv).all()
                or np.any(industry_rv <= 0) or np.any(aggregate_rv <= 0)):
            raise ValueError("Monthly realized variances must be finite and positive.")

        # 4. z_m = log(industry RV_m), u_m = log(aggregate RV_m), y_m = u_{m+1}.
        log_industry = np.log(industry_rv)
        log_aggregate = np.log(aggregate_rv)
        data[name] = {
            "z": log_industry[:-1],
            "u": log_aggregate[:-1],
            "y": log_aggregate[1:],
            "predictor_months": months[:-1].astype(str).to_numpy(dtype=str),
            "response_months": months[1:].astype(str).to_numpy(dtype=str),
            "industry_names": np.asarray(returns.columns, dtype=str),
            "monthly_industry_rv": industry_rv,
            "monthly_aggregate_rv": aggregate_rv,
            "trading_day_counts": np.bincount(month_index, minlength=len(months)),
        }
    return data, b, sd


def read_daily(root=ROOT):
    """Read the frozen French return block; the factor file supplies DATES only."""
    if config.portfolio_weighting not in ("value", "equal"):
        raise ValueError("Portfolio weighting must be 'value' or 'equal'.")
    title = f"Average {config.portfolio_weighting.title()} Weighted Returns -- Daily"
    sources = {}
    date_grids = []
    for name, filename, expected_hash in (
        ("returns", config.returns_file, config.returns_sha256),
        ("calendar", config.calendar_file, config.calendar_sha256),
    ):
        path = root / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(f"{name}: source hash differs from config.py.")
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != 1 or not members[0].lower().endswith(".txt"):
                raise ValueError("Expected one TXT file per ZIP.")
            lines = archive.read(members[0]).decode("utf-8-sig").splitlines()
        release = re.search(r"\b(\d{6}) CRSP database\.", lines[0])
        if release is None:
            raise ValueError("Missing CRSP release month.")
        release = release.group(1)
        marker = title if name == "returns" else "Mkt-RF SMB HML RF"
        positions = [i for i, line in enumerate(lines) if " ".join(line.split()) == marker]
        if len(positions) != 1:
            raise ValueError(f"Expected one {name} table.")
        start = positions[0] + 1
        if name == "returns":
            industries = lines[start].split()
            if len(industries) != config.dimension or len(set(industries)) != config.dimension:
                raise ValueError("Industry columns differ from the configured dimension.")
            start += 1
        rows = []
        for line in lines[start:]:
            if not line.strip():
                break
            cells = line.split()
            width = config.dimension + 1 if name == "returns" else 5
            if len(cells) != width or re.fullmatch(r"\d{8}", cells[0]) is None:
                raise ValueError(f"Malformed daily row in {name}.")
            rows.append(cells)
        dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in rows], format="%Y%m%d"))
        if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("Daily dates must be nonempty, unique and increasing.")
        if dates[-1].to_period("M") > pd.Period(release, freq="M"):
            raise ValueError("Daily dates extend beyond the declared release.")
        date_grids.append(dates)
        sources[name] = {"file": filename, "sha256": digest, "crsp_release": release}
        if name == "returns":
            r_percent = np.asarray([row[1:] for row in rows], dtype=np.float64)
            if not np.isfinite(r_percent).all():
                raise ValueError("Unexpected nonfinite raw return.")
            r_percent[np.isin(r_percent, (-99.99, -999.0))] = np.nan
            if np.any(r_percent < -100):
                raise ValueError("Simple return below -100 percent.")
            returns = pd.DataFrame(r_percent / 100, index=dates, columns=industries)
    if (not date_grids[0].equals(date_grids[1])
            or sources["returns"]["crsp_release"] != sources["calendar"]["crsp_release"]):
        raise ValueError("Returns and reference calendar differ in dates or release.")
    return returns, sources


def main():
    returns, sources = read_daily()
    data, b, sd = prepare_monthly(
        returns, config.training_months, config.validation_months,
        config.test_months, config.benchmark_ddof,
    )
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    for name, arrays in data.items():
        np.savez(output / f"{name}.npz", **arrays)
        print(f"{name}: z {arrays['z'].shape}, u {arrays['u'].shape}, y {arrays['y'].shape}")
    metadata = {
        "sources": sources, "data_vintage": config.data_vintage,
        "portfolio_weighting": config.portfolio_weighting,
        "training_months": config.training_months, "validation_months": config.validation_months,
        "test_months_reserved": config.test_months, "test_targets_constructed": False,
        "industry_names": data["training"]["industry_names"].tolist(),
        "benchmark_weights": b.tolist(), "training_daily_sd": sd.tolist(),
        "benchmark_ddof": config.benchmark_ddof,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
