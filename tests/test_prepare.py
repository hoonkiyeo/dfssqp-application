"""Small independent checks of the daily-to-monthly calculation."""

import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

import config
from data.prepare import prepare_monthly, read_daily


def example_returns():
    dates = pd.to_datetime([
        "2020-01-02", "2020-01-03", "2020-01-31", "2020-02-03", "2020-02-28",
        "2020-03-02", "2020-03-03", "2020-03-31", "2020-04-01", "2020-04-30",
        "2020-05-01", "2020-05-29", "2020-06-01", "2020-07-01",
    ])
    a = 0.01 * np.array([1, -1, 2, -2, 3, -3, 4, -4, 1, -2, 2, -3, 5, 6])
    return pd.DataFrame({"A": a, "B": 2 * a}, index=dates)


def prepare_example(frame):
    return prepare_monthly(frame, ("2020-01", "2020-03"),
                           ("2020-04", "2020-05"), ("2020-06", "2020-07"), ddof=1)


def test_monthly_formulas_and_next_month_pairing():
    data, b, _ = prepare_example(example_returns())
    np.testing.assert_allclose(b, [2/3, 1/3])
    # B=2*A, so daily aggregate return is (4/3)*A, BEFORE squaring.
    industry_rv = 0.0001 * np.array([[6, 24], [13, 52], [41, 164]])
    aggregate_rv = (16/9) * 0.0001 * np.array([6, 13, 41])
    train = data["training"]
    np.testing.assert_allclose(train["monthly_industry_rv"], industry_rv)
    np.testing.assert_allclose(train["monthly_aggregate_rv"], aggregate_rv)
    np.testing.assert_allclose(train["z"], np.log(industry_rv[:-1]))
    np.testing.assert_allclose(train["u"], np.log(aggregate_rv[:-1]))
    np.testing.assert_allclose(train["y"], np.log(aggregate_rv[1:]))
    assert train["predictor_months"].tolist() == ["2020-01", "2020-02"]
    assert train["response_months"].tolist() == ["2020-02", "2020-03"]
    assert data["validation"]["predictor_months"].tolist() == ["2020-04"]
    assert data["validation"]["response_months"].tolist() == ["2020-05"]


def test_benchmark_uses_training_only_and_test_is_unused():
    frame = example_returns()
    original, b, sd = prepare_example(frame)
    frame.loc["2020-04":"2020-05", "A"] *= 7
    changed, b_new, sd_new = prepare_example(frame)
    np.testing.assert_array_equal(b, b_new)
    np.testing.assert_array_equal(sd, sd_new)
    for key in original["training"]:
        np.testing.assert_array_equal(original["training"][key], changed["training"][key])
    assert not np.array_equal(original["validation"]["y"], changed["validation"]["y"])
    frame.loc["2020-06":] = np.nan
    after_test_change, _, _ = prepare_example(frame)
    for split in changed:
        for key in changed[split]:
            np.testing.assert_array_equal(changed[split][key], after_test_change[split][key])


@pytest.mark.parametrize("problem", ["missing_return", "missing_month", "zero_rv"])
def test_invalid_monthly_inputs_are_rejected(problem):
    frame = example_returns()
    if problem == "missing_return":
        frame.iloc[0, 0] = np.nan
    elif problem == "missing_month":
        frame = frame.loc[frame.index.month != 2]
    else:
        frame.loc["2020-04"] = 0
    with pytest.raises(ValueError):
        prepare_example(frame)


def test_source_units_sentinels_calendar_and_hash(tmp_path, monkeypatch):
    texts = {
        "returns": "This file uses the 202007 CRSP database.\n"
                   "Average Value Weighted Returns -- Daily\nA B\n"
                   "20200102 1 -2\n20200103 -99.99 -999\n\n",
        "calendar": "This file uses the 202007 CRSP database.\n"
                    "Mkt-RF SMB HML RF\n20200102 999 999 999 999\n"
                    "20200103 999 999 999 999\n\n",
    }
    monkeypatch.setattr(config, "dimension", 2)
    monkeypatch.setattr(config, "portfolio_weighting", "value")

    def write_source(name, text):
        path = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{name}.txt", text)
        monkeypatch.setattr(config, f"{name}_file", path.name)
        monkeypatch.setattr(config, f"{name}_sha256", hashlib.sha256(path.read_bytes()).hexdigest())

    for name, text in texts.items():
        write_source(name, text)
    frame, _ = read_daily(tmp_path)
    np.testing.assert_array_equal(frame.iloc[0].to_numpy(), [0.01, -0.02])
    assert frame.iloc[1].isna().all()
    write_source("calendar", texts["calendar"].replace("20200103 999 999 999 999\n", ""))
    with pytest.raises(ValueError, match="calendar differ"):
        read_daily(tmp_path)
    monkeypatch.setattr(config, "returns_sha256", "wrong")
    with pytest.raises(ValueError, match="source hash"):
        read_daily(tmp_path)
