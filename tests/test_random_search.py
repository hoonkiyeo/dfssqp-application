"""Checks for the agreed two-candidate random-search iteration."""

import json

import numpy as np
import pytest

from algorithms.df_ssqp import initialize as initialize_df_ssqp
from algorithms.random_search import initialize, step


def test_shared_initial_direction():
    state = initialize()
    assert state["k"] == 0
    np.testing.assert_array_equal(state["w"], initialize_df_ssqp()["w"])


@pytest.mark.parametrize("losses,selected", [((1., 2.), "+"), ((2., 1.), "-"), ((1., 1.), "+")])
def test_raw_paired_evaluations_and_selection(losses, selected):
    state = initialize()
    original_w = state["w"].copy()
    calls = []

    def loss(w, seed):
        calls.append((w.copy(), seed))
        return losses[len(calls) - 1]

    result, trace = step(state, 71, 0.1, loss)
    assert len(calls) == 2
    assert calls[0][1] == calls[1][1] == trace["seed"]
    assert trace["seed"] >= 2**32
    np.testing.assert_allclose((calls[0][0] + calls[1][0]) / 2, original_w, rtol=0, atol=1e-16)
    for candidate, _ in calls:
        np.testing.assert_allclose(np.linalg.norm(candidate - original_w), 0.1, rtol=1e-14)
    # Both probes cannot already have unit norm when eta > 0.
    assert not all(np.isclose(np.linalg.norm(candidate), 1, rtol=0, atol=1e-14) for candidate, _ in calls)
    chosen = calls[0 if selected == "+" else 1][0]
    np.testing.assert_array_equal(result["w"], chosen / np.linalg.norm(chosen))
    np.testing.assert_allclose(np.linalg.norm(result["w"]), 1, rtol=0, atol=3e-16)
    assert trace["selected"] == selected
    assert result["k"] == 1
    assert state["k"] == 0
    np.testing.assert_array_equal(state["w"], original_w)


def test_eta_shares_direction_and_seed_but_new_iteration_and_run_do_not():
    state = initialize()
    rng_before = np.random.get_state()
    traces = [step(state, 71, eta, lambda w, seed: 1.)[1] for eta in (0.01, 0.1, 1.)]
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace["s"], traces[0]["s"])
        assert trace["seed"] == traces[0]["seed"]
    for other in (dict(state, k=1), state):
        outer_seed = 71 if other["k"] == 1 else 72
        trace = step(other, outer_seed, 0.1, lambda w, seed: 1.)[1]
        assert trace["seed"] != traces[0]["seed"]
        assert not np.array_equal(trace["s"], traces[0]["s"])
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


@pytest.mark.parametrize("eta", [0.01, 0.1, 1.])
def test_resume_from_serialized_state_is_identical(eta):
    a = np.linspace(-1, 1, len(initialize()["w"]))

    def loss(w, seed):
        return float(a @ w) + float(seed % 97) / 1000

    continuous = initialize()
    for _ in range(12):
        continuous, _ = step(continuous, 71, eta, loss)
    resumed = initialize()
    for _ in range(4):
        resumed, _ = step(resumed, 71, eta, loss)
    resumed = json.loads(json.dumps({"k": resumed["k"], "w": resumed["w"].tolist()}))
    for _ in range(8):
        resumed, _ = step(resumed, 71, eta, loss)
    assert resumed["k"] == continuous["k"] == 12
    np.testing.assert_array_equal(resumed["w"], continuous["w"])


@pytest.mark.parametrize("eta", [0., -1., np.nan, np.inf])
def test_invalid_eta_does_not_evaluate(eta):
    def loss(w, seed):
        pytest.fail("Invalid eta must fail before training.")

    with pytest.raises(ValueError, match="eta"):
        step(initialize(), 71, eta, loss)


def test_nonfinite_loss_is_rejected():
    with pytest.raises(FloatingPointError, match="losses"):
        step(initialize(), 71, 0.1, lambda w, seed: np.nan)


def test_zero_selected_candidate_is_rejected():
    # Recover this iteration's direction, then make its negative probe exactly zero.
    _, trace = step(initialize(), 71, 1., lambda w, seed: 1.)
    state = {"k": 0, "w": trace["s"].copy()}
    losses = iter((2., 1.))
    with pytest.raises(FloatingPointError, match="selected candidate"):
        step(state, 71, 1., lambda w, seed: next(losses))
