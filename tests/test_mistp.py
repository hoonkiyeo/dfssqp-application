"""Checks for the agreed three-point MiSTP update with final projection."""

import json

import numpy as np
import pytest

from algorithms.df_ssqp import initialize as initialize_df_ssqp
from algorithms.mistp import initialize, step


def test_shared_initial_direction():
    state = initialize()
    assert state["k"] == 0
    np.testing.assert_array_equal(state["w"], initialize_df_ssqp()["w"])


@pytest.mark.parametrize(
    "losses,selected,index",
    [
        ((0., 1., 2.), "current", 0),
        ((2., 0., 1.), "+", 1),
        ((2., 1., 0.), "-", 2),
        ((1., 1., 2.), "current", 0),
        ((1., 2., 1.), "current", 0),
        ((2., 1., 1.), "+", 1),
        ((1., 1., 1.), "current", 0),
    ],
)
def test_three_fresh_evaluations_selection_and_ties(losses, selected, index):
    state = initialize()
    original_w = state["w"].copy()
    calls = []

    def loss(w, seed):
        calls.append((w.copy(), seed))
        return losses[len(calls) - 1]

    result, trace = step(state, 71, 0.1, loss)
    assert len(calls) == 3
    assert all(seed == trace["seed"] for _, seed in calls)
    assert trace["seed"] >= 2**32
    np.testing.assert_array_equal(calls[0][0], original_w)
    np.testing.assert_array_equal(calls[1][0], trace["w_plus"])
    np.testing.assert_array_equal(calls[2][0], trace["w_minus"])
    np.testing.assert_allclose(
        (calls[1][0] + calls[2][0]) / 2, original_w, rtol=0, atol=1e-16
    )
    for candidate, _ in calls[1:]:
        np.testing.assert_allclose(np.linalg.norm(candidate - original_w), 0.1, rtol=1e-14)
    # Probe evaluations happen before the added projection, not after it.
    assert not all(
        np.isclose(np.linalg.norm(candidate), 1., rtol=0, atol=1e-14)
        for candidate, _ in calls[1:]
    )
    chosen = calls[index][0]
    np.testing.assert_array_equal(result["w"], chosen / np.linalg.norm(chosen))
    np.testing.assert_allclose(np.linalg.norm(result["w"]), 1., rtol=0, atol=3e-16)
    assert trace["selected"] == selected
    assert (trace["L_current"], trace["L_plus"], trace["L_minus"]) == losses
    assert result["k"] == 1
    assert state["k"] == 0
    np.testing.assert_array_equal(state["w"], original_w)


def test_indexed_streams_match_the_agreed_seed_map():
    state = initialize()
    outer_seed = 71
    streams = np.random.SeedSequence([outer_seed, 0, 20260910]).spawn(2)
    gaussian = np.random.default_rng(streams[0]).normal(size=len(state["w"]))
    expected_direction = gaussian / np.linalg.norm(gaussian)
    expected_seed = int(
        np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max)
    )
    _, trace = step(state, outer_seed, 0.1, lambda w, seed: 1.)
    np.testing.assert_array_equal(trace["s"], expected_direction)
    assert trace["seed"] == expected_seed
    np.testing.assert_allclose(np.linalg.norm(trace["s"]), 1., rtol=0, atol=3e-16)


def test_alpha_candidates_share_randomness_but_iterations_and_runs_do_not():
    state = initialize()
    rng_before = np.random.get_state()
    traces = [step(state, 71, alpha, lambda w, seed: 1.)[1] for alpha in (0.01, 0.1, 1.)]
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace["s"], traces[0]["s"])
        assert trace["seed"] == traces[0]["seed"]
    for outer_seed, other_state in ((71, dict(state, k=1)), (72, state)):
        trace = step(other_state, outer_seed, 0.1, lambda w, seed: 1.)[1]
        assert trace["seed"] != traces[0]["seed"]
        assert not np.array_equal(trace["s"], traces[0]["s"])
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_current_is_retrained_with_fresh_randomness_even_when_it_stays():
    state = initialize()
    calls = []

    def loss(w, seed):
        calls.append((w.copy(), seed))
        return (0., 1., 2.)[(len(calls) - 1) % 3]

    for _ in range(3):
        state, trace = step(state, 71, 0.1, loss)
        assert trace["selected"] == "current"
    assert state["k"] == 3
    assert len(calls) == 9
    seeds = [calls[index][1] for index in (0, 3, 6)]
    assert len(set(seeds)) == 3
    for start in (0, 3, 6):
        assert len({seed for _, seed in calls[start:start + 3]}) == 1


def test_next_iteration_starts_from_projected_state():
    calls = []

    def loss(w, seed):
        calls.append((w.copy(), seed))
        return (2., 0., 1.)[(len(calls) - 1) % 3]

    first, trace = step(initialize(), 71, 1., loss)
    raw_selected = trace["w_plus"]
    assert not np.isclose(np.linalg.norm(raw_selected), 1., rtol=0, atol=1e-14)
    second, _ = step(first, 71, 1., loss)
    np.testing.assert_array_equal(calls[3][0], first["w"])
    np.testing.assert_array_equal(calls[3][0], raw_selected / np.linalg.norm(raw_selected))
    assert second["k"] == 2


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.])
def test_multiple_updates_match_a_direct_mathematical_reference(alpha):
    actual = initialize()
    expected = initialize()
    target = np.linspace(-1., 1., len(actual["w"]))

    def loss(w, seed):
        return float((w - target) @ (w - target)) + float(seed % 97) / 1000

    for k in range(12):
        streams = np.random.SeedSequence([71, k, 20260910]).spawn(2)
        gaussian = np.random.default_rng(streams[0]).normal(size=len(target))
        direction = gaussian / np.linalg.norm(gaussian)
        seed = int(np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max))
        candidates = [expected["w"], expected["w"] + alpha * direction,
                      expected["w"] - alpha * direction]
        values = [loss(w, seed) for w in candidates]
        winner = int(np.argmin(values))
        chosen = candidates[winner]
        expected = {"k": k + 1, "w": chosen / np.linalg.norm(chosen)}
        actual, trace = step(actual, 71, alpha, loss)
        assert trace["selected"] == ("current", "+", "-")[winner]
        np.testing.assert_array_equal(actual["w"], expected["w"])
    assert actual["k"] == expected["k"] == 12


@pytest.mark.parametrize("alpha", [0.01, 0.1, 1.])
def test_resume_from_serialized_state_is_identical(alpha):
    target = np.linspace(-1., 1., len(initialize()["w"]))

    def loss(w, seed):
        return float(target @ w) + float(seed % 97) / 1000

    continuous = initialize()
    for _ in range(12):
        continuous, _ = step(continuous, 71, alpha, loss)
    resumed = initialize()
    for _ in range(4):
        resumed, _ = step(resumed, 71, alpha, loss)
    resumed = json.loads(json.dumps({"k": resumed["k"], "w": resumed["w"].tolist()}))
    for _ in range(8):
        resumed, _ = step(resumed, 71, alpha, loss)
    assert resumed["k"] == continuous["k"] == 12
    np.testing.assert_array_equal(resumed["w"], continuous["w"])


@pytest.mark.parametrize("alpha", [0., -1., np.nan, np.inf])
def test_invalid_alpha_does_not_evaluate(alpha):
    def loss(w, seed):
        pytest.fail("Invalid alpha must fail before training.")

    with pytest.raises(ValueError, match="alpha"):
        step(initialize(), 71, alpha, loss)


@pytest.mark.parametrize("bad_index", [0, 1, 2])
@pytest.mark.parametrize("bad_loss", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidate_loss_is_rejected(bad_index, bad_loss):
    values = [1., 2., 3.]
    values[bad_index] = bad_loss
    losses = iter(values)
    with pytest.raises(FloatingPointError, match="losses"):
        step(initialize(), 71, 0.1, lambda w, seed: next(losses))


def test_zero_selected_candidate_is_rejected():
    _, trace = step(initialize(), 71, 1., lambda w, seed: 1.)
    # Reusing the same indexed direction makes the negative probe exactly zero.
    state = {"k": 0, "w": trace["s"].copy()}
    losses = iter((2., 1., 0.))
    with pytest.raises(FloatingPointError, match="selected candidate"):
        step(state, 71, 1., lambda w, seed: next(losses))
