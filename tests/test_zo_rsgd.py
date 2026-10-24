"""Independent formula and reproducibility checks for ZO-RSGD with m=1."""

import json

import numpy as np
import pytest

import config
from algorithms.df_ssqp import initialize as initialize_df_ssqp
from algorithms.df_ssqp import step as df_ssqp_step
from algorithms.zo_rsgd import initialize, step


ETA = 1 / np.sqrt(5_000)
MU = 0.01


def quadratic_loss(w, seed):
    """A deterministic, inexpensive stand-in for one stochastic loss sample."""
    target = np.linspace(-0.4, 0.8, len(w))
    return float(np.sum((w - target) ** 2) + (seed % 97) / 1_000)


def test_shared_initial_direction():
    state = initialize()
    assert state["k"] == 0
    np.testing.assert_array_equal(state["w"], initialize_df_ssqp()["w"])
    np.testing.assert_allclose(np.linalg.norm(state["w"]), 1, rtol=0, atol=4e-16)


@pytest.mark.parametrize("outer_seed", [71, 72, config.outer_seeds[0]])
def test_iterations_match_independent_matrix_projection_reference(outer_seed):
    state = initialize()
    reference_w = state["w"].copy()

    for k in range(8):
        calls = []
        original_w = state["w"].copy()

        def loss(w, seed):
            calls.append((w.copy(), seed))
            return quadratic_loss(w, seed)

        streams = np.random.SeedSequence([outer_seed, k, 20260910]).spawn(2)
        gaussian = np.random.default_rng(streams[0]).normal(size=config.dimension)
        seed = int(np.random.default_rng(streams[1]).integers(2**32, np.iinfo(np.int64).max))
        projector = np.eye(config.dimension) - np.outer(reference_w, reference_w)
        tangent = projector @ gaussian
        probe = reference_w + MU * tangent
        probe /= np.linalg.norm(probe)
        loss_current = quadratic_loss(reference_w, seed)
        loss_probe = quadratic_loss(probe, seed)
        gradient = (loss_probe - loss_current) * tangent / MU
        updated = reference_w - ETA * gradient
        updated /= np.linalg.norm(updated)

        result, trace = step(state, outer_seed, ETA, MU, loss)
        assert len(calls) == 2
        assert calls[0][1] == calls[1][1] == trace["seed"] == seed
        assert seed >= 2**32
        np.testing.assert_array_equal(calls[0][0], original_w)
        np.testing.assert_allclose(calls[1][0], probe, rtol=0, atol=2e-14)
        np.testing.assert_allclose(trace["u"], tangent, rtol=0, atol=2e-13)
        np.testing.assert_allclose(trace["w_probe"], probe, rtol=0, atol=2e-14)
        np.testing.assert_allclose(trace["g_hat"], gradient, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(result["w"], updated, rtol=0, atol=2e-13)
        assert trace["L_current"] == pytest.approx(loss_current, rel=2e-14)
        assert trace["L_probe"] == pytest.approx(loss_probe, rel=2e-14)
        assert abs(original_w @ trace["u"]) < 2e-14
        assert abs(original_w @ trace["g_hat"]) < 2e-13
        for point in (calls[0][0], calls[1][0], result["w"]):
            np.testing.assert_allclose(np.linalg.norm(point), 1, rtol=0, atol=4e-16)
        # Keep the Gaussian length; this is not a normalized search direction.
        assert not np.isclose(np.linalg.norm(trace["u"]), 1)
        assert trace["eta"] == ETA and trace["mu"] == MU
        assert state["k"] == k and result["k"] == k + 1
        np.testing.assert_array_equal(state["w"], original_w)
        state, reference_w = result, updated


@pytest.mark.parametrize("k", [0, 1, 19])
def test_training_seed_matches_df_ssqp_and_is_independent_of_direction_draw(k):
    outer_seed = config.outer_seeds[0]
    state = initialize()
    state["k"] = k
    df_state = initialize_df_ssqp()
    df_state["k"] = k
    _, trace = step(state, outer_seed, ETA, MU, quadratic_loss)
    _, df_trace = df_ssqp_step(df_state, outer_seed, "AI", quadratic_loss)
    assert trace["seed"] == df_trace["seed"]


def test_replay_resume_new_randomness_and_global_rng():
    original_rng = np.random.get_state()
    continuous = initialize()
    traces = []
    for _ in range(12):
        continuous, trace = step(continuous, 71, ETA, MU, quadratic_loss)
        traces.append(trace)

    resumed = initialize()
    for _ in range(4):
        resumed, _ = step(resumed, 71, ETA, MU, quadratic_loss)
    stored = json.loads(json.dumps({"k": resumed["k"], "w": resumed["w"].tolist()}))
    resumed = {"k": stored["k"], "w": np.asarray(stored["w"])}
    for _ in range(8):
        resumed, _ = step(resumed, 71, ETA, MU, quadratic_loss)
    assert resumed["k"] == continuous["k"] == 12
    np.testing.assert_array_equal(resumed["w"], continuous["w"])

    _, replay = step(initialize(), 71, ETA, MU, quadratic_loss)
    np.testing.assert_array_equal(replay["u"], traces[0]["u"])
    assert replay["seed"] == traces[0]["seed"]
    assert len({trace["seed"] for trace in traces}) == len(traces)
    assert not np.array_equal(traces[0]["u"], traces[1]["u"])
    _, other_run = step(initialize(), 72, ETA, MU, quadratic_loss)
    assert other_run["seed"] != replay["seed"]
    assert not np.array_equal(other_run["u"], replay["u"])

    current_rng = np.random.get_state()
    assert original_rng[0] == current_rng[0]
    np.testing.assert_array_equal(original_rng[1], current_rng[1])
    assert original_rng[2:] == current_rng[2:]


def test_zero_gradient_retains_direction_and_advances_iteration():
    state = {"k": 0, "w": np.eye(1, config.dimension).ravel()}
    original_w = state["w"].copy()
    seeds = []
    for k in range(3):
        calls = []

        def constant_loss(w, seed):
            calls.append(seed)
            return 3.0

        state, trace = step(state, 71, ETA, MU, constant_loss)
        assert len(calls) == 2 and calls[0] == calls[1] == trace["seed"]
        np.testing.assert_array_equal(trace["g_hat"], np.zeros(config.dimension))
        np.testing.assert_array_equal(state["w"], original_w)
        assert state["k"] == k + 1
        seeds.append(trace["seed"])
    assert len(set(seeds)) == 3
