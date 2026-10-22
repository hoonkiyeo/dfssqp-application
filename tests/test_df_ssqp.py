"""Independent formula, zero-step and reproducibility checks for the outer update."""

from copy import deepcopy

import numpy as np
import pytest

import config
from algorithms.df_ssqp import initialize, step


def test_ai_matches_closed_form_and_unprojected_update():
    # Linear f and w=e1: the AI direction removes g_hat's first coordinate.
    state = initialize()
    state.update(w=np.array([1., 0.]), lam=np.zeros(1), g_bar=np.zeros(2), G_bar=np.zeros((1, 2)))
    q, calls = np.array([.3, .7]), []

    def loss(w, seed):
        calls.append((w.copy(), seed))
        return float(q @ w)

    nxt, t = step(state, config.outer_seeds[0], 'AI', loss)
    g = (q @ t['delta']) * t['delta']
    np.testing.assert_allclose(t['g_hat'], g, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t['d'], [0, -g[1]], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t['eta'], [-g[0]/2], rtol=1e-14, atol=1e-15)
    assert t['tau'] == 1
    assert t['nu'] == pytest.approx(.45)
    assert t['s'] == pytest.approx(.15)
    np.testing.assert_allclose(nxt['w'], state['w'] + .15*t['d'])
    assert np.linalg.norm(nxt['w']) > 1  # No projection after the update.
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    np.testing.assert_array_equal(calls[0][0], state['w'] + t['b']*t['delta'])
    np.testing.assert_array_equal(calls[1][0], state['w'] - t['b']*t['delta'])
    np.testing.assert_array_equal(t['G'], [[2., 0.]])
    assert not t['clipped']


@pytest.mark.parametrize('initial_seed', [config.initialization_seed, 1, 10])
def test_feasible_zi_cancellation_advances_all_state_then_moves(initial_seed):
    state = initialize()
    v = np.random.default_rng(initial_seed).normal(size=config.dimension)
    state['w'] = v / np.linalg.norm(v)
    before = deepcopy(state)
    loss = lambda w, seed: float(np.sum((w - np.linspace(-.4, .8, len(w)))**2))
    nxt, t = step(state, config.outer_seeds[0], 'ZI', loss)
    assert t['roundoff']['zero_step']
    assert np.linalg.norm(t['d']) < 1e-12  # Test observation, not algorithm threshold.
    assert np.isinf(t['nu_trial']) and nxt['nu'] == state['nu']
    assert nxt['k'] == 1
    np.testing.assert_array_equal(nxt['g_bar'], t['g_hat'])
    np.testing.assert_array_equal(nxt['G_bar'], t['G_hat'])
    np.testing.assert_array_equal(nxt['w'], state['w'] + t['s']*t['d'])
    np.testing.assert_array_equal(nxt['lam'], state['lam'] + t['s']*t['eta'])
    assert np.any(nxt['lam'] != state['lam'])
    after, t2 = step(nxt, config.outer_seeds[0], 'ZI', loss)
    assert after['k'] == 2 and not t2['roundoff']['zero_step']
    assert t2['seed'] != t['seed'] and np.linalg.norm(t2['d']) > 1e-8
    np.testing.assert_array_equal(t2['g_bar'], (1-t2['beta'])*nxt['g_bar'] + t2['beta']*t2['g_hat'])
    for key in state:
        np.testing.assert_array_equal(state[key], before[key])


def test_exact_zero_continues_without_division():
    state = initialize()
    state['w'] = np.eye(1, config.dimension).ravel()
    nxt, t = step(state, config.outer_seeds[0], 'ZI', lambda w, seed: 0.)
    assert np.all(t['d'] == 0) and t['roundoff']['zero_step']
    assert t['reduction'] == 0 and np.isinf(t['tau_trial']) and np.isinf(t['nu_trial'])
    assert nxt['tau'] == state['tau'] and nxt['nu'] == state['nu']
    assert nxt['k'] == 1


@pytest.mark.parametrize('scale', [0., 1e-5, 1000.])
def test_safeguard_bounds_and_raw_average_storage(scale):
    state = initialize()
    state['w'] = scale * state['w']
    nxt, t = step(state, config.outer_seeds[0], 'AI', lambda w, seed: float(w[0]))
    assert t['clipped']
    singular = np.linalg.svd(t['G'], compute_uv=False)[0]
    assert singular == pytest.approx(config.jacobian_lower if scale < 1 else config.jacobian_upper)
    np.testing.assert_array_equal(nxt['G_bar'], 2*state['w'][None, :])
    assert not np.array_equal(nxt['G_bar'], t['G'])
    np.testing.assert_allclose(t['K'] @ np.r_[t['d'], t['eta']], t['rhs'], atol=1e-10)


def test_replay_resume_and_global_rng():
    loss = lambda w, seed: float(np.sum(w**2) + .2*w[0])
    original_rng = np.random.get_state()
    state = initialize()
    for _ in range(3):
        state, _ = step(state, config.outer_seeds[2], 'AI', loss)
    checkpoint = deepcopy(state)
    resumed, trace = step(checkpoint, config.outer_seeds[2], 'AI', loss)
    # Intervening calls with other seeds/modes cannot alter this indexed stream.
    step(initialize(), config.outer_seeds[3], 'ZI', loss)
    replay, other = step(state, config.outer_seeds[2], 'AI', loss)
    for key in resumed:
        np.testing.assert_array_equal(resumed[key], replay[key])
    np.testing.assert_array_equal(trace['delta'], other['delta'])
    assert trace['seed'] == other['seed'] >= 2**32
    assert trace['seed'] not in config.evaluation_seeds
    after = np.random.get_state()
    np.testing.assert_array_equal(original_rng[1], after[1])
    assert original_rng[0] == after[0] and original_rng[2:] == after[2:]


def test_rejects_nonfinite_loss_and_bad_solve_without_advancing(monkeypatch):
    state = initialize()
    with pytest.raises(FloatingPointError, match='losses'):
        step(state, config.outer_seeds[0], 'AI', lambda w, seed: np.nan)
    assert state['k'] == 0
    monkeypatch.setattr(np.linalg, 'solve', lambda K, rhs: np.zeros(len(rhs)))
    with pytest.raises(FloatingPointError, match='predicted reduction'):
        step(state, config.outer_seeds[0], 'AI', lambda w, seed: float(w[0]))
    assert state['k'] == 0
