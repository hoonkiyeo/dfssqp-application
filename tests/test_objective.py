"""Check the mathematical fit, information separation and reproducibility."""

import random

import numpy as np
import pytest
import torch

import config
from models.objective import evaluate
from models.single_index import SingleIndexModel


@pytest.fixture
def data(monkeypatch):
    monkeypatch.setattr(config, "training_steps", 11)
    monkeypatch.setattr(config, "batch_size", 2)
    z = np.array([[20., -4.], [22., 2.], [17., 1.], [26., -1.], [19., 3.]])
    u = np.array([-12., -10., -8., -9., -11.])
    y = .6*u + .3*z[:, 0] - .2*z[:, 1] + 100.
    train = {"z": z, "u": u, "y": y}
    valid = {"z": z[[1, 3]], "u": u[[1, 3]], "y": y[[1, 3]] + .1}
    return train, valid


def test_fit_matches_independent_numpy_adam(data):
    """Manual gradients and Adam moments; no Torch autograd in the reference."""
    train, valid = data
    w = np.array([1.4, -.2])
    fit = evaluate(w, 421, train, valid, details=True)
    params = {key: value.copy() for key, value in fit["initial_parameters"].items()}
    m = {key: np.zeros_like(value) for key, value in params.items()}
    v = {key: np.zeros_like(value) for key, value in params.items()}
    x = train["z"] @ w
    mx, sx = x.mean(), x.std()
    mu, su, my, sy = train["u"].mean(), train["u"].std(), train["y"].mean(), train["y"].std()
    x, u, y = (x-mx)/sx, (train["u"]-mu)/su, (train["y"]-my)/sy
    beta1, beta2 = config.adam_betas
    for step, indices in enumerate(fit["batches"], start=1):
        bx, bu = x[indices], u[indices]
        pre = bx[:, None] * params["W1"].T + params["b1"]
        hidden = np.logaddexp(0., pre)
        prediction = params["beta"]*bu + (hidden @ params["W2"].T).ravel() + params["b2"][0]
        residual_gradient = 2*(prediction-y[indices])/len(indices)
        hidden_gradient = residual_gradient[:, None]*params["W2"]/(1+np.exp(-pre))
        gradients = {
            "beta": np.asarray(residual_gradient @ bu),
            "W1": (hidden_gradient*bx[:, None]).sum(axis=0)[:, None],
            "b1": hidden_gradient.sum(axis=0),
            "W2": (residual_gradient @ hidden)[None, :],
            "b2": np.asarray([residual_gradient.sum()]),
        }
        for key in params:
            m[key] = beta1*m[key] + (1-beta1)*gradients[key]
            v[key] = beta2*v[key] + (1-beta2)*gradients[key]**2
            params[key] -= config.learning_rate*(m[key]/(1-beta1**step)) / (
                np.sqrt(v[key]/(1-beta2**step)) + config.adam_epsilon)
    for key in params:
        np.testing.assert_allclose(params[key], fit["parameters"][key], rtol=1e-12, atol=1e-13)
    vx = ((valid["z"] @ w - mx)/sx)[:, None]
    hidden = np.logaddexp(0., vx @ params["W1"].T + params["b1"])
    normalized = params["beta"]*(valid["u"]-mu)/su + (hidden @ params["W2"].T).ravel() + params["b2"][0]
    prediction = my + sy*normalized
    np.testing.assert_allclose(prediction, fit["validation_predictions"], rtol=1e-13, atol=1e-13)
    assert fit["mse"] == pytest.approx(np.mean((prediction-valid["y"])**2), abs=1e-12)
    assert evaluate(w, 421, train, valid) == fit["mse"]
    assert [len(batch) for batch in fit["batches"]] == [2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2]


def test_paired_randomness_replay_and_global_state(data):
    train, valid = data
    w = np.array([1.4, -.2])
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    original_w = w.copy()
    plus = evaluate(w+.01, 123, train, valid, details=True)
    minus = evaluate(w-.01, 123, train, valid, details=True)
    with torch.no_grad():
        replay = evaluate(w+.01, 123, train, valid, details=True)
    for key in plus["initial_parameters"]:
        np.testing.assert_array_equal(plus["initial_parameters"][key], minus["initial_parameters"][key])
        np.testing.assert_array_equal(plus["parameters"][key], replay["parameters"][key])
    for a, b in zip(plus["batches"], minus["batches"]):
        np.testing.assert_array_equal(a, b)
    assert plus["statistics"]["index_mean"] != minus["statistics"]["index_mean"]
    np.testing.assert_array_equal(plus["validation_predictions"], replay["validation_predictions"])
    np.testing.assert_array_equal(torch.get_rng_state(), torch_rng)
    for a, b in zip(np.random.get_state(), numpy_rng):
        np.testing.assert_array_equal(a, b)
    assert random.getstate() == python_rng
    np.testing.assert_array_equal(w, original_w)


def test_validation_does_not_change_training(data):
    train, valid = data
    a = evaluate([1.4, -.2], 123, train, valid, details=True)
    changed = {key: values + 900 for key, values in valid.items()}
    b = evaluate([1.4, -.2], 123, train, changed, details=True)
    assert a["statistics"] == b["statistics"]
    for key in a["parameters"]:
        np.testing.assert_array_equal(a["parameters"][key], b["parameters"][key])
    np.testing.assert_array_equal(a["training_predictions"], b["training_predictions"])
    assert a["mse"] != b["mse"]


def test_constant_columns_and_input_conversion(data):
    train, valid = data
    for split in (train, valid):
        split["u"][:] = 3
        split["y"][:] = 7
    fit = evaluate([0., 0.], 123, train, valid, details=True)
    assert all(fit["statistics"][name+"_scale"] == 1 for name in ("index", "control", "response"))
    assert np.isfinite(fit["mse"])
    as_lists = [{key: value.tolist() for key, value in split.items()} for split in (train, valid)]
    assert evaluate([0., 0.], 123, *as_lists) == fit["mse"]


def test_nonfinite_input_and_training_loss_fail(data, monkeypatch):
    train, valid = data
    with pytest.raises(ValueError, match="finite vector"):
        evaluate([np.nan, 0.], 123, train, valid)
    valid_nan = {key: value.copy() for key, value in valid.items()}
    valid_nan["y"][0] = np.nan
    with pytest.raises(ValueError, match="data must be finite"):
        evaluate([1., 0.], 123, train, valid_nan)
    monkeypatch.setattr(SingleIndexModel, "forward", lambda self, x, u: torch.full_like(u, float("nan")))
    with pytest.raises(FloatingPointError, match="training loss"):
        evaluate([1., 0.], 123, train, valid)
