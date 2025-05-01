from src.turborocket.solvers.solver import adjoint, integrator
import numpy as np
import pytest


def dummy_function(x):
    return x**2 - 4


def test_adjoint():
    assert adjoint(dummy_function, 4, 0.1, 100, 0.6, 0) == pytest.approx(2, 1e-3)


def test_integrator():
    assert np.isclose(integrator(dummy_function, 0, 10, 100), 293.3333, rtol=1e-3)
