from src.turborocket.profiling.Supersonic.constraints import (
    M_star_func,
    M_star_func_int,
    starting_test,
    k_star_max,
    Q_func,
    Q_func_int,
    Q,
    C,
    shock_pressure_rat,
    M_star_i_max_rhs,
    M_star_i_max,
    mass_flow,
    inv_mass_flow,
    wf_parameter,
    r_star,
)

import numpy as np


# Testing the M_star functional expression
def test_M_star_func():
    assert np.isclose(M_star_func(1, 3, 1.5, 1.4), 0.4871392896, rtol=1e-3)


# Testing the M_star integral
def test_M_star_func_int():
    assert np.isclose(
        M_star_func_int(0.4, 1, 2, 1.4, 100), 0.2587260620893371, rtol=1e-3
    )


def test_starting_test():
    assert np.isclose(starting_test(0.4, 1, 2, 1.4, 100), -0.31020702, rtol=1e-3)


def test_k_star_max():
    assert np.isclose(k_star_max(1, 2, 1.4, 100), 0.2845509748347924, rtol=1e-3)


def test_Q_func():
    assert np.isclose(Q_func(2, 1.4), 0.05059644256, rtol=1e-3)


def test_Q_func_int():
    assert np.isclose(Q_func_int(1, 2, 1.4, 100), 0.3897895947060327, rtol=1e-3)


def test_Q():
    assert np.isclose()


def test_C():
    assert 1


def test_shock_pressure_rat():
    assert 1


def test_M_star_i_max_rhs():
    assert 1


def test_M_star_i_max():
    assert 1


def test_mass_flow():
    assert 1


def test_inv_mass_flow():
    assert 1


def test_wf_parameter():
    assert 1


def test_r_star():
    assert 1
