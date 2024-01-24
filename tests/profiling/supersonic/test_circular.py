from src.turborocket.profiling.Supersonic.circular import (
    M_star,
    prandtl_meyer,
    arc_angles_lower,
    arc_angles_upper,
    beta_o,
    A_rat,
)

import numpy as np


def test_M_star():
    assert np.isclose(M_star(1.4, 2), 1.63299316, rtol=1e-3)


def test_prandtl_meyer():
    assert np.isclose(prandtl_meyer(1.4, 1.63), 0.457243, rtol=1e-3)


def test_arc_angles_lower():
    alpha_l_i, alpha_l_o = arc_angles_lower(
        -np.pi / 3, np.pi / 3, np.pi / 3, -np.pi / 3, np.pi / 2
    )

    assert np.isclose(alpha_l_i, 1.570796327, rtol=1e-3)

    assert np.isclose(alpha_l_o, -3.665191429, rtol=1e-3)


def test_arc_angles_upper():
    alpha_u_i, alpha_u_o = arc_angles_upper(
        -np.pi / 3, np.pi / 3, np.pi / 3, -np.pi / 3, np.pi / 2
    )

    assert np.isclose(alpha_u_i, 0.5235987759, rtol=1e-3)

    assert np.isclose(alpha_u_o, 1.570796327, rtol=1e-3)


def test_beta_o():
    assert np.isclose(beta_o(2, 1.5, 1.4, np.pi / 4), -1.055440078, rtol=1e-3)


def test_A_rat():
    assert np.isclose(A_rat(np.pi / 4, -np.pi / 4), 1, rtol=1e-3)
