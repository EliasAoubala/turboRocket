from src.turborocket.profiling.Supersonic.transition import (
    func_R_star,
    func_R_star_k,
    R_star,
    phi_k,
    vortex_coords,
    mue_k,
    mach_slope,
    wall_slope,
    wall_coords,
    transform_coord,
    moc,
)

import numpy as np


def test_func_R_star():
    assert np.isclose(func_R_star(0.9967412602371031, 1.4), -2.276853164, rtol=1e-3)


def test_func_R_star_k():
    assert np.isclose(func_R_star_k(1.4, np.pi, 2, 3), -3.993667857, rtol=1e-3)


def test_R_star():
    assert np.isclose(R_star(1.4, np.pi, 1.5708, 3, 1), 1, rtol=1e-3)


def test_phi_k():
    assert np.isclose(phi_k(np.pi / 2, np.pi / 4, 3, 1.57), -2.354601837, rtol=1e-3)


def test_vortex_coords():
    xk, yk = vortex_coords(0.8, np.pi / 2)

    assert np.isclose(xk, -4 / 5, rtol=1e-3)

    assert np.isclose(yk, 0, rtol=1e-3)


def test_mue_k():
    assert np.isclose(mue_k(1.4, 0.8), -0.8536095489, rtol=1e-3)


def test_mach_slope():
    assert np.isclose(
        mach_slope(np.pi / 2, np.pi / 4, np.pi / 3, np.pi / 2), -0.767326988, rtol=1e-3
    )


def test_wall_slope():
    assert np.isclose(wall_slope(np.pi / 4), 1, rtol=1e-3)


def test_wall_coords():
    xkl, ykl = wall_coords(1, 1, 0.8, 0.8, -0.76, 1)

    assert np.isclose(xkl, 1, rtol=1e-3)

    assert np.isclose(ykl, 1, rtol=1e-3)


def test_transform_coord():
    xklt, yklt = transform_coord(2, 3, np.pi / 3)

    assert np.isclose(xklt, -1.598076, rtol=1e-3)

    assert np.isclose(yklt, 3.232050808, rtol=1e-3)
