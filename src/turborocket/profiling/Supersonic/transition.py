"""
transition.py

This source file encapsulates all the functions required for sizing the transition arc sections of the turbine blade, utilising the method of characteristics.

The following equations follows the method for supersonic turbine design presented in NASA TN D-4421.



# noqa: E501

"""

from src.turborocket.solvers.solver import adjoint

import numpy as np


def func_R_star(R_star, gamma):
    """
    Function of R* ( f(R*) )

    This equation describes the generalised function of the non-dimentionalised radius
    defining the expansion line angle.

    Variables:
        gamma:  Specific Heat Ratio
        R_star: Non-dimentionalised Radius

    """

    f_r_star = ((gamma + 1) / (gamma - 1)) ** (1 / 2) * np.arcsin(
        (gamma - 1) / R_star**2 - gamma
    ) + np.arcsin((gamma + 1) * R_star**2 - gamma)

    return f_r_star


def func_R_star_k(gamma, v_i, dv, k):
    f_r_star_k = (
        2 * v_i
        - np.pi / 2 * (((gamma + 1) / (gamma - 1)) ** (1 / 2) - 1)
        - 2 * (k - 1) * dv
    )

    return f_r_star_k


def R_star(gamma, v_i, dv, k, guess):
    # We use the ajoint equation

    target = func_R_star_k(gamma, v_i, dv, k)

    r_star = adjoint(func_R_star, guess, -0.01, 50, 0.1, target, params=[gamma])

    return r_star


def phi_k(v_i, v_l, k, dv):
    phi_k_i = v_i - v_l - (k - 1) * dv

    return phi_k_i


def vortex_coords(R_star_k, phi_k):
    x_star = -R_star_k * np.sin(phi_k)
    y_star = R_star_k * np.cos(phi_k)

    return [x_star, y_star]


def mue_k(gamma, R_star_k):
    mach_angle = -np.arcsin(
        (((gamma + 1) / 2) * R_star_k**2 - ((gamma - 1) / 2)) ** (1 / 2)
    )

    return mach_angle


def mach_slope(phi_k_1, phi_k_2, mue_k_1, mue_k_2):
    # This function computes the gradient of the mach line

    m_k = np.tan(((phi_k_1 + phi_k_2) / 2) + ((mue_k_1 + mue_k_2) / 2))

    return m_k


def wall_slope(phi_k_1):
    # This function computes the slope the wall at a given segment

    m_bar_k = np.tan(phi_k_1)

    return m_bar_k


def wall_coords(x_star_l_k_1, y_star_l_k_1, y_star_k, x_star_k, m_bar_k, m_k):
    x_star_l_k = (
        (y_star_l_k_1 - m_bar_k * x_star_l_k_1) - (y_star_k - m_k * x_star_k)
    ) / (m_k - m_bar_k)
    y_star_l_k = (
        m_k * (y_star_l_k_1 - m_bar_k * x_star_l_k_1)
        - m_bar_k * (y_star_k - m_k * x_star_k)
    ) / (m_k - m_bar_k)

    return [x_star_l_k, y_star_l_k]


def transform_coord(x_star_l_k, y_star_l_k, alpha_l_i):
    # This function does the co-ordinate transformation for the transition co-ords

    x_star_l_k_t = x_star_l_k * np.cos(alpha_l_i) - y_star_l_k * np.sin(alpha_l_i)

    y_star_l_k_t = x_star_l_k * np.sin(alpha_l_i) + y_star_l_k * np.cos(alpha_l_i)

    return [x_star_l_k_t, y_star_l_k_t]


def moc(k_max, v_i, v_l, gamma, alpha_l_i):
    # This function encapsulates the overall method of characteristics procedure for solving
    # for the shape of the transition arcs

    # First we need to define the k_max value, this needs to be an integer,
    # so what we do is we can invert this

    delta_v = (v_i - v_l) / k_max

    # We setup our history arrays accordingly
    phi_hist = []
    r_star_hist = []
    xk_hist = []
    yk_hist = []
    mue_hist = []
    mk_hist = []
    m_bar_k_hist = []
    xlk_hist = []
    ylk_hist = []
    xlkt_hist = []
    ylkt_hist = []

    # We can now do the intial look to get the intial values of the loop.
    phi = phi_k(v_i, v_l, k_max + 1, delta_v)

    phi_hist.append(phi)

    # We now must solver R* using the adjoint method
    r_star = R_star(gamma, v_i, delta_v, k_max + 1, 1)

    r_star_hist.append(r_star)

    # We can now compute the co-ordinates of the expansion line
    [xk, yk] = vortex_coords(r_star, phi)

    xk_hist.append(xk)
    yk_hist.append(yk)

    # We need to now compute the mach angle
    mue = mue_k(gamma, r_star)

    mue_hist.append(mue)

    # Gradients need to be 0 at the intial point which is vertical
    mk_hist.append(0)
    m_bar_k_hist.append(0)

    # Wall co-ordinates are the same as the intial co-ordinates
    xlk_hist.append(xk)
    ylk_hist.append(yk)

    # Finally, we do a co-ordinate transformation

    [xlkt, ylkt] = transform_coord(xk, yk, alpha_l_i)

    xlkt_hist.append(xlkt)
    ylkt_hist.append(ylkt)

    # Having established the delta_v, we can then loop through k values

    for k in np.linspace(k_max, 1, num=k_max):
        # We evaluate our phi angle
        phi = phi_k(v_i, v_l, k, delta_v)

        phi_hist.append(phi)

        # We now must solver R* using the adjoint method
        r_star = R_star(gamma, v_i, delta_v, k, 1)

        r_star_hist.append(r_star)

        # We can now compute the co-ordinates of the expansion line
        [xk, yk] = vortex_coords(r_star, phi)

        xk_hist.append(xk)
        yk_hist.append(yk)

        # We need to now compute the mach angle
        mue = mue_k(gamma, r_star)

        mue_hist.append(mue)

        # We must thus compute the slope of the mach line, based on the average of the current point and previous (k+1)
        m_k = mach_slope(phi_hist[k_max - k], phi, mue_hist[k_max - k], mue)

        mk_hist.append(m_k)

        # We can now compute the wall segment slope
        m_bar_k = wall_slope(phi_hist[k_max - k])

        m_bar_k_hist.append(m_bar_k)

        # Wall Co-ordinates
        [xlk, ylk] = wall_coords(xlk_hist, ylk_hist, yk_hist, xk_hist, m_bar_k, m_k)

        xlk_hist.append(xlk)
        ylk_hist.append(ylk)

        # Finally, we do a co-ordinate transformation

        [xlkt, ylkt] = transform_coord(xlk, ylk, alpha_l_i)

        xlkt_hist.append(xlkt)
        ylkt_hist.append(ylkt)

    # Finally, we return the transformed co-ordinates

    return [xlkt_hist, ylkt_hist]
