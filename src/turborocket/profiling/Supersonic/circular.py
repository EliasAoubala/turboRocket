"""
Circular.py

This source file encapsulates all the functions required for sizing the circular sections of the tubrine blade, following the free vortex assumption.

The following equations follows the method for supersonic turbine design presented in NASA TN D-4421.

Free vortex flow is assumed for the blade flow.

# noqa: E501
"""

import numpy as np


def M_star(gamma, M):
    """
    Critical Velocity Ratio

    This function computes the critical velocity ratio of the flow as a function of Mach number and specific heat ratio.

    Variables:
        gamma:  Specific Heat Ratio
        M:      Mach Number

    # noqa: E501
    """

    crit_vel_rat = (
        (((gamma + 1) / 2) * M**2) / (1 + ((gamma - 1) / 2) * M**2)
    ) ** (1 / 2)

    return crit_vel_rat


def inv_M_star(gamma, M_star):
    """
    This function computes the mach number based on the critical velocity ratio.

    Args:
        gamma (float): Specific Heat Ratio
        M_star (float): Crtical Velocity ratio
    """
    M = M_star / (((gamma + 1) / 2) - ((gamma - 1) / 2) * M_star**2) ** (1 / 2)

    return M


def prandtl_meyer(gamma, crit_vel_rat):
    """
    Prandtl Meyer Angle (v)

    This function computes the Prandtl-Meyer angle v based on the mach number of a flow.

    # noqa: E501
    """

    v = np.pi / 4 * (((gamma + 1) / (gamma - 1)) ** (1 / 2) - 1) + (1 / 2) * (
        (((gamma + 1) / (gamma - 1)) ** (1 / 2))
        * np.arcsin((gamma - 1) * crit_vel_rat**2 - gamma)
        + np.arcsin((gamma + 1) / crit_vel_rat**2 - gamma)
    )

    return v


def arc_angles_upper(beta_o, beta_i, v_i, v_o, v_u):
    """
    Upper Circular Arc Angles Alpha

    This function computes the circular arc angles based on the inlet and outlet angles,
    coupled with their prantl-meyer angles

    Variables:
        alpha_l_i:  Lower Arc Inlet Angle
        alpha_l_o:  Lower Arc Outlet Angle
        alpha_u_i:  Upper Arc Inlet Angle
        alpha_u_o:  Upper Arc Outlet Angle
        beta_i:     Inlet relative flow angle
        beta_o:     Exit relative

    # noqa: E501
    """

    alpha_u_i = beta_i - (v_u - v_i)

    alpha_u_o = beta_o + (v_u - v_o)

    return [alpha_u_i, alpha_u_o]


def arc_angles_lower(beta_o, beta_i, v_i, v_o, v_l):
    """
    Lower Circular Arc Angles Alpha

    This function computes the arc angles based on the inlet and outlet angles, coupled with their prantl-meyer angles

    Variables:
        alpha_l_i:  Lower Arc Inlet Angle
        alpha_l_o:  Lower Arc Outlet Angle
        alpha_u_i:  Upper Arc Inlet Angle
        alpha_u_o:  Upper Arc Outlet Angle
        beta_i:     Inlet relative flow angle
        beta_o:     Exit relative

    # noqa: E501
    """

    alpha_l_i = beta_i - (v_i - v_l)

    alpha_l_o = beta_o + (v_o - v_l)

    return [alpha_l_i, alpha_l_o]


def beta_o(M_i, M_o, gamma, beta_i):
    """ """

    exit_o = -np.arccos(
        (
            (M_i / M_o)
            * ((1 + (gamma - 1) / 2 * M_o**2) / (1 + (gamma - 1) / 2 * M_i**2))
            ** ((gamma + 1) / (2 * (gamma - 1)))
        )
        * np.cos(beta_i)
    )

    return exit_o


def A_rat(beta_i, beta_o):
    """
    The Area Ratio (A_i / A_o)

    This equation computes the area ratio of the supersonic turbine blade, assuming that the spacings are equal on the inlet and exit sides of the turbine (Axial turbine)

    # noqa: E501
    """

    return np.cos(beta_i) / np.cos(beta_o)
