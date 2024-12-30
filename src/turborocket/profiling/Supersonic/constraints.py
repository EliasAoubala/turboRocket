""" 
This file encapsulates all the major constraint solvers for supersonic turbine blades

The major constraints required to be considered are:

-> Supersonic Turbine startability
-> Flow Seperation
"""


from turborocket.solvers.solver import adjoint, integrator


def M_star_func(M_star, M_star_l, k_star, gamma):
    integral = (1 - ((k_star / M_star_l) * M_star) ** 2) ** (1 / (gamma - 1)) / M_star

    return integral


def M_star_func_int(k_star, M_star_l, M_star_u, gamma, n):
    # This is the integral of the M_star_func expression
    parameters = [M_star_l, k_star, gamma]
    integral = integrator(M_star_func, M_star_l, M_star_u, n, parameters)

    return integral


def starting_test(k_star, M_star_l, M_star_u, gamma, n):
    # We can design the loop for solving the loop

    # Firstly we need to evaluate the range of the integral function values,
    # between the upper and lower M* values

    LHS = M_star_func_int(k_star, M_star_l, M_star_u, gamma, n)

    # We then compute the RHS of the equation
    RHS = (1 - k_star**2) ** (1 / (gamma - 1)) - (
        1 - (k_star * (M_star_u / M_star_l)) ** 2
    ) ** (1 / (gamma - 1))

    error = LHS - RHS

    return error


def k_star_max(M_star_l, M_star_u, gamma, n):
    # We then solve for k_star

    parameters = [M_star_l, M_star_u, gamma, n]

    k_star = adjoint(starting_test, 0.1, 0.1, 1000, 0.8, 0, parameters)

    return k_star


def Q_func(M_star, gamma):
    integral = (((gamma + 1) / 2) - ((gamma - 1) / 2) * M_star**2) ** (
        1 / (gamma - 1)
    ) / M_star

    return integral


def Q_func_int(M_star_l, M_star_u, gamma, n):
    # This is the integral of the Q_func expression
    parameters = [gamma]
    integral = integrator(Q_func, M_star_l, M_star_u, n, parameters)

    return integral


def Q(M_star_l, M_star_u, gamma, n):
    # First we need to spread out our Q values

    integral = Q_func_int(M_star_l, M_star_u, gamma, n)

    # Solving for Q_val
    Q_val = ((M_star_l * M_star_u) / (M_star_u - M_star_l)) * integral

    return Q_val


def C(M_star_l, M_star_u, gamma, n, k_star):
    # First we need to spread out our Q values

    integral = M_star_func_int(k_star, M_star_l, M_star_u, gamma, n)

    C_val = (
        1
        - (((gamma + 1) / (gamma - 1)) ** (1 / 2))
        * (((gamma + 1) / 2) ** (1 / (gamma - 1)))
        * (M_star_u / (M_star_u - M_star_l))
        * integral
    )

    return C_val


def shock_pressure_rat(Q, C):
    # This equation solves for the pressure ratio across the shock

    return Q / (1 - C)


def M_star_i_max_rhs(M_star_i_max, gamma):
    # This equation solves for the rhs of the M_star_inlet_max equation

    val = (M_star_i_max ** (2 * gamma / (gamma - 1))) * (
        (1 - ((gamma - 1) / (gamma + 1)) * (M_star_i_max**2))
        / ((M_star_i_max**2) - ((gamma - 1) / (gamma + 1)))
    ) ** (1 / (gamma - 1))

    return val


def M_star_i_max(p_rat):
    # This equation now solves for the M_star_inlet_max

    # Here we are using adjoint to solve for the equation
    sol = adjoint(M_star_i_max_rhs, 2, 0.1, 1000, 0.8, p_rat)

    return sol


def mass_flow(M_star_l, M_star_u, gamma, n, wf_parameter):
    # This equation solves for the flow rate in the passage

    integral = Q_func_int(M_star_l, M_star_u, gamma, n)

    mass_flow = (
        wf_parameter
        * ((2 / (gamma + 1)) ** (1 / 2))
        * ((2 / (gamma + 1)) ** (1 / (gamma - 1)))
        * integral
    )

    return mass_flow


def inv_mass_flow(M_star_l, M_star_u, gamma, n, mass_flow):
    # This function solves for the wf parameter based on the profile designed

    integral = Q_func_int(M_star_l, M_star_u, gamma, n)

    wf_parameter = mass_flow / (
        ((2 / (gamma + 1)) ** (1 / 2))
        * ((2 / (gamma + 1)) ** (1 / (gamma - 1)))
        * integral
    )

    return wf_parameter


def wf_parameter(r_star, h, a_total_inlet, rho_total_inlet):
    # This function evaluates the weight-flow parameter

    return r_star * h * a_total_inlet * rho_total_inlet


def r_star(wf_parameter, h, a_total_inlet, rho_total_inlet):
    # This function computes the sonic velocity parameter

    return wf_parameter / (h * a_total_inlet * rho_total_inlet)
