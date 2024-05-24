"""

This file contains a series of loss-model utilities functions for mean-line design of turbines

Each Loss model is encapsulated into an object with a series of functions calculating loss contributions.

Loss-Models currently implemented:

    - craigcox: Loss Model by Craig and Cox
    - kackerOkapuu: Loss Model by Kacker and Okapuu (1982)
    - Aungier: Loss Modely by Aungier 
    - Andreas: Loss Model for Radial Turbines https://api.semanticscholar.org/CorpusID:218640940
}  

"""

import numpy as np


class Andreas:
    def __init__(self):
        return

    def nozzlePhi(self):
        # This equation computed the Supersonic Nozzle Velocity Coefficient
        # Velocity coefficient here is c_No/c_No_is (Actual Velocity / Isentropic Velocity)

        self.phi2 = (
            1
            - (0.0029 * self.Ma1**3 - 0.0502 * self.Ma1**2 + 0.2241 * self.Ma1 - 0.0877)
        ) ** (0.5)

        return

    def bladePhi(self):
        # This equation computes the supersonic turbine blade velocity coefficient
        # This is defined as w_exit/ w_inlet

        self.phi3 = (
            0.957
            - 0.000362 * self.deltaB
            - 0.0258 * self.Ma1_rel
            + 0.00000639 * self.deltaB**2
            + 0.0674 * self.Ma1_rel**2
            - 0.0000000753 * self.deltaB**3
            - 0.043 * self.Ma1_rel**3
            - 0.000238 * self.deltaB * self.Ma1_rel
            + 0.00000145 * self.deltaB**2 * self.Ma1_rel
            + 0.0000425 * self.deltaB * self.Ma1_rel**2
        )

        return

    def ventilationPower(self):

        # This calculates the ventilation losses due to partial admission

        self.p_v = (1.85 / 2) * (
            (1 - self.epsilon)
            * self.rho_exit
            * (self.n / 60) ** 3
            * self.D_mean**4
            * 4.5
            * self.h_blade
        )

        return


class Aungier:
    """

    The Aungier Loss Model

    Variables:
        Y_p:    Profile Loss parameters
        K_mod:  Experience factor (derived from the kackerOkapuu method)
        K_inc:  Off-design incidence factor
        K_m:    Mach Number Factor
        K_p:    Compressibility Factor
        K_RE:   Reynolds Number Factor

        Y_p1:   Profile Loss coefficient for nozzle blades (beta_1 = 0)
        Y_p2:   Profile Loss coefficent for rotor blades (beta_1 = alpha_2)

        Y_s:    Secondary Flow Losses for low aspect ratio

    """

    def __init__():
        # Intialising core parameters of the loss model
        return

    def Y(self):
        """
        Total Pressure Loss Coefficient (Y)

        This function solves for the profile loss coefficient of the turbine stage, characterising the increase in total pressure of the turbine exit stage

        Y = (P_t1 - P_t2) / (P_t2 - P_2)

        Variables:
            Y_p:    The Profile Loss Coefficient
            Y_s:    The Secondary Flow loss coefficient
            Y_tcl:  The blade clearnace loss coefficient
            Y_te:   The trailing edge loss coefficient
            Y_ex:   The supersonic expansion loss coefficient
            Y_sh:   The shock loss coefficient
            T_lw:   The lashing wire loss coefficient (rotors)
        """

        return Y_p + Y_s + Y_tcl + T_te + Y_ex + Y_sh + Y_lw

    def delta_h_par(self):
        """
        Parasitic Losses (delta_h_par)

        The parasitic losses accounts for waster work due to losses in disk friction, shear forces and partial admission.

        These are reflected in an increase in total enthalph of the discharge flow
        relative to the value produced when the flow passes through the blade row.

        In contrast to the pressure loss coefficient, these losses do not affect total pressure, but do influence stage efficiency.

        Variables:
            delta_h_adm:    Partial admission losses (rotors)
            delta_h_df:     Disk friction work (diaphragm-disk rotors)
            delta_h_seal:   Leakage bypass loss (shourded rotors and rotor balance holes)
            delta_h_gap:    Clearnace gap windage loss (shrouded blades and nozzles of diaphragm-disk rotors)
            delta_h_q:      moisture work loss (rotors)
        """

        return delta_h_adm + delta_h_df + delta_h_seal + delta_h_gap + delta_h_q

    def yp(self):
        """
        The Profile Loss Coefficient (Y_p)

        This pressure loss coefficient characterises the profile losses of the turbine stage

        Variables:
            K_mod:  kackerOkapuu Experience factor
            K_inc:  Correction for off-design incidence effects
            K_m:    Correction for Mach Number effects
            K_p:    correction for Compressibility effects
            K_re:   correction for Reynolds Number effects
            Y_p1:   profile loss coefficient for nozzle blades (beta_1 = 90)
            Y_p2:   profile loss coefficients for impulse blades (alpha_2 = beta_1)

        Returns:
            _type_: _description_
        """

        prt = (y_p1 + ((beta_1 / alpha_2) ** 2) * (y_p2 - y_p1)) * (
            (5 * t / c) ** (beta_1 / alpha_2)
        ) - delta_y_te

        y_p = k_mod * k_inc * k_m * k_p * k_re * prt

        return y_p

    def F_ar(self):
        if h_c < 2:
            f_ar = 0.5 * (2 * c / h) ** (0.7)
        else:
            f_ar = c / h

        return f_ar

    def Y_s(self):
        y_bar_s = (
            0.0334
            * F_ar
            * (C_l / (s_c)) ** 2
            * (np.cos(alpha_2) / np.cos(beta_1))
            * (np.cos(alpha_2) ** 2 / np.cos(alpha_m) ** 3)
        )

        y_s = k_s * k_re * (y_bar_s**2 / (1 + 7.5 * y_bar_s)) ** (1 / 2)

        return y_s

    def Y_sh(self):
        y_bar_sh = 0.8 * x_1**2 + x_2**2

        y_sh = (y_bar_sh**2 / (1 + y_bar_sh**2)) ** (1 / 2)

        return y_sh

    def Y_ex(self):
        y_ex = ((M2 - 1) / M2) ** 2

        return y_ex

    def Y_te(self):
        y_te = ((t_2 / s) * np.sin(beta_g) - t_2) ** 2

        return y_te

    def Y_tcl(self):
        if TIP_TYPE == "shrouded":
            B = 0.37
        else:
            B = 0.47

        Y_tcl = (
            B
            * (c / h)
            * (t_cl / c) ** (0.78)
            * (C_l / (s / c)) ** 2
            * (np.cos(alpha_2) ** 2 / np.cos(alpha_m) ** 3)
        )
