"""
This file contains the main objects used for the design of the turbine stage
"""

from turborocket.fluids.ideal_gas import IdealFluid
import numpy as np
from typing import List


class TurbineStageDesign:
    """
    Class defining a generic turbine stage design and performance analysis class
    """

    def __init__(
        self, gas: IdealFluid, m_dot: float, omega: float, alpha: float
    ) -> None:
        """Constructor for the Turbine Stage Design Object

        Args:
            gas (IdealFluid): Gas Class Describing the Upstream Inlet gas properties
            m_dot (float): Mass-flow rate through turbine stage
            omega (float): Stage rotational speed (RPM)
            alpha (float): Nozzle Inlet Angle (degrees)
        """

        self._gas = gas
        self._m_dot = m_dot

        self._omega = omega * (2 * np.pi) / 60  # RPM to rad/s
        self._alpha = alpha * (np.pi / 180)  # degrees to rad

        return

    def set_operating_point(self, u_cis: float, Rt: float) -> None:
        """Function to define the turbine stage operating set-point, based on the pressure ratio and blade speed ratio

        Args:
            u_cis (float): Normalised Blade Speed (by Isentropic Velocity)
            Rt (float): Pressure ratio across turbine stage (P_o/P_1)
        """
        # Parameter validaiton
        if u_cis < 0:
            raise ValueError(f"U/c_is must be a positive number: {u_cis} <= 0")

        if Rt < 0:
            raise ValueError(f"Rt must be a postiive number: {Rt} <= 0")

        self._u_cis = u_cis
        self._rt = Rt

        return

    def get_p1(self) -> float:
        """Function that solves for the downstream pressure of the system

        Returns:
            float: Downstream Pressure of the turbine stage
        """

        if self._rt is None:
            raise TypeError(
                "Stage Pressure Ratio not defined, call `set_operating_point` first!"
            )

        self._p1 = self._gas.get_pressure() / self._rt

        return self._p1

    def get_blade_speed(self) -> float:
        """Function that solves for the blade speed of the mean-line of the turbine

        Returns:
            float: Blade Speed (m/s)
        """
        if self._cis is None:
            self._cis = self._gas.get_cis(p1=self.get_p1())

        self._u = self._u_cis * self._cis

        return self._u

    def get_mean_diameter(self) -> float:
        """Function that solves for the turbine mean diameter

        Returns:
            float: Mean Turbine Diameter (m)
        """

        if self._u is None:
            self._d_mean = 2 * self._omega / self.get_blade_speed()

        else:
            self._d_mean = 2 * self._omega / self._u

        return self._d_mean

    def get_isentropic_spouting(self) -> float:
        """This function solves for the spouting velocity of the gas

        Returns:
            float: Isentropic spouting velocity of the gas
        """

        self._cis = self._gas.get_cis(p1=self.get_p1())

        return self._cis

    def get_actual_spouting(self, phi: float = 0.9) -> float:
        """This function solves for the actual spouting velocity for the turbine stage nozzle

        Args:
            phi (float, optional): Nozzle Loss Coefficient. Defaults to 0.9.

        Returns:
            float: Actual Spouting Velocity (m/s)
        """
        self._phi = phi

        if self._cis is None:
            self._cis = self._gas.get_cis(p1=self.get_p1())

        self._c1 = phi * self._cis

        return self._c1

    def get_m_star(self, a_star: float, c: float) -> float:
        """Function for solving for the critical mach number of a gas

        Args:
            c (float): Velocity of the gas (m/s)

        Returns:
            float: Critical Mach Number of the gas (N/D)
        """

        return c / a_star

    def get_t1(self, m_star: float | None = None) -> float:
        """This function solves for the nozzle exit temperature

        Args:
            m_star (float | None, optional): Critical Mach Number at Nozzle Exit. Defaults to None.

        Returns:
            float: Nozzle Exit Temperature
        """

        if m_star is not None:
            # We use the m_star provided

            return self._gas.get_expanded_t(M_star=m_star)

        self._t1 = self._gas.get_expanded_t(m_star=self.get_m_star(c=self._c1))

    def get_p1_o(self) -> float:
        """Function that solves for nozzle exit stagnation pressure conditions

        Returns:
            float: Nozzle Exit Stagnation Pressure
        """
        po = self._gas.get_pressure()

        p_rat_nom = self._gas.get_expanded_p(M_star=self.get_m_star(c=self._c1)) / po

        p_rat_act = (
            self._gas.get_expanded_p(M_star=(self.get_m_star(c=self._c1) / self._phi))
            / po
        )

        self._po_1 = self._gas.get_pressure() * (p_rat_act / p_rat_nom)

        return self._po_1

    def get_A1(self) -> float:
        """Solves for the Exit Area of the Nozzle

        Returns:
            float: Exit Area of the Nozzle (m^2)
        """
        self._rho_1 = self.get_p1() / (self._gas.get_R() * self._t1)

        self._a1 = self._m_dot / (self._rho_1 * self._c1)

        return self._a1

    def get_nozzle_height(self, N: int) -> float:
        """Function Solving for the height of the Nozzles

        Args:
            N (int): Number of Nozzles

        Returns:
            float: Nozzle Height
        """

        self._N1 = N

        self._sc = np.sqrt((4 * self._a1) / (np.pi * self._N1))

        return self._sc

    def get_blade_height(
        self, delta_s_p: float = 1.5e-3, delta_s_bt: float = 1e-3
    ) -> float:
        """This function solves for the blade height, based on the rotor height and offsets

        Args:
            delta_s_p (float, optional): Partial Admission Overlap distance, which varies from 1 to 2 mm. Defaults to 1.5e-3.
            delta_s_bt (float, optional): Blade Tip Overalp Distance, which varies from 0 to 1 mm. Defaults to 1e-3.

        Returns:
            float: Blade Height (m)
        """

        self._sb = self._sc + delta_s_bt + delta_s_p

        return self._sb

    def get_rotor_diameters(self) -> List[float]:
        """This function solves for the rotor diameters,

        Returns:
            List[float]: Rotor Diameters [D_hub, D_tip] , (m)
        """

        self._d_hub = self._d_mean - self._sb / 2
        self._d_tip = self._d_mean + self._sb / 2

        return [self._d_hub, self._d_tip]

    def q(self, M_crit: float) -> float:
        """This function solves for the "q(M_c1^*)" function

        Args:
            M_crit (float): Critical Mach number of the gas

        Returns:
            float: M_crit
        """

        g = self._gas.get_gamma()

        return M_crit * (((g + 1) / 2) * (1 - ((g - 1) / (g + 1)) * M_crit)) ** (
            1 / (g - 1)
        )

    def k(self) -> float:
        """Solves for the `k` factor of the equation

        Returns:
            float: K factor (N/D)
        """

        g = self._gas.get_gamma()

        return np.sqrt(g * (2 / (g + 1)) ** ((g + 1) / (g - 1)))

    def get_partial_admission(self) -> float:
        """Get the Partial Admission

        Returns:
            float: Partial Admission Ratio of the Turbine Stage
        """
        R = self._gas.get_R()
        To = self._gas.get_temperature()
        M_c1 = self.get_m_star(c=self._c1)

        self._eps = (self._m_dot * np.sqrt(R * To)) / (
            self._sc
            * np.pi
            * self._po_1
            * self.q(M_crit=M_c1)
            * self.k()
            * (np.sin(self._alpha))
            * self._d_mean
        )

        return self._eps

    def get_relative_speed(
        self,
        c1: float,
    ) -> float:
        """Function solves for the relative speed

        Args:
            c1 (float): Absolute speed of the gas (m/s)

        Returns:
            float: Relative Speed of the gas (m/s)
        """

        self._w1 = np.sqrt(
            (c1 * np.cos(self._alpha) - self._u) ** 2 + (c1 * np.sin(self._alpha)) ** 2
        )

        return self._w1

    def get_beta_1(self) -> float:
        """Function solves for the relative inlet angle

        Returns:
            float: Relative Inlet Angle (rad)
        """

        self._beta_1 = np.arctan(
            np.sin(self._alpha) / (np.cos(self._alpha) - (self._u / self._c1))
        )

        return self._beta_1

    def get_relative_temp(self) -> float:
        """Function that solves for the relative stagnation temperature of the gas

        Returns:
            float: Relative Stagnation Temperature of the Gas (K)
        """
        g = self._gas.get_gamma()
        R = self._gas.get_R()

        self._t01_rel = self._t1 + ((g - 1) / (2 * g * R)) * self._w1**2

        return self._t01_rel

    def get_p1_or(self) -> float:
        """Function that solves for the rotor relative stagnation pressure

        Returns:
            float: Rotor Relative stagnation Pressure
        """
        po = self._gas.get_pressure()

        p_rat_c1 = self._gas.get_expanded_p(M_star=self.get_m_star(c=self._c1)) / po

        p_rat_w1 = self._gas.get_expanded_p(M_star=self.get_m_star(c=self._w1)) / po

        self._po_1r = self._po_1 * (p_rat_c1 / p_rat_w1)

        return self._po_1r

    def get_a_star_w(self) -> float:
        """This Function solves for the relative critical velocity of the gas

        Returns:
            float: a_star_w
        """
        g = self._gas.get_gamma()
        R = self._gas.get_R()

        self._a_star_2 = np.sqrt(((2 * g) / (g + 1)) * R * self._t01_rel)

        return self._a_star_2

    def phi_r(
        self,
        AR: float,
        beta_2: float,
    ) -> float:
        """Function

        Args:
            AR (float): Aspect Ratio (N/D)

        Returns:
            float: Aspect Ratio of the blade.
        """

        self._phi_r = (
            (1 - 0.23 * (1 - (self._beta_1 + beta_2) / np.pi) ** 3)
            * (1 - 0.05 * (M_w_1 - 1) ** 2)
            * (1 - 0.06 * AR)
            * (1 - t / (2 * np.pi * self._eps * self._d_mean))
        )

        return self._phi_r

    def get_w2(self) -> float:
        """Function that get relative velocity

        Returns:
            float: Relative Exit Velocity (m/s)
        """

        self._w2 = self._phi_r * self._w1

        return self._w2

    def get_p2_or(self) -> float:
        """Function that solves for the rotor relative stagnation pressure

        Returns:
            float: Rotor Relative stagnation Pressure
        """
        po = self._gas.get_pressure()

        p_rat_w1 = self._gas.get_expanded_p(M_star=self.get_m_star(c=self._w1)) / po

        p_rat_w2 = self._gas.get_expanded_p(M_star=self.get_m_star(c=self._w2)) / po

        self._po_2r = self._po_1r * (p_rat_w1 / p_rat_w2)

        return self._po_2r

    def get_beta_2(self) -> float:
        """Function that gets the relative exit angle (Beta_2)

        Returns:
            float: Relative Exit Angle of the rotor (rad)
        """
        R = self._gas.get_R()

        m_star_w2 = self.get_m_star(a_star=self._a_star_2, c=self._w2)

        q_w2 = self.q(m_star_w2)

        k = self.k()

        sin_beta_2 = (self._m_dot * np.sqrt(R * self._t01_rel)) / (
            self._eps * np.pi * self._po_2r * q_w2 * k * self._sb * self._d_mean
        )

        self._beta_2 = np.arctan(sin_beta_2)

        return self._beta_2

    def get_rotor_exit_temp(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """

        g = self._gas.get_gamma()

        R = self._gas.get_R()

        self._t2 = self._t02_rel - ((g - 1) / (2 * g * R)) * self._w2**2

        return self._t2

    def get_c2(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """

        self._c2 = np.sqrt(
            (self._w2 * np.sin(self._beta_2)) ** 2
            + (self._w2 * np.cos(self._beta_2) - self._u) ** 2
        )

        return self._c2

    def get_alpha_2(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """

        self._alpha_2 = np.tan(
            (self._w2 * np.sin(self._beta_2))
            / (self._w2 * np.cos(self._beta_2) - self._u)
        )

        return self._alpha_2

    def get_t_o2(self) -> float:
        """_summary_

        Returns:
            float: _description_
        """
        g = self._gas.get_gamma()

        self._to2 = self._t2 + ((g - 1) / (2 * g * R)) * self._c2**2

        return self._to2

    def get_po2(self) -> float:
        """Function for getting the exit stagnation pressure of the turbine stage

        Returns:
            float: Exit Stagnation Pressure of the Turbine Stage
        """
        self._p2 = self._p1 / self._rt

        m_star_c2 = self.get_m_star(a_star=self._a_star_2, c=self._c2)

        p_rat_c2 = self._gas.get_expanded_p(M_star=m_star_c2) / self._gas.get_pressure()

        self._po_2 = self._p2 / p_rat_c2

        return self._po_2

    def get_phi_l(self, delta_r: float, alpha_t: float = 0) -> float:
        """This function solves for the leakage loss coefficient for the mass_flow

        Args:
            delta_r (float): Leakage Radius of the Turbine
            alpha_t (float, optional): Empirical leakage parameter, is 0 for impulse turbines. Defaults to 0.

        Returns:
            float: Leakage loss mass-flow factor (N/D)
        """

        self._phi_l = (
            np.sqrt(1 + alpha_t * (1 / (self._phi * np.sin(self._alpha)) ** 2 - 1))
            * (1 + (self._sb / self._d_mean))
            * (delta_r / self._sb)
        )

        return self._phi_l

    def get_m_leakage(self) -> float:
        """This function solves for the leakage mass flow rate of the system

        Returns:
            float: Leakage mass-flow rate of turbine stage (kg/s)
        """

        self._m_dot_l = self._m_dot * self._phi_l

        return self._m_dot_l

    def get_leakage_loss(self) -> float:
        """This function solves for the leakage loss

        Returns:
            float: Efficiency loss factor caused by leakage
        """

        self._eta_l = (self._m_dot - self._m_dot_l) / self._m_dot

        return self._eta_l

    def get_eta_h(self) -> float:
        """This function solves for the hydraulic efficiency factor

        Returns:
            float: Efficiency factor of the system
        """

        self._eta_h = (
            2
            * (self._phi**2)
            * (self._u / self._c1)
            * (1 + self._phi_r * (np.cos(self._beta_2) / np.cos(self._beta_1)))
        )

        return self._eta_h

    def get_zeta_eps(self) -> float:
        """Function that solves for the partial admission loss factor

        Returns:
            float: Partial Admission Loss Factor
        """

        self._zeta_eps = (
            0.35
            * (
                (0.3 * self._u_cis) / (self._d_mean * np.sin(self._alpha))
                + ((1 - self._eps) / self._eps)
            )
            * (self._u_cis**2)
            * self._eta_h
        )

        return self._zeta_eps

    def get_eta(self) -> float:
        """This function solves for the global efficiency

        Returns:
            float: Overall system efficiency
        """

        self._eta = self._eta_h * self._eta_l - self._zeta_eps

        return self._eta

    def get_power(self) -> float:
        """This function solves for the power produced by the turbine

        Returns:
            float: Power Produced from Turbine
        """

        self._p = self._eta * self._gas.get_enthalpy_drop(p1=self._p1)

        return self._p

    def solve_performance(self) -> dict:
        """This function solves for the turbine stage performance

        Returns:
            float: [P_power, eta_stage] [W, %]
        """
        # Initialise our dicts
        pressure_dict = {}
        temperature_dict = {}
        geometry_dict = {}
        angles_dict = {}
        velocities_dict = {}
        performance_dict = {}

        pressure_dict["p_1"] = self.get_p1()

        velocities_dict["u"] = self.get_blade_speed()

        geometry_dict["D_m"] = self.get_mean_diameter()

        velocities_dict["c_1s"] = self.get_isentropic_spouting()

        velocities_dict["c_1"] = self.get_actual_spouting()

        temperature_dict["t_1"] = self.get_t1()

        pressure_dict["p_1o"] = self.get_p1_o()

        geometry_dict["A_1"] = self.get_A1()

        geometry_dict["s_b"] = self.get_nozzle_height()

        geometry_dict["s_r"] = self.get_blade_height()

        geometry_dict["D_hub"], geometry_dict["D_tip"] = self.get_rotor_diameters()

        performance_dict["eps"] = self.get_partial_admission()

        angles_dict["beta_1"] = self.get_beta_1()

        temperature_dict["t_1o_r"] = self.get_relative_temp()

        pressure_dict["p_1o_r"] = self.get_p1_or()

        self.get_a_star_w()

        velocities_dict["w_2"] = self.get_w2()

        pressure_dict["p_2o_r"] = self.get_p2_or()

        angles_dict["beta_2"] = self.get_beta_2()

        temperature_dict["t_2"] = self.get_rotor_exit_temp()

        velocities_dict["c_2"] = self.get_c2()

        angles_dict["alpha_2"] = self.get_alpha_2()

        temperature_dict["t_2o"] = self.get_t_o2()

        pressure_dict["p_2o"] = self.get_po2()

        performance_dict["phi_l"] = self.get_phi_l()

        performance_dict["m_leakage"] = self.get_m_leakage()

        performance_dict["eta_l"] = self.get_leakage_loss()

        performance_dict["eta_h"] = self.get_eta_h()

        performance_dict["zeta_eps"] = self.get_zeta_eps()

        performance_dict["eta_o"] = self.get_eta()

        performance_dict["Power"] = self.get_power()

        final_dict = {
            "performance": performance_dict,
            "velocity": velocities_dict,
            "pressures": pressure_dict,
            "temperature": temperature_dict,
            "geometry": geometry_dict,
        }

        return final_dict
