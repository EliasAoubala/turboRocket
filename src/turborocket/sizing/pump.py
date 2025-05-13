"""This file contains the authors objects for the sizing of liquid propellant pumps"""

from turborocket.fluids.fluids import IncompressibleFluid
import numpy as np


class Barske:
    """This object represents a Partial Emission Pump of a Barske Style for Liquid Propellants"""

    def __init__(self, fluid: IncompressibleFluid, dp: float, m_dot: float, N: float):
        """Constructor for the Pump Object

        Args:
            fluid (IncompressibleFluid): Working Fluid of the Pump
            dp (float): Desired Pressure Rise Across the Pump (Pa)
            m_dot (float): Mass Flow Rate through the Pump (kg/s)
            N (float): Pump Nominal Shaft Speed (rpm)
        """

        self._fluid = fluid
        self._dp = dp
        self._m_dot = m_dot
        self._N = N

        self._METER_TO_INCH = 39.3701
        self._KG_M3_TO_LB_CUFT = 0.062428
        self._M2_S_TO_FT2_S = 10.7639

        return

    def size_pump(self) -> None:
        """This function sizes the pump"""

    def get_frictional_loss(self, N: float | None = None) -> float:
        """This function gets the frictional loss of the pump

        Args:
            N (float): Pump Shaft Speed (RPM). Defaults to Design Shaft Speed

        Returns:
            float: Frictional Power Loss (W)
        """

        if N is None:
            N = self._N

        rho = self._fluid.get_density() * self._KG_M3_TO_LB_CUFT
        v = (self._fluid.get_viscosity() / rho) * self._M2_S_TO_FT2_S

        d1 = self._d1 * self._METER_TO_INCH
        d2 = self._d2 * self._METER_TO_INCH
        l1 = self._l1 * self._METER_TO_INCH

        P_hp = (
            0.60e-6
            * rho
            * (v**0.2)
            * (N / 1000) ** 2.8
            * (
                d2**4.6 * ((1 / np.sin(self._alpha_1)) + (1 / np.sin(self._alpha_2)))
                + 9.2 * d1**3.6 * l1
            )
        )

        return P_hp * 745.7

    def get_pressure_rise(
        self, psi: float | None = None, N: float | None = None
    ) -> float:
        """This function solves for the pressure rise across the pump

        Args:
            psi (float | None): Pressure Coefficient (n.d). Defaults to Design Pressure Coefficient
            N (float | None): Shaft Speed of the Pump (rpm). Defaults to Design Shaft Speed.

        Returns:
            float: Pressure Rise Across Pump (Pa)
        """
        # We need to solve for the blade speeds at the tips

        if N is None:
            N = self._N
        if psi is None:
            psi = self._psi

        u_2 = N * np.pi * (self._d2) / 60

        u_1 = N * np.pi * (self._d1) / 60

        rho = self._fluid.get_density()

        dp = rho * (1 / 2) * ((1 + psi) * u_2**2 - u_1**2)

        return dp

    def get_hydraulic_power(
        self, psi: float | None, N: float | None = None, m_dot: float | None = None
    ) -> float:
        """This function solve for the hydraulic power produced by the pump

        Args:
            psi (float): Pressure Coefficient (n.d). Defaults to design Pressure Coefficient.
            N (float): Pump Shaft Speed (rpm). Defaults to Design Shaft Speed.
            m_dot (float): Mass Flow Rate Through the Pump (kg/s). Defaults to design Mass Flow Rate.

        Returns:
            float: Hydraulic Power of the Pump (W)
        """

        if m_dot is None:
            m_dot = self._m_dot

        q = m_dot / self._fluid.get_density()

        dp = self.get_pressure_rise(psi=psi, N=N)

        P = q * dp

        return P

    def get_efficiency(
        self,
        psi: float | None = None,
        N: float | None = None,
        m_dot: float | None = None,
    ) -> float:
        """This function solves for the hydraulic efficiency of the pump

        Args:
            N (float | None): Shaft Speed of the Pump (rpm). Defaults to Design Shaft Speed.
            m_dot (float | None): Mass Flow Rate Through the Pump (kg/s). Defaults to Design Mass Flow Rate.
            psi (float | None, optional): _description_. Defaults to None. Defaults to Design Pressure Coefficient

        Returns:
            float: System Efficiency (x/100)
        """

        if psi is None:
            psi = self._psi

        if N is None:
            N = None

        if m_dot is None:
            m_dot = self._m_dot

        # Firstly we get the hypothetical power
        P_s = self.get_hydraulic_power(psi=1, N=N, m_dot=m_dot)

        # Then we get the actual hydraulic power
        P_a = self.get_hydraulic_power(psi=psi, N=N, m_dot=m_dot)

        # Then we get the frictional power loss
        P_f = self.get_frictional_loss(N=N)

        # We can then calculate our total efficiency

        eta = P_a / (P_f + P_s)

        return eta
