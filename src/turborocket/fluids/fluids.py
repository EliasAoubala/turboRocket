import numpy as np


# We can create a generic class characterising an ideal gas


class IdealGas:
    def __init__(
        self,
        p: float,
        t: float,
        gamma: float,
        R: float | None = None,
        cp: float | None = None,
    ) -> None:
        """Constructor for an Ideal Gas Object

        Args:
            p (float): Gas Pressure (Pa)
            t (float): Gas Temperature (K)
            R (float): Gas Constant (J/kg K)
            gamma (float): Specific Heat Ratio (N/D)
            cp (float): Specific Heat Capacity (J/kg K)
        """

        self._p = p
        self._t = t
        self._R = R
        self._gamma = gamma
        self._cp = cp

        return

    # Getter functions

    def get_pressure(self) -> float:
        """Function for getting the gas pressure

        Returns:
            float: Gas Pressure (Pa)
        """

        return self._p

    def get_temperature(self) -> float:
        """Function for getting the gas temperature

        Returns:
            float: Gas Temperature (K)
        """

        return self._t

    def get_R(self) -> float:
        """Function for getting the Gas Constant

        Returns:
            float: Gas Constant (J / kg K)
        """

        return self._R

    def get_gamma(self) -> float:
        """Function for getting the gas specific heat ratio

        Returns:
            float: Gas specific heat ratio (N/D)
        """

        return self._gamma

    def get_cp(self) -> float:
        """Function for getting the gas specific heat capacity

        Returns:
            float: Gas specific heat capacity (J/ kg K)
        """

        return self._cp

    def get_density(self) -> float:
        """Function for getting the gas density

        Returns:
            float: Gas Density (kg /m^3)
        """

        self._rho = self._p / (self._R * self._t)

        return self._rho

    def speed_of_sound(self, T: float | None = None) -> float:
        """This function Solves for the Speed of Sound of the gas

        Args:
            T (float | None, optional): Gas Temperature. Defaults to None.

        Returns:
            float: Speed of Sound of the gas (m/s)
        """
        if T is not None:
            return np.sqrt(self._gamma * self._R * T)

        self._c = np.sqrt(self._gamma * self._R * self._t)

        return self._c

    def get_enthalpy_drop(self, p1: float) -> float:
        """This function solves for the drop in static enthalpy as a result of a pressure expansion

        Args:
            p1 (float): Downstream Pressure

        Returns:
            float: Enthalpy Drop (J / kg)
        """

        self._dh = (
            self._cp
            * self._t
            * (1 - (p1 / self._p) ** ((self._gamma - 1) / self._gamma))
        )

        return self._dh

    def get_cis(self, p1: float | None) -> float:
        """Get's the isentropic velocity caused by an enthalpy drop

        Args:
            p1 (float | None): Downstream Pressure

        Returns:
            float: Isentropic Spouting Velocity (m/s)
        """

        if p1 is not None:
            dh = self.get_enthalpy_drop(p1)

            return np.sqrt(2 * dh)

        elif self._dh is not None:

            return np.sqrt(2 * dh)

        else:
            raise TypeError(
                "Missing Downstream Pressure to solve for spounting Velocity!"
            )

    def get_critical_speed(self) -> float:
        """This function solves for the critical speed of the gas (M=1)

        Returns:
            float: Critical Speed of the gas (c*) [m/s]
        """

        self._c_star = self.speed_of_sound(T=(self._t * 2 / (self._gamma + 1)))

        return self._c_star

    def get_expanded_t(self, M_star: float) -> float:
        """This function solves for the expanded gas temperature, based on the critical mach number provided

        Args:
            M_star (float): Critical Mach Number of the gas M* (N/D)

        Returns:
            float: Expanded Gas Temperature (K)
        """

        return self._t * (1 - ((self._gamma - 1) / (self._gamma + 1)) * M_star**2)

    def get_expanded_p(self, M_star: float) -> float:
        """This function solves for the expanded gas pressure, based on the critical mach number provided

        Args:
            M_star (float): Critical Mach Number of the gas M* (N/D)

        Returns:
            float: Exit Pressure of the gas (Pa)
        """

        return self._p * (
            1 - ((self._gamma - 1) / (self._gamma + 1)) * M_star**2
        ) ** (self._gamma / (self._gamma - 1))


class IncompressibleFluid:
    """Generic Function Defining the Properties of an Incompressible Fluid"""

    def __init__(
        self,
        rho: float,
        P: float,
        T: float | None = None,
        mue: float | None = None,
        B: float | None = None,
    ) -> None:
        """Constructor for the Incompressible Fluid

        Args:
            rho (float): Fluid Density (kg/m^3)
            P (float): Fluid Pressure (kg/m^3)
            T (float): Fluid Temperature (kg/m^3)
            mue (float | None, optional): Fluid Viscosity (Pa s). Defaults to None.
            B (float | None, optional): Fluid Bulk Modulus (Pa). Defaults to None.
        """

        self._rho = rho
        self._P = P
        self._T = T
        self._mue = mue
        self._B = B

        return

    def get_density(self) -> float:
        """Getter Function for fluid Density

        Returns:
            float: Fluid Density (kg/m^3)
        """

        return self._rho

    def set_density(self, rho: float) -> None:
        """Setter Function for fluid density

        Args:
            rho (float): Fluid Density (kg/m^3)
        """

        self._rho = rho

        return

    def get_pressure(self) -> float:
        """Getter function for fluid pressure

        Returns:
            float: Fluid Pressure (Pa)
        """

        return self._P

    def set_pressure(self, P: float) -> None:
        """Setter Function for fluid Pressure

        Args:
            P (float): Fluid Pressure (Pa)
        """

        self._P = P

        return

    def get_temperature(self) -> float:
        """Getter function for fluid temperature

        Returns:
            float: Fluid Temperature (K)
        """

        return self._T

    def get_viscosity(self) -> float:
        """Getter function for fluid viscosity

        Returns:
            float: Fluid viscosity (Pa s)
        """
        return self._mue

    def get_bulk_modululs(self) -> float:
        """Getter function for the fluid's Bulk Modulus

        Returns:
            float: Fluid Bulk Modulus (Pa)
        """

        return self._B
