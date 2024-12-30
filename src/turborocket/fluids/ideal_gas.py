
import numpy as np


# We can create a generic class characterising an ideal gas


class IdealFluid():
    def __init__(self,
                 P: float | None = None,
                 T: float | None = None,
                 R_gas: float | None = None,
                 rho: float | None = None,
                 gamma: float = 1.4,
                 c: float | None = None,
                 total: bool = False):
        self._P = P
        self._T = T
        self._R_gas = R_gas
        self._rho = rho
        self._gamma = gamma
        self._total = total
        self._c = c

    def get_density(self) -> float:
        """This function computes density of an ideal gas

        Args:
            P (float): Gas Pressure (Pa)
            R (float): Specific Gas Constant (J/kg K)
            T (float): Gas Temperature (K)

        Returns:
            rho (float): Resultant gas Density
        """
        if not self._rho:
            self.rho = self._P/(self._R_gas*self._T)

        return self.rho

    def get_gamma(self) -> float:
        return self._gamma

    def speed_of_sound(self) -> float:
        """This function computes the speed of sound of an ideal gas

        Args:
            gamma (float): Specific Heat Ratio of the Gas (N.D)
            R (float): Specific Gas Constant (J/kg K)
            T (float): Gas Temperature (K)

        Returns:
            float: Resultant gas speed of sound
        """
        if not self._c:
            self._c = np.sqrt(self._gamma * self._R_gas * self._T)

        return self._c
