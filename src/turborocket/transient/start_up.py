"""This file contains the relations and objects as it relates to the evaluation of gas-generator startup conditions"""

from turborocket.fluids.fluids import IncompressibleFluid
from turborocket.solvers.solver import adjoint
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj


class GasGenerator:
    """
    Object Defining Gas Generator characteristics and behaviours
    """

    def __init__(self, Ox: str, Fu: str, Pcc: float, MR: float) -> None:
        """Constructor for the Gas Generator Object

        Args:
            Ox (str): Name of the Oxidiser Used for the Gas Generator
            Fu (str): Name of the Fule Used for the Gas Generator
            Pcc (float): Chamber Pressure of the Gas Generator [Pa]
            MR (float): Propellant Mixture Ratio
        """

        self._ox_name = Ox
        self._fu_name = Fu
        self._pcc = Pcc
        self._mr = MR

        self.cea_object()

        return

    def injector_cond(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        cdo: float,
        cdf: float,
    ) -> None:
        """This function generates the injector

        Args:
            ox_in (IncompressibleFluid): Fluid Object Defining The Inlet Oxidiser Conditions
            fu_in (IncompressibleFluid): Fluid Object Defining the Inlet Fuel Conditions
            Cdo (float): Injector Discharge Coefficient for the Oxidiser Orifices
            Cdf (float): Injector Discharge Coefficient for the Fuel Orifices
        """

        self._ox_obj = ox_in
        self._fu_obj = fu_in
        self._cdo = cdo
        self._cdf = cdf

        self._rho_ox = self._ox_obj.get_density()
        self._rho_fu = self._fu_obj.get_density()

        self._pox_inlet = self._ox_obj.get_pressure()
        self._pfu_inlet = self._fu_obj.get_pressure()

        # We can now evaluate for the specific injector areas
        self._a_m_ox = 1 / (
            self._cdo * np.sqrt(2 * self._rho_ox * (self._pox_inlet - self._pcc))
        )

        self._a_m_fu = 1 / (
            self._cdf * np.sqrt(2 * self._rho_fu * (self._pfu_inlet - self._pcc))
        )

        # We can also derfine the relative alpha factor between the two injectors
        self._alpha = (
            np.sqrt((self._pfu_inlet - self._pcc) / (self._pox_inlet - self._pcc))
            * self._mr
        )

        return

    def cea_object(self) -> None:
        """This function defines the cea object used for combustion modelling"""

        self._cea = CEA_Obj(
            oxName=self._ox_name,
            fuelName=self._fu_name,
            pressure_units="Bar",
            temperature_units="K",
            density_units="kg/m^3",
            specific_heat_units="J/kg-K",
        )

        return

    def combustion_param(self, mr: float, pcc: float, eta_c: float = 0.8) -> float:
        """This function solves for the combustion parameter

        Args:
            mr (float): Mixture Ratio of propellants
            pcc (float): Chamber Pressure of the Gas
            eta_c (float): C* efficiency of the combustion

        Returns:
            float: Combustion Parameter
        """

        # Getting combustion c_star value (this is converted from ft/s to m/s)
        c_star = self._cea.get_Cstar(Pc=pcc, MR=mr) * 0.3048 * eta_c

        return pcc / c_star

    def size_system(self, m_dot: float, eta_c: float = 0.8) -> dict:
        """This function sizes the gas generator

        Args:
            m_dot (float): Mass Flow Rate Through Gas Generator (kg/s)
            eta_c (float): C* efficiency of the combustion (%)

        Returns:
            dict
        """

        self._m_dot = m_dot
        self._eta_c = eta_c

        param = self.combustion_param(mr=self._mr, pcc=self._pcc, eta_c=eta_c)

        self._a_cc = self._m_dot / (param)

        # Solving for new injection conditions
        self._m_dot_fu = self._m_dot / (self._mr + 1)

        self._a_fu = self._a_m_fu * self._m_dot_fu

        self._m_dot_ox = self._m_dot * (self._mr / (self._mr + 1))

        self._a_ox = self._a_m_ox * self._m_dot_ox

        dic = self.get_cond(
            ox_inlet=self._ox_obj,
            fu_inlet=self._fu_obj,
            pcc=self._pcc,
            eta_c=self._eta_c,
        )

        return dic

    def get_cond(
        self,
        ox_inlet: IncompressibleFluid,
        fu_inlet: IncompressibleFluid,
        pcc: float,
        eta_c: float,
    ) -> dict:
        """This function returns a dictionary of the key parameters for the system at a given condition

        Args:
            ox_inlet (IncompressibleFluid): Object Representing the state of the Oxidiser Inlet
            fu_inlet (IncompressibleFluid): Object Representing the state of the Fuel Inlet
            pcc (float): Chamber Pressure for the system

        Returns:
            dict: Dictionary of Key Parameters for the Combustion [Pcc, MR, To, m_dot_t, ox_stiff, fu_stiff, gamma, cp]
        """

        p_ox = ox_inlet.get_pressure()
        p_fu = fu_inlet.get_pressure()

        mr = self._alpha * np.sqrt((p_ox - pcc) / (p_fu - pcc))

        param = self.combustion_param(mr=mr, pcc=pcc, eta_c=eta_c)
        m_dot_t = self._a_cc * param

        m_dot_o = m_dot_t * mr / (mr + 1)
        m_dot_f = m_dot_t / (mr + 1)

        ox_stiff = (p_ox - pcc) / pcc
        fu_stiff = (p_fu - pcc) / pcc

        # Finally, we can get the combustion gas thermal conditions - this assumes ideal gas application
        to = self._cea.get_Temperatures(Pc=pcc, MR=mr)[0] * eta_c**2

        # Getting combustion gas properties
        cp = self._cea.get_Chamber_Cp(Pc=pcc, MR=mr)
        gamma = self._cea.get_Chamber_MolWt_gamma(Pc=pcc, MR=mr)[1]

        # We create our Dict and finally return it

        dic = {
            "P_cc": pcc,
            "MR": mr,
            "T_o": to,
            "Cp": cp,
            "gamma": gamma,
            "ox_stiffness": ox_stiff,
            "fu_stiffness": fu_stiff,
            "m_dot_t": m_dot_t,
            "m_dot_o": m_dot_o,
            "m_dot_f": m_dot_f,
        }

        return dic

    def get_geometry(self) -> dict:
        """Function that gets the injector geometry for the user

        Returns:
            dict: Dictionary Describing key parameters
        """

        dic = {
            "CdA_ox": self._cdo * self._a_ox,
            "CdA_fu": self._cdf * self._a_fu,
            "A_fu": self._a_fu,
            "A_ox": self._a_ox,
            "Acc": self._a_cc,
        }

        return dic

    def perturb_error(
        self,
        pcc: float,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        k_f: float,
        alpha: float,
        eta_c: float,
    ) -> float:
        """This function solves for the error in a given perturbation

        Args:
            pcc (float): Chamber Pressure of the gas (Pa)
            ox_in (IncompressibleFluid): Inlet Oxidiser Object
            fu_in (IncompressibleFluid): Inlet Fuel Object
            k_f (float): Fuel Injector Orifice Factor (kg / s Pa**(1/2))
            alpha (float): Injector Orifice Ratio Factor (N/D)
            eta_c (float): C* Efficiency

        Returns:
            float: Error in pertubation
        """
        p_ox = ox_in.get_pressure()
        p_fu = fu_in.get_pressure()

        mr = alpha * ((p_ox - pcc) / (p_fu - pcc)) ** (1 / 2)

        # We can now solve the combustion parameter at this condition
        param = self.combustion_param(pcc=pcc, mr=mr, eta_c=eta_c)

        # We can solve for the rhs of the equation
        rhs = k_f * (alpha * np.sqrt(p_ox - pcc) + np.sqrt(p_fu - pcc))

        error = rhs - param

        return error

    def solve_perturb(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
    ) -> dict:
        """This function solves for the updated gas generator conditions based on changes to the oxidiser and fuel inlet conditions

        Args:
            ox_in (IncompressibleFluid): Oxidiser Incompressible Fluid
            fu_in (IncompressibleFluid): Fuel Incompressible Fluid

        Returns:
            dict: Dictionary of Key Parameters for the Combustion [Pcc, MR, To, m_dot_t, ox_stiff, fu_stiff, gamma, cp]
        """

        # We evaluate for our K_f term
        self._kf = self._cdf * self._a_fu * np.sqrt(2 * self._rho_fu) / self._a_cc

        # We can estimate the intial chamber pressure as 2/3 of the lower pressure
        p_ox = ox_in.get_pressure()
        p_fu = fu_in.get_pressure()

        pcc_guess = min(p_ox, p_fu) * (2 / 3)

        # We can now setup the adjoint optimisation
        pcc_new = adjoint(
            func=self.perturb_error,
            x_guess=pcc_guess,
            dx=0.1e5,
            n=500,
            relax=1,
            target=0,
            params=[ox_in, fu_in, self._kf, self._alpha, self._eta_c],
        )

        # We can now solve for all the key conditions
        dic = self.get_cond(
            ox_inlet=ox_in, fu_inlet=fu_in, pcc=pcc_new, eta_c=self._eta_c
        )

        return dic
