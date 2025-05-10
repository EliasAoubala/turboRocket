"""This file contains the relations and objects as it relates to the evaluation of transient start-ups of turbopumps"""

from turborocket.fluids.fluids import IncompressibleFluid
from turborocket.solvers.solver import adjoint
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj


class CombustionChamber:
    """
    Object Defining Combustion Chamber characteristics and behaviours
    """

    def __init__(self, Ox: str, Fu: str, Pcc: float, MR: float) -> None:
        """Constructor for the Combustion Chamber Object

        Args:
            Ox (str): Name of the Oxidiser Used for the Combustion Chamber
            Fu (str): Name of the Fule Used for the Combustion Chamber
            Pcc (float): Chamber Pressure of the Combustion Chamber [Pa]
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
        c_star = self._cea.get_Cstar(Pc=pcc / 1e5, MR=mr) * 0.3048 * eta_c

        return pcc / c_star

    def size_system(self, m_dot: float, eta_c: float = 0.8) -> dict:
        """This function sizes the Combustion Chamber

        Args:
            m_dot (float): Mass Flow Rate Through Combustion Chamber (kg/s)
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
        to = self._cea.get_Temperatures(Pc=pcc / 1e5, MR=mr)[0] * eta_c**2

        # Getting combustion gas properties
        gamma = self._cea.get_Chamber_MolWt_gamma(Pc=pcc / 1e5, MR=mr)[1]
        R = 8314 / self._cea.get_Chamber_MolWt_gamma(Pc=pcc / 1e5, MR=mr)[0]

        # Specific Heat Capacity is calculated based on gamma and R, as CEA will account for the solid fraction which is unrealistic for this.
        cp = R / ((gamma - 1) / gamma)

        # We create our Dict and finally return it

        dic = {
            "P_cc": pcc,
            "MR": mr,
            "T_o": to,
            "Cp": cp,
            "gamma": gamma,
            "R": R,
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

    def solve_perturb_ss(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
    ) -> dict:
        """This function solves for the updated combustion chamber conditions based on changes to the oxidiser and fuel inlet conditions for a steady state

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

        # We can also attach the error
        error = self.perturb_error(
            pcc=pcc_new,
            ox_in=ox_in,
            fu_in=fu_in,
            k_f=self._kf,
            alpha=self._alpha,
            eta_c=self._eta_c,
        )
        dic["error"] = error

        return dic

    def set_pcc_transient(self, P_cc_transient: float) -> None:
        """This function sets the transient set pressure used for transient modelling of the combustion chamber

        Args:
            P_cc_transient (float): Transient Chamber Pressure (Pa)
        """

        self._pcc_transient = P_cc_transient

        return

    def get_pcc_transient(self) -> float:
        """This is a "getter" function for the transient set pressure of the combustion chamber

        Returns:
            float: Transient Chamber Pressure (Pa)
        """

        return self._pcc_transient

    def set_l_star(self, L_star: float) -> None:
        """This function sets L* of the chamber

        Args:
            L_star (float): L_star of the combustion chamber (m)
        """

        self._l_star = L_star

        self._v_cc = self._a_cc * self._l_star

        return

    def get_c_star(self, Pcc: float, MR: float, eta_c: float) -> float:
        """This Function Gets the C* of the Combustion Chamber Based on Given Efficiencies

        Args:
            Pcc (float): Combustion Chamber Pressure of Gas
            Mr (float): Mixture Ratio of Propellants

        Returns:
            float: C* of the combustion (m/s)
        """

        c_star = self._cea.get_Cstar(Pc=Pcc / 1e5, MR=MR) * 0.3048 * eta_c

        return c_star

    def get_density(self, Pcc: float, MR: float, eta_c: float) -> float:
        """This function gets the Combustion gas density
        - Assumption: Ideal Gas

        Args:
            Pcc (float): Chamber Pressure (Pa)
            MR (float): Mixture Ratio
            eta_c (float): C* Efficiency of the Gas

        Returns:
            float: Density of the Gas (kg/s)
        """
        # Getting combustion gas properties
        R = 8314 / self._cea.get_Chamber_MolWt_gamma(Pc=(Pcc / 1e5), MR=MR)[0]
        T = self._cea.get_Temperatures(Pc=Pcc / 1e5, MR=MR, frozen=1)[0] * eta_c**2

        rho = Pcc / (R * T)

        return rho

    def get_injector_flow(
        self, ox_in: IncompressibleFluid, fu_in: IncompressibleFluid, Pcc: float
    ) -> tuple:
        """This function solves for the oxidiser and fuel injector mass_flows based on the transient chamber pressure

        Args:
            ox_in (IncompressibleFluid): _description_
            fu_in (IncompressibleFluid): _description_

        Returns:
            tuple: (m_dot_ox, m_dot_fu)
        """

        p_ox = ox_in.get_pressure()
        p_fu = fu_in.get_pressure()

        rho_ox = ox_in.get_density()
        rho_fu = fu_in.get_density()

        dp_ox = p_ox - Pcc

        dp_fu = p_fu - Pcc

        if dp_ox <= 0:
            m_dot_ox = 0
        else:
            m_dot_ox = self._cdo * self._a_ox * (2 * rho_ox * (p_ox - Pcc)) ** (1 / 2)

        if dp_fu <= 0:
            m_dot_fu = 0
        else:
            m_dot_fu = self._cdf * self._a_fu * (2 * rho_fu * (p_fu - Pcc)) ** (1 / 2)

        return (m_dot_ox, m_dot_fu)

    def transient_engine_nofire(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        m_dot_ox: float,
        m_dot_fu: float,
        eta_c: float = 0.85,
    ) -> dict:
        """Function for the case where the engine hasnt lit yet.

        Args:
            ox_in (IncompressibleFluid): _description_
            fu_in (IncompressibleFluid): _description_
            m_dot_ox (float): _description_
            m_dot_fu (float): _description_
            eta_c (float, optional): _description_. Defaults to 0.85.

        Returns:
            dict: Dictionary
        """

        dp_dt = 0
        self._pcc_transient = 1e5

        dic = {
            "dp_dt": dp_dt,
            "P_cc": self._pcc_transient,
            "MR": 0,
            "T_o": 0,
            "Cp": 1005,
            "gamma": 1.4,
            "ox_stiffness": 0,
            "fu_stiffness": 0,
            "m_dot_t": 0,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        return dic

    def transient_startup(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        m_dot_ox: float,
        m_dot_fu: float,
        eta_c: float = 0.85,
    ) -> dict:

        MR_current = m_dot_ox / m_dot_fu

        # We need to solve for the combustion density at the current time
        rho_c = self.get_density(Pcc=self._pcc_transient, MR=MR_current, eta_c=eta_c)

        # W need to get the c_star of the current condition
        c_star = self.get_c_star(Pcc=self._pcc_transient, MR=MR_current, eta_c=eta_c)

        # Finally we can solve for the pressure gradient
        dp_dt = (self._pcc_transient / (rho_c * self._v_cc)) * (
            m_dot_ox + m_dot_fu - (self._pcc_transient * self._a_cc) / c_star
        )

        # We need to now evaluate for the system performance paramets

        R = (
            8314
            / self._cea.get_Chamber_MolWt_gamma(
                Pc=self._pcc_transient / 1e5, MR=MR_current
            )[0]
        )
        gamma = self._cea.get_Chamber_MolWt_gamma(
            Pc=self._pcc_transient / 1e5, MR=MR_current
        )[1]

        dic = {
            "dp_dt": dp_dt,
            "P_cc": self._pcc_transient,
            "MR": MR_current,
            "T_o": self._cea.get_Temperatures(
                Pc=self._pcc_transient / 1e5, MR=MR_current, frozen=1
            )[0]
            * eta_c**2,
            "Cp": R / ((gamma - 1) / gamma),
            "gamma": gamma,
            "R": R,
            "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "m_dot_t": m_dot_fu + m_dot_ox,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        # We store the last MR for locking
        self._MR_transient = MR_current

        return dic

    def transient_shutdown(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        m_dot_ox: float,
        m_dot_fu: float,
        eta_c: float = 0.85,
    ) -> dict:
        """Function characterinsing the shutdown transient

        Args:
            ox_in (IncompressibleFluid): Inlet Oxidiser Object
            fu_in (IncompressibleFluid): Fuel Injector Object
            m_dot_ox (float): Oxidiser Mass Flow
            m_dot_fu (float): Fuel Mass Flow
            eta_c (float, optional): Combustion Efficiency. Defaults to 0.85.

        Returns:
            dict: _description_
        """
        c_star = self.get_c_star(
            Pcc=self._pcc_transient, MR=self._MR_transient, eta_c=eta_c
        )

        rho_c = self.get_density(
            Pcc=self._pcc_transient, MR=self._MR_transient, eta_c=eta_c
        )

        dp_dt = (self._pcc_transient / (rho_c * self._v_cc)) * (
            -(self._pcc_transient * self._a_cc) / c_star
        )

        R = (
            8314
            / self._cea.get_Chamber_MolWt_gamma(
                Pc=self._pcc_transient / 1e5, MR=self._MR_transient
            )[0]
        )
        gamma = self._cea.get_Chamber_MolWt_gamma(
            Pc=self._pcc_transient / 1e5, MR=self._MR_transient
        )[1]

        dic = {
            "dp_dt": dp_dt,
            "P_cc": self._pcc_transient,
            "MR": self._MR_transient,
            "T_o": self._cea.get_Temperatures(
                Pc=self._pcc_transient / 1e5, MR=self._MR_transient, frozen=1
            )[0]
            * eta_c**2,
            "Cp": R / ((gamma - 1) / gamma),
            "R": R,
            "gamma": gamma,
            "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "m_dot_t": m_dot_fu + m_dot_ox,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        return dic

    def transient_time_step(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        eta_c: float = 0.85,
    ) -> dict:
        """Conducts a Transient Time Step for the Combustion Chamber Performance

        Args:
            ox_in (IncompressibleFluid): _description_
            fu_in (IncompressibleFluid): _description_
            dt (float): _description_
            eta_c (float): C* efficiency of the combustion. Defaults to 0.85.

        Returns:
            dict: Dictionary of Performance metrics of the combustion
        """
        m_dot_ox, m_dot_fu = self.get_injector_flow(
            ox_in=ox_in, fu_in=fu_in, Pcc=self._pcc_transient
        )

        # We check for the condition for engine ignition, otherwise we leave the system as is.

        if (m_dot_ox == 0) or (m_dot_fu == 0):
            # Engine is not lit, hence we dont consider mass-flows, we just deplete the engine until it goes back to ambient

            # We check if the chamber is at ambient conditions
            if self._pcc_transient <= 1e5:
                dic = self.transient_engine_nofire(
                    ox_in=ox_in,
                    fu_in=fu_in,
                    m_dot_fu=m_dot_fu,
                    m_dot_ox=m_dot_ox,
                    eta_c=eta_c,
                )
            else:
                # We deplete the chamber using the c* value
                dic = self.transient_shutdown(
                    ox_in=ox_in,
                    fu_in=fu_in,
                    m_dot_fu=m_dot_fu,
                    m_dot_ox=m_dot_ox,
                    eta_c=eta_c,
                )

        else:
            # We need to check if our mass flow rates are enough for ignition - arbitrary criterion of 5%
            m_dot_t = m_dot_fu + m_dot_ox

            if m_dot_t < self._m_dot * 0.05:
                dic = self.transient_engine_nofire(
                    ox_in=ox_in,
                    fu_in=fu_in,
                    m_dot_fu=m_dot_fu,
                    m_dot_ox=m_dot_ox,
                    eta_c=eta_c,
                )

            else:
                dic = self.transient_startup(
                    ox_in=ox_in,
                    fu_in=fu_in,
                    m_dot_fu=m_dot_fu,
                    m_dot_ox=m_dot_ox,
                    eta_c=eta_c,
                )

        return dic


class GasGenerator(CombustionChamber):
    """This Object Defines the Characteristics of the Gas Generator Object"""

    def __init__(self, Ox, Fu, Pcc, MR):
        super().__init__(Ox, Fu, Pcc, MR)


class MainEngine(CombustionChamber):
    """This Object Defines the Characteristics of the Main Engine Object"""

    def __init__(self, Ox, Fu, Pcc, MR):
        super().__init__(Ox, Fu, Pcc, MR)


class LiquidValve:
    """Object Defining the Behaviour of Liquid Propellant Valves"""

    def __init__(
        self,
        cda: float,
        tau: float,
        s_pos_init: float = 0,
        epsilon: float = 100,
    ):
        """Constructor for the liquid propellant valve

        Args:
            cda (float): Flow Area of the valve
            tau (float): Opening/Closing Time of the valve
            s_pos_init (float): Initial Position of the valve
            epsilon (float): Normalisation Parmaeter (Pa). Defaults to 100.
            L_eff (float): Effective flow length within valve for Damping (m). Defaults to 0.015 m
        """

        self._cda = cda
        self._tau = tau
        self._s_pos = s_pos_init
        self._pos = s_pos_init
        self._epsilon = epsilon

        return

    def actuate(self, position: float) -> None:
        """Set's the commanded position of the liquid valve

        Args:
            position (float): Active position of the liquid valve
        """

        self._s_pos = position

        return

    def update_pos(self, dt: float) -> None:
        """This function updates the valve position using a first order model

        Args:
            dt (float): Time Step
        """

        ds_dt = (self._s_pos - self._pos) / self._tau

        self._pos += ds_dt * dt

        return

    def get_mdot(
        self, upstr: IncompressibleFluid, downstr: IncompressibleFluid, dt: float
    ) -> float:
        """This function gets the massflow rate through the valve based on the upstream and downstream conditions

        Args:
            upstr (IncompressibleFluid): Upstream Fluid Object
            downstr (IncompressibleFluid): Downstream Fluid Object
            dt (float): Integration Time Step

        Returns:
            float: Mass Flow Rate (kg/s)
        """

        p1 = upstr.get_pressure()

        p2 = downstr.get_pressure()

        if p1 > p2:
            rho = upstr.get_density()

        else:
            rho = downstr.get_density()

        a = self._cda * self._pos

        dpe = p1 - p2

        # We normalise the flow equation to allow for non-infinite fradients at low dps
        dp_a = ((dpe) ** 2 + self._epsilon**2) ** (1 / 2)

        m_dot = a * ((dpe) / dp_a) * (2 * rho * dp_a) ** (1 / 2)

        return m_dot

    def get_pos(self) -> float:
        """Function that gets the position of the valve

        Returns:
            float: Position of the valve
        """

        return self._pos

    def get_inertial_param(
        self, upstr: IncompressibleFluid, downstr: IncompressibleFluid
    ) -> float:
        """This function solves for the inertial flow parameter of the valve (used for modelling inertial flows in transient conditions)

        Args:
            upstr (IncompressibleFluid): Upstream Flow Object
            downstr (IncompressibleFluid): Downstream Flow Object

        Returns:
            float: Inertial Parameter Pressure Drop (Pa)
        """

        m_dot = self.get_mdot(upstr=upstr, downstr=downstr)

        a = self._cda * self._pos

        rho = upstr.get_density()

        dp = m_dot**2 / (2 * rho * (a) ** 2)

        return dp


class Turbine:
    """Object That Defines the Turbine Transient Performance"""

    def __init__(
        self,
        delta_b: float,
        a_rat: float,
        D_m: float,
        eta: float,
        I: float | None = None,
    ):
        """Constructor for the Transient Turbine Object

        Args:
            I (float): Moment of Inertia of Blisk (kg m^2)
            delta_b (float): Change in Angle of Turbine Blade (Degrees)
            a_rat (float): Area Ratio of Nozzle
            d_m (float): Mean Diameter of Turbine (m)
            eta (float): Turbine Efficiency
        """

        self._I = I
        self._delta_b = (delta_b / 180) * np.pi
        self._a_rat = a_rat
        self._rm = D_m / 2
        self._eta = eta

        return

    def area_error(self, M: float, gamma: float):
        """Error Function for the Nozzle Expansion Ratio Relationship

        Args:
            M (float): Mach number at the Exit of the Nozzle
            gamma (float): Specific Heat Ratio of the Gas
        """

        rhs = (1 / M) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2)) ** (
            (gamma + 1) / (2 * (gamma - 1))
        )

        error = rhs - self._a_rat

        return error

    def get_subsonic_mach(self, gamma):
        """This Function Solves for the subsonic Mach Number solution

        Args:
            gamma (_type_): Specific Heat Ratio of the Gas

        Returns:
            _type_: Subsonic Mach number
        """
        M = adjoint(
            func=self.area_error,
            x_guess=0.4,
            dx=0.01,
            n=500,
            relax=1,
            target=0,
            params=[gamma],
        )

        return M

    def get_supersonic_mach(self, gamma):
        """This function solves for the Supersonic Mach Number solution

        Args:
            gamma (_type_): Specific Heat Ratio of the Gas

        Returns:
            _type_: Supersonic Mach Number
        """
        M = adjoint(
            func=self.area_error,
            x_guess=2,
            dx=0.01,
            n=500,
            relax=1,
            target=0,
            params=[gamma],
        )

        return M

    def get_static_temp(self, T_o: float, M: float, gamma: float) -> float:
        """This function solves for the static temperature of the gas based on the mach number.

        Args:
            T_o (float): Stagnation temperature of Gas (K)
            M (float): Mach Number of Gas Flow

        Returns:
            float: Static Temperature of Gas
        """

        T = T_o / (1 + ((gamma - 1) / 2) * M**2)

        return T

    def get_exit_pressure(self, P_o: float, M: float, gamma: float) -> float:
        """This function solves for the static temperature of the gas based on the mach number.

        Args:
            P_o (float): Stagnation pressure of the gas (Pa)
            M (float): Mach Number of the Gas Flow
            gamma (float): Specific Heat Ratio of the Gas

        Returns:
            float: Static Pressure of the Gas
        """

        P = P_o / (1 + ((gamma - 1) / 2) * M**2) ** (gamma / (gamma - 1))

        return P

    def get_exit_velocity(self, T: float, gamma: float, R: float, M: float) -> float:
        """This function solves for the exit velocity of the gas

        Args:
            T (float): Static Temperature of the gas at the nozzle exit
            gamma (float): Specific Heat Ratio of the Gas
            R (float): Specific Gas Constant
            M (float): Mach Number of the gas at the nozzle exit

        Returns:
            float: Exit Velocity of the gas
        """

        v = M * (gamma * R * T) ** (1 / 2)

        return v

    def get_choke_prat(self, gamma: float) -> float:
        """This function solves for the choking pressure ratio of a fluid

        Args:
            gamma (float): Specific Heat Ratio of the gas

        Returns:
            float: Choking Pressure Ratio (P_o/P)
        """

        p_rat = 1 / (self.get_exit_pressure(P_o=1, M=1, gamma=gamma))

        return p_rat

    def torque_subsonic(
        self, T_o: float, P_o: float, gamma: float, R: float, P_exit: float
    ) -> float:
        """Gets the specific torque for a subsonic solution

        Args:
            T_o (float): _description_
            P_o (float): _description_
            gamma (float): _description_
            R (float): _description_
            P_exit (float): _description_

        Returns:
            float: Specific Torque produced by turbine (N m s / kg)
        """

        M_sub = self.get_subsonic_mach(gamma=gamma)

        # We then get the exit temperature
        T_s = self.get_static_temp(T_o=T_o, M=M_sub, gamma=gamma)

        P_s = self.get_exit_pressure(P_o=P_o, M=M_sub, gamma=gamma)

        if P_s < P_exit:
            v_e = 0
        else:
            v_e = self.get_exit_velocity(T=T_s, gamma=gamma, R=R, M=M_sub)

        T = self._eta * v_e * self._rm * (1 + np.cos(self._delta_b))

        return T

    def torque_supersonic(
        self, T_o: float, P_o: float, gamma: float, R: float, P_exit: float
    ) -> float:
        """Gets the specific torque for a supersonic solution

        Args:
            T_o (float): _description_
            P_o (float): _description_
            gamma (float): _description_
            R (float): _description_
            P_exit (float): _description_

        Returns:
            float: Specific Torque produced by turbine (N m s / kg)
        """

        M_sus = self.get_supersonic_mach(gamma=gamma)

        # We then get the exit temperature
        T_s = self.get_static_temp(T_o=T_o, M=M_sus, gamma=gamma)

        P_s = self.get_exit_pressure(P_o=P_o, M=M_sus, gamma=gamma)

        if P_s < P_exit:
            v_e = 0
        else:
            v_e = self.get_exit_velocity(T=T_s, gamma=gamma, R=R, M=M_sus)

        T = self._eta * v_e * self._rm * (1 + np.cos(self._delta_b))

        return T

    def get_torque(
        self, T_o: float, P_o: float, gamma: float, R: float, P_exit: float
    ) -> float:
        """This function solves for the Torque produced by the Turbine Stage

        Args:
            T_o (float): Stagnation Temperature of the Gas (K)
            P_o (float): Stagnation Pressure of the Gas (Pa)
            gamma (float): Specific Heat Ratio of the Gas
            R (float): Specific Gas Constant of the Gas (J/kg)
            P_exit (float): Exit Static Pressure of the turbine stage

        Returns:
            float: Torque produced by the turbine stage
        """

        # We solve for the torques produced by both solutions, then select the largest one

        T_sub = self.torque_subsonic(T_o=T_o, P_o=P_o, gamma=gamma, R=R, P_exit=P_exit)

        T_sus = self.torque_supersonic(
            T_o=T_o, P_o=P_o, gamma=gamma, R=R, P_exit=P_exit
        )

        T = max(T_sub, T_sus)

        return T


class Pump:
    """Object representing the transient functionality of the pump"""

    def __init__(
        self,
        D: float,
        Q_nom,
        eta_nom: float,
        N_nom: float,
        I: float | None = None,
        Q_max_factor: float = 1.5,
        k_factor: float = 0.25,
    ):

        self._D = D
        self._I = I

        self._g = 9.18

        self._Q_bep_d = Q_nom
        self._Q_max_d = Q_nom * Q_max_factor
        self._N_nom = N_nom

        self._eta_bep = eta_nom
        self._k = k_factor

        return

    def shut_off_head(self, N: float) -> float:
        """This function estimates the the theoretical shut off head of a pump

        Args:
            N (float): Rotational Rate for the Pump (rad/s)
        """

        u_o = self._D * N / 2

        H_o = (u_o**2) / (2 * self._g)

        return H_o

    def get_q_bep(self, N: float) -> float:
        """This function get sthe best operation point for the turbine at the selected shaft speed

        Args:
            N (_type_): Shaft Speed (Rad/s)

        Returns:
            float: Best Operating Point Flow Rate (m^3/s)
        """

        return self._Q_bep_d * (N / self._N_nom)

    def get_q_max(self, N: float) -> float:
        """This function gets the maximum flow operating point for the turbine at the selected shaft speed

        Args:
            N (float): Shaft Speed (Rad/s)

        Returns:
            float: Maximum Flow Operating Point (m^3/s)
        """

        return self._Q_max_d * (N / self._N_nom)

    def get_eta(self, Q: float, N: float) -> float:
        """Simplified function that solves for the efficiency of the Pump

        Args:
            Q (float): Flow Rate of Fluid Through the Pump (m^3/s)

        Returns:
            float: Efficiency of the Turbine
        """
        # We need to get the best operating point
        Q_bep = self.get_q_bep(N=N)

        eta = self._eta_bep * (1 - (((Q - Q_bep) ** 2) / self._k))

        if eta < 0:
            eta = 0

        return eta

    def get_head(self, Q, N) -> float:
        """This function solves for the head produced by the pump

        Args:
            Q (_type_): Volumetric Flow Rate Through the Pump (m^3 /s)
            N (_type_): Rotational Rate for the Pump (Rad/s)

        Returns:
            float: Head Produced by Pump (m)
        """

        # We firstly need to solve for the shut_off head of the pump
        H_o = self.shut_off_head(N=N)

        # We need to get the maximum flow operating point
        Q_max = self.get_q_max(N=N)

        if Q_max == 0:
            # Pump is not spinning at all, hence no head.
            H = 0
            return 0

        # We can now solve for the head produced by the pump
        H = H_o * (1 - (Q / Q_max) ** 2)

        if H < 0:
            H = 0

        return H

    def get_exit_condition(
        self, inlet: IncompressibleFluid, N: float, m_dot: float
    ) -> IncompressibleFluid:
        """This function solves for the exit conditions of the pump

        Args:
            inlet (IncompressibleFluid): Inlet Fluid Object
            N (float): Rotational Rate of the Pump
            m_dot (float): Mass Flow Rate Through the Pump

        Returns:
            IncompressibleFluid: Exit Fluid Object
        """
        rho = inlet.get_density()
        Q = m_dot / rho
        p_inlet = inlet.get_pressure()

        H = self.get_head(Q=Q, N=N)

        # We get the pump efficiency
        eta = self.get_eta(Q=Q, N=N)

        if eta < 0:
            eta = 0

        dp = eta * (H * self._g * rho)

        outlet = IncompressibleFluid(rho=rho, P=p_inlet + dp)

        return outlet

    def get_torque(self, inlet: IncompressibleFluid, N: float, m_dot: float) -> float:
        """This function solves for the torque produced from the pump

        Args:
            inlet (IncompressibleFluid): Inlet Fluid of the pump
            N (float): Rotational Rate of the Pump (rad/s)
            m_dot (float): Mass Flow Rate Through the pump

        Returns:
            float: Torque Produced from the Pump
        """
        rho = inlet.get_density()
        Q = m_dot / rho

        H = self.get_head(Q=Q, N=N)

        P_shaft = m_dot * self._g * H

        if N == 0:
            # Shaft not spinning, hence no torque
            T = 0
            return T

        T = P_shaft / N

        return T


class Cavity:
    """Object Defining the characteristics of liquid incompressible cavities"""

    def __init__(self, fluid: IncompressibleFluid, V: float) -> None:
        """Constructor for the cavity object

        Args:
            fluid (IncompressibleFluid): Initial fluid state within cavity
        """

        self._fluid = fluid
        self._v = V

    def update_pressure(self, m_dot: float) -> None:
        """This function updates the pressure within the cavity, using the bulk modulus approach

        Args:
            m_dot (float): Mass-flow entering/exiting cavity (kg/s)
        """
        B = self._fluid.get_bulk_modululs()
        rho = self._fluid.get_density()

        dv = m_dot / rho

        dp = B * dv / self._v

        p2 = self._fluid.get_pressure() + dp

        self._fluid.set_pressure(P=p2)

        return

    def get_fluid(self) -> IncompressibleFluid:
        """Function that gets the fluid class of the cavity

        Returns:
            IncompressibleFluid: Fluid Subclass of the Cavity
        """

        return self._fluid
