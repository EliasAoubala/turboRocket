"""This file contains the relations and objects as it relates to the evaluation of transient start-ups of turbopumps"""

from turborocket.fluids.fluids import IncompressibleFluid, IdealGas
from turborocket.solvers.solver import adjoint
import numpy as np
from turborocket.combustion.comb_solver import CombustionCantera


class CombustionChamber:
    """
    Object Defining Combustion Chamber characteristics and behaviours
    """

    def __init__(
        self,
        Ox: str,
        Fu: str,
        Pcc: float,
        MR: float,
    ) -> None:
        """Constructor for the Combustion Chamber Object

        Args:
            Ox (str): Name of the Oxidiser Used for the Combustion Chamber
            Fu (str): Name of the Fule Used for the Combustion Chamber
            Pcc (float): Chamber Pressure of the Combustion Chamber [Pa]
            MR (float): Propellant Mixture Ratio
            look_up (bool, optional): Look Up Flag for Cantera (Defaults to False)
            look_up_file (str | None, optional): Look Up File Directory (Defaults to None)
        """

        self._ox_name = Ox
        self._fu_name = Fu
        self._pcc = Pcc
        self._mr = MR

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

    def comb_object(
        self,
        look_up: bool = False,
        look_up_file: str | None = None,
        combustion_file: str | None = None,
    ) -> None:
        """This function sets up the combustion object used for the comuting of combustion conditions

        Args:
            look_up (bool, optional): Look Up Flag for whether interpolation based approach is used. Defaults to False.
            look_up_file (str | None, optional): Name of the Lookup file to be loaded for the interpolation. Defaults to None.
            combustion_file (str | None, optional): Mechanism File Used for Combustion Modelling. Defaults to None.
        """

        self._comb = CombustionCantera(
            fuel=self._fu_name,
            oxidiser=self._ox_name,
            species_file=combustion_file,
            look_up=look_up,
            look_up_file=look_up_file,
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

        c_star = self.get_c_star(Pcc=pcc, MR=mr, eta_c=eta_c)

        return pcc / c_star

    def area_rat(self,
                 gas: IdealGas,
                 P_a: float) -> float:
        """This function evaluates for the ideal nozzle area ratio for the chamber

        Args:
            gas (IdealGas): Gas Component for the chamber conditions
            P_a (float): Ambient Pressure (Bar)

        Returns:
            float: Required Area ratio for selected expansion
        """
        
        gamma = gas.get_gamma()
        P_c = gas.get_pressure()
        
        A_rat = (
            ((gamma - 1)/2)**(1/2) 
            * (2 / (gamma + 1)) ** ((gamma + 1)/(2 * (gamma - 1)))
            * (P_a / P_c) ** (-1 / gamma)
            * (1 - (P_a / P_c) ** ((gamma - 1) / gamma)) ** (- (1/2))
        )
        
        return A_rat
    
    

    def size_system(self, m_dot: float, eta_c: float = 0.8, P_e: float = 1e5) -> dict:
        """This function sizes the Combustion Chamber

        Args:
            m_dot (float): Mass Flow Rate Through Combustion Chamber (kg/s)
            eta_c (float, optional): C* efficiency of the combustion (%). Defaults to 0.8.
            P_e (float, optional): Nozzle Exit Pressure (Pa). Defaults to 1 Bar (Atmosphere).

        Returns:
            dict
        """

        self._m_dot = m_dot
        self._eta_c = eta_c

        param = self.combustion_param(mr=self._mr, pcc=self._pcc, eta_c=eta_c)

        self._a_cc = self._m_dot / (param)

        gas = self._comb.get_thermo_prop(Pcc=self._pcc, MR=self._mr)
        
        self._a_rat = self.area_rat(gas = gas, P_a = P_e)
        
        self._a_e = self._a_cc * self._a_rat

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
    
    def get_supersonic_mach(self,
                            gas: IdealGas):
        """This function solves for the Supersonic Mach Number solution

        Args:
            gas (IdealGas): Combustion Chamber Ideal Gas

        Returns:
            float: Supersonic Mach Number
        """
        gamma = gas.get_gamma()
        
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
    
    def get_P_e(self,
                gas: IdealGas) -> float:
        """This function solves for the exit pressure of the chamber

        Args:
            gas (IdealGas): Combustion Gas Object Inside the Chamber

        Returns:
            float: Exit Pressure for the Combustion Chamber
        """
        
        # First we get the supersonic Mach Number at the exit of the chamber
        M_e = self.get_supersonic_mach(gas = gas)
        
        # We get the chamber conditions
        P_c = gas.get_pressure()
        gamma = gas.get_gamma()
        
        # We can now solve for the exit pressure of the chamber
        P_e = P_c * (1 + (gamma - 1)/2 * M_e**2) ** (- gamma / (gamma - 1)) 
        
        return P_e
    
    def get_cf(self, 
               gas: IdealGas,
               P_a: float = 1e5
               ) -> float:
        """This function gets the Thrust Coefficient for the Chamber

        Args:
            gas (IdealGas): Combustion Chamber Ideal Gas Object
            P_a (float, optional): Ambient Pressure (Pa). Defaults to 1 Bar.

        Returns:
            float: Thrust Coefficient
        """
        # We get the specific heat ratio and chamber pressure
        P_c = gas.get_pressure()
        gamma = gas.get_gamma()
        
        # We get the exit pressure 
        P_e = self.get_P_e(gas = gas)

        c_f = (
            (((2 * gamma**2)/(gamma - 1)) * 
              (2 / (gamma + 1))**((gamma + 1)/(gamma - 1)) * 
              (1 - (P_e / P_c)**((gamma - 1)/ gamma)) ) ** (1/2) +
              (self._a_rat) * ((P_e - P_a)/P_c)  
        )

        return c_f
    
    def get_thrust(self,
                   gas: IdealGas,
                   P_a: float = 1e5):
        """This function solves for the thrust of the chamber

        Args:
            gas (IdealGas): Combustion Chamber Gas Condition
            P_a (float, optional): Ambient Pressure (Pa). Defaults to 1 Bar.
        """
        
        # We get the chamber pressure
        P_c = gas.get_pressure()
        
        c_f = self.get_cf(gas=gas)

        F = c_f * P_c * self._a_cc
        
        return F

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
        gas = self._comb.get_thermo_prop(Pcc=pcc, MR=mr)

        to = gas.get_temperature() * eta_c**2
        
        # We get the thrust of the system
        F = self.get_thrust(gas = gas)

        # Getting combustion gas properties
        gamma = gas.get_gamma()
        R = gas.get_R()
        cp = gas.get_cp()

        # We create our Dict and finally return it

        dic = {
            "P_cc": pcc,
            "MR": mr,
            "T_o": to,
            "Cp": cp,
            "gamma": gamma,
            "R": R,
            "F": F,
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

        p_min = min(p_fu, p_ox)
        # We check if either is cooked, if so we pull a big penalty
        if p_min < pcc:
            return 1000

        mr = alpha * ((p_ox - pcc) / (p_fu - pcc)) ** (1 / 2)

        # We can now solve the combustion parameter at this condition
        param = self.combustion_param(pcc=pcc, mr=mr, eta_c=eta_c)

        # We can solve for the rhs of the equation
        rhs = k_f * (alpha * np.sqrt(p_ox - pcc) + np.sqrt(p_fu - pcc))

        error = rhs - param

        return error

    def set_stochastic_parameters(
        self,
        Cd_o: float | None = None,
        Cd_f: float | None = None,
        eta_c: float | None = None,
    ) -> None:
        """This function sets all the key stochasting parameters that are relevant for this component.

        Args:
            Cd_o (float | None, optional): Oxidiser Injector Discharge Coefficient. Defaults to None.
            Cd_f (float | None, optional): Fuel Injector Discharge Coefficient. Defaults to None.
            eta_c (float | None, optional): Combustion Efficiency of the Engine. Defaults to None.
        """

        if Cd_o is not None:
            self._cdo = Cd_o

        if Cd_f is not None:
            self._cdf = Cd_f

        if eta_c is not None:
            self._eta_c = eta_c

        return

    def solve_perturb_ss(
        self,
        ox_in: IncompressibleFluid,
        fu_in: IncompressibleFluid,
        obj_flag: bool = False,
    ) -> dict:
        """This function solves for the updated combustion chamber conditions based on changes to the oxidiser and fuel inlet conditions for a steady state

        Args:
            ox_in (IncompressibleFluid): Oxidiser Incompressible Fluid
            fu_in (IncompressibleFluid): Fuel Incompressible Fluid

        Optional:
            obj_flag (bool): IdealGas Object Output Flag. Defaults to None

        Returns:
            dict: Dictionary of Key Parameters for the Combustion [Pcc, MR, To, m_dot_t, ox_stiff, fu_stiff, gamma, cp]
        """

        # We evaluate for our K_f term
        self._kf = self._cdf * self._a_fu * np.sqrt(2 * self._rho_fu) / self._a_cc

        self._ko = self._cdo * self._a_ox * np.sqrt(2 * self._rho_ox) / self._a_cc

        self._alpha = self._ko / self._kf

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

        if obj_flag:
            # We form our ideal gas object
            comb_gas = IdealGas(
                p=dic["P_cc"],
                t=dic["T_o"],
                cp=dic["Cp"],
                gamma=dic["gamma"],
                R=dic["R"],
            )

            dic["gas_obj"] = comb_gas

        # We can compact alot of the parameters we used

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
        c_star = self._comb.get_thermo_prop(Pcc=Pcc, MR=MR).get_c_star() * eta_c

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
        gas = self._comb.get_thermo_prop(Pcc=Pcc, MR=MR)

        R = gas.get_R()
        T = gas.get_temperature() * eta_c**2

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
            "R": 287,
            "gamma": 1.4,
            "ox_stiffness": 0,
            "fu_stiffness": 0,
            "m_dot_t": 0,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        comb_gas = IdealGas(
            p=dic["P_cc"],
            t=dic["T_o"],
            cp=dic["Cp"],
            gamma=dic["gamma"],
            R=dic["R"],
        )

        dic["gas_obj"] = comb_gas

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
        gas = self._comb.get_thermo_prop(Pcc=self._pcc_transient, MR=MR_current)

        dic = {
            "dp_dt": dp_dt,
            "P_cc": self._pcc_transient,
            "MR": MR_current,
            "T_o": gas.get_temperature() * eta_c**2,
            "Cp": gas.get_cp(),
            "R": gas.get_R(),
            "gamma": gas.get_gamma(),
            "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "m_dot_t": m_dot_fu + m_dot_ox,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        comb_gas = IdealGas(
            p=dic["P_cc"],
            t=dic["T_o"],
            cp=dic["Cp"],
            gamma=dic["gamma"],
            R=dic["R"],
        )

        dic["gas_obj"] = comb_gas

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
        gas = self._comb.get_thermo_prop(Pcc=self._pcc_transient, MR=self._MR_transient)

        dic = {
            "dp_dt": dp_dt,
            "P_cc": self._pcc_transient,
            "MR": self._MR_transient,
            "T_o": gas.get_temperature() * eta_c**2,
            "Cp": gas.get_cp(),
            "R": gas.get_R(),
            "gamma": gas.get_gamma(),
            "ox_stiffness": (ox_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "fu_stiffness": (fu_in.get_pressure() - self._pcc_transient)
            / self._pcc_transient,
            "m_dot_t": m_dot_fu + m_dot_ox,
            "m_dot_o": m_dot_ox,
            "m_dot_f": m_dot_fu,
        }

        comb_gas = IdealGas(
            p=dic["P_cc"],
            t=dic["T_o"],
            cp=dic["Cp"],
            gamma=dic["gamma"],
            R=dic["R"],
        )

        dic["gas_obj"] = comb_gas

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
        super().__init__(
            Ox,
            Fu,
            Pcc,
            MR,
        )


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

    def get_exit_condition(
        self, upstr: IncompressibleFluid, m_dot: float
    ) -> IncompressibleFluid:
        """This function solves for the exit condition of the valve, based on an inlet and a mass flow rate.

        Args:
            upstr (IncompressibleFluid): Upstream Fluid Object of Valve
            m_dot (float): Mass Flow Rate Through the valve (kg/s)

        Returns:
            IncompressibleFluid: Exit Fluid Object of the Valve
        """

        # For this, we need to re-arrange the incompressible fluid flow equation to figure out what our dp is based on the mass flow rate of the valve.
        rho = upstr.get_density()
        p1 = upstr.get_pressure()

        a = self._cda * self._pos

        dp = (m_dot / a) ** 2 * (1 / (2 * rho))

        # We can now evaluate for our exit pressure and create our return object accordingly
        p2 = p1 - dp

        exit = IncompressibleFluid(rho=rho, P=p2)

        return exit

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

    def __init__(self, a_rat: float, D_m: float, eta_nom: float, u_co_nom: float):
        """Constructor for the Transient Turbine Object

        Args:
            a_rat (float): Area Ratio of Nozzle
            d_m (float): Mean Diameter of Turbine (m)
            eta_nom (float): Nominal Turbine Efficiency (%)
            u_co_nom (float): Nominal Blade Speed Ratio for Turbine
        """
        self._a_rat = a_rat
        self._rm = D_m / 2
        self._eta_nom = eta_nom
        self._u_co_nom = u_co_nom

        return

    def set_performance(
        self,
        eta_nom: float,
        u_co_nom: float,
    ) -> None:
        """Function that sets the turbines performance

        Args:
            eta_nom (float): Nominal Effiency
            u_co_nom (float): Nominal Blade Speed Ratio for Turbine
        """

        self._eta_nom = eta_nom
        self._u_co_nom = u_co_nom

        return

    def get_isentropic_velocity(
        self,
        combustion_gas: IdealGas,
        p_exit: float,
    ) -> float:
        """This function solves for the insentropic velocity of the gas, based on the expansion ratio of the gas

        Args:
            combustion_gas (IdealGas): Combustion Gas Object Produced by the Gas Generator
            p_exit (float): Exit Static Pressure for the Turbine

        Returns:
            float: Isentropic Expansion Velocity of the gas (m/s)
        """
        # We simply call the gas function to resolve for the isentropic expansion velocity

        v_is = combustion_gas.get_cis(p1=p_exit)

        return v_is

    def get_mean_speed(self, N: float) -> float:
        """Need to get the blade speed of the turbine

        Args:
            N (float): Rotational Rate of the Speed (rad/s)

        Returns:
            float: Blade Speed of the Turbine (m/s)
        """
        U = self._rm * N

        return U

    def get_efficiency(
        self, combustion_gas: IdealGas, N: float, p_exit: float
    ) -> float:
        """This function gets the efficiency of the turbine stage at off-design performance

        Assumptions:
            - Based on the Goldman Paper, a linear relationship has been assumed as correlelated to the blade speed ratio.
            - The efficiency is augmented driven based on the maximum expansion ratio of the nozzles at a given shaft speed,
              where if the expansion ratio is higher than as designed, the efficiency will be naturally depreciated.

        Args:
            N (float): Shaft Speed of the Turbopump (rad/s)
            c_o (float): Isentropic Expansion Velocity (m/s)
            p_exit (float): Exit Static Pressure of the GG Stage (Pa)

        Returns:
            float: Total to static efficiency of the turbine (%)
        """
        # We need to intially solve for the meanline blade speed
        u_m = self._rm * N

        # We get the isentropic velocity of the gas
        c_o = self.get_isentropic_velocity(combustion_gas=combustion_gas, p_exit=p_exit)

        u_co_a = u_m / c_o

        if u_co_a > self._u_co_nom:

            eta_bep = self._eta_nom
        else:

            eta_bep = self._eta_nom * u_co_a / self._u_co_nom

        # Finally we need to evaluate what the maxium expansion velocity of the gas is based on the Mach Number
        M = self.get_supersonic_mach(gamma=combustion_gas.get_gamma())

        # We solve our expansion pressur ratio
        P_min = self.get_exit_pressure(
            P_o=combustion_gas.get_pressure(), M=M, gamma=combustion_gas.get_gamma()
        )

        dh_max = combustion_gas.get_enthalpy_drop(p1=P_min)

        dh_theo = combustion_gas.get_enthalpy_drop(p1=p_exit)

        if dh_max == 0:
            return 0
        # We check if the nozzle is underexpanded
        if P_min > p_exit:
            # We then augment the efficiency accordingly to match expectations by clamping power output

            eta = eta_bep * (dh_max / dh_theo) ** 2

        else:
            # We invert this plot and adjust the efficiency based on the difference in enthalpy expansions

            eta = eta_bep * (dh_theo / dh_max) ** 2

        return eta

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

    def get_torque(self, combustion_gas: IdealGas, P_exit: float, N: float) -> float:
        """This function solves for the Torque produced by the Turbine Stage

        Args:
            combustion_gas (IdealGas): Combustion Gas Products Produced by the Gas Generator
            P_exit (float): Exit Pressure of the Turbine (Pa)
            N (float): Shaft Speed of the Turbine (rad/s)

        Returns:
            float: Torque produced by the Turbine
        """

        # We can now solve for the expected efficiency of the system
        eta = self.get_efficiency(combustion_gas=combustion_gas, N=N, p_exit=P_exit)

        # We can now solve for the power being produced by the turbine
        Pw = eta * combustion_gas.get_enthalpy_drop(p1=P_exit)

        # We can finally solve for the torque produced by the turbine by dividing it by the current shaft speed.
        T = Pw / N

        return T


class Pump:
    """Object representing the transient functionality of the pump"""

    def __init__(
        self,
        D_1: float,
        D_2: float,
        D_3: float,
    ):
        """_summary_

        Args:
            D_1 (float): Inner Diameter of Pump Eye (m)
            D_2 (float): Outer Diameter of the Pump Tip (m)
            D_3 (float): Diffuser Oulet Diameter (m)
        """

        self._D_1 = D_1
        self._D_2 = D_2
        self._D_3 = D_3

        self._g = 9.18

        return

    def set_performance(
        self,
        C_c: float,
        psi: float,
        eta_bep: float,
        N_nom: float,
    ) -> None:
        """This Function Sets the Pump Performance

        Args:
            C_c (float): Diffuser Vena-Contra Factor (%)
            psi (float): Pressure Coefficient
            eta_bep (float): Best Efficiency Point (%)
            N_nom (float): Nominal Shaft Speed (rad/s)
        """
        self._C_c = C_c
        self._psi = psi
        self._eta_bep = eta_bep
        self._N_nom = N_nom

        return

    def shut_off_head(self, N: float) -> float:
        """This function estimates the the theoretical shut off head of a pump

        Args:
            N (float): Rotational Rate for the Pump (rad/s)
        """

        u_1 = N * (self._D_1 / 2) ** 2
        u_2 = N * (self._D_2 / 2) ** 2

        H_o = (1 / (2 * self._g)) * ((1 + self._psi) * u_2**2 - u_1**2)

        return H_o

    def get_q_max(self, N: float) -> float:
        """This function gets the maximum flow operating point for the turbine at the selected shaft speed

        Args:
            N (float): Shaft Speed (Rad/s)

        Returns:
            float: Maximum Flow Operating Point (m^3/s)
        """

        u_1 = N * (self._D_1 / 2) ** 2
        u_2 = N * (self._D_2 / 2) ** 2

        v_3 = ((1 + self._psi) * u_2**2 - u_1**2) ** (1 / 2)

        a_3 = np.pi * (self._D_3 / 2) ** 2

        return a_3 * v_3 * self._C_c

    def get_eta_bep(self, N: float) -> float:
        """Simpliefied Model for Identifying what the best operating efficiency of the pump is

        Args:
            N (float): Shaft Speed (Rad/s)

        Returns:
            float: Best Operating Efficiency of Pump (%)
        """

        return self._eta_bep * (N / self._N_nom)

    def get_eta(self, Q: float, N: float, fluid: IncompressibleFluid) -> float:
        """Simplified function that solves for the efficiency of the Pump

        Args:
            Q (float): Flow Rate of Fluid Through the Pump (m^3/s)

        Returns:
            float: Efficiency of the Turbine
        """
        # We need to get the fixed shaft power
        Q_max = self.get_q_max(N=N)
        H_o = self.get_head(Q=0, N=N)

        P_max = fluid.get_density() * self._g * H_o * Q_max

        # We can then figure out what our shaft power is
        P_shaft = P_max / self.get_eta_bep(N=N)

        # We can then evaluate for what the actual power is
        H_a = self.get_head(Q=Q, N=N)

        P_actual = fluid.get_density() * self._g * H_a * Q

        # We can thus solve for the efficiency
        eta = P_actual / P_shaft

        if N == 0:
            eta = 0

        return eta

    def get_head(self, Q: float, N: float) -> float:
        """This function solves for the head produced by the pump

        Args:
            Q (float): Volumetric Flow Rate Through the Pump (m^3 /s)
            N (float): Rotational Rate for the Pump (Rad/s)

        Returns:
            float: Head Produced by Pump (m)
        """

        # We firstly need to solve for the shut_off head of the pump
        H_o = self.shut_off_head(N=N)
        print(f"Shut off Head: {H_o} m")
        # We need to get the maximum flow operating point
        Q_max = self.get_q_max(N=N)

        if Q_max == 0:
            # Pump is not spinning at all, hence no head.
            H = 0
            return 0

        # We will model similar to the standard Pump Head Curves presented
        # in the Barske paper, which is that the pump head remains relatively
        # constant across all flow rates, but once a critical flow rate is reached it falls off.

        # This will be modelled simply as the shut off head of the pump being constant, but once the critical
        # flow rate is achieved, a parboal will be modelled

        if Q <= Q_max:
            H = H_o
        else:
            H = -100 * ((Q / Q_max) ** 2 - 1) ** 2 + H_o

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

        dp = H * self._g * rho

        outlet = IncompressibleFluid(rho=rho, P=p_inlet + dp)

        return outlet

    def get_torque(
        self,
        inlet: IncompressibleFluid,
        N: float,
    ) -> float:
        """This function solves for the torque produced from the pump

        Args:
            inlet (IncompressibleFluid): Inlet Fluid of the pump
            N (float): Rotational Rate of the Pump (rad/s)

        Returns:
            float: Torque Produced from the Pump (N m)
        """
        # We solve for the shaft power which is constant
        Q_max = self.get_q_max(N=N)
        H_o = self.get_head(Q=0, N=N)

        P_max = inlet.get_density() * self._g * H_o * Q_max

        print(f"Shaft Seed: {N*60/(2*np.pi)}")
        print(f"Actual Head: {H_o} m")

        # We can then evaluate for the torque of the system, by dividing our max power by best effiency point and shaft speed
        T = P_max / (self.get_eta_bep(N=N) * N)

        if N == 0:
            T = 0

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


class MechanicalLosses:
    """This object represents the Mechanical Losses of the System"""

    def __init__(self, m_bearing: list[float], n_bearing: list[int]) -> None:
        """Constructor for the Bearing Object

        Args:
            m_bearing (list[float]): Array of Selected Bearing Moments (Nm)
            n_bearing (list[float]): Array of Selected Bearings and Number of each type
        """

        self._m_bearing = m_bearing
        self._n_bearing = n_bearing

        if len(self._m_bearing) != len(self._n_bearing):
            raise ValueError(
                "Number of Bearings listed dow not equal the number of bearings"
            )

        return

    def get_torque(self) -> float:
        # This function gets the total induced torque by the bearing system
        T = 0

        i = 0
        for n in self._n_bearing:
            T += self._m_bearing[i] * n

            i += 1

        return T
