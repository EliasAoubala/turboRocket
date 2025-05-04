"""This file contains transient loader objects for the system"""

import yaml
from scipy.interpolate import interp1d
import numpy as np


class PropellantLoader:
    """This class is used for the loading of propellant information"""

    def __init__(
        self,
        name: str | None = None,
        Density: float | None = None,
        Compressibility: float | None = None,
        P_deliver: float | None = None,
    ):
        self._density = Density
        self._B = Compressibility
        self._name = name
        self._P_deliver = P_deliver

        return


class ValveLoader:
    """This class is used for the loading of valve information"""

    def __init__(
        self,
        Name: str | None = None,
        OD: float | None = None,
        Cd: float | None = None,
        Tau: float | None = None,
        s_pos_init: float | None = None,
    ):
        self._name = Name
        self._od = OD
        self._cd = Cd
        self._tau = Tau
        self._s_pos_init = s_pos_init

        self._cda = self._cd * np.pi * (self._od / 2) ** 2

        return


class EngineLoader:

    def __init__(
        self,
        P_cc: float | None = None,
        P_init: float | None = None,
        P_inj_ox: float | None = None,
        P_inj_ox_init: float | None = None,
        P_inj_fu: float | None = None,
        P_inj_fu_init: float | None = None,
        m_dot: float | None = None,
        MR: float | None = None,
        eta_c: float | None = None,
        L_star: float | None = None,
        Cd_o: float | None = None,
        Cd_f: float | None = None,
        V_ox_inj: float | None = None,
        V_fu_inj: float | None = None,
    ):

        self._P_cc = P_cc
        self._P_init = P_init
        self._P_inj_ox = P_inj_ox
        self._P_inj_ox_init = P_inj_ox_init
        self._P_inj_fu = P_inj_fu
        self._P_inj_fu_init = P_inj_fu_init
        self._m_dot = m_dot
        self._MR = MR
        self._eta_c = eta_c
        self._L_star = L_star
        self._Cd_o = Cd_o
        self._Cd_f = Cd_f
        self._V_ox_inj = V_ox_inj
        self._V_fu_inj = V_fu_inj

        return


class TurbineLoader:
    """This Class is used for loading in Turbine Related Information"""

    def __init__(
        self,
        delta_b: float | None = None,
        a_rat: float | None = None,
        D_m: float | None = None,
        eta: float | None = None,
    ):
        self._delta_b = delta_b
        self._a_rat = a_rat
        self._D_m = D_m
        self._eta = eta

        return


class PumpLoader:
    """This Class is used for the loading of Pump Related Information"""

    def __init__(
        self,
        D_nom: float | None = None,
        Q_nom: float | None = None,
        eta_nom: float | None = None,
    ):
        self._D_nom = D_nom
        self._Q_nom = Q_nom
        self._eta_nom = eta_nom

        return


class TransientLoader:
    """Object used for loading of yaml files for key parameters"""

    def __init__(self, config_file: str):

        with open(config_file) as f:
            self._file = yaml.safe_load(f)

        # Initialising and type hinting the arrays
        self._engines: dict[str, EngineLoader] = {}
        self._valves: dict[str, ValveLoader] = {}
        self._propellants: dict[str, PropellantLoader] = {}
        self._turbo_pumps: dict[
            str, dict[str, dict[str, PumpLoader | TurbineLoader] | float]
        ] = {}

        if "SIMULATION" in self._file.keys():
            self.load_simulation(**self._file["SIMULATION"])

        if "SIZING" in self._file.keys():

            for engine in self._file["SIZING"]:

                dic = self._file["SIZING"][engine]

                print(dic)

                self._engines[engine] = self.engine_sizing(**dic)

        if "VALVES" in self._file.keys():

            for valve in self._file["VALVES"]:

                dic = self._file["VALVES"][valve]

                self._valves[valve] = self.valve_definition(**dic)

        if "TURBOPUMPS" in self._file.keys():

            for turbopump in self._file["TURBOPUMPS"]:

                tb_dic = self._file["TURBOPUMPS"][turbopump]
                self._turbo_pumps[turbopump] = {}
                self._turbo_pumps[turbopump]["Pumps"] = {}
                self._turbo_pumps[turbopump]["Turbines"] = {}

                for component in tb_dic:

                    cp_dic = tb_dic[component]

                    if component == "SYSTEM":
                        # We get the turbopump high level parameters

                        self._turbo_pumps[turbopump]["N_nom"] = (
                            cp_dic["N_nom"] * (2 * np.pi) / 60
                        )
                        self._turbo_pumps[turbopump]["N_init"] = (
                            cp_dic["N_init"] * (2 * np.pi) / 60
                        )

                        self._turbo_pumps[turbopump]["I"] = cp_dic["I"]

                    elif cp_dic["type"] == "Pump":
                        _ = cp_dic.pop("type", None)

                        self._turbo_pumps[turbopump]["Pumps"][component] = (
                            self.pump_definition(**cp_dic)
                        )

                    elif cp_dic["type"] == "Turbine":
                        _ = cp_dic.pop("type", None)

                        self._turbo_pumps[turbopump]["Turbines"][component] = (
                            self.turbine_definition(**cp_dic)
                        )

                    else:

                        raise ValueError(f"Invalid Type Specified for {component}")

        if "SEQUENCE" in self._file.keys():

            self.load_sequence(self._file["SEQUENCE"])

        if "PROPELLANTS" in self._file.keys():

            for propellant in self._file["PROPELLANTS"]:

                dic = self._file["PROPELLANTS"][propellant]

                self._propellants[propellant] = self.propellant_definition(**dic)

        return

    def load_simulation(
        self,
        t_start: float | None = None,
        t_stop: float | None = None,
        dt_fix: float | None = None,
        dt_init: float | None = None,
        max_dp: float | None = None,
        alpha: float | None = None,
    ) -> None:

        self._t_start = t_start
        self._t_stop = t_stop
        self._dt_fix = dt_fix
        self._dt_init = dt_init
        self._max_dp = max_dp
        self._alpa = alpha

        return

    def engine_sizing(
        self,
        P_cc: float | None = None,
        P_init: float | None = None,
        P_inj_ox: float | None = None,
        P_inj_ox_init: float | None = None,
        P_inj_fu: float | None = None,
        P_inj_fu_init: float | None = None,
        m_dot: float | None = None,
        MR: float | None = None,
        eta_c: float | None = None,
        L_star: float | None = None,
        Cd_o: float | None = None,
        Cd_f: float | None = None,
        V_ox_inj: float | None = None,
        V_fu_inj: float | None = None,
    ) -> EngineLoader:
        engine_obj = EngineLoader(
            P_cc=P_cc,
            P_init=P_init,
            P_inj_ox=P_inj_ox,
            P_inj_ox_init=P_inj_ox_init,
            P_inj_fu=P_inj_fu,
            P_inj_fu_init=P_inj_fu_init,
            m_dot=m_dot,
            MR=MR,
            eta_c=eta_c,
            L_star=L_star,
            Cd_o=Cd_o,
            Cd_f=Cd_f,
            V_ox_inj=V_ox_inj,
            V_fu_inj=V_fu_inj,
        )

        return engine_obj

    def valve_definition(
        self,
        Name: str,
        OD: float | None = None,
        Cd: float | None = None,
        Tau: float | None = None,
        s_pos_init: float | None = None,
    ) -> ValveLoader:

        valve_obj = ValveLoader(Name=Name, OD=OD, Cd=Cd, Tau=Tau, s_pos_init=s_pos_init)

        return valve_obj

    def pump_definition(
        self,
        D_nom: float | None = None,
        Q_nom: float | None = None,
        eta_nom: float | None = None,
    ) -> PumpLoader:

        pump_obj = PumpLoader(D_nom=D_nom, Q_nom=Q_nom, eta_nom=eta_nom)

        return pump_obj

    def turbine_definition(
        self,
        delta_b: float | None = None,
        a_rat: float | None = None,
        D_m: float | None = None,
        eta: float | None = None,
    ) -> TurbineLoader:

        turbine_obj = TurbineLoader(delta_b=delta_b, a_rat=a_rat, D_m=D_m, eta=eta)

        return turbine_obj

    def propellant_definition(
        self,
        Name: str | None = None,
        Density: float | None = None,
        Compressibility: float | None = None,
        P_deliver: float | None = None,
    ) -> PropellantLoader:

        propellant_obj = PropellantLoader(
            name=Name,
            Density=Density,
            Compressibility=Compressibility,
            P_deliver=P_deliver,
        )

        return propellant_obj

    def load_sequence(self, seq_dict) -> None:
        """This function loads in the valve sequences and generates time regressions

        Args:
            dict (_type_): Dictionary of valve on time sequqnces
        """

        self._sequences = {}

        for valve in seq_dict:

            # We create our x_array

            x_array = [item for sublist in seq_dict[valve] for item in sublist]

            # We check if the initial actuation time is at T=0, if not, we add a period before it opens

            if x_array[0] > 0:
                x_array.insert(0, -10)
                x_array.insert(1, 0)

            if len(x_array) % 2 != 0:
                raise Exception(f"Sequence for {valve} has incomplete intervals!")

            y_array = [(i + 1) % 2 for i in range(len(x_array))]

            self._sequences[valve] = interp1d(x_array, y_array, kind="zero")

        return
