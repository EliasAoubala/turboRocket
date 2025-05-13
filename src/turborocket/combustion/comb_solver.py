"""
This file contains the objects that are used for solving for combustion conditions of the gas:

    - A custom combustion solver has been written using cantera, to allow for more accurate combustion modelling.
    - This solver should have comparable results to RocketCEA, however it will discount condensed fractions (solid)
      In the evaluation of the specific heat capacity and specific heat ratios of the gasses, which is critical to know.
"""

import numpy as np
import cantera as ct
from pint import UnitRegistry


class Combustion:
    """Object representing the combustion solving object"""

    def __init__(self, fuel: str, oxidiser: str) -> None:
        """Constructor for the Combustion Object

        Args:
            fuel (str): Molecular Name of the Fuel
            oxidiser (str): Molecular Name of the Oxidiser
        """

        self._fuel = fuel
        self._oxidiser = oxidiser

        self._ureg = UnitRegistry()
        self._Q_ = ureg.Quantity

        self._full_specieis = {
            S.name: S for S in ct.Species.list_from_file("nasa_gas.yaml")
        }

        self._gas = ct.Solution(thermo="ideal-gas", species=self._full_specieis)
