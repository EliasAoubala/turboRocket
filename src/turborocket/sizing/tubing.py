"""
This function contains the loss relationships for evaluating the losses induced by tubing.
"""

from enum import Enum
from turborocket.fluids.fluids import IncompressibleFluid

import numpy as np

class LossType(Enum):
    """Object Describing the Types of Losses to be Considered
    """

    MAJOR = "MAJOR"
    MINOR = "MINOR"

class MinorLoss(Enum):
    """Minor Loss Object
    """

    ELBOW_90 = 1.3
    ELBOW_45 = 0.35

    BEND_90 = 0.45
    BEND_180 = 1.5

    TEE_STRAIGHT = 0.4
    TEE_ELBOW = 1.0

    UNION = 0.04

    EXPANSION = "Expansion Loss"

    CONTRACTION = "Contraction Loss"


def contraction_loss(self,
                     A_1: float,
                     A_2: float
                     ) -> float:
    """Utility Function for a sudden Contraction Loss Evaluation

    Args:
        A_1 (float): Inlet Area
        A_2 (float): Exit Area

    Returns:
        float: Head loss Coefficient (K)
    """

    if A_1 < A_2:
        raise ValueError("Cannot Solve for Contraction Loss when Inlet Area is bigger than Exit")

    K = 0.5 * (1 - A_1/A_2)

    return K


def expansion_loss(self,
                   A_1: float,
                   A_2: float
                   ) -> float:
    """Utility Function for Expansion Loss Evaluation

    Args:
        A_1 (float): Inlet Area
        A_2 (float): Exit Area

    Returns:
        float: Expansion Loss of the Fluid
    """

    if A_2 < A_1:
        raise ValueError("Cannot compute the expansion loss of the tubing when Inlet Area is larger than exit area!")
    
    K = (1 - A_1/A_2)**2

    return K

class FeedSystem():
    """This object represents the Feed System of the TurboPump
    """

    def __init__(self,
                 fluid: IncompressibleFluid
                 ):
        """Constructor for the FeedSystem Object

        Args:
            fluid (IncompressibleFluid): Incompressible Fluid Object
        """

        self._fluid = fluid

        return
    
    def setup_description(self,
                          description: dict[float, dict[str, LossType | float]]
                          ) -> None:
        """This function sets-up the description of the FeedSystem A

        - This function will iterate through a user provided dictionary and solve for the required losses
        
        Args:
            description (dict[float, dict[str, LossType | float]]): Feed System Description
        
        """
        



def Re(fluid: IncompressibleFluid, 
       L: float,
       m_dot: float,
       A: float | None = None,
       ) -> float:
    """This function gets the Reynolds Number of the Fluid 

    Args:
        fluid (IncompressibleFluid): Incompresible Fluid Object
        L (float): Characteristic Diameter of the Tubing [m]
        m_dot (float): Mass Flow of the Tubing [kg/s]
        A (float | None, optional): Flow Area of the Tubing [m^2]. Defaults to none
    """

    rho = fluid.get_density()
    mu = fluid.get_viscosity()

    if A is not None:
        # If user specifies the area, we use it.
        v = m_dot/(rho * A)

    else:

        # Assume a circular profile
        A = np.pi * (L/2)**2

        v = m_dot/(rho * A)

    Re = (rho * v * L) /  mu

    return Re

def fanning_factor(Re: float,)
