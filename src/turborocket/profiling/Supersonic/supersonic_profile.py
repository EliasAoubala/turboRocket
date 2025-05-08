# This file encapsulates the main function for computing the supersonic profile

from turborocket.profiling.Supersonic.circular import (
    prandtl_meyer,
    M_star,
    arc_angles_upper,
    arc_angles_lower,
    inv_M_star,
)

from turborocket.profiling.Supersonic.transition import moc, moc_2

from turborocket.profiling.Supersonic.constraints import inv_mass_flow, r_star
from turborocket.profiling.Supersonic.constraints import (
    k_star_max,
    Q,
    C,
    shock_pressure_rat,
    M_i_star_max,
)
from turborocket.profiling.Supersonic.constraints import M_star_u_max, M_star_l_min

from turborocket.fluids.fluids import IdealGas

from turborocket.profiling.Supersonic.fixed_edge import get_m_e

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


class SupersonicProfile:

    def __init__(
        self,
        beta_i: float,
        beta_o: float,
        M_i: float,
        M_o: float,
        M_u: float,
        M_l: float,
        m_dot: float,
        h: float,
        fluid: IdealGas,
    ) -> None:

        # Constant for converting angles to radians
        ANGLE_CONVERSION = np.pi / 180

        self._beta_i = beta_i * ANGLE_CONVERSION
        self._beta_o = beta_o * ANGLE_CONVERSION
        self._M_i = M_i
        self._M_o = M_o
        self._M_u = M_u
        self._M_l = M_l
        self._m_dot = m_dot
        self._h = h
        self._fluid = fluid

    def prantl_meyer(self):
        # The first process required for the design of the supersonic blade profile is the definition of the upper and lower circular arcs.

        # We must firstly compute the Prantl Meyer Angle of the Upper and Lower surfaces based on the predefined mach-numbers.

        # Calculating our critical velocity ratios

        GAMMA = self._fluid.get_gamma()

        self._M_u_star = M_star(GAMMA, self._M_u)

        self._M_l_star = M_star(GAMMA, self._M_l)

        self._M_i_star = M_star(GAMMA, self._M_i)

        self._M_o_star = M_star(GAMMA, self._M_o)

        # Calculating our Prantl Meyer values

        self._v_u = prandtl_meyer(GAMMA, self._M_u_star)

        self._v_l = prandtl_meyer(GAMMA, self._M_l_star)

        self._v_i = prandtl_meyer(GAMMA, self._M_i_star)

        self._v_o = prandtl_meyer(GAMMA, self._M_o_star)

    def circular_section(self):

        # Calculating our circular arc angles

        [self._alpha_u_i, self._alpha_u_o] = arc_angles_upper(
            beta_o=self._beta_o,
            beta_i=self._beta_i,
            v_i=self._v_i,
            v_o=self._v_o,
            v_u=self._v_u,
        )

        [self._alpha_l_i, self._alpha_l_o] = arc_angles_lower(
            beta_o=self._beta_o,
            beta_i=self._beta_i,
            v_i=self._v_i,
            v_o=self._v_o,
            v_l=self._v_l,
        )

        # We can calculate the non-dimentional radii of the upper and lower
        # circles of the vortex sections of the plot

        self._R_l_star = 1 / self._M_l_star

        self._R_u_star = 1 / self._M_u_star

        return

    def r_star(self):

        # We specify the number of steps for our integration

        INTEGRAL_NUMBER = 100  # TODO: Fix this magic number
        GAMMA = self._fluid.get_gamma()

        self._wf_parameter = inv_mass_flow(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
            mass_flow=self._m_dot,
        )

        # We meed to now calculate the total density assuming ideal gas
        density_i_total = self._fluid.get_density()

        # Calculating thet local speed of sound of the ideal gas.
        a_i_total = self._fluid.speed_of_sound()

        # Based on the weight-flow parameter, we can compute the sonic radius
        self._r_star_a = r_star(
            wf_parameter=self._wf_parameter,
            h=self._h,
            a_total_inlet=a_i_total,
            rho_total_inlet=density_i_total,
        )

        return

    def inlet_lower_transition(self):

        # We specify an arbritrary number of points for the MoC
        K_MAX = 10

        # We specify an arbritary inlet absolute angle of zero
        ALPHA_I = 0

        GAMMA = self._fluid.get_gamma()

        [self._xlkt_il, self._ylkt_il] = moc(
            k_max=K_MAX,
            v_i=self._v_i,
            v_l=self._v_l,
            gamma=GAMMA,
            alpha_l_i=self._alpha_l_i,
        )

        return

    def outlet_lower_transition(self):

        # We specify an arbritrary number of points for the MoC
        K_MAX = 10

        # We specify an arbritary inlet absolute angle of zero

        GAMMA = self._fluid.get_gamma()

        [self._xlkt_ol, self._ylkt_ol] = moc_2(
            k_max=K_MAX,
            v_i=self._v_o,
            v_l=self._v_l,
            gamma=GAMMA,
            alpha_l_i=self._alpha_l_o,
        )  # self._alpha_l_o)

        return

    def inlet_upper_transition(self):

        # We specify an arbritrary number of points for the MoC
        K_MAX = 100

        # We specify an arbritary inlet absolute angle of zero

        GAMMA = self._fluid.get_gamma()

        [self._xlkt_iu, self._ylkt_iu] = moc_2(
            k_max=K_MAX,
            v_i=self._v_i,
            v_l=self._v_u,
            gamma=GAMMA,
            alpha_l_i=self._alpha_u_i,
        )

        return

    def outlet_upper_transition(self):

        # We specify an arbritrary number of points for the MoC
        K_MAX = 10

        # We specify an arbritary inlet absolute angle of zero

        GAMMA = self._fluid.get_gamma()

        [self._xlkt_ou, self._ylkt_ou] = moc(
            k_max=K_MAX,
            v_i=self._v_o,
            v_l=self._v_u,
            gamma=GAMMA,
            alpha_l_i=self._alpha_u_o,
        )  # self._alpha_u_o)

        return

    def generate_transitions(self):
        """This function generates the transition points for the tubrine pro"""

    def straight_line_segments(self):
        # The final section for the creation of these aerofoils is to draw in the straight line segments
        # These are parallel to the flow inlet and outlet and start at the last points of the transition.
        # The straight line continues until it reaches the equivalent x-coordinate of the lower transition.

        ############### Inlet Straight Line Segment ###############
        self._x_i_start = self._xlkt_iu[-1]
        self._y_i_start = self._ylkt_iu[-1]

        self._x_i_end = self._xlkt_il[-1]

        # We get our required offset
        delta_y_i = (self._x_i_end - self._x_i_start) * np.tan(self._beta_i)

        # Hence we can get our last point of the segment
        self._y_i_end = self._y_i_start + delta_y_i

        # We can define our inlet co-ordinate arrays
        self._x_i_line = np.array([self._x_i_start, self._x_i_end])
        self._y_i_line = np.array([self._y_i_start, self._y_i_end])

        ############### Outlet Straight Line Segment ###############
        self._x_o_start = self._xlkt_ou[-1]
        self._y_o_start = self._ylkt_ou[-1]

        self._x_o_end = self._xlkt_ol[-1]

        # We get our required offset
        delta_y_o = (self._x_o_end - self._x_o_start) * np.tan(self._beta_o)

        # Hence we can get our last point of the segment
        self._y_o_end = self._y_o_start + delta_y_o

        # We can define our inlet co-ordinate arrays
        self._x_o_line = np.array([self._x_o_start, self._x_o_end])
        self._y_o_line = np.array([self._y_o_start, self._y_o_end])

    def get_g_star(self) -> None:
        """This function gets the blades spacing for the generated profile"""

        self._g_star = self._ylkt_ol[-1] - self._y_o_end

        return

    def get_c_star(self) -> None:
        """This function gets the blades chord length for the generated profile"""

        # We take the first point and last point for the transition region
        self._c_star = self._xlkt_ol[-1] - self._xlkt_il[-1]

        return

    def get_solidity(self) -> None:
        """This function gets the blade solidity for the generated profile"""
        self.get_g_star()
        self.get_c_star()

        self._sigma = self._c_star / self._g_star

        return

    def generate_blade(self, NUMBER_OF_POINTS):

        # This function creates the blade profile
        # Now that we know the final values of the y co-ordinates, we can generate our blade profiles

        # Discretising our circular sections
        alpha_l_array = np.linspace(self._alpha_l_i, self._alpha_l_o, NUMBER_OF_POINTS)

        alpha_u_array = np.linspace(self._alpha_u_i, self._alpha_u_o, NUMBER_OF_POINTS)

        # We can thus generate our x and y co-ordinates accordingly.

        self._x_l_array = -self._R_l_star * np.sin(alpha_l_array)
        self._y_l_array = self._R_l_star * np.cos(alpha_l_array)

        self._x_u_array = -self._R_u_star * np.sin(alpha_u_array)
        self._y_u_array = self._R_u_star * np.cos(alpha_u_array)

        # Now Shifting all our array points for the upper blade profiling accordingly

        self._y_o_end_sf = self._y_o_end + self._g_star
        self._y_i_end_sf = self._y_i_end + self._g_star

        self._y_o_start_sf = self._y_o_start + self._g_star
        self._y_i_start_sf = self._y_i_start + self._g_star

        # Shifting transition points
        self._ylkt_iu_sf = self._ylkt_iu + self._g_star
        self._ylkt_ou_sf = self._ylkt_ou + self._g_star

        # Shifting Circular Points
        self._y_u_array_sf = self._y_u_array + self._g_star

    def plot_circles(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We need to now create our x and y co-ordinates for the upper and lower arcs

        # Firstly we need to discretise our arrays

        alpha_l_array = np.linspace(self._alpha_l_i, self._alpha_l_o, NUMBER_OF_POINTS)

        alpha_u_array = np.linspace(self._alpha_u_i, self._alpha_u_o, NUMBER_OF_POINTS)

        # We can thus generate our x and y co-ordinates accordingly.

        x_l_array = -self._R_l_star * np.sin(alpha_l_array)
        y_l_array = self._R_l_star * np.cos(alpha_l_array)

        x_u_array = -self._R_u_star * np.sin(alpha_u_array)
        y_u_array = self._R_u_star * np.cos(alpha_u_array)

        # We then plot our results

        ax.plot(x_l_array, y_l_array)
        ax.plot(x_u_array, y_u_array)
        ax.set_aspect("equal")
        plt.show()

    def plot_transition(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu, label="Inlet Upper")
        # ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        # ax.plot(self._xlkt_ou, self._ylkt_ou, label="Outlet Upper")
        ax.legend()
        ax.set_aspect("equal")
        plt.show()

    def plot_all(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array)
        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu, label="Inlet Upper")
        ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        ax.plot(self._xlkt_ou, self._ylkt_ou, label="Outlet Upper")
        ax.plot(self._x_i_line, self._y_i_line)
        ax.plot(self._x_o_line, self._y_o_line)

        print(f"Y - Outlet: {self._y_o_line}")
        print(f"X - Outlet: {self._x_o_line}")

        print(f"Y - Inlet: {self._y_i_line}")
        print(f"X - Inlet: {self._x_i_line}")

        ax.set_aspect("equal")
        ax.legend()
        plt.show()

    def plot_all_shift(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array_sf)
        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu_sf, label="Inlet Upper")
        ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        ax.plot(self._xlkt_ou, self._ylkt_ou_sf, label="Outlet Upper")
        ax.plot(self._x_i_line, self._y_i_line + self._g_star)
        ax.plot(self._x_o_line, self._y_o_line + self._g_star)
        ax.set_aspect("equal")
        ax.legend()
        plt.show()

    def plot_all_shift_to_scale(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array * self._r_star_a, self._y_l_array * self._r_star_a)
        ax.plot(self._x_u_array * self._r_star_a, self._y_u_array_sf * self._r_star_a)
        ax.plot(
            self._xlkt_il * self._r_star_a,
            self._ylkt_il * self._r_star_a,
            label="Inlet Lower",
        )
        ax.plot(
            self._xlkt_iu * self._r_star_a,
            self._ylkt_iu_sf * self._r_star_a,
            label="Inlet Upper",
        )
        ax.plot(
            self._xlkt_ol * self._r_star_a,
            self._ylkt_ol * self._r_star_a,
            label="Outlet Lower",
        )
        ax.plot(
            self._xlkt_ou * self._r_star_a,
            self._ylkt_ou_sf * self._r_star_a,
            label="Outlet Upper",
        )
        ax.plot(
            self._x_i_line * self._r_star_a,
            self._y_i_line * self._r_star_a,
        )
        ax.plot(
            self._x_o_line * self._r_star_a,
            self._y_o_line * self._r_star_a,
        )
        ax.set_aspect("equal")
        ax.legend()
        plt.show()

    def generate_xy(self, NUMBER_OF_POINTS):

        # This function creates an .txt file containing x,y co-ordinates for the blade profiles automatically generated.

        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # x_array = np.append(x_array, [self._x_i_start * self._r_star_a][::-1]) # , self._x_i_end * self._r_star_a
        # y_array = np.append(y_array, [self._y_i_start_sf * self._r_star_a][::-1]) # , self._y_i_end_sf * self._r_star_a

        x_array = np.append(x_array, (self._xlkt_il * self._r_star_a)[-1])
        y_array = np.append(y_array, (self._ylkt_il * self._r_star_a)[-1])

        x_array = np.append(x_array, (self._xlkt_iu * self._r_star_a)[:1:-1])
        y_array = np.append(y_array, (self._ylkt_iu_sf * self._r_star_a)[:1:-1])

        x_array = np.append(x_array, self._x_u_array * self._r_star_a)
        y_array = np.append(y_array, self._y_u_array_sf * self._r_star_a)

        x_array = np.append(x_array, (self._xlkt_ou * self._r_star_a)[1:-1])
        y_array = np.append(y_array, (self._ylkt_ou_sf * self._r_star_a)[1:-1])

        # x_array = np.append(x_array, [self._x_o_start * self._r_star_a]) #, self._x_o_end * self._r_star_a])
        # y_array = np.append(y_array, [self._y_o_start_sf * self._r_star_a]) #, self._y_o_end_sf * self._r_star_a])

        x_array = np.append(x_array, (self._xlkt_ol * self._r_star_a)[:1:-1])
        y_array = np.append(y_array, (self._ylkt_ol * self._r_star_a)[:1:-1])

        x_array = np.append(x_array, (self._x_l_array * self._r_star_a)[::-1])
        y_array = np.append(y_array, (self._y_l_array * self._r_star_a)[::-1])

        x_array = np.append(x_array, (self._xlkt_il * self._r_star_a)[1:-2])
        y_array = np.append(y_array, (self._ylkt_il * self._r_star_a)[1:-2])

        # x_array = np.append(x_array, (self._xlkt_il * self._r_star_a)[-1])
        # y_array = np.append(y_array, (self._ylkt_il * self._r_star_a)[-1])

        z_array = np.zeros(x_array.size)

        df = pd.DataFrame(data={"x": x_array * 1e3, "y": y_array * 1e3, "z": z_array})

        df.to_csv("blade.txt", header=False, index=False, sep="\t")

    def M_i_max(self):
        """
        This function solves for the critical inlet mach number for the profile to ensure the geometry can be started succesfully.

        In supersonic turbines, it is critical that the geometry can be started up under low flow conditions.

        This particularly important at startup conditions as the relative inlet velocities are at their highest levels (due to blade speeds being low).

        We can solve for the maximum acceptable inlet mach number/ prantl meyer angle and see if self-starting is possible for the turbine.

        """

        # First we need to solve for our k_star_max based on our upper and lowe mach numbers we have selected
        INTEGRAL_NUMBER = 100  # TODO: Fix this magic number
        GAMMA = self._fluid.get_gamma()

        self._k_star = k_star_max(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
        )

        # Now we know our k_star max, we can figure out what our Q and C are for the turbine accordingly

        self._Q_blade = Q(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
        )

        self._C_blade = C(
            M_star_l=self._M_l_star,
            M_star_u=self._M_u_star,
            gamma=GAMMA,
            n=INTEGRAL_NUMBER,
            k_star=self._k_star,
        )

        # From this, we can figure out what our shock pressure ratio is for the gas.

        self._p_rat = shock_pressure_rat(Q=self._Q_blade, C=self._C_blade)

        # Now that we know our shock pressure ratio, we can now calculate back our M_star_i_max value

        self._M_i_star_max = M_i_star_max(p_rat=self._p_rat, gamma=GAMMA)

        # We can back calculate for what this mach number

        self._M_i_max = inv_M_star(gamma=GAMMA, M_star=self._M_i_star_max)

        self._v_i_max = prandtl_meyer(GAMMA, self._M_i_star_max)

        return self._M_i_max

    def M_u_max(self):
        """
        This function solves for the maximum upper surface mach number inorder to prevent flow seperation
        """
        GAMMA = self._fluid.get_gamma()

        self._M_u_star_max = M_star_u_max(M_star_o=self._M_o_star, gamma=GAMMA)

        self._M_u_max = inv_M_star(gamma=GAMMA, M_star=self._M_u_star_max)

        return

    def M_l_min(self):
        """
        This function solves for the minimum lower surface mach number to avoid flow seperation of the gas
        """
        GAMMA = self._fluid.get_gamma()

        self._M_l_star_min = M_star_l_min(m_star_i=self._M_i_star, gamma=GAMMA)

        self._M_l_min = inv_M_star(gamma=GAMMA, M_star=self._M_l_star_min)

        return


class SymmetricFiniteEdge(SupersonicProfile):
    """This object represents a symmetric finite edge supersonic profile w/boundary layer correction.

    We assume

    Args:
        SupersonicProfile (_type_): Infinite Edge Supersonic Profile Object
    """

    def __init__(
        self,
        beta_ei: float,
        beta_i: float,
        M_i: float,
        M_u: float,
        M_l: float,
        m_dot: float,
        h: float,
        t_g_rat: float,
        g_expand: float,
        le_angle: float,
        fluid: IdealGas,
    ):
        # First thing to do is get the acutal entry conditions for the turbine based on the farfiedl
        ANGLE_CONVERSION = np.pi / 180

        # We need to firstly solve for what the Mach number at the inlet of the turbine will be
        M_e = get_m_e(
            t_g_rat=t_g_rat,
            beta_e=beta_ei * ANGLE_CONVERSION,
            beta_i=beta_i * ANGLE_CONVERSION,
            M_i=M_i,
            gamma=fluid.get_gamma(),
        )

        # We can then Initialise for our turbine
        super().__init__(
            beta_i=beta_ei,
            beta_o=-beta_ei,
            M_i=M_e,
            M_o=-M_e,
            M_u=M_u,
            M_l=M_l,
            m_dot=m_dot,
            h=h,
            fluid=fluid,
        )

        # We then log the information as it relates to the finite leading edges and trailing edges - along with boundary layer computations.
        self._t_g_rat = t_g_rat
        self._g_expand = g_expand
        self._le_angle = le_angle * ANGLE_CONVERSION

        return

    def get_turbine_profile(self) -> dict[str, float | list]:
        """This Function Performs the Geometry generation for the Turbine Blade Profile

        Returns:
            dict[str, float]: Dictionary of parameters describing the turbine profile
        """
        # We firstly solve for our Prandtl Meyer Numebers
        self.prantl_meyer()

        # We then get our circular section parameters
        self.circular_section()

        # We solve for the upper maximum and lower minimum mach numbers to prevent flow speeration
        self.M_u_max()
        self.M_l_min()

        # We solve for the maximum inlet mach number (at turbine inlet) before the turbine would unstarts
        self.M_i_max()

        # We solve for the key geometries of the turbine
        self.inlet_lower_transition()
        self.inlet_upper_transition()
        self.outlet_lower_transition()
        self.outlet_upper_transition()

        # We define the straight line segments on the upper surface pretending their is no
        self.straight_line_segments()

        # We can now get the blade spacing (G*) and chord length (C*) to calculate our solidity
        self.get_g_star()
        self.get_c_star()
        self.get_solidity()

        # We can solve for our blade thickness at this stage now and generate the co-ordinates for our leading edge

    #     # We can now discretise our blade
    #     self.generate_blade(50)

    # print(f"v_i {super._v_i}")
    # print(f"v_o {super._v_o}")
    # print(f"v_l {super._v_l}")
    # print(f"v_u {super._v_u}")

    # print(f"alpha_u_o {super._alpha_u_o} rad")
    # print(f"alpha_u_o {super._alpha_u_o*ANGLE_CONVERSION} Degrees")
    # print("\n")

    # print(f"alpha_u_i {super._alpha_u_i} rad")
    # print(f"alpha_u_i {super._alpha_u_i*ANGLE_CONVERSION} Degrees")
    # print("\n")

    # print(f"alpha_l_o {super._alpha_l_o} rad")
    # print(f"alpha_l_o {super._alpha_l_o*ANGLE_CONVERSION} Degrees")
    # print("\n")

    # print(f"alpha_l_i {super._alpha_l_i} rad")
    # print(f"alpha_l_i {super._alpha_l_i*ANGLE_CONVERSION} Degrees")
    # print("\n")

    # # Now we go onto get our non-dimentionalised sonic radii

    # # super.r_star()

    # # print(f"R_l_star {super._R_l_star}")
    # # print(f"R_u_star {super._R_u_star}")
    # # print(f"r_star {super._r_star_a}")
    # # print(f"R_l {super._R_l_star*super._r_star_a} m")
    # # print(f"R_u {super._R_u_star*super._r_star_a} m")

    # # We now can confirm if the turbine can be started or not

    # # Assessing flow serperation

    # print("\n----------------------------------------- \n")

    # print("FLOW Serperation")

    # print("\n-----------------------------------------\n")

    # super.M_u_max()
    # super.M_l_min()

    # print(f"Maximum Possible M_u: {super._M_u_max:.3f}")
    # print(f"Current M_u: {super._M_u:.3f}")
    # print(f"Margin: {(super._M_u_max - super._M_u)/super._M_u}")

    # if super._M_u_max < super._M_u:
    #     raise ValueError(
    #         f"Mach number too high on upper surface: {super._M_u} > {super._M_u_max}"
    #     )

    # print("\n----------------------------------------- \n")

    # print(f"Minimum Possible M_l: {super._M_l_min:.3f}")
    # print(f"Curren M_l: {super._M_l:.3f}")
    # print(f"Margin: {(super._M_l - super._M_l_min)/super._M_l}")

    # if super._M_l < super._M_l_min:
    #     raise ValueError(
    #         f"Mach number too low on lower surface: {super._M_l} < {super._M_l_min}"
    #     )

    # # Assessing startability

    # print("\n----------------------------------------- \n")

    # print("STARTABILITY")

    # print("\n----------------------------------------- \n")

    # super.M_i_max()

    # print(f"Maximum M_i: {super._M_i_max:.3f}")
    # print(f"Nominal M_i: {super._M_i:.3f}")

    # print(
    #     f"V_i_max {super._v_i_max*ANGLE_CONVERSION}, v_u: {super._v_u*ANGLE_CONVERSION}, v_l: {super._v_l*ANGLE_CONVERSION}"
    # )

    # if super._M_i_max < super._M_i:
    #     raise ValueError(
    #         f"Mach number exceeded Nominally: {super._M_i} > {super._M_i_max}"
    #     )

    # super.plot_circles(50)

    # # Now we get the transition co-ordinates for the lower area
    # super.inlet_lower_transition()
    # super.inlet_upper_transition()
    # super.outlet_lower_transition()
    # super.outlet_upper_transition()
    # super.straight_line_segments()
    # super.generate_blade(50)

    # # super.plot_transition()

    # # print(super._g_star)
    # # print(super._R_l_star - super._R_u_star)

    # super.plot_all_shift(50)

    def generate_finite_edge(self) -> None:
        """This function generates the finite leading and trailing edges for the turbine"""

        # We need to figure out what the blade thickness is
        self._t = self._t_g_rat * self._g_star

        # We need to fogire out now what the x displacement is based on the angle
        theta = self._le_angle + self._beta_i

        if theta == np.pi / 2:
            self._dx_edge = 0
        elif theta > np.pi / 2:
            raise ValueError("Leading Edge Angle leading to a protruding LE shape!")
        else:
            self._dx_edge = self._t / (
                np.tan(self._beta_i + self._le_angle) - np.tan(self._beta_i)
            )

        # We can then adjust our points on the straight lines at the inlet and outlet of the turbine
