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
import math
import matplotlib.pyplot as plt

import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import minimize, Bounds


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
        """This function generates all the transition points for the turbine"""

        # We do the inlet transition arcs
        self.inlet_lower_transition()
        self.inlet_upper_transition()

        # We now need to compare our conditions for the inlet and lower, and decide if we need to do to re-do the MoC
        if (self._v_o != self._v_i) and (self._alpha_u_o != self._alpha_u_i):
            # We generate the outlet transition
            self.outlet_upper_transition()
        else:
            # Same as the first one, so we can invert our inital co-ordinates.
            self._xlkt_ou = -self._xlkt_iu
            self._ylkt_ou = self._ylkt_iu

        if (self._v_o != self._v_i) and (self._alpha_l_o != self._alpha_l_i):
            # We generate the outlet transition
            self.outlet_lower_transition()
        else:
            # Same as the first one, so we can invert our inital co-ordinates.
            self._xlkt_ol = -self._xlkt_il
            self._ylkt_ol = self._ylkt_il

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

    def discretise_circular(self, N: float) -> None:
        """This function discretises the circular sections of the turbine profiles

        Args:
            N (float): Number of Points for discretisation of the turbine Profile
        """
        alpha_l_array = np.linspace(self._alpha_l_i, self._alpha_l_o, N)

        alpha_u_array = np.linspace(self._alpha_u_i, self._alpha_u_o, N)

        # We can thus generate our x and y co-ordinates accordingly.

        self._x_l_array = -self._R_l_star * np.sin(alpha_l_array)
        self._y_l_array = self._R_l_star * np.cos(alpha_l_array)

        self._x_u_array = -self._R_u_star * np.sin(alpha_u_array)
        self._y_u_array = self._R_u_star * np.cos(alpha_u_array)

    def generate_surface_maps(self) -> None:
        """This function generates the surface maps for the s_position of the upper and lower surface of the blade profiles

        - We calculate the straight line distance between the points
        """
        ########## Upper Surface ##########

        # We create our upper surface arrays
        df_upper = self.generate_upper_xy()

        # We get the start points
        first_point = df_upper.iloc[0]

        df_upper["s"] = ""

        # Setting the Initial Position
        df_upper.loc[0, "s"] = 0

        for index, row in df_upper.iterrows():

            if index == 0:
                continue

            x_1_u = row["x"]
            y_1_u = row["y"]

            s_0_u = df_upper.loc[index - 1, "s"]
            x_0_u = df_upper.loc[index - 1, "x"]
            y_0_u = df_upper.loc[index - 1, "y"]

            df_upper.loc[index, "s"] = s_0_u + np.sqrt(
                (x_0_u - x_1_u) ** 2 + (y_0_u - y_1_u) ** 2
            )

        ########## Lower Surface ##########

        # We create our upper surface arrays
        df_lower = self.generate_lower_xy()

        # We get the start points
        first_point = df_lower.iloc[0]

        df_lower["s"] = ""

        # Setting the Initial Position
        df_lower.loc[0, "s"] = 0

        for index, row in df_lower.iterrows():

            if index == 0:
                continue

            x_1_l = row["x"]
            y_1_l = row["y"]

            s_0_l = df_lower.loc[index - 1, "s"]
            x_0_l = df_lower.loc[index - 1, "x"]
            y_0_l = df_lower.loc[index - 1, "y"]

            df_lower.loc[index, "s"] = s_0_l + np.sqrt(
                (x_0_l - x_1_l) ** 2 + (y_0_l - y_1_l) ** 2
            )

        # We can now normalise our "s" array from 0 to 1.
        df_upper["s"] = df_upper["s"] / df_upper["s"].max()
        df_lower["s"] = df_lower["s"] / df_lower["s"].max()

        # We can now prepare our interpolation arrays for the x and y positions of the upper and lower surface using linear interpolation.
        self._s_x_u = interp1d(df_upper["s"], df_upper["x"], kind="linear")
        self._s_y_u = interp1d(df_upper["s"], df_upper["y"], kind="linear")

        # We can now do the same process for the lower surface
        self._s_x_l = interp1d(df_lower["s"], df_lower["x"], kind="linear")
        self._s_y_l = interp1d(df_lower["s"], df_lower["y"], kind="linear")

        return

    def get_upper_surface_position(self, s_u: float) -> tuple:
        """Gets the co-ordinates of the upper surface position

        Args:
            s_u (float): Streamline Length along upper surface [0 - 1]. 0 is LE, 1 is TE

        Returns:
            tuple: x,y position of the surface
        """
        # We can get the upper surface position
        x_u = self._s_x_u(s_u)
        y_u = self._s_y_u(s_u)

        return (x_u, y_u)

    def get_lower_surface_position(self, s_l: float) -> tuple:
        """Gets the co-ordinates of the lower surface position

        Args:
            s_l (float): Streamline Length along lower surface [0 - 1]. 0 is LE, 1 is TE

        Returns:
            tuple: x,y position of the surface
        """
        # We can get the lower surface position
        x_l = self._s_x_l(s_l)
        y_l = self._s_y_l(s_l)

        return (x_l, y_l)

    def get_distance_upper(
        self, s_u: float, s_l: float, shift_flag: bool = False
    ) -> float:
        """Gets the distance between two points on the upper lower surface

        Args:
            s_u (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            s_l (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            shift_flag (bool): Whether we shift the position of the y for matching.

        Returns:
            float: Distance between the upper and lower surface blade profiles
        """
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        if shift_flag:
            y_l += self._g_star * self._sf

        dl = ((x_u - x_l) ** 2 + (y_u - y_l) ** 2) ** (1 / 2)

        return dl

    def get_distance_lower(
        self, s_l: float, s_u: float, shift_flag: bool = False
    ) -> float:
        """Gets the distance between two points on the upper lower surface

        Args:
            s_u (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            s_l (float): Position along the Lower Surface [0 - 1]. 0 is LE, 1 is TE
            shift_flag (bool): Whether we shift the position of the y for matching.

        Returns:
            float: Distance between the upper and lower surface blade profiles
        """
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        if shift_flag:
            y_l += self._g_star * self._sf

        dl = ((x_u - x_l) ** 2 + (y_u - y_l) ** 2) ** (1 / 2)

        return dl

    def fit_surface_circle(
        self, s_u: float, s_l: float, offset: float = 0
    ) -> dict[str, float]:
        """This function fits a cirle based on the upper and lower surface points

        Args:
            s_u (float): Upper Surface Point
            s_l (float): Lower Surface Point
            offset (float, optional): Offset for lower surface positioning

        Returns:
            dict[str, float]: Dictionary of the camber center co-ordinates and the spacing
        """
        if s_l >= 0.999 or s_u >= 0.999:
            ds = -0.001
        else:
            ds = 0.001

        # We get the upper surface position and gradient
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_up, y_up = self.get_upper_surface_position(s_u=s_u + ds)

        dy_dx_u = (y_up - y_u) / (x_up - x_u)

        # We get the lower surface position and gradient
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)
        x_lp, y_lp = self.get_lower_surface_position(s_l=s_l + ds)

        # y_l += offset
        # y_lp += offset

        dy_dx_l = (y_lp - y_l) / (x_lp - x_l)

        # We can now convert this into an angle
        phi_u = math.atan2(-1, dy_dx_u)
        phi_l = math.atan2(-1, dy_dx_l)

        # We can then solve for our radius
        r = (x_u - x_l) / (np.cos(phi_u) + np.cos(phi_l))

        # We can then solve for our centroid accordingly
        x_0 = x_l + r * np.cos(phi_l)
        y_0 = y_l + r * np.sin(phi_l)

        dic = {"x": x_0, "y": y_0, "dl": 2 * r}

        return dic

    def error_circle(self, s_u: float, s_l: float) -> float:
        """This function gets the radius error between two points on a curve when fitting a circle

        Args:
            s_u (float): Upper Position
            s_l (float): Lower Position

        Returns:
            float: Error in Radii
        """
        if s_l >= 0.999 or s_u >= 0.999:
            ds = -0.001
        else:
            ds = 0.001

        # We get the upper surface position and gradient
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)
        x_up, y_up = self.get_upper_surface_position(s_u=s_u + ds)

        dy_dx_u = (y_up - y_u) / (x_up - x_u)

        # We get the lower surface position and gradient
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)
        x_lp, y_lp = self.get_lower_surface_position(s_l=s_l + ds)

        # y_l += offset
        # y_lp += offset

        dy_dx_l = (y_lp - y_l) / (x_lp - x_l)

        # We can now convert this into an angle
        phi_u = math.atan2(-1, dy_dx_u)
        phi_l = math.atan2(-1, dy_dx_l)

        # We can then solve for our radii
        r_x = (x_u - x_l) / (np.cos(phi_u) + np.cos(phi_l))
        r_y = (y_u - y_l) / (np.sin(phi_u) + np.sin(phi_l))

        return abs(r_x - r_y)

    def camber_position(self, s_l: float) -> dict[str, float]:
        """Gets the location of the camber Line

        Args:
            s_l (float): Surface Position on lower surface

        Returns:
            dict[str, float]: Dictionary of the x and y position of the camber, along with the thickness at his point
        """
        # We get the tangent location on the upper surface
        x_u, y_u = self.get_upper_surface_position(s_u=s_l)
        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        # We can get the mid point between these two points accordingly.
        dic = {"x": (x_u + x_l)/2,
               "y": (y_u + y_l)/2}

        dic["x_1"] = x_u
        dic["y_1"] = y_u

        dic["x_2"] = x_l
        dic["y_2"] = y_l

        return dic

    def get_cad_shift(self) -> float:
        """This function gets the offset for the CAD

        Returns:
            float: CAD Offset
        """
        
        df = self.generate_xy()
        
        offset = - df["y"].min() - 0.5 * (df["y"].max() - df["y"].min())
        
        return offset

    def passage_position(self, s_l: float) -> dict[str, float]:
        """Gets the location of the passage center line for machining

        Args:
            s_l (float): Surface Position on lower surface

        Returns:
            dict[str, float]: Dictionary of the x and y position of the camber, along with the thickness at his point
        """

        # We get the upper surface position
        bnds = Bounds(lb=0, ub=1)
        res = minimize(
            self.get_distance_upper,
            s_l * 0.8,
            args=(s_l, True),
            bounds=bnds,
            method="Nelder-Mead",
        )

        s_u = res.x

        # We then need to get the x and y positions of the upper and lower surfaces
        x_u, y_u = self.get_upper_surface_position(s_u=s_u)

        x_l, y_l = self.get_lower_surface_position(s_l=s_l)

        y_l += self._g_star * self._sf

        dic = {}

        dic["x_1"] = x_u
        dic["y_1"] = y_u

        dic["x_2"] = x_l
        dic["y_2"] = y_l

        return dic

    def get_passage_spacing(self) -> float:
        """This function evaluates for the blade Passage spacing for machining

        Returns:
            float: Blade Passage Spacing
        """
        up = self.get_upper_surface_position(s_u=0.5)
        down = self.get_lower_surface_position(s_l=0.5)

        y_u = (up / self._sf - self._g_star)[1]

        y_d = (down / self._sf)[1]

        s = y_d - y_u

        return s * self._sf

    def generate_blade(self):

        # Now Shifting all our array points for the upper blade profiling accordingly

        self._y_i_line_up = self._y_i_line + self._g_star
        self._y_o_line_up = self._y_o_line + self._g_star

        # Shifting transition points
        self._ylkt_iu_up = self._ylkt_iu + self._g_star
        self._ylkt_ou_up = self._ylkt_ou + self._g_star

        # Shifting Circular Points
        self._y_u_array_up = self._y_u_array + self._g_star

        return

    def plot_circles(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array)
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

    def plot_passage(self):
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

        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")

        ax.set_aspect("equal")
        ax.set_title(f"Normalised Flow Passage Profile")

        ax.legend()
        plt.show()

    def plot_normalised(self):
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array, self._y_l_array)
        ax.plot(self._x_u_array, self._y_u_array_up)
        ax.plot(self._xlkt_il, self._ylkt_il, label="Inlet Lower")
        ax.plot(self._xlkt_iu, self._ylkt_iu_up, label="Inlet Upper")
        ax.plot(self._xlkt_ol, self._ylkt_ol, label="Outlet Lower")
        ax.plot(self._xlkt_ou, self._ylkt_ou_up, label="Outlet Upper")
        ax.plot(self._x_i_line, self._y_i_line_up)
        ax.plot(self._x_o_line, self._y_o_line_up)
        ax.set_ylabel(r"y* ($\frac{y}{r^*}$)")
        ax.set_xlabel(r"x* ($\frac{x}{r^*}$)")
        ax.set_aspect("equal")
        ax.set_title(f"Normalised Blade Profile")
        ax.legend()
        plt.show()

    def plot_scaled(self):

        if not hasattr(self, "_x_l_array_sf"):
            self.scale_coords(sf=self._r_star_a)
            scaling_text = r"$r*$"
        else:
            scaling_text = r"$sf$"
        # This function plots the circular arcs for visual inspection

        fig, ax = plt.subplots()

        # We then plot our results

        ax.plot(self._x_l_array_sf * 1e3, self._y_l_array_sf * 1e3)
        ax.plot(self._x_u_array_sf * 1e3, self._y_u_array_sf * 1e3)
        ax.plot(
            self._xlkt_il_sf * 1e3,
            self._ylkt_il_sf * 1e3,
            label="Inlet Lower",
        )
        ax.plot(
            self._xlkt_iu_sf * 1e3,
            self._ylkt_iu_sf * 1e3,
            label="Inlet Upper",
        )
        ax.plot(
            self._xlkt_ol_sf * 1e3,
            self._ylkt_ol_sf * 1e3,
            label="Outlet Lower",
        )
        ax.plot(
            self._xlkt_ou_sf * 1e3,
            self._ylkt_ou_sf * 1e3,
            label="Outlet Upper",
        )
        ax.plot(self._x_i_line_sf * 1e3, self._y_i_line_sf * 1e3, label="Leading Edge")
        ax.plot(self._x_o_line_sf * 1e3, self._y_o_line_sf * 1e3, label="Trailing Edge")

        ax.set_ylabel(r"y (mm)")
        ax.set_xlabel(r"x (mm)")

        ax.set_aspect("equal")
        ax.set_title(f"Blade Profile Scaled by {scaling_text}")
        ax.legend()
        plt.show()

    def generate_upper_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the upper surface

        Returns:
            pd.DataFrame: Data frame of profile co-ordinates of the Upper Surface
        """

        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # We plot the Leading Edge Array,
        x_array = np.append(x_array, self._x_i_line_sf[::-1])
        y_array = np.append(y_array, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array = np.append(x_array, (self._xlkt_iu_sf)[-2:1:-1])
        y_array = np.append(y_array, (self._ylkt_iu_sf)[-2:1:-1])

        # # We then do the inlet Upper Circular element
        x_array = np.append(x_array, self._x_u_array_sf)
        y_array = np.append(y_array, self._y_u_array_sf)

        # # We then do the outlet Upper Transition
        x_array = np.append(x_array, self._xlkt_ou_sf[1:-1])
        y_array = np.append(y_array, self._ylkt_ou_sf[1:-1])

        # # We plote the Trailing Edge Array,
        x_array = np.append(x_array, self._x_o_line_sf)
        y_array = np.append(y_array, self._y_o_line_sf)

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.

        df = pd.DataFrame(data={"x": x_array, "y": y_array, "z": z_array})

        return df

    def generate_lower_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the lower surface

        Returns:
            pd.DataFrame: Data frame of profile co-ordinates of the Lower Surface.
        """
        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # # # We then do the outlet lower transition
        x_array = np.append(x_array, self._xlkt_ol_sf[-1:1:-1])
        y_array = np.append(y_array, self._ylkt_ol_sf[-1:1:-1])

        # # We then do the lower circular element
        x_array = np.append(x_array, (self._x_l_array_sf)[::-1])
        y_array = np.append(y_array, (self._y_l_array_sf)[::-1])

        # # We then do the inlet lower transition element
        x_array = np.append(x_array, (self._xlkt_il_sf)[1:-2])
        y_array = np.append(y_array, (self._ylkt_il_sf)[1:-2])

        x_array = np.append(x_array, self._x_i_line_sf[-1])
        y_array = np.append(y_array, self._y_i_line_sf[-1])

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.

        df = pd.DataFrame(
            data={"x": x_array[::-1], "y": y_array[::-1], "z": z_array[::-1]}
        )

        return df

    def generate_xy(self) -> pd.DataFrame:
        """Function that generates an x-y data frame of the co-ordinates of the turbine, that can be either plotted or used accordingly.

        Returns:
            pd.DataFrame: Dataframe of Profile Co-ordinates
        """

        # We simply need to create a master x-array and y-array, create a pandas dataframe, then export as csv
        x_array = np.array([])
        y_array = np.array([])
        z_array = np.array([])

        # We plot the Leading Edge Array,
        x_array = np.append(x_array, self._x_i_line_sf[::-1])
        y_array = np.append(y_array, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array = np.append(x_array, (self._xlkt_iu_sf)[-2:1:-1])
        y_array = np.append(y_array, (self._ylkt_iu_sf)[-2:1:-1])

        # # We then do the inlet Upper Circular element
        x_array = np.append(x_array, self._x_u_array_sf)
        y_array = np.append(y_array, self._y_u_array_sf)

        # # We then do the outlet Upper Transition
        x_array = np.append(x_array, self._xlkt_ou_sf[1:-1])
        y_array = np.append(y_array, self._ylkt_ou_sf[1:-1])

        # # We plote the Trailing Edge Array,
        x_array = np.append(x_array, self._x_o_line_sf)
        y_array = np.append(y_array, self._y_o_line_sf)

        # # # We then do the outlet lower transition
        x_array = np.append(x_array, self._xlkt_ol_sf[-2:1:-1])
        y_array = np.append(y_array, self._ylkt_ol_sf[-2:1:-1])

        # # We then do the lower circular element
        x_array = np.append(x_array, (self._x_l_array_sf)[::-1])
        y_array = np.append(y_array, (self._y_l_array_sf)[::-1])

        # # We then do the inlet lower transition element
        x_array = np.append(x_array, (self._xlkt_il_sf)[1:-2])
        y_array = np.append(y_array, (self._ylkt_il_sf)[1:-2])

        x_array = np.append(x_array, self._x_i_line_sf[-1])
        y_array = np.append(y_array, self._y_i_line_sf[-1])

        z_array = np.zeros(x_array.size)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.
        # y_array = y_array - y_array.min() - 0.5 * (y_array.max() - y_array.min())

        df = pd.DataFrame(data={"x": x_array * 1e3, "y": y_array * 1e3, "z": z_array})

        return df

    def get_xy_mean_line(self) -> pd.DataFrame:
        """Function that gets the mean line between the upper and lower surfaces of the turbine

        Returns:
            pd.DataFrame: Dataframe containing the mean_line co-ordinates of the upper and lower surface
        """
        
        ####################################### Upper Surface Profile #######################################
        
        x_array_u = np.array([])
        y_array_u = np.array([])

        # We plot the Leading Edge Array,
        x_array_u = np.append(x_array_u, self._x_i_line_sf[::-1])
        y_array_u = np.append(y_array_u, self._y_i_line_sf[::-1])

        # The then go to the inlet upper Transition
        x_array_u = np.append(x_array_u, (self._xlkt_iu_sf)[-2:1:-1])
        y_array_u = np.append(y_array_u, (self._ylkt_iu_sf)[-2:1:-1])

        # We then do the inlet Upper Circular element
        x_array_u = np.append(x_array_u, self._x_u_array_sf)
        y_array_u = np.append(y_array_u, self._y_u_array_sf)

        # We then do the outlet Upper Transition
        x_array_u = np.append(x_array_u, self._xlkt_ou_sf[1:-1])
        y_array_u = np.append(y_array_u, self._ylkt_ou_sf[1:-1])

        # We plote the Trailing Edge Array,
        x_array_u = np.append(x_array_u, self._x_o_line_sf)
        y_array_u = np.append(y_array_u, self._y_o_line_sf)

        ####################################### Lower Surface Profile #######################################

        x_array_l = np.array([])
        y_array_l = np.array([])

        # We then do the outlet lower transition
        x_array_l = np.append(x_array_l, self._xlkt_ol_sf[-1:1:-1])
        y_array_l = np.append(y_array_l, self._ylkt_ol_sf[-1:1:-1])

        # We then do the lower circular element
        x_array_l = np.append(x_array_l, (self._x_l_array_sf)[::-1])
        y_array_l = np.append(y_array_l, (self._y_l_array_sf)[::-1])

        # We then do the inlet lower transition element
        x_array_l = np.append(x_array_l, (self._xlkt_il_sf)[1:-2])
        y_array_l = np.append(y_array_l, (self._ylkt_il_sf)[1:-2])

        x_array_l = np.append(x_array_l, self._x_i_line_sf[-1])
        y_array_l = np.append(y_array_l, self._y_i_line_sf[-1])

        # We can now inverse the order of the array now
        x_array_l = x_array_l[::-1]
        y_array_l = y_array_l[::-1]
        
        ######################################### Normalising the distances #########################################

        y_array = np.append(y_array_l, y_array_u)

        # We need to center in the y_axis- to do this, we will get the maximum and minimum value for the y, half it and shift accordingly.
        y_array_l = y_array_l - y_array.min() - 0.5 * (y_array.max() - y_array.min())
        y_array_u = y_array_u - y_array.min() - 0.5 * (y_array.max() - y_array.min())

        ######################################### Interpolation #########################################
        

        df = pd.DataFrame(data={"x": x_array * 1e3, "y": y_array * 1e3, "z": z_array})
        

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

    def scale_coords(self, sf: float) -> None:
        """This function scales the geometry, based on a scaling factor for the geometry (either R_star_a or a chord based scale factor)

        Args:
            sf (float): Scale Factor for the Geometry
        """

        self._sf = sf

        self._x_l_array_sf = self._x_l_array * sf
        self._y_l_array_sf = self._y_l_array * sf

        self._x_u_array_sf = self._x_u_array * sf
        self._y_u_array_sf = self._y_u_array_up * sf

        self._xlkt_il_sf = self._xlkt_il * sf
        self._ylkt_il_sf = self._ylkt_il * sf

        self._xlkt_ol_sf = self._xlkt_ol * sf
        self._ylkt_ol_sf = self._ylkt_ol * sf

        self._xlkt_iu_sf = self._xlkt_iu * sf
        self._ylkt_iu_sf = self._ylkt_iu_up * sf

        self._xlkt_ou_sf = self._xlkt_ou * sf
        self._ylkt_ou_sf = self._ylkt_ou_up * sf

        self._x_i_line_sf = self._x_i_line * sf
        self._y_i_line_sf = self._y_i_line_up * sf

        self._x_o_line_sf = self._x_o_line * sf
        self._y_o_line_sf = self._y_o_line_up * sf

        return
    
    def generate_mesh_upper(self, n: int = 1000) -> pd.DataFrame:
        """This function gets the upper surface mesh contour
        
        Args:
            n (int, optional): Number of Points on the Upper Surface. Defaults to 1000.

        Returns:
            pd.DataFrame: Contour of the Upper Surface
        """
        
        # We get the offset
        offset = self.get_cad_shift()
        
        self.generate_surface_maps()
        
        x_c = []
        y_c = []
        
        for x in np.linspace(0, 1, n):
            camber = self.camber_position(x)
            
            x_c.append(camber["x"])
            y_c.append(camber["y"])
            
        data ={
            "x": np.array(x_c)*1e3, 
            "y": (np.array(y_c) + self._t/2)*1e3 - offset,
            "z": np.zeros(np.array(x_c).size)
            }
            
        df = pd.DataFrame(data)
        
        return df
    
    def generate_mesh_lower(self, n: float = 1000) -> pd.DataFrame:
        """This function gets the lower surface mesh contour
        
        Args:
            n (float, optional): Number of Points on the Upper Surface. Defaults to 1000.

        Returns:
            pd.DataFrame: Contour of the Upper Surface
        """
        
        # We get the offset
        offset = self.get_cad_shift()
        
        self.generate_surface_maps()
        
        x_c = []
        y_c = []
        z_c = []
        
        for x in np.linspace(0, 1, n):
            camber = self.camber_position(x)
            
            x_c.append(camber["x"])
            y_c.append(camber["y"])
            
        data = {
            "x": (np.array(x_c))*1e3, 
            "y": (np.array(y_c) - self._t/2)*1e3 - offset,
            "z": np.zeros(np.array(x_c).size)
            }
        
        df = pd.DataFrame(data)
        
        return df


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

    def generate_turbine_profile(self) -> None:
        """This Function Performs the Geometry generation for the Turbine Blade Profile"""
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
        self.generate_transitions()

        # We discretise the circulate sections
        self.discretise_circular(50)

        # We define the straight line segments on the upper surface pretending their is no
        self.straight_line_segments()

        # We can now get the blade spacing (G*) and chord length (C*) to calculate our solidity
        self.get_g_star()
        self.get_c_star()
        self.get_solidity()
        print(f"Initial Solidity: {self._sigma}")

        # We can now generate the finite edge thickness
        self.generate_finite_edge()

        # We then further expand the blade spacing based on user input
        self.adjust_blade_spacing(b_factor=self._g_expand)

        # Finally we can generate our blade
        self.generate_blade()

        # We can re_caclulate the solidity
        self._sigma = self._c_star / self._g_star

        print(f"Final Solidity: {self._sigma}")

        return

    def get_performance(self) -> dict[str, float]:
        """This function gets the key performance parameters as it relates to the turbine, namely "startability" and margin till flow seperation

        Returns:
            (dict): Dictionary of Performance Parameters
        """

        # Firstly we evaluate what our maximum possible upper surface mach number is along with minimum lower surface mach number
        self.M_u_max()
        self.M_l_min()

        # We check if we are in compliance
        if self._M_u_max < self._M_u:
            raise ValueError(
                f"Mach number too high on upper surface for flow seperation: {self._M_u} > {self._M_u_max}"
            )

        if self._M_l < self._M_l_min:
            raise ValueError(
                f"Mach number too low on lower surface for flow seperation: {self._M_l} < {self._M_l_min}"
            )

        # We now check if both of these are higher than the inlet conditions

        if self._M_u < self._M_i:
            raise ValueError(
                f"Mach Number too low on upper surface and is decelerating from inlet! {self._M_u} < {self._M_i}"
            )

        if self._M_l > self._M_i:
            raise ValueError(
                f"Mach Number too high on lower surface and is accelerating from inlet! {self._M_l} > {self._M_i}"
            )

        # We then evaluate for the maximum possible Inlet Mach Number
        self.M_i_max()

        # We check if we are in compliance
        if self._M_i_max < self._M_i:
            raise ValueError(
                f"Inlet Mach number exceeded Maximum For Starting: {self._M_i} > {self._M_i_max}"
            )

        # We then assemble our array and dictionary of key parameters
        dic = {
            "M_u_max": self._M_u_max,
            "M_u_margin": (self._M_u_max - self._M_u) / self._M_u,
            "M_l_min": self._M_l_min,
            "M_l_margin": (self._M_l - self._M_l_min) / self._M_l,
            "M_e_max": (self._M_i_max),
            "M_e_margin": (self._M_i_max - self._M_i) / self._M_i,
        }

        return dic

    def size_geometry(
        self, D_m: float, N: int | None = None, b: float | None = None
    ) -> dict[str, float]:
        """This function is used for scaling the geometry based on the solidity we use for the blade design.

        Args:
            D_m (float): Mean Diameter of the Turbine
            N (int | None): Number of Blades at the meanline Diameter (m). Defaults to None
            b (float | None): Blade Chord Length (m). Defaults to None
        """

        if N is not None and b is not None:
            raise ValueError("Problem is Over defined!")

        elif b is not None:
            # We calcualte the blade spacing based on the chord length
            self._t = b / self._sigma

            self._N = round(D_m / self._t)

        elif N is not None:
            # We calculate the chord length based on the
            self._N = N
        else:
            raise ValueError(
                "Not enough information. Require either the blade chord length or number of profiles at the mean diameter"
            )

        # We can now solve for the chord length based on the number of blades we intend to have
        self._t = np.pi * D_m / self._N

        # From this, we can figure out what the blade chord should be

        self._b = self._sigma * self._t

        # We can then generate our scaled components based on the distances fromt he last points.
        b_normal = self._x_o_line[-1] - self._x_i_line[-1]

        self._sf = self._b / b_normal

        # We can scale accordingly for both the x and y axis for all the key dimensions
        self.scale_coords(sf=self._sf)

        # We can finally return a dictionary containing the key Properties of the turbine
        dic = {
            "sigma": self._sigma,
            "b": self._b,
            "t": self._t,
        }

        return dic

    def generate_finite_edge(self, N: int = 100) -> None:
        """This function generates a finite leading edge for the Turbine

        Args:
            N (int, optional): Number of Points to discretise on each line segment. Defaults to 100.

        Raises:
            ValueError: Leading Edge Angle is Protruding
        """

        # We need to figure out what the blade thickness is
        self._t = self._t_g_rat * self._g_star

        # We need to fogire out now what the x displacement is based on the angle
        theta = self._le_angle + self._beta_i

        append_flag = True

        if theta == np.pi / 2:
            self._dx_edge = 0

            self._dy_edge = self._t

        elif theta > np.pi / 2:
            raise ValueError("Leading Edge Angle leading to a protruding LE shape!")

        elif self._le_angle == 0:
            # No leading edge Angle
            append_flag = False

        else:
            self._dx_edge = self._t / (
                np.tan(self._beta_i + self._le_angle) - np.tan(self._beta_i)
            )

            self._dy_edge = self._dx_edge * np.tan(self._beta_i)

        if append_flag:
            # We can generate our intersection points

            x_i_new = self._x_i_line[-1] + self._dx_edge
            y_i_new = self._y_i_line[-1] + self._dy_edge

            self._x_i_line = np.append(
                np.linspace(self._x_i_line[0], x_i_new, N),
                np.linspace(x_i_new, self._x_i_line[-1], N)[1:],
            )
            self._y_i_line = np.append(
                np.linspace(self._y_i_line[0], y_i_new, N),
                np.linspace(y_i_new, self._y_i_line[-1] - self._t, N)[1:],
            )

            x_o_new = self._x_o_line[-1] - self._dx_edge
            y_o_new = self._y_o_line[-1] + self._dy_edge

            self._x_o_line = np.append(
                np.linspace(self._x_o_line[0], x_o_new, N),
                np.linspace(x_o_new, self._x_o_line[-1], N)[1:],
            )
            self._y_o_line = np.append(
                np.linspace(self._y_o_line[0], y_o_new, N),
                np.linspace(y_o_new, self._y_o_line[-1] - self._t, N)[1:],
            )

        # Update Leading Edge Thickness
        self._g_star += self._t

        return

    def adjust_blade_spacing(self, b_factor: float) -> None:
        """This function adjusts the blade spacing, along with all with all the co-ordinates of the system

        Args:
            b_factor (float): Blade Spacing Factor (% change in geometry)
        """

        # Firstly we adjust our g_star value based on the b_factor and get the displacement distance

        dy = self._g_star * b_factor

        self._g_star = self._g_star * (1 + b_factor)

        # We can then decrease all the suction surface co-ordinates accordingly.
        self._y_i_line -= dy
        self._y_o_line -= dy

        # Shifting transition points
        self._ylkt_iu -= dy

        # Shifting Circular Points
        self._y_u_array -= dy

        return
