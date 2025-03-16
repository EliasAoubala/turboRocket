# This file encapsulates the main function for computing the supersonic profile

from turborocket.profiling.Supersonic.circular import prandtl_meyer, M_star, arc_angles_upper, arc_angles_lower, inv_M_star

from turborocket.profiling.Supersonic.transition import moc, moc_2

from turborocket.profiling.Supersonic.constraints import inv_mass_flow, r_star
from turborocket.profiling.Supersonic.constraints import k_star_max, Q, C, shock_pressure_rat, M_i_star_max
from turborocket.profiling.Supersonic.constraints import M_star_u_max, M_star_l_min

from turborocket.fluids.ideal_gas import IdealFluid

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

class SupersonicProfile():
    
    def __init__(self,
                 beta_i: float,
                 beta_o: float,
                 M_i: float,
                 M_o: float,
                 M_u: float,
                 M_l: float,
                 m_dot: float,
                 h: float,
                 fluid: IdealFluid) -> None:
        
        # Constant for converting angles to radians
        ANGLE_CONVERSION = np.pi/180
        
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
                                                v_u=self._v_u
                                                )

        [self._alpha_l_i, self._alpha_l_o] = arc_angles_lower(
                                                beta_o=self._beta_o,
                                                beta_i=self._beta_i,
                                                v_i=self._v_i,
                                                v_o=self._v_o,
                                                v_l=self._v_l
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
        
        self._wf_parameter = inv_mass_flow(M_star_l=self._M_l_star,
                                           M_star_u=self._M_u_star,
                                           gamma=GAMMA,
                                           n=INTEGRAL_NUMBER,
                                           mass_flow=self._m_dot)
        
        # We meed to now calculate the total density assuming ideal gas
        density_i_total = self._fluid.get_density()
        
        # Calculating thet local speed of sound of the ideal gas.
        a_i_total = self._fluid.speed_of_sound()
        
        # Based on the weight-flow parameter, we can compute the sonic radius
        self._r_star_a = r_star(wf_parameter=self._wf_parameter,
                                h=self._h,
                                a_total_inlet=a_i_total,
                                rho_total_inlet=density_i_total)
        
        return
    
    def inlet_lower_transition(self):
        
        
        # We specify an arbritrary number of points for the MoC
        K_MAX= 10
        
        # We specify an arbritary inlet absolute angle of zero
        ALPHA_I = 0
        
        GAMMA = self._fluid.get_gamma()
        
        [self._xlkt_il, self._ylkt_il] = moc(k_max=K_MAX,
                                     v_i=self._v_i,
                                     v_l=self._v_l,
                                     gamma=GAMMA,
                                     alpha_l_i=self._alpha_l_i)
        
    def inlet_upper_transition(self):
        
        
        # We specify an arbritrary number of points for the MoC
        K_MAX= 100
        
        # We specify an arbritary inlet absolute angle of zero
        
        GAMMA = self._fluid.get_gamma()
        
        [self._xlkt_iu, self._ylkt_iu] = moc_2(k_max=K_MAX,
                                     v_i=self._v_i,
                                     v_l=self._v_u,
                                     gamma=GAMMA,
                                     alpha_l_i=self._alpha_u_i)
        
    
    def outlet_upper_transition(self):
        
        
        # We specify an arbritrary number of points for the MoC
        K_MAX= 10
        
        # We specify an arbritary inlet absolute angle of zero
        
        GAMMA = self._fluid.get_gamma()
        
        [self._xlkt_ou, self._ylkt_ou] = moc(k_max=K_MAX,
                                     v_i=self._v_o,
                                     v_l=self._v_u,
                                     gamma=GAMMA,
                                     alpha_l_i=self._alpha_u_o) # self._alpha_u_o)  

    def outlet_lower_transition(self):
        
        
        # We specify an arbritrary number of points for the MoC
        K_MAX= 10
        
        # We specify an arbritary inlet absolute angle of zero
        
        GAMMA = self._fluid.get_gamma()
        
        [self._xlkt_ol, self._ylkt_ol] = moc_2(k_max=K_MAX,
                                     v_i=self._v_o,
                                     v_l=self._v_l,
                                     gamma=GAMMA,
                                     alpha_l_i=self._alpha_l_o) # self._alpha_l_o)  
        
    def straight_line_segments(self):
        # The final section for the creation of these aerofoils is to draw in the straight line segments
        # These are parallel to the flow inlet and outlet and start at the last points of the transition.
        # The straight line continues until it reaches the equivalent x-coordinate of the lower transition.
        
        # Inlet Straight Line Segment
        self._x_i_start = self._xlkt_iu[-1]
        self._y_i_start = self._ylkt_iu[-1]
        
        self._x_i_end = self._xlkt_il[-1]
        
        # We get our required offset
        delta_y_i = (self._x_i_end - self._x_i_start)*np.tan(self._beta_i)
        
        # Hence we can get our last point of the segment
        self._y_i_end = self._y_i_start + delta_y_i
        
        # Repeating for the exit straight line segment
        self._x_o_start = self._xlkt_ou[-1]
        self._y_o_start = self._ylkt_ou[-1]
        
        self._x_o_end = self._xlkt_ol[-1]
        
        # We get our required offset
        delta_y_o = (self._x_o_end - self._x_o_start)*np.tan(self._beta_o)
        
        # Hence we can get our last point of the segment
        self._y_o_end = self._y_o_start + delta_y_o    
    
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
        self._y_u_array  = self._R_u_star * np.cos(alpha_u_array)
        
        self._g_star = self._ylkt_ol[-1] - self._y_o_end # Blade Spacing
        
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
        y_u_array  = self._R_u_star * np.cos(alpha_u_array)
        
        # We then plot our results
        
        ax.plot(x_l_array, y_l_array)
        ax.plot(x_u_array, y_u_array)
        ax.set_aspect('equal')
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
        ax.set_aspect('equal')
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
        ax.plot([self._x_i_start, self._x_i_end], [self._y_i_start, self._y_i_end])
        ax.plot([self._x_o_start, self._x_o_end], [self._y_o_start, self._y_o_end])
        ax.set_aspect('equal')
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
        ax.plot([self._x_i_start, self._x_i_end], [self._y_i_start_sf, self._y_i_end_sf])
        ax.plot([self._x_o_start, self._x_o_end], [self._y_o_start_sf, self._y_o_end_sf])
        ax.set_aspect('equal')
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
        ax.plot([self._x_i_start, self._x_i_end], [self._y_i_start_sf, self._y_i_end_sf])
        ax.plot([self._x_o_start, self._x_o_end], [self._y_o_start_sf, self._y_o_end_sf])
        ax.set_aspect('equal')
        ax.legend()
        plt.show()
        
    def plot_all_shift_to_scale(self, NUMBER_OF_POINTS):
        # This function plots the circular arcs for visual inspection
        
        fig, ax = plt.subplots()

        # We then plot our results
        
        ax.plot(self._x_l_array * self._r_star_a, self._y_l_array * self._r_star_a)
        ax.plot(self._x_u_array * self._r_star_a, self._y_u_array_sf * self._r_star_a)
        ax.plot(self._xlkt_il * self._r_star_a, self._ylkt_il * self._r_star_a, label="Inlet Lower")
        ax.plot(self._xlkt_iu * self._r_star_a, self._ylkt_iu_sf * self._r_star_a, label="Inlet Upper")
        ax.plot(self._xlkt_ol * self._r_star_a, self._ylkt_ol * self._r_star_a, label="Outlet Lower")
        ax.plot(self._xlkt_ou * self._r_star_a, self._ylkt_ou_sf * self._r_star_a, label="Outlet Upper")
        ax.plot([self._x_i_start * self._r_star_a, self._x_i_end * self._r_star_a], [self._y_i_start_sf * self._r_star_a, self._y_i_end_sf * self._r_star_a])
        ax.plot([self._x_o_start * self._r_star_a, self._x_o_end * self._r_star_a], [self._y_o_start_sf * self._r_star_a, self._y_o_end_sf * self._r_star_a])
        ax.set_aspect('equal')
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
        
        df = pd.DataFrame(data={"x": x_array*1e3, "y": y_array*1e3, "z": z_array})
        
        df.to_csv("blade.txt", header=False, index = False, sep="\t")
        
        
        
        
        
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
        
        self._k_star = k_star_max(M_star_l = self._M_l_star,
                            M_star_u = self._M_u_star,
                            gamma = GAMMA,
                            n = INTEGRAL_NUMBER
                            )
        
        # Now we know our k_star max, we can figure out what our Q and C are for the turbine accordingly
        
        self._Q_blade = Q(M_star_l = self._M_l_star,
                    M_star_u = self._M_u_star,
                    gamma = GAMMA,
                    n = INTEGRAL_NUMBER)
        
        self._C_blade = C(M_star_l = self._M_l_star,
                    M_star_u = self._M_u_star,
                    gamma = GAMMA,
                    n = INTEGRAL_NUMBER,
                    k_star = self._k_star)
        
        # From this, we can figure out what our shock pressure ratio is for the gas.
        
        self._p_rat = shock_pressure_rat(Q = self._Q_blade,
                                   C = self._C_blade)
        
        # Now that we know our shock pressure ratio, we can now calculate back our M_star_i_max value
        
        self._M_i_star_max = M_i_star_max(p_rat = self._p_rat,
                                          gamma = GAMMA)
        
        # We can back calculate for what this mach number 
        
        self._M_i_max = inv_M_star(gamma = GAMMA,
                                   M_star = self._M_i_star_max)
        
        self._v_i_max = prandtl_meyer(GAMMA, self._M_i_star_max)
        
        return self._M_i_max
        
    def M_u_max(self):
        """
        This function solves for the maximum upper surface mach number inorder to prevent flow seperation
        """
        GAMMA = self._fluid.get_gamma()
        
        self._M_u_star_max = M_star_u_max(M_star_o = self._M_o_star,
                                     gamma = GAMMA)
        
        self._M_u_max = inv_M_star(gamma = GAMMA,
                                   M_star = self._M_u_star_max)
        
        return
        
    def M_l_min(self):
        """
        This function solves for the minimum lower surface mach number to avoid flow seperation of the gas
        """
        GAMMA = self._fluid.get_gamma()
        
        self._M_l_star_min = M_star_l_min(m_star_i = self._M_i_star,
                                     gamma = GAMMA)
        
        self._M_l_min = inv_M_star(gamma = GAMMA,
                                   M_star = self._M_l_star_min)
        
        return