from turborocket.profiling.Supersonic.supersonic_profile import SupersonicProfile
from turborocket.fluids.ideal_gas import IdealFluid
import numpy as np

if __name__ == "__main__":

    ANGLE_CONVERSION = 180 / np.pi

    target_fluid = IdealFluid(P=20e5, T=816.495, R_gas=1305.355, gamma=1.148 )

    super = SupersonicProfile(
        beta_i=57.194,                  # Relative angles
        beta_o=-57.194,                 # Relative Angles
        M_i = 2.08,                     # Relative Mach Numbers
        M_o = 2.08,                     # Relative Mach Numbers
        M_u = 3,
        M_l = 2,
        m_dot = 0.0092,
        h = 15e-3,
        fluid=target_fluid,
    )

    super.prantl_meyer()

    super.circular_section()

    print(f"v_i {super._v_i}")
    print(f"v_o {super._v_o}")
    print(f"v_l {super._v_l}")
    print(f"v_u {super._v_u}")

    print(f"alpha_u_o {super._alpha_u_o} rad")
    print(f"alpha_u_o {super._alpha_u_o*ANGLE_CONVERSION} Degrees")
    print("\n")

    print(f"alpha_u_i {super._alpha_u_i} rad")
    print(f"alpha_u_i {super._alpha_u_i*ANGLE_CONVERSION} Degrees")
    print("\n")

    print(f"alpha_l_o {super._alpha_l_o} rad")
    print(f"alpha_l_o {super._alpha_l_o*ANGLE_CONVERSION} Degrees")
    print("\n")

    print(f"alpha_l_i {super._alpha_l_i} rad")
    print(f"alpha_l_i {super._alpha_l_i*ANGLE_CONVERSION} Degrees")
    print("\n")

    # Now we go onto get our non-dimentionalised sonic radii

    super.r_star()

    print(f"R_l_star {super._R_l_star}")
    print(f"R_u_star {super._R_u_star}")
    print(f"r_star {super._r_star_a}")
    print(f"R_l {super._R_l_star*super._r_star_a} m")
    print(f"R_u {super._R_u_star*super._r_star_a} m")
    
    # We now can confirm if the turbine can be started or not

    # Assessing flow serperation

    
    print("\n----------------------------------------- \n")
    
    print("FLOW Serperation")
    
    print("\n-----------------------------------------")
    
    
    super.M_u_max()
    super.M_l_min()
    
    
    print(f"Maximum Possible M_u: {super._M_u_max:.3f}")
    print(f"Current M_u: {super._M_u:.3f}")
    print(f"Margin: {(super._M_u_max - super._M_u)/super._M_u}")
    
    print("\n----------------------------------------- \n")
     
    print(f"Minimum Possible M_l: {super._M_l_min:.3f}")
    print(f"Curren M_l: {super._M_l:.3f}")
    print(f"Margin: {(super._M_l - super._M_l_min)/super._M_l}")
    
        
    if super._M_l < super._M_l_min:
        raise ValueError(f"Mach number too low on lower surface: {super._M_l} < {super._M_l_min}")
    
    # Assessing startability
    
    print("\n----------------------------------------- \n")
    
    print("STARTABILITY")
    
    print("\n----------------------------------------- \n")
    
    super.M_i_max()
    
    print(f"Maximum M_i: {super._M_i_max:.3f}")
    print(f"Nominal M_i: {super._M_i:.3f}")
    print(f"Startup M_i: {super._M_i_start:.3f}")
    
    if super._M_i_max < super._M_i:
        raise ValueError(f"Mach number exceeded: {super._M_i} > {super._M_i_max}")

    # super.plot_circles(100)

    # # Now we get the transition co-ordinates for the lower area
    # super.inlet_lower_transition()
    # super.inlet_upper_transition()
    # super.outlet_lower_transition()
    # super.outlet_upper_transition()
    # super.straight_line_segments()
    # super.generate_blade(100)

    # super.plot_transition()

    # print(super._g_star)
    # print(super._R_l_star - super._R_u_star)

    # super.plot_all_shift(100)
    
    # super.plot_all_shift_to_scale(100)
