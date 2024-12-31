from turborocket.profiling.Supersonic.supersonic_profile import SupersonicProfile
from turborocket.fluids.ideal_gas import IdealFluid
import numpy as np

if __name__ == "__main__":

    ANGLE_CONVERSION = 180 / np.pi

    target_fluid = IdealFluid(P=20e5, T=816.495, R_gas=1305.355, gamma=1.3 )

    super = SupersonicProfile(
        beta_i=68.34,                  # Relative angles
        beta_o=-68.34,                 # Relative Angles
        M_i = 1.846,                     # Relative Mach Numbers Nominally
        M_o = 1.846,                     # Relative Mach Numbers Nominally
        M_u = 2.45,
        M_l = 1.5,
        m_dot = 0.0184,
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
    
    print("\n-----------------------------------------\n")
    
    
    super.M_u_max()
    super.M_l_min()

    
    print(f"Maximum Possible M_u: {super._M_u_max:.3f}")
    print(f"Current M_u: {super._M_u:.3f}")
    print(f"Margin: {(super._M_u_max - super._M_u)/super._M_u}")

     
    if super._M_u_max < super._M_u:
        raise ValueError(f"Mach number too high on upper surface: {super._M_u} > {super._M_u_max}")

    
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
    
    
    print(f"V_i_max {super._v_i_max*ANGLE_CONVERSION}, v_u: {super._v_u*ANGLE_CONVERSION}, v_l: {super._v_l*ANGLE_CONVERSION}")
    
    if super._M_i_max < super._M_i:
        raise ValueError(f"Mach number exceeded Nominally: {super._M_i} > {super._M_i_max}")

    super.plot_circles(100)

    # Now we get the transition co-ordinates for the lower area
    super.inlet_lower_transition()
    super.inlet_upper_transition()
    super.outlet_lower_transition()
    super.outlet_upper_transition()
    super.straight_line_segments()
    super.generate_blade(100)

    super.plot_transition()

    print(super._g_star)
    print(super._R_l_star - super._R_u_star)

    super.plot_all_shift(100)
    
    super.plot_all_shift_to_scale(100)
