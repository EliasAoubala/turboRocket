from turborocket.profiling.Supersonic.supersonic_profile import SupersonicProfile
from turborocket.fluids.ideal_gas import IdealFluid
import numpy as np

if __name__ == "__main__":

    ANGLE_CONVERSION = 180 / np.pi

    target_fluid = IdealFluid(P=20e5, T=800, R_gas=283, gamma=1.4)

    super = SupersonicProfile(
        beta_i=70,
        beta_o=-70,
        M_i=2,
        M_o=2,
        M_u=4,
        M_l=1.5,
        m_dot=1,
        h=30e-3,
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

    super.plot_circles(100)

    # Now we get the transition co-ordinates for the lower area
    super.inlet_lower_transition()
    super.inlet_upper_transition()
    super.outlet_lower_transition()
    super.outlet_upper_transition()
    super.straight_line_segments()
    super.generate_blade(100)

    # super.plot_transition()

    print(super._g_star)
    print(super._R_l_star - super._R_u_star)

    super.plot_all(100)
