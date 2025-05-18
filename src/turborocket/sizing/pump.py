"""This file contains the authors objects for the sizing of liquid propellant pumps"""

#from turborocket.fluids.fluids import IncompressibleFluid
import numpy as np
import matplotlib.pyplot as plt


class Barske:
    #This object represents a Partial Emission Pump of a Barske Style for Liquid Propellants

    def __init__(self, dp_metric: float, M_metric: float, N: float, W_metric: float, v_metric: float, alpha: float):

        "Converted values"
        self.H_prime = 3.28084 * ((dp_metric*100000)/(W_metric*9.81)) #ft
        self.M = M_metric * 2.20462 # lbs/s
        self.W = W_metric * 0.062428 # lbs/ft3
        self.Q = self.M/self.W # ft3/s
        self.G = self.Q * 448.8309375 # GPM
        self.v = v_metric * 10.764 # ft2/s
        self.alpha = alpha
        self.N = N # rpm
        #self.N_s = (self.N * (self.G) ** 0.5)/(); " rpm, GPM, ft"

        self.size_pump()

    def size_pump(self): #Generates geometric parameters for the pump
        
        #Inlet Conditions
        self.v_0 = 12 #ft/s, range of 5 - 12
        self.d_0 = (183.5 * (self.Q/self.v_0)) ** 0.5 # inch
        self.d_1 = self.d_0 # inch

        #Velocities and Ideal Heads
        self.u_1 = self.N * self.d_1 * 0.00435 #ft/s
        self.u_2 = (((self.H_prime*2*32.2) + (self.u_1 ** 2))/2) ** 0.5 #ft/s

        self.H_prime_d = (self.u_2**2)/(2*32.2) # ft
        self.H_prime_s = (self.u_2 ** 2 - self.u_1 ** 2)/(2*32.2) # ft

        #Ideal pressures
        self.p_prime_s = self.W*(self.u_2**2 - self.u_1**2)/9300 # psi
        self.p_prime_d = (self.W*self.u_2**2)/9300 # psi
        self.p_prime = self.p_prime_s + self.p_prime_d # psi

        #Blade geometry
        self.d_2 = self.u_2/(0.00435*self.N)
        self.l_1 = self.d_1*0.25
        self.l_2 = self.l_1*(self.d_1/self.d_2)

        #Diffuser geometry
        self.v_3 = 0.85 * self.u_2 # ft/s
        self.a_3 = (144*self.M)/(1*self.v_3*self.W) # in^2
        self.a_4 = self.a_3 * 3.5 # in^2
        self.d_3 = ((4/np.pi)*self.a_3) ** 0.5 # inch
        self.d_4 = ((4/np.pi)*self.a_4) ** 0.5 # inch
        self.delta = 10 # degree
        self.l_3 =(self.d_4-self.d_3)/np.tan(np.deg2rad(self.delta)) # inch
        self.v_4 = (self.Q*144)/self.a_4 # ft/s

    def get_pump_performance(self, psi: float):
        self.psi = psi
        self.H = self.H_prime_s + (self.psi * self.H_prime_d)
        self.p = self.p_prime_s + (self.psi * self.p_prime_d)

        self.H_metric = self.H * 0.3048 # m
        self.p_metric = self.p * 0.0689476  # bar

        self.specific_speed = (self.N*(self.G)**0.5)/(self.H**0.75)

        return self.H_metric, self.p_metric


    def get_instantaneous_efficiency(self):
        self.P_prime = (self.H_prime * self.M)/550
        self.P = (self.H * self.M)/550
        self.P_F = 0.6e-6 * (self.W) * (self.v**0.2) * ((self.N/1000)**2.8) * (((2 * (1/np.sin(np.deg2rad(self.alpha)))) * self.d_2**4.6) + (self.l_1 * 9.2 * self.d_1 ** 3.6))
        self.eta = 1/((self.p_prime/self.p) + (self.P_F/self.P))

    def get_sweep_efficiency(self, N_start: float, N_end: float, generate_graph: bool):
        #pump performance must be run first
        rpm_step = 1000
        rpm_values = np.arange(N_start, N_end + rpm_step, rpm_step)
        
        sweep_u_1 = []
        sweep_u_2 = []
        sweep_p = []
        sweep_p_prime = []
        sweep_P_F = []
        sweep_H = []
        sweep_Q = []
        sweep_P = []
        sweep_eta = []

        for rpm in rpm_values:
            new_u_1 = 0.00435 * self.d_1 * rpm; sweep_u_1.append(new_u_1)
            new_u_2 = 0.00435 * self.d_2 * rpm; sweep_u_2.append(new_u_2)
            new_p = (self.W/9300) * ((1 + self.psi) * new_u_2 ** 2 - new_u_1 ** 2); sweep_p.append(new_p)
            new_p_prime = (self.W / 9300) * ((2 * new_u_2 ** 2) - new_u_1 ** 2); sweep_p_prime.append(new_p_prime)
            new_P_F = 0.6e-6 * (self.W) * (self.v**0.2) * ((rpm/1000)**2.8) * (((2 * (1/np.sin(np.deg2rad(self.alpha)))) * self.d_2**4.6) + (self.l_1 * 9.2 * self.d_1 ** 3.6))
            sweep_P_F.append(new_P_F)
            new_H = (144*new_p)/self.W; sweep_H.append(new_H)
            new_Q = (((self.specific_speed*new_H**0.75)/rpm)**2) * 0.002228010407594; sweep_Q.append(new_Q)
            new_P = 0.262 * new_p * new_Q; sweep_P.append(new_P)
            new_eta = 1/((new_p_prime/new_p) + (new_P_F/new_P)); sweep_eta.append(new_eta)


        if generate_graph:
            plt.figure(figsize=(8, 5))
            plt.plot(rpm_values, sweep_eta, marker='o', linestyle='-', color='blue', label='Pump Efficiency')
            plt.xlabel('RPM')
            plt.ylabel('Efficiency (%)')
            plt.title('Pump Efficiency vs RPM')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        #return rpm_values, sweep_eta
        return sweep_eta


#EXAMPLE CODE
pump = Barske(35, 0.25, 24000, 786, 1.462e-6, 80) #initialise pump model
pump.get_pump_performance(0.2) #generate performance figures for given psi value
pump_efficiency = pump.get_sweep_efficiency(1000, 50000, True) #generate efficiency plot for given psi value