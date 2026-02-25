import numpy as np

from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim
from fipy.tools import numerix as nx
import scipy as sp



I_0 = 2.5e15 # [s^-1] 0.5 muAmps
E_0 = 4e6  # [MeV]
r = 0.5e-2 # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0, dim = 3)

Lx_Al = 50e-6
Ly_Al = 2e-2
Lz_Al = Ly_Al

Lx_Ol = 50e-6
Ly_Ol = Ly_Al
Lz_Ol = Lz_Al

Ly = Ly_Al
Lz = Lz_Al
dx = 1e-6
dy = 0.5e-3
dz = dy


def C_Al(T):
    M_Al = 0.02698  # kg/mol
    # Prevent T from going to extreme values that break polynomials
    T_safe = nx.clip(T, 0, 2200)

    Ts_low = np.array([1, 2, 3, 4, 6, 8, 10, 15, 20, 25,
              30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
              180, 200, 220, 240, 260, 280, 300])
    Cp_low = np.array([0.000051, 0.000108, 0.000176, 0.000261,
              0.00050, 0.00088, 0.0014, 0.0040, 0.0089, 0.0175,
              0.0315, 0.0515, 0.0775, 0.142, 0.214, 0.287, 0.357, 0.422,
              0.481, 0.580, 0.654, 0.713, 0.760, 0.797, 0.826, 0.849,
              0.869, 0.886, 0.902]) * 1e3

    f_low = sp.interpolate.PchipInterpolator(Ts_low, Cp_low )
    cond1 = (T_safe < 300)
    cond2 = (T_safe >= 300) & (T_safe < 933)
    cond3 = (T_safe >= 933)

    val1 = f_low(T_safe)

    A = 28.08920
    B = -5.414849
    C = 9.560423
    D = 3.427370
    E = -0.277375
    val2 = A + B*T_safe + C*T_safe**2 + D*T_safe**3 + E/(T_safe**2)
    val2 /= M_Al

    A = 31.75104
    B = 3.935826e-8
    C = -1.786515e-8
    D = 2.694171e-9
    E = 5.480037e-9
    val3 = A + B * T_safe + C * T_safe ** 2 + D * T_safe ** 3 + E / (T_safe ** 2)
    val3 /= M_Al
    Cp = cond1 * val1 + cond2 * val2 + cond3 * val3
    # Return J/kg*K and ensure it's always positive
    return nx.maximum(Cp, 0.0)

# https://link.springer.com/article/10.1007/BF00514474
# https://inis.iaea.org/records/n4gzz-mpp69
def k_Al(T):
    # Prevent T from going to extreme values that break polynomials
    T_safe = nx.clip(T, 0, 2200)

    Ts_solid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                123.2, 150, 173.2, 200, 223.2, 250, 273.2, 298.2, 300, 323.2,
                350, 373.2, 400, 473.2, 500, 573.2, 600, 673.2, 700, 773.2,
                800, 873.2, 900, 933.52]
    Ks_solid = [0, 41.1, 81.8, 121, 157, 188, 213, 229, 237, 239,
                235, 226, 214, 201, 189, 176, 138, 117, 75.2, 49.5,
                33.8, 24.0, 17.7, 13.5, 8.50, 5.85, 4.32, 3.42, 3.02,
                2.62, 2.48, 2.41, 2.37, 2.35, 2.35, 2.35,
                2.36, 2.37, 2.37, 2.39, 2.40, 2.40, 2.40, 2.37, 2.36, 2.33,
                2.31, 2.26, 2.25, 2.19, 2.18, 2.12, 2.10, 2.08]
    Ts_liquid = [933.52, 973.2, 1000, 1073.2, 1100, 1173.2, 1200, 1273.2,
                 1300, 1372.2, 1400, 1473.2, 1500, 1573.2, 1600, 1673.2, 1700,
                 1773.2, 1800, 1873.2, 1900, 1973.2, 2000, 2073.2, 2173.2, 2200]
    Ks_liquid = [0.907, 0.921, 0.390, 0.955, 0.964, 0.986, 0.994, 1.01, 1.02, 1.04, 1.05, 1.07,
                 1.07, 1.08, 1.09, 1.10, 1.11, 1.11, 1.12, 1.13, 1.13, 1.14, 1.14, 1.14, 1.15, 1.15]

    f_sol = sp.interpolate.PchipInterpolator(Ts_solid, Ks_solid)
    f_liq = sp.interpolate.PchipInterpolator(Ts_liquid, Ks_liquid)
    cond1 = (T_safe < 933.52)
    cond2 = (T_safe >= 933.52)
    val1 = f_sol(T_safe)
    val2 = f_liq(T_safe)

    k_combined = (cond1 * val1 + cond2 * val2) * 1e2
    # Final safety floor (k should never be 0 or negative)
    return nx.maximum(k_combined, 1.0)

# Define aluminum
Z_Al = 13
A_Al = 27 #g/mol
Al = Atom('Al', Z_Al, A_Al, 1)
rho_Al = 2700 # kg/m^3
x0_Al = 0

aluminum = Medium(rho_Al, C_Al, k_Al,
              Al, Lx_Al, Ly_Al, Lz_Al,
              "Al//Hydrogen in Aluminum.txt", x0=Lx_Ol, name = 'Aluminum')

def C_olivine(T):
    M = 0.14069  # kg/mol
    T_safe = nx.clip(T, 200, 3000)

    a, b, c, d, e = 87.36, 8.717e-2, -3.699e6, 8.436e2, -2.237e-5

    Cp = a + b * T_safe + c / (T_safe ** 2) + d / (T_safe ** 0.5) + e * (T_safe ** 2)
    return nx.maximum(Cp / M, 10.0)

def k_olivine(T):
    # ensure float and keep T in a numerically safe range
    Tf = nx.clip(1.0 * T, 160, 6000)
    k = 1.7 * nx.exp(-(Tf - 298.0) / 300.0)

    return nx.clip(k, 1e-4, 50.0)

rho_olivine = 3.3e3  # kg/m^3

# Parameters for the medium and values
Z_Mg = 12
A_Mg = 24.305  # g/mol
Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

Z_Fe = 26
A_Fe = 55.845  # g/mol
Fe = Atom('Fe', Z_Fe, A_Fe, 0.2222)

Z_Si = 14
A_Si = 28.086  # g/mol
Si = Atom('Si', Z_Si, A_Si, 0.1111)

Z_O = 8
A_O = 15.999  # g/mol
O = Atom('O', Z_O, A_O, 0.4444)


olivine = Medium(rho_olivine, C_olivine, k_olivine,
                [Mg, Fe, Si, O],
                Lx_Ol, Ly_Ol, Lz_Ol,
                "Olivine//Hydrogen in Olivine.txt",
                x0 = 0, name = 'Olivine')

# Actual Sim
mediums = [olivine, aluminum]

BC = BoundaryConditions(['BBR', 'BBR', 'Fixed', 'BBR', 'BBR', 'BBR'], 298, 298, eps= 0.05)
ts, Ts = sim.heateq_solid_3d(beam, mediums, BC, Ly, Lz,10000, dt =  1e-6, alpha = 0, beta = 0,
                             dx=dx, dy=dy, dz=dz, view=True, dt_ramp=2, dT_target = 500, x_units='µm', y_units = 'cm', z_units='cm')