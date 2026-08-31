from auto_test import *
import numpy as np
import os
import subprocess
import shutil

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = current_date()

# Physical Constants
c = 340.0
rho = 1.2
r_obs = 7.28
n_observer = 72

# Dense Array Kinematics
n_blades = 3       
r_eff = 1.3   
f_fundamental = 107.25
Mr = 0

# Target DFT Arrays
amps = np.array([2.24403691, 2.51785082, 2.82507509, 2.51785082, 2.11850745,
       1.78250188, 1.49978842, 1.33668784, 1.12468265, 0.94630252,
       0.75167481, 0.63245553, 0.53214501, 0.47427474, 0.37672982,
       0.31697864, 0.26670429, 0.22440369, 0.18881218, 0.15886565,
       0.13366878, 0.11246827, 0.10023745, 0.0843393 , 0.07096268,
       0.06324555, 0.05636766, 0.05023773, 0.04477442, 0.03990525,
       0.03556559, 0.0168279 ])
phases = np.array([-0.52553536,  2.15796003, -1.44996874,  1.24464607, -2.37549141,
        0.30196642,  2.96014376, -0.66065072,  2.00782602, -1.59883573,
        1.06966977, -2.59377759,  0.07278902,  2.75945052, -0.8349312 ,
        1.79199056, -1.78825856,  0.86393839, -2.74948072,  0.04738869,
        2.63735093, -0.97903073,  1.8146542, -1.87510113,  0.58388223,
       -3.11710788, -0.1341826 ,  2.16487423, -0.93496501,  1.44772272,
       -2.89526461,  0.61418295])
num_harmonics = len(amps)
n_source = num_harmonics * 6

output_dir = f"data_{date}/{n_source}Sources_{num_harmonics}Harmonics_SinglePeriod_Y_Observers"
os.makedirs(output_dir, exist_ok=True)

# Orienting observer ring to match source coordinates (X-Y plane)
observer_setup(observer, D_obs=-r_obs, n=n_observer, normal='y')

lines = [
    f"# Automatically generated {n_source}-Source Dense Phased Array\n",
    "# Number of source sets\n",
    f"{n_source * num_harmonics}\n"
]

for s in range(n_source):
    # Full 360-degree spatial sweep
    theta_s = s * (2 * np.pi / n_source)
    x_b = r_eff * np.cos(theta_s)
    y_b = r_eff * np.sin(theta_s)

    # Superimpose all 12 harmonics at this spatial coordinate
    for idx in range(num_harmonics):
        n = idx + 1
        omega_n = 2 * np.pi * n * f_fundamental
        
        # Distribute the target pressure amplitude evenly across the ring
        amp_s = amps[idx] / (n_blades)
        strength_n = (4 * np.pi * r_obs * amp_s  * (1 - Mr)) / (omega_n)
        
        # Tyler-Sofrin stagger: Current phase - (harmonic_number * blades * current_angle)
        phase_shift = (phases[idx] - (n * n_blades * theta_s)) #% (2 * np.pi)

        lines.extend([
            "# Source type\n",
            "Monopole Harmonic\n",
            "# Source frequency [rad/s], source strength, and source phase [rad]\n",
            f"{omega_n:.6f} {strength_n:.8e} {phase_shift:.6f}\n",
            "# Source definition\n",
            "Point\n",
            "# Source initial location [m] (x, y, z)\n",
            f"{x_b:.6f} {y_b:.6f} 0.000000\n",
            "# Source motion type and duration [s]\n",
            "Stationary 1000.0\n"
            # "# Source Motion Definition\n",
            # f"{ux} 0.0 0.0 0.0 0.0 0.0\n"
        ])

with open(source, 'w') as f:
    f.writelines(lines)

# P_values = [10, 20, 30, 40, 50, 60, 70, 80]
# for p in P_values:
timestep_sizing(config, M=0.0, P=100, f_s=num_harmonics * f_fundamental, T_sim=(1)/(f_fundamental))

# Bump MPI processes to handle 864 individual source elements
subprocess.run(["mpirun", "-np", "1", "../bin/FWH_Sphere"], check=True)

# Post-Process and Move Files
for i in range(1, n_observer + 1):
    obs_str = f"{i:02d}"
    ana_file = f"FWH_observer_time_{obs_str}_validation.dat" 
    dest_file = os.path.join(output_dir, f"analytical_{obs_str}.dat")

    if os.path.exists(ana_file):
        shutil.move(ana_file, dest_file)
        print(f"Success. Analytical output saved to: {dest_file}")
    else:
        print(f"Error: Analytical output {ana_file} not found. Check solver execution.")

print("Spatial Simulation Complete.")