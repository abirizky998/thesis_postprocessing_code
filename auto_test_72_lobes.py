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
r_obs = 50.0
n_observer = 36

# Dense Array Kinematics
n_source = 72       
n_blades = 3        
r_eff = 1.3         
f_fundamental = 107.25

# Target DFT Arrays
amps = np.array([3.39017904, 4.41553483, 4.33133909, 4.05853316, 3.68401838,
                 3.00466103, 2.60251833, 2.20761803, 1.81316128, 1.49356841,
                 1.56660969, 1.11126176])
phases = np.array([-0.53425042,  2.19246152, -1.4319258 ,  1.29567134, -2.37311503,
                    0.30186215,  2.99472323, -0.62744329,  2.08961611, -1.36787151,
                    1.35893209, -2.542671])
num_harmonics = len(amps)

output_dir = f"data_{date}/72Sources_12Harmonics"
os.makedirs(output_dir, exist_ok=True)

# Orienting observer ring to match source coordinates (X-Y plane)
observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')
timestep_sizing(config, M=0.0, P=40, f_s=num_harmonics * f_fundamental, T_sim=1/f_fundamental)

lines = [
    "# Automatically generated 72-Source Dense Phased Array\n",
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
        amp_s = amps[idx] / n_source
        strength_n = (4 * np.pi * r_obs * amp_s) / (rho * omega_n)
        
        # Tyler-Sofrin stagger: Current phase - (harmonic_number * blades * current_angle)
        phase_shift = (phases[idx] - (n * n_blades * theta_s)) % (2 * np.pi)

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
        ])

with open(source, 'w') as f:
    f.writelines(lines)

# Bump MPI processes to handle 864 individual source elements
subprocess.run(["mpirun", "-np", "1", "../../bin/FWH_Sphere"], check=True)

# Post-Process and Move Files
for i in range(1, n_observer + 1):
    obs_str = f"{i:02d}"
    ana_file = f"FWH_observer_time_{obs_str}_validation.dat" 
    dest_file = os.path.join(output_dir, f"ana_spatial_{obs_str}.dat")

    if os.path.exists(ana_file):
        shutil.move(ana_file, dest_file)
        print(f"Success. Analytical output saved to: {dest_file}")
    else:
        print(f"Error: Analytical output {ana_file} not found. Check solver execution.")

print("Spatial Simulation Complete.")