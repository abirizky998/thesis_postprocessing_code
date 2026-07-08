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
n_source = 4
r_eff = 1.5
freq = 100 # Hz
omega = 2 * np.pi * freq

output_dir = f"data_{date}/4lobes-0deg"
os.makedirs(output_dir, exist_ok=True)

observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')
timestep_sizing(config, M=0.0, P=40, f_s=freq, T_sim=1/freq)

lines = [
    "# Automatically generated 3-Blade Spatial Harmonic Cluster\n",
    "# Number of source sets\n",
    f"{n_source}\n"
]

for n in range(n_source):
    x_b = r_eff * np.cos(n * 2 * np.pi / n_source)
    y_b = r_eff * np.sin(n * 2 * np.pi / n_source)

    strength_n = 1.0
    phase_shift = 0 #n * 2 * np.pi / n_source

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega:.6f} {strength_n:.8e} {phase_shift:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z)\n",
        f"{x_b:.6f} {y_b:.6f} 0.0\n",
        "# Source motion type and duration [s]\n",
        "Stationary 1000.0\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

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