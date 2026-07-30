from auto_test import *
import numpy as np
import os
import subprocess
import shutil
import time

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = current_date()

# Physical Constants
c = 340.0
rho = 1.2
r_obs = 7.28
n_observer = 36

# Dense Array Kinematics
n_source = 18
n_blades = 3       
r_eff = 1.3         
f_fundamental = 107.25
processes = 32


# Target DFT Arrays
amps = np.array([3.39017904, 4.41553483, 4.33133909])
phases = np.array([-0.53425042,  2.19246152, -1.4319258 ])
num_harmonics = len(amps)

output_dir = f"data_{date}/{n_source}Sources_{num_harmonics}Harmonics_xpsweep"
os.makedirs(output_dir, exist_ok=True)

# Orienting observer ring to match source coordinates (X-Y plane)
observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')

lines = [
    f"# Automatically generated {n_source}-Source Dense Phased Array\n",
    "# Number of source sets\n",
    f"{n_source * num_harmonics}\n"
]

for s in range(n_source):
    theta_s = s * (2 * np.pi / n_source)
    x_b = r_eff * np.cos(theta_s)
    y_b = r_eff * np.sin(theta_s)

    for idx in range(num_harmonics):
        n = idx + 1
        omega_n = 2 * np.pi * n * f_fundamental
        
        amp_s = amps[idx] # / (3*n_source)
        strength_n = (4 * np.pi * r_obs * amp_s) / (omega_n)
        
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
            # "# Source Motion Definition\n",
            # f"{ux} 0.0 0.0 0.0 0.0 0.0\n"
        ])

with open(source, 'w') as f:
    f.writelines(lines)

x_values = [80, 100]
p_values = [20, 40, 60, 80, 100]
T_sim = 1 / f_fundamental

total_runs = len(x_values) * len(p_values)
current_run = 1

for x in x_values:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c / f_fundamental, R=1.3 * r_eff)
    for p in p_values:
        timestep_sizing(config, M=0.0, P=p, f_s=f_fundamental, T_sim=T_sim)

        print(f"--- Running Case {current_run} / {total_runs} | X = {x} P = {p} | Grid Count = {n**2} Timestep Count = {T_sim * p * f_fundamental}")
        subprocess.run(["mpirun", "-np", str(processes), '../bin/FWH_Sphere'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(10)

        for i in range(1, n_observer + 1):
            obs_str = f"{i:02d}"

            ana_file = f"FWH_observer_time_{obs_str}_validation.dat"
            ana_dest_file = os.path.join(output_dir, f"ana_spatial_{obs_str}.dat")

            if p==p_values[-1]:
                shutil.move(ana_file, ana_dest_file)
            elif os.path.exists(ana_file):
                os.remove(ana_file)

            num_file = f"FWH_observer_time_{obs_str}_RT.dat"
            num_dest_file = os.path.join(output_dir, f"X{int(x)}_P{int(p)}_{obs_str}.dat")

            if os.path.exists(num_file):
                shutil.move(num_file, num_dest_file)

        current_run += 1

print("Spatial Simulation Complete.")