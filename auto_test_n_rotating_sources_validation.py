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
n_observer = 72
mach_number = 0.5

# Dense Array Kinematics
n_source = 2
f_ratio = 4
# n_blades = 3       
r_eff = 1.3

f_rot = (mach_number * c / r_eff) / (2 * np.pi)
f_source = f_ratio * f_rot

output_dir = f"data_{date}/{n_source}Sources_M{mach_number}_fratio{f_ratio}"
os.makedirs(output_dir, exist_ok=True)

# Orienting observer ring to match source coordinates (X-Y plane)
observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')

lines = [
    f"# Automatically generated {n_source}-Source Dense Phased Array\n",
    "# Number of source sets\n",
    f"{n_source}\n"
]

for s in range(n_source):
    theta_s = s * (2 * np.pi / n_source)
    x_b = r_eff * np.cos(theta_s)
    y_b = r_eff * np.sin(theta_s)


    omega_n = 2 * np.pi * f_source    
    phase_shift = (- (theta_s / n_source))
    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega_n:.6f} 1.0 {phase_shift:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z)\n",
        f"{x_b:.6f} {y_b:.6f} 0.000000\n",
        "# Source motion type and duration [s]\n",
        "General 1000.0\n"
        "# Source Motion Definition\n",
        f"0.0 0.0 0.0 0.0 0.0 {f_rot * 2 * np.pi}\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

x_values = [100]
p_values = [100]
T_sim = 1 / f_rot

total_runs = len(x_values) * len(p_values)
current_run = 1

for p in p_values:
    timestep_sizing(config, M=0.0, P=p, f_s=f_source, T_sim=T_sim)
    processes = 1
    print(f"--- Running Case {current_run} / {total_runs} | P = {p} | Timestep Count = {T_sim * p * f_source}")
    # subprocess.run(["mpirun", "-np", str(processes), '../../bin/FWH_Sphere'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # time.sleep(10)
    subprocess.run(['../../bin/FWH_Sphere'], check=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(1, n_observer + 1):
        obs_str = f"{i:02d}"

        ana_file = f"FWH_observer_time_{obs_str}_validation.dat"
        ana_dest_file = os.path.join(output_dir, f"rotating_{obs_str}.dat")

        if p==p_values[-1]:
            shutil.move(ana_file, ana_dest_file)
        elif os.path.exists(ana_file):
            os.remove(ana_file)

        # num_file = f"FWH_observer_time_{obs_str}_RT.dat"
        # num_dest_file = os.path.join(output_dir, f"X{int(x)}_P{int(p)}_{obs_str}.dat")

        # if os.path.exists(num_file):
        #     shutil.move(num_file, num_dest_file)

    current_run += 1

print("Spatial Simulation Complete.")