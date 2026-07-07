from auto_test import *
import numpy as np
import os
import subprocess

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = current_date()


c = 340.0
rho = 1.2
r_obs = 7.28
n_observer = 36

T_period = 0.009324
base_freq = 1.0 / T_period

amps = np.array([14.802918  ,  9.39540517,  2.79875548,  0.74677086,
        0.38365698,  0.66020024,  0.55781105,  0.23309134,  0.20060146,
        0.27991645])

phases = np.array([ -0.57408072,  2.01884969, -1.88204196, -1.04759197,
        1.8760436 , -0.33453731,  2.47233509, -1.40745429,  0.3626003 ,
        2.61284136])

output_dir = f"data_{date}/WoanGregorekValidation"
os.makedirs(output_dir, exist_ok=True)

observer_setup(observer, D_obs=r_obs, n=n_observer, normal='x')
timestep_sizing(config, M=0.0, P=40.0, f_s=base_freq, T_sim=T_period)

lines = [
    "# Automatically generated harmonic cluster\n",
    "# Number of source sets\n",
    f"{len(amps)}\n"
]

for idx in range(len(amps)):
    n = idx + 1
    omega_n = 2 * np.pi * n * base_freq
    strength_n = (4 * np.pi * r_obs * amps[idx]) / (rho * omega_n)
    delta_phase = omega_n * (r_obs / c) # correct phase for retarded time
    phase_n = phases[idx] + delta_phase 

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega_n:.6f} {strength_n:.8e} {phase_n:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z) - Stationary at origin\n",
        "0.000000 0.000000 0.000000\n",
        "# Source motion type and duration [s]\n",
        "Stationary 1000.0\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

subprocess.run(["mpirun", "-np", "6", "../../bin/FWH_Sphere"], check=True)

for n in range(1, n_observer+1):
    obs_str = f"{n:02d}"
    ana_file = f"FWH_observer_time_{obs_str}_validation.dat"
    dest_file = os.path.join(output_dir, f"ana_{obs_str}.dat")

    if os.path.exists(ana_file):
        shutil.move(ana_file, dest_file)
        print(f"Success. Analytical output saved to: {dest_file}")
    else:
        print("Error: Analytical output not found. Check solver execution.")

print("Run Complete.")