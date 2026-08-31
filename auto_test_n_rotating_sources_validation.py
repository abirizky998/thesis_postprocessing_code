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

c = 340.0
rho = 1.2
r_obs = 7.28
n_observer = 360
mach_number = [0.5, 0.8]

n_source = [1, 2, 4, 8, 16]
f_ratio = [2, 3, 5, 8]
r_eff = 1.3

observer_setup_scatter(observer, D_obs=r_obs, n_observer=n_observer)
total_runs = len(n_source) * len(mach_number) * len(f_ratio)
current_run = 1

for n in n_source:
    for M in mach_number:
        for nu in f_ratio:
            f_rot = (M * c / r_eff) / (2 * np.pi)
            f_source = nu * f_rot
            T_sim = 1 / f_rot

            output_dir = f"data_{date}/{n}Sources_M{M}_fratio{nu}_singlecount"
            os.makedirs(output_dir, exist_ok=True)

            lines = [
                f"# Automatically generated {n}-Source\n",
                "# Number of source sets\n",
                f"{n}\n"
            ]

            for s in range(n):
                theta_s = s * (2 * np.pi / n)
                x_b = r_eff * np.cos(theta_s)
                y_b = r_eff * np.sin(theta_s)

                omega_n = 2 * np.pi * f_source    
                phase_shift = 0.0

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
                    "General 1000.0\n",
                    "# Source Motion Definition\n",
                    f"0.0 0.0 0.0 0.0 0.0 {f_rot * 2 * np.pi}\n"
                ])

            with open(source, 'w') as f:
                f.writelines(lines)

            timestep_sizing(config, M=0.0, P=100, f_s=f_source, T_sim=T_sim)
            processes = 1
            print(f"--- Running Case {current_run} / {total_runs} | Timestep Count = {T_sim * 100 * f_source}")
            subprocess.run(['../bin/FWH_Sphere'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # for i in range(1, n_observer + 1):
            #     obs_str = f"{i:02d}"

                # ana_file = f"FWH_observer_time_{obs_str}_validation.dat"
            ana_file = "FWH_observer_time_01_validation.dat"
            slice_scatter_output(ana_file, output_dir)
            # ana_dest_file = os.path.join(output_dir, f"rotating_{obs_str}.dat")

            # shutil.move(ana_file, ana_dest_file)
            os.remove(ana_file)

            current_run += 1

print("Sweep Complete.")