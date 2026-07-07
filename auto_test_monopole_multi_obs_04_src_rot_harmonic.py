from auto_test import *

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'

date = current_date()
M = 0.0 # not moving in translation
m_str = f"{int(M * 100):02d}"


c = 340.0 
f_rot_x = 100 # Hz
Rx = 2 * np.pi * f_rot_x # rotating frequency of the "rotor"
T_sim = 2 / f_rot_x # two period of rotation

harmonic_mult = [4, 8, 16]

for i in harmonic_mult:
    f_source = f_rot_x * i
    # T_sim = 2 / f_source
    
    omega = 2 * np.pi * f_source # harmonic frequency of the monopole source(s)
    lambda_s = c / f_source
    R = lambda_s 

    n_source = 4

    ux = source_speed_omega(config_path=source, M=M, speed_of_sound=c, omega=omega,
                            strength=1.0, Omega_rot=Rx, delta=5e-1, n_sources=n_source)

    output_dir = f"data_{date}/rot_mono_{n_source}src_{m_str}M_{int(f_rot_x)}Hz"
    os.makedirs(output_dir, exist_ok=True)

    N = grid_sizing(config_path=config, M=M, X=20.0, lambda_s=lambda_s, R=R) # doens't matter
    dt = timestep_sizing(config_path=config, M=M, P=40.0, f_s=f_source, T_sim=T_sim) # make sure f_s is set to highest system frequ, frot

    n_observer = 12

    observer_distance = 50.0
    observer_setup(observer, D_obs=observer_distance, n=n_observer, normal = 'x')

    subprocess.run(["mpirun", "-np", "8", "../bin/FWH_Sphere"], check=True)

    for n in range(1, n_observer + 1):

        obs_str = f"{n:02d}"
        ana_file = f"FWH_observer_time_{obs_str}_validation.dat"

        dest_file = os.path.join(output_dir, f"ana_rot_{int(f_source)}Hz_obs{obs_str}.dat")

        if os.path.exists(ana_file):
            shutil.move(ana_file, dest_file)
            print(f"Success. Analytical output saved to: {dest_file}")
        else:
            print("Error: Analytical output not found. Check solver execution.")

print("Batch Complete.")
