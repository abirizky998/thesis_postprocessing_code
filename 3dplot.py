import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd
# %load_ext autoreload
# %autoreload 2

from fwh_analysis import FWHDataset, FWHPlotter

# 1. Define your batch folders
folders = [
    "data_260528/test_cfl_PPP10",
    "data_260528/test_cfl_PPP15",
    "data_260528/test_cfl_PPP20",
    "data_260528/test_cfl_PPW10",
    "data_260528/test_cfl_PPW15",
    "data_260528/test_cfl_PPW20"
]

# 2. Compile everything into one master list
master_data = []
for folder in folders:
    folder_data = FWHDataset.compile_fwh_batch(folder, use_doppler=True)
    master_data.extend(folder_data)
FWHPlotter.plot_3d_parameter_surface(master_data, use_doppler=True, shape_by='elements')