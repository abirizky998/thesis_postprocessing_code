import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
# pandas not required in this script
# %load_ext autoreload
# %autoreload 2

from fwh_analysis import FWHDataset, FWHPlotter

# 1. Define your batch folders
folders = [
    "data_260528/test_cfl_PPP10",
    "data_260528/test_cfl_PPP15",
    "data_260528/test_cfl_PPP20",
    'data_260602/test_cfl_PPP16',
    'data_260602/test_cfl_PPP17',
    'data_260602/test_cfl_PPP18',
    'data_260602/test_cfl_PPP19',
    'data_260602/test_cfl_PPP21',
    'data_260602/test_cfl_PPP22',
    'data_260602/test_cfl_PPP23',
    'data_260602/test_cfl_PPP24',
    'data_260602/test_cfl_PPP25',
    'data_260602/test_cfl_PPP26',
    'data_260602/test_cfl_PPP27',
    'data_260602/test_cfl_PPP28',
    'data_260602/test_cfl_PPP29',
    'data_260602/test_cfl_PPP30',
    'data_260602/test_cfl_PPP31',
    'data_260602/test_cfl_PPP32',
    'data_260602/test_cfl_PPP33',
    'data_260602/test_cfl_PPP34',
    'data_260602/test_cfl_PPP35',

    "data_260528/test_cfl_PPW10",
    "data_260602/test_cfl_PPW11",
    "data_260602/test_cfl_PPW12",
    "data_260602/test_cfl_PPW13",
    "data_260602/test_cfl_PPW14",
    "data_260528/test_cfl_PPW15",
    "data_260602/test_cfl_PPW16",
    "data_260602/test_cfl_PPW17",
    "data_260602/test_cfl_PPW18",
    "data_260602/test_cfl_PPW19",
    "data_260528/test_cfl_PPW20",
    "data_260602/test_cfl_PPW21",
    "data_260602/test_cfl_PPW22",
    "data_260602/test_cfl_PPW23",
    "data_260602/test_cfl_PPW24",
    "data_260602/test_cfl_PPW25",
    "data_260602/test_cfl_PPW26",
    "data_260602/test_cfl_PPW27",
    "data_260602/test_cfl_PPW28",
    "data_260602/test_cfl_PPW29",
    'data_260602/test_cfl_PPW30',
    "data_260602/test_cfl_PPW31",
    "data_260602/test_cfl_PPW32",
    'data_260602/test_cfl_PPW33',
    'data_260602/test_cfl_PPW34',
    'data_260602/test_cfl_PPW35'
]

# 2. Compile everything into one master list
master_data = []

for folder in folders:
    folder_data = FWHDataset.compile_fwh_batch(folder, use_doppler=True)
    master_data.extend(folder_data)
FWHPlotter.plot_3d_parameter_surface(master_data, use_doppler=True, shape_by='elements')