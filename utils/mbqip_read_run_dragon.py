import numpy as np
import glob
from dragonnet.dragonnet import DragonNet
from exdragonnet import EXdragonnet
import torch
from torch import cuda

def load_and_format_covariates_mbqip(file_path):

    data = np.loadtxt(file_path, delimiter=',')

    contfeats = [1]
    binfeats = [i for i in range(2,49) if i not in contfeats]

    perm = contfeats + binfeats
    x = data[:, perm]
    return x

def load_all_other_crap(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    perm = 49
    t, y = data[:, 50], data[:,perm][:, None]
    
    return  t.reshape(-1,), y.reshape(-1,)

def run_mbqip(data_base_dir):
    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []

    # Check if GPU is available
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    for idx, simulation_file in enumerate(simulation_files):
        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)

        # Create the model and move it to the appropriate device
        model = DragonNet(x.shape[1])
        model = model.to(device)  # This moves the model to the GPU if available

        # Assuming that your EXdragonnet is designed to work with DragonNet
        # and accepts it as a parameter, then it should also make sure to move
        # everything to the appropriate device (either GPU or CPU).
        est = EXdragonnet(model)
        est.fit(y, t, X=x)

        # Make sure to also move any data you're using for prediction or evaluation
        # to the same device as your model.
        x_device = x.to(device) if isinstance(x, torch.Tensor) else torch.Tensor(x).to(device)
        ans.append((est.ate(x_device), est.ate_interval(x_device)))

    return ans
