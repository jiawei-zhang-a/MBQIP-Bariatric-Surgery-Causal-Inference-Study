import numpy as np
import glob
from dragonnet.dragonnet import DragonNet
from exdragonnet import EXdragonnet
import torch

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
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        # Check if GPU is available
        model = DragonNet(x.shape[1], epochs = 20)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        est = EXdragonnet(model)
        est.fit(y,t,X = x)
        ans.append((est.ate(x),est.ate_interval(x)))
    return ans