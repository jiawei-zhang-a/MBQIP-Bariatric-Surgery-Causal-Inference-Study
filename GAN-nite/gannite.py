import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from ganite import Ganite
from ganite.datasets import load
from ganite.utils.metrics import sqrt_PEHE_with_diff

from sklearn.model_selection import train_test_split
import numpy as np
import scipy.special
import glob
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

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
    return t.reshape(-1, 1), y

def run_mbqip(data_base_dir='/', output_dir='~/result/'):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    for idx, simulation_file in enumerate(simulation_files):

        simulation_output_dir = os.path.join(output_dir, str(idx))

        os.makedirs(simulation_output_dir, exist_ok=True)

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=5000, random_state=1) #5000 per 50000 cases, 10%
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]

        train_output_dir = simulation_output_dir
        os.makedirs(train_output_dir, exist_ok=True)

        model = Ganite(x_train, t_train, y_train, num_iterations=500)

        pred = model(x_test).numpy()

        print(idx)
        print(np.mean(pred))
        print("#################")

def main():
    #print("\nBPD_DS")    
    #run_mbqip("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/BPD_DS","/Users/jiaweizhang/med/dragonnet/result")
    print("\nRYGB")
    run_mbqip("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/RYGB","/Users/jiaweizhang/med/dragonnet/result")
    print("\nSADI-S")
    run_mbqip("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/SADI-S","/Users/jiaweizhang/med/dragonnet/result")

if __name__ == '__main__':
    main()