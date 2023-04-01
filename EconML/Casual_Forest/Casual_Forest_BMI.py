from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.special
import glob
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from XGBoost import XGBoost

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


def run_mbqip(data_base_dir='/'):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        est = CausalForestDML(
            n_jobs=-1, 
            model_y=RegressionForest(), 
            model_t=RegressionForest(), 
            model_final=RegressionForest()
        )

        est.fit(y,t,X = x,W=None)
        print("#################")
        print(x.shape)
        print(est.ate(x))
        print(est.ate_interval(x))
        print("#################")
    return ans

def main():
    run_mbqip("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/BMI")


if __name__ == '__main__':
    main()