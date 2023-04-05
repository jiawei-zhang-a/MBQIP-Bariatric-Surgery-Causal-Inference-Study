from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.special
from sklearn.ensemble import RandomForestRegressor
import glob
from sklearn.tree import plot_tree
import xgboost as xgb

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
    
    return t.reshape(-1,), y.reshape(-1,)


def run_mbqip(data_base_dir='/'):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        est = CausalForestDML(model_y=RandomForestRegressor(),
                       model_t=RandomForestRegressor(),
                       criterion='mse', n_estimators=1000,
                       min_impurity_decrease=0.001,
                       random_state=123)
        
        est.fit(y,t,X = x,W=None)
        print("Ate of the model is: ")
        print(est.ate(x))
        print("Confidence Interval of the model is:")
        print(est.ate_interval(x))
    return ans

def main():
    PATH = "/scratch/jz4721/dragonnet/"
    run_mbqip(PATH + "dat/mbqip/csv/BMI")


if __name__ == '__main__':
    main()