from econml.orf import DROrthoForest
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from econml.sklearn_extensions.linear_model import WeightedLassoCV
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.special
import glob
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os


f = open('result.txt', 'w')

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
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=5000, random_state=1) #5000 per 50000 cases, 10%
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        t_train, t_test = t[train_index], t[test_index]

        y_train = y_train.reshape((y_train.shape[0],))
        est3 = DROrthoForest(model_Y=Lasso(alpha=0.01),
                            propensity_model=LogisticRegression(C=1),
                            model_Y_final=WeightedLassoCV(cv=3),
                            propensity_model_final=LogisticRegressionCV(cv=3),
                            n_trees=1000, min_leaf_size=10)
        est3.fit(y_train, t_train, X=x_train)
        print(str(np.mean(est3.effect(x_test))))
        f.write(str(idx))
        f.write(str(np.mean(est3.effect(x_test))))
        f.write("#################")

def run_full(simulation_file):
    x = load_and_format_covariates_mbqip(simulation_file)
    t, y = load_all_other_crap(simulation_file)
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=5000, random_state=1) #5000 per 50000 cases, 10%
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]
    y_train = y_train.reshape((y_train.shape[0],))
    est3 = DROrthoForest(model_Y=Lasso(alpha=0.01),
                        propensity_model=LogisticRegression(C=1),
                        model_Y_final=WeightedLassoCV(cv=3),
                        propensity_model_final=LogisticRegressionCV(cv=3),
                        n_trees=1000, min_leaf_size=10)
    est3.fit(y_train, t_train, X=x_train)
    print(str(np.mean(est3.effect(x_test))))
    f.write(str(np.mean(est3.effect(x_test))) + "\n")
    f.write("#################")
    
def main():
    """
    f.write("\nBand\n")
    run_full("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/Band/mbqip_0.csv")
    f.close()
    f.write("\nBPD_DS")    
    run_full("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/BPD_DS/mbqip_2.csv")
    f.close()
    """
    f.write("\nRYGB")
    run_full("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/RYGB/mbqip_0.csv")
    f.close()
    """
    f.write("\nSADI-S")
    run_full("/Users/jiaweizhang/med/dragonnet/dat/mbqip/csv/diff_BMI/SADI-S/mbqip_0.csv")
    f.close()
    """
if __name__ == '__main__':
    main()
    #turn_knob("/home/zc2157/zc2157/dragonnet/dat/mbqip/csv/diff_BMI/BPD_DS", "dragonnet", "/home/zc2157/zc2157/dragonnet/BPD_DS")
    #turn_knob("/home/zc2157/zc2157/dragonnet/dat/mbqip/csv/diff_BMI/RYGB", "dragonnet", "/home/zc2157/zc2157/dragonnet/RYGB")
    #turn_knob("/home/zc2157/zc2157/dragonnet/dat/mbqip/csv/diff_BMI/SADI-S", "dragonnet", "/home/zc2157/zc2157/dragonnet/SADI-S")
