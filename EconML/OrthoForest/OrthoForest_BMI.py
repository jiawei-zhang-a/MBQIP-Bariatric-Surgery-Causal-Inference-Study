from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
from econml.orf import DROrthoForest
from sklearn.model_selection import train_test_split
from econml.dr import ForestDRLearner
import numpy as np
import scipy.special
import glob
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

f = open('new.txt', 'w')

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


def run_mbqip(data_base_dir='/'):

    simulation_files = sorted(glob.glob("{}/*.csv".format(data_base_dir)))
    ans = []
    for idx, simulation_file in enumerate(simulation_files):

        x = load_and_format_covariates_mbqip(simulation_file)
        t, y = load_all_other_crap(simulation_file)
        print(t)
        est = DROrthoForest(model_Y = RandomForestRegressor())
        est.fit(y,t,X = x,W=None)

        print(est.ate(x))
        print(est.ate_interval(x))

    return ans

def main():
    PATH = "/scratch/jz4721/dragonnet/"
    run_mbqip(PATH + "dat/mbqip/csv/BMI")

if __name__ == '__main__':
    main()