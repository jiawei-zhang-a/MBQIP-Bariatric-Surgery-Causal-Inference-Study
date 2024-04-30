import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
import mbqip_read_run as mbqip_utils
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.linear_model import Lasso
from econml.sklearn_extensions.linear_model import  WeightedLasso

np.random.seed(0)

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("DMLOrthoForest")
    est = DMLOrthoForest()
    print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()
