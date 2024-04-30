from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
from econml.dr import ForestDRLearner, LinearDRLearner
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

PATH = "/scratch/jz4721/SCI/"

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("ForestDRLearner with random forest")
    est1 = ForestDRLearner(model_regression=RandomForestRegressor(),
                               model_propensity=RandomForestClassifier())
    print(mbqip_utils.run_mbqip(est1, PATH+"data/mbqip/csv/BMI"))

    print("ForestDRLearner with lightgbm")
    est1 = ForestDRLearner(model_regression=lgb.LGBMRegressor(verbosity = -1),
                            model_propensity=lgb.LGBMClassifier(verbosity = -1))  
    print(mbqip_utils.run_mbqip(est1, PATH+"data/mbqip/csv/BMI"))

    print("ForestDRLearner with xgboost")
    est3 = ForestDRLearner(model_regression=xgb.XGBRegressor(),
                            model_propensity=xgb.XGBClassifier())
    print(mbqip_utils.run_mbqip(est3, PATH+"data/mbqip/csv/BMI"))


if __name__ == '__main__':
    main()

