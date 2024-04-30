import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("DMLOrthoForest with random forest")
    est = DMLOrthoForest(model_T=RandomForestRegressor(),
                         model_Y=RandomForestRegressor())
    print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))
    
    print("DMLOrthoForest with lightgbm")
    est = DMLOrthoForest(model_T=lgb.LGBMRegressor(verbosity = -1),
                         model_Y=lgb.LGBMRegressor(verbosity = -1))
    print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))
    
    print("DMLOrthoForest with xgboost")
    est = DMLOrthoForest(model_T=xgb.XGBRegressor(),
                         model_Y=xgb.XGBRegressor())
    print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()
