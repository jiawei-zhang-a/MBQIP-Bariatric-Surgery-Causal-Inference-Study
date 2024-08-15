import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import mbqip_read_run as mbqip_utils
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    
    # model propensity and outcome by random forest
    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        )
    print("CausalForestDML with random forest")                  
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

    # model propensity and outcome by gradient boosting
    est = CausalForestDML(model_y=lgb.LGBMRegressor(verbosity = -1),
                        model_t=lgb.LGBMRegressor(verbosity = -1),
    )

    print("CausalForestDML with lightgbm")
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

    # model propensity and outcome by XGBoost
    est = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor(),
    )
    print("CausalForestDML with xgboost")
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

if __name__ == '__main__':

    main()
