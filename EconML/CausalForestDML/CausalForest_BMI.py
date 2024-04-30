from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

PATH = "/scratch/jz4721/SCI/"

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    print("CausalForestDML with random forest")
    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        )
                      
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))
    
    print("CausalForestDML with lightgbm")
    est = CausalForestDML(model_y=lgb.LGBMRegressor(verbosity = -1),
                        model_t=lgb.LGBMRegressor(verbosity = -1),
    )
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

    print("CausalForestDML with xgboost")
    est = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor(),
    )
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

if __name__ == '__main__':

    main()
