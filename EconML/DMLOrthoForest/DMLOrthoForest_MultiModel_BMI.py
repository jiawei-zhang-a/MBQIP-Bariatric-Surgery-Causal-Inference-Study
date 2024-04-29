from econml.dml import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import sys

sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    print("DMLOrthoForest with random forest")
    est = DMLOrthoForest(model_T=RandomForestRegressor(random_state=123),
                         model_Y=RandomForestRegressor(random_state=123))
    print(mbqip_utils.run_mbqip(est, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))
    
    print("DMLOrthoForest with lightgbm")
    est = DMLOrthoForest(model_T=lgb.LGBMRegressor(),
                         model_Y=lgb.LGBMRegressor())
    print(mbqip_utils.run_mbqip(est, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))
    
    print("DMLOrthoForest with xgboost")
    est = DMLOrthoForest(model_T=xgb.XGBRegressor(),
                         model_Y=xgb.XGBRegressor())
    print(mbqip_utils.run_mbqip(est, "/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))

if __name__ == '__main__':
    main()
