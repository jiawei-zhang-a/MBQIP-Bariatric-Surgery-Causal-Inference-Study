import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.orf import DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor
import mbqip_risk_rate as mbqip_risk
import mbqip_bootstrap as mbqip_bootstrap
import lightgbm as lgb
import xgboost as xgb
import numpy as np
np.random.seed(0)

def main():
    
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    
    print("DMLOrthoForest with random forest")
    run_for_each_outcome_and_model(RandomForestRegressor())
    
    print("DMLOrthoForest with lightgbm")
    run_for_each_outcome_and_model(lgb.LGBMRegressor(verbosity = -1))
    
    print("DMLOrthoForest with xgboost")
    run_for_each_outcome_and_model(xgb.XGBRegressor())

def run_for_each_outcome_and_model(model):
    outcomes = ["Death", "intervention", "readmission", "reoperation"]
    for outcome in outcomes:
        print(f"\n{outcome}")
        est = DMLOrthoForest(model_Y=model, model_T=model)
        print(mbqip_risk.run_mbqip_risk(est, f"{PATH}data/mbqip/csv/{outcome}"))
        
if __name__ == '__main__':
    main()
