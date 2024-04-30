from econml.dml import LinearDML, DML
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import sys
import numpy as np
np.random.seed(0)

sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk  # Adjust to your specific module and function
#PATH = "/scratch/jz4721/Observational-Study"
PATH = "../../"

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")

    run_linear_dml_for_each_outcome_and_model(RandomForestRegressor(), "Random Forest")
    run_linear_dml_for_each_outcome_and_model(lgb.LGBMRegressor(verbosity = -1), "LightGBM")
    run_linear_dml_for_each_outcome_and_model(xgb.XGBRegressor(), "XGBoost")

def run_linear_dml_for_each_outcome_and_model(model, model_name):
    outcomes = ["Death", "intervention", "readmission", "reoperation"]
    for outcome in outcomes:
        print(f"\nLinearDML with {model_name} for {outcome}")
        est = LinearDML(model_y=model, model_t=model)
        print(mbqip_risk.run_mbqip_risk(est, f"{PATH}/data/mbqip/csv/{outcome}"))
    #for outcome in outcomes:
        #print(f"\nrDML with {model_name} for {outcome}")
        #est = DML(model_Y=model, model_T=model)
        #mbqip_risk.run_mbqip_risk(est, f"{PATH}/data/mbqip/csv/{outcome}")

if __name__ == '__main__':
    main()
