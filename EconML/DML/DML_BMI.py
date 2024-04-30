from econml.dml import LinearDML, DML
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import sys
import numpy as np
np.random.seed(0)

sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils  # Adjust to your specific module and function

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")

    run_linear_dml_with_model(RandomForestRegressor(), "Random Forest")
    run_linear_dml_with_model(lgb.LGBMRegressor(verbosity = -1), "LightGBM")
    run_linear_dml_with_model(xgb.XGBRegressor(), "XGBoost")

PATH = "../../"
def run_linear_dml_with_model(model, model_name):
    print(f"LinearDML with {model_name}")
    est = LinearDML(model_y=model, model_t=model)
    print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))
    #print(f"DML with {model_name}")
    #est = DML(model_y=model, model_t=model, model_final=model) Only point estimates are returned
    #print(mbqip_utils.run_mbqip(est, PATH + "data/mbqip/csv/BMI"))
    
if __name__ == '__main__':
    main()
