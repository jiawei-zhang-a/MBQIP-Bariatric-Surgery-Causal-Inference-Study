from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils

def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)
    mbqip_utils.run_mbqip(est,"/scratch/jz4721/dragonnet/dat/mbqip/csv/BMI")
    
if __name__ == '__main__':
    main()