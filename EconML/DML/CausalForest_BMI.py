from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_read_run as mbqip_utils
import lightgbm as lgb


def main():
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    print("CausalForestDML with random forest")
    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(),
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        random_state=123)
                      
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))
    
    print("CausalForestDML with lightgbm")
    est = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor(),
    )
    print(mbqip_utils.run_mbqip(est,"/scratch/jz4721/Observational-Study/data/mbqip/csv/BMI"))
if __name__ == '__main__':

    main()
