import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import mbqip_read_run as mbqip_utils
import numpy as np
np.random.seed(0)

def main():
    
    est = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor(), # model propensity and outcome by random forest
                        criterion='mse', n_estimators=1000,
                        min_impurity_decrease=0.001,
                        )
    
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \BMI treatment effect\n")
    print("CausalForestDML with random forest")                  
    print(mbqip_utils.run_mbqip(est,PATH + "data/mbqip/csv/BMI"))

if __name__ == '__main__':

    main()
