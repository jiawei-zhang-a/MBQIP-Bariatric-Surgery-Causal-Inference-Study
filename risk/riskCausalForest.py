
import sys
PATH = "/scratch/jz4721/SCI/"
sys.path.append(PATH+'utils')
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

import mbqip_risk_rate as mbqip_risk
import os
import mbqip_bootstrap as mbqip_bootstrap
import lightgbm as lgb

np.random.seed(0)

def main():

    est1 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est2 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est3 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est4 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())       

    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    print("causal forest with random forest")
    print("\nDeath")
    print(mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death"))

    print("\nintervention")    
    print(mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention"))

    print("\nreadmission") 
    print(mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission"))

    print("\nreoperation") 
    print(mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation"))
        

if __name__ == '__main__':
    main()


