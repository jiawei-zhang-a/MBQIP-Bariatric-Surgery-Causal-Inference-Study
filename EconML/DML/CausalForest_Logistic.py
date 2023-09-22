
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
import os
import mbqip_bootstrap as mbqip_bootstrap
import lightgbm as lgb
import xgboost as xgb


def main():

    est1 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est2 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est3 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())   
    est4 = CausalForestDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestRegressor())       

    PATH = "/scratch/jz4721/Observational-Study"
    print("\list \n(1)RYGB\n(2)Band\n(3)BPD-DS\n(4)SADI-S \nrelative treatment effect")
    print("causal forest with random forest")
    print("\nDeath")
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death")

    print("\nintervention")    
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention")

    print("\nreadmission") 
    mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission")

    print("\nreoperation") 
    mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation")
    
    print("\ncausal forest with lightgbm")
    est1 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est2 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est3 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est4 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    print("\nDeath")
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death")

    print("\nintervention")    
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention")

    print("\nreadmission") 
    mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission")

    print("\nreoperation") 
    mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation")

    print("\ncausal forest with xgboost")
    est1 = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor())
    est2 = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor())
    est3 = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor())
    est4 = CausalForestDML(model_y=xgb.XGBRegressor(),
                        model_t=xgb.XGBRegressor())
    print("\nDeath")
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death")

    print("\nintervention")
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention")

    print("\nreadmission")
    mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission")

    print("\nreoperation")
    mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation")

        
    


if __name__ == '__main__':
    main()


