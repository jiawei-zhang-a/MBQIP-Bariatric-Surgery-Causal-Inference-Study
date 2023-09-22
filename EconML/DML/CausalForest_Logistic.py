
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('../../utils')
import mbqip_risk_rate as mbqip_risk
import os
import mbqip_bootstrap as mbqip_bootstrap
import lightgbm as lgb


def main():

    if len(sys.argv) == 2:
        task_id = int(sys.argv[1])
    else:
        print("Please add the job number like this\nEx.python CausalForest_logistic.py 1")
        exit()

    if os.path.exists("Result") == False:
        os.mkdir("Result")

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
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death", "Death", task_id)

    print("\nintervention")    
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention", "intervention", task_id)

    print("\nreadmission") 
    mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission", "readmission", task_id)

    print("\nreoperation") 
    mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation", "reoperation", task_id)
    
    print("\ncausal forest with lightgbm")
    est1 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est2 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est3 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    est4 = CausalForestDML(model_y=lgb.LGBMRegressor(),
                        model_t=lgb.LGBMRegressor())
    mbqip_risk.run_mbqip_risk(est1, PATH + "/data/mbqip/csv/Death", "Death", task_id)

    print("\nintervention")    
    mbqip_risk.run_mbqip_risk(est2, PATH + "/data/mbqip/csv/intervention", "intervention", task_id)

    print("\nreadmission") 
    mbqip_risk.run_mbqip_risk(est3, PATH + "/data/mbqip/csv/readmission", "readmission", task_id)

    print("\nreoperation") 
    mbqip_risk.run_mbqip_risk(est4, PATH + "/data/mbqip/csv/reoperation", "reoperation", task_id)
    


if __name__ == '__main__':
    main()


